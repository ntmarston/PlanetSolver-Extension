import pandas as pd
import numpy as np
from astropy import units as u
from scipy.interpolate import PchipInterpolator

class evolution_track:

    df = pd.DataFrame()

    def __init__(self):
        pass

    @staticmethod
    def load_track(efile):
        input_col_names = ['Age', 'Atmospheric Entropy', 'Radius', 'Mass', 'delta_M', 'AMF', 'T_eff']
        with open(efile, 'r') as file:
            lines = file.readlines()
        data_start_idx = None
        header_lines = []
            
        for i, line in enumerate(lines):
            line = line.strip()
            # Look for the column headers
            if 'Age (Myr)' in line and 'Entropy' in line and 'Radius' in line:
                data_start_idx = i+1 #skips heading column
                break
            else:
                header_lines.append(line)

        evodf = pd.read_table(efile, skiprows=data_start_idx, names=input_col_names, sep=" ")
        for i in range(0, len(evodf)):
            density = evolution_track.__pl_density_cgs(radius=evodf["Radius"][i], mass=evodf["Mass"][i])
            evodf.at[i, "Density"] = density.value

        evodf = evolution_track.make_age_monotone(evodf)

        return evodf


    @staticmethod
    def make_age_monotone(df):
        i = 0
        while i < len(df)-1:
            if df.at[i, "Age"] == df.at[i+1, "Age"]:
                steps_to_unique = 0
                for j in range(i, len(df)):
                    if df.at[j, "Age"] == df.at[i, "Age"]:
                        steps_to_unique += 1
                    elif (i+j) == len(df)-1:
                        break
                    else:
                        break
                
                next_unique = df.at[i+steps_to_unique, "Age"]
                timestep = next_unique - df.at[i, "Age"]# Time entries are not evenly spaced in the first place, and the duplicates are probably a precision issue.
                Delta_T = min(1, (timestep-1e-4))  # Prefer spacing the points out over 1Myr unless timestep is smaller
                dt_approx = np.linspace(0, Delta_T, steps_to_unique) # Probably best approximated by logspace but don't need that much accuracy here to break the degeneracy

                for j in range(0, steps_to_unique):
                    adjusted_age = df.at[(i+j), "Age"] + dt_approx[j]
                    df.at[(i+j), "Age"] = adjusted_age
                    
                    #print(f"Adjusted age at {i}+{j} = {adjusted_age:.5f}")

                i = i + steps_to_unique
                #print(f"GOTO next unique at {df.at[i, "Age"]}")


            else:
                i += 1

        
        x = df["Age"]
        dx = np.diff(x)
        assert len(np.where(dx <= 0)[0]) < 1, "Failed to convert to monotonic"

        return df

    @staticmethod
    def __pl_density_cgs(radius, mass, rad_err = None, mass_err = None):
        """
        Calculate the bulk density of a planet in cgs units (g/cm³) with error propagation.

        Parameters
        ----------
        radius : float or `~astropy.units.Quantity`
            Planetary radius. If a float is provided, it is assumed to be in Earth radii.
            If a `Quantity` is provided, it must be compatible with length units.

        mass : float or `~astropy.units.Quantity`
            Planetary mass. If a float is provided, it is assumed to be in Earth masses.
            If a `Quantity` is provided, it must be compatible with mass units.

        rad_err : float, array-like, `~astropy.units.Quantity`, or None, optional
            Uncertainty in planetary radius. If a float is provided, it is assumed to be
            in the same units as radius. If array-like of length 2, treated as
            asymmetric errors [lower, upper] and averaged for symmetric approximation.
            If None, no error propagation is performed.

        mass_err : float, array-like, `~astropy.units.Quantity`, or None, optional
            Uncertainty in planetary mass. If a float is provided, it is assumed to be
            in the same units as mass. If array-like of length 2, treated as
            asymmetric errors [lower, upper] and averaged for symmetric approximation.
            If None, no error propagation is performed.

        Returns
        -------
        density : `~astropy.units.Quantity`
            Planetary bulk density in grams per cubic centimeter (g/cm³).

        density_err : `~astropy.units.Quantity`, array, or None
            Uncertainty in planetary bulk density in g/cm³. Returns None if neither
            rad_err nor mass_err is provided. If input errors are asymmetric (length-2 arrays),
            returns asymmetric errors as array [lower_error, upper_error]. Otherwise returns
            symmetric error as scalar Quantity.

        Notes
        -----
        - If both inputs are floats, the function assumes they are in Earth units.
        - Automatically converts units to CGS (centimeters and grams) before computing.
        - Volume is computed assuming a spherical planet.
        - Error propagation uses standard uncertainty propagation formulas for
            density = mass / ((4/3) * π * radius³). For asymmetric input errors,
            the method properly propagates them to asymmetric density errors by
            considering the sign of partial derivatives.
        """
        #----Unit handling-----#
        # Handle radius units
        if not isinstance(radius, u.Quantity):
            radius = radius * u.R_earth  # assume in Earth radii

        # Handle mass units
        if not isinstance(mass, u.Quantity):
            mass = mass * u.M_earth  # assume in Earth masses

        # Handle radius error units
        if rad_err is not None:
            if not isinstance(rad_err, u.Quantity):
                # Assume same units as radius before conversion
                if not isinstance(radius, u.Quantity):
                    rad_err = rad_err * u.R_earth
                else:
                    rad_err = rad_err * radius.unit

        # Handle mass error units
        if mass_err is not None:
            if not isinstance(mass_err, u.Quantity):
                # Assume same units as mass before conversion
                if not isinstance(mass, u.Quantity):
                    mass_err = mass_err * u.M_earth
                else:
                    mass_err = mass_err * mass.unit

        # Convert to cgs
        radius_cgs = radius.to(u.cm)
        mass_cgs = mass.to(u.g)
        mass_err_cgs = None
        rad_err_cgs = None
        if rad_err is not None:
            rad_err_cgs = rad_err.to(u.cm)
        if mass_err is not None:
            mass_err_cgs = mass_err.to(u.g)

        # Calculate density
        V = (4 / 3) * np.pi * radius_cgs ** 3
        density = mass_cgs / V

        # Calculate error if requested
        density_err = None
        if rad_err is not None or mass_err is not None:
            # Partial derivatives for error propagation
            # ρ = M / ((4/3)πR³)
            # ∂ρ/∂M = 1 / ((4/3)πR³) = ρ/M
            # ∂ρ/∂R = -3M / ((4/3)πR⁴) = -3ρ/R

            # Check if we have asymmetric errors
            mass_is_asymmetric = mass_err is not None and hasattr(mass_err_cgs, '__len__') and len(mass_err_cgs) == 2
            rad_is_asymmetric = rad_err is not None and hasattr(rad_err_cgs, '__len__') and len(rad_err_cgs) == 2

            if mass_is_asymmetric or rad_is_asymmetric:
                # Handle asymmetric error propagation
                dρ_dM = density / mass_cgs if mass_err is not None else 0
                dρ_dR = -3 * density / radius_cgs if rad_err is not None else 0

                # Calculate lower and upper bounds
                err_lower_terms = []
                err_upper_terms = []

                if mass_err is not None:
                    if mass_is_asymmetric:
                        # For mass: positive derivative, so lower mass error -> lower density error
                        err_lower_terms.append((dρ_dM * mass_err_cgs[0]) ** 2)
                        err_upper_terms.append((dρ_dM * mass_err_cgs[1]) ** 2)
                    else:
                        # Symmetric mass error
                        mass_term = (dρ_dM * mass_err_cgs) ** 2
                        err_lower_terms.append(mass_term)
                        err_upper_terms.append(mass_term)

                if rad_err is not None:
                    if rad_is_asymmetric:
                        # For radius: negative derivative, so lower radius error -> upper density error
                        err_lower_terms.append((dρ_dR * rad_err_cgs[1]) ** 2)  # Note: switched indices
                        err_upper_terms.append((dρ_dR * rad_err_cgs[0]) ** 2)  # Note: switched indices
                    else:
                        # Symmetric radius error
                        rad_term = (dρ_dR * rad_err_cgs) ** 2
                        err_lower_terms.append(rad_term)
                        err_upper_terms.append(rad_term)

                density_err_lower = np.sqrt(sum(err_lower_terms))
                density_err_upper = np.sqrt(sum(err_upper_terms))

                # Return as array [lower, upper] with proper units
                density_err = np.array([density_err_lower.value, density_err_upper.value]) * density.unit

            else:
                # Handle symmetric error propagation
                err_terms = []

                if mass_err is not None:
                    dρ_dM = density / mass_cgs
                    err_terms.append((dρ_dM * mass_err_cgs) ** 2)

                if rad_err is not None:
                    dρ_dR = -3 * density / radius_cgs
                    err_terms.append((dρ_dR * rad_err_cgs) ** 2)

                if err_terms:
                    density_err = np.sqrt(sum(err_terms))

        if density_err is None:
            return density
        else:
            return density, density_err
        
    @staticmethod
    def calc_observable_timescale(df, dfcontrol, impact_time):
        obs_pchip = PchipInterpolator(x=df["Age"], y=df["Radius"])
        control_pchip = PchipInterpolator(x=dfcontrol["Age"], y=dfcontrol["Radius"])
        modelAge = np.linspace(impact_time, 14000, 400)
        observability_threshold = control_pchip(modelAge) + 0.5 #Change based on the instrumental uncertainty that we choose
        if len(np.where(obs_pchip(modelAge) > observability_threshold)[0]) == 0:
            return 0

        zerocurve = obs_pchip(modelAge) - observability_threshold
        
        zero_crossings = np.where(np.diff(np.sign(zerocurve)))[0]
        if len(zero_crossings) == 1: #If the planet does not ever cool below the threshold
            observable_timescale = 14000-impact_time
        observable_timescale = modelAge[zero_crossings[1]]-modelAge[zero_crossings[0]]

        return observable_timescale