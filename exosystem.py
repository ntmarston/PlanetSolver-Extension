"""
Filename: exosystem.py
Author: Nicholas Marston
Date: 2025-07-23
Version: 1.0
Description:
    Tool for visualization and analysis of exoplanet systems, pulls data from the IPAC Exoplanet Archive and SIMBAD.
    Written for python version 3.13

License: GNU GPL 3.0
Contact: ntmarston@gmail.com
Dependencies: astropy.units, astropy.constants, astroquery, matplotlib, math, numpy

References:
[1]Howe AR, Becker JC, Stark CC, Adams FC (2025) Architecture Classification for Extrasolar Planetary Systems. AJ 169:149.
https://doi.org/10.3847/1538-3881/adabdb
[2] R. kumar Kopparapu, R. M. Ramirez, J. SchottelKotte, J. F. Kasting, S. Domagal-Goldman, and V. Eymet, “Habitable Zones Around Main-Sequence Stars: Dependence on Planetary Mass,”
ApJ, vol. 787, no. 2, p. L29, May 2014, doi: 10.1088/2041-8205/787/2/L29.

"""

import math
import matplotlib.axes
import pandas as pd
import re
from astropy import units as u
from astropy.units import Quantity, UnitTypeError
from astropy.constants import R_earth, M_earth, R_sun, M_sun, G
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.simbad import Simbad
import matplotlib.pyplot as plt
from math import gcd, pi
import random
import numpy as np


class Planet:

    def __init__(self, name: str, properties: dict = None, mass = None, radius = None,
                 orbital_period=None, density = None, system=None):
        """
        Constructor for Planet object. Allows manual assignment for mass, radius, orbital_period, density, system.
        Prefers parameters to be passed as a dictionary where entries correspond to Exoplanet Archive ps/pscomppars table column names.
        Expected parameters are pl_bmasse, pl_dens, pl_rade, pl_orbper, hostname, pl_letter. These and any associated
        errors are stored in the extended_properties attribute.
        Properties dict entries which do not have keys matching one of the preceding values will be stored in
        the custom_properties attribute. Any keys matching the pattern '<property>_reflink' will be stored in the
        reference_map attribute
        as the reference entry for '<property>'.

        :param name: Planet name
        :param properties: Dict containing planetary parameters in form {['param name']: value}.
        Intended use requires 'param name' to match a column in the Exoplanet Archive ps table
        :param mass: Mass in EarthMass. Can be set with the pl_bmasse key in the properties dict
        :param radius: Radius in EarthRadii. Can be set with the pl_rade key in the properties dict
        :param orbital_period: Orbital Period in days. Can be set with the pl_orbper key in the properties dict.
        :param density: Density in g/cm3. Can be set with the pl_dens key in the properties dict.

        """

        if properties is None:
            properties = {}
        self.name = name
        self.system = None #parent system
        self.is_resonant = False #Default value
        #Store extended parameter set as dictionary
        self.extended_properties = {}
        #Store references for specific values in the form d[parameter] = reflink
        self.reference_map = {}
        #For specific un-thought-of use cases
        self.custom_properties = {}
        #Dict of boolean special attributes such as super-puff, resonant, in habitable zone, etc
        self.spec_attributes = {}

        #Unpack input dictionary
        for key, value in properties.items():
            match key:
                case 'pl_bmasse':
                    try:
                        value = self.__ensure_unit(value, u.M_earth)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.mass = value
                    self.extended_properties[key] = value

                case "pl_bmasseerr1":
                    try:
                        value = self.__ensure_unit(value, u.M_earth)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.extended_properties[key] = value

                case "pl_bmasseerr2":
                    try:
                        value = self.__ensure_unit(value, u.M_earth)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.extended_properties[key] = value

                case "pl_rade":
                    try:
                        value = self.__ensure_unit(value, u.R_earth)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.extended_properties[key] = value
                    self.radius = value

                case "pl_radeerr1":
                    try:
                        value = self.__ensure_unit(value, u.R_earth)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.extended_properties[key] = value
                case "pl_radeerr2":
                    try:
                        value = self.__ensure_unit(value, u.R_earth)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.extended_properties[key] = value


                case "pl_dens":
                    try:
                        if value is None:
                            self.extended_properties[key] = value
                            self.density = value
                        elif not isinstance(value, Quantity):
                            value = value * (u.g / u.cm ** 3)
                        elif not value.unit.is_equivalent(u.g / u.cm ** 3):
                            value = value.value * (u.g / u.cm ** 3)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.extended_properties[key] = value
                    self.density = value

                case "pl_denserr1":
                    try:
                        if value is None:
                            self.extended_properties[key] = value
                            self.density = value
                        elif not isinstance(value, Quantity):
                            value = value * (u.g / u.cm ** 3)
                        elif not value.unit.is_equivalent(u.g / u.cm ** 3):
                            value = value.value * (u.g / u.cm ** 3)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.extended_properties[key] = value

                case "pl_denserr2":
                    try:
                        if value is None:
                            self.extended_properties[key] = value
                            self.density = value
                        elif not isinstance(value, Quantity):
                            value = value * (u.g / u.cm ** 3)
                        elif not value.unit.is_equivalent(u.g / u.cm ** 3):
                            value = value.value * (u.g / u.cm ** 3)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.extended_properties[key] = value



                case "pl_orbper":
                    try:
                        value = self.__ensure_unit(value, u.d)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.extended_properties[key] = value
                    self.orbital_period = value

                case "pl_orbpererr1":
                    try:
                        value = self.__ensure_unit(value, u.d)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.extended_properties[key] = value

                case "pl_orbpererr2":
                    try:
                        value = self.__ensure_unit(value, u.d)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                            f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.extended_properties[key] = value


                case "hostname":
                    self.extended_properties[key] = value
                    self.system = value


                case "pl_letter":
                    self.extended_properties[key] = value

                case "discoverymethod":
                    self.extended_properties[key] = value
                case "system": #for manual overwrite
                    self.extended_properties["hostname"] = value
                    self.system = value
                case _:
                    if "reflink" in key:
                        html_ref = value
                        href_match = re.search(r'href=([^\s>]+)', html_ref)
                        text_match = re.search(r'>([^<]+)<', html_ref)
                        if href_match and text_match:
                            link = href_match.group(1)
                            text = text_match.group(1).strip()
                            ref_string = f"{text} ({link})"
                        else:
                           ref_string = value
                        self.reference_map[key.replace("_reflink", "")] = ref_string
                    else:
                        self.custom_properties[key] = value



        #basic properties for easy access, specified properties in constructor overwrite values in the dictionary
        try:
            mass = self.__ensure_unit(mass, u.M_earth)
        except UnitTypeError as uce:
            raise UnitTypeError(f"Mass cannot be provided as {mass.unit}, "
                                f"values must be provided in units with compatible dimensions, or as generic numbers")
        except ValueError as ve:
            raise ValueError(f"Mass cannot be provided as {type(mass)}, "
                             f"values must be provided in units with compatible dimensions, or as generic numbers")
        if not mass is None:
            self.mass = mass
            self.extended_properties['pl_bmasse'] = mass

        try:
            radius = self.__ensure_unit(radius, u.R_earth)
        except UnitTypeError as uce:
            raise UnitTypeError(f"Radius cannot be provided as {radius.unit}, "
                                f"values must be provided in units with compatible dimensions, or as generic numbers")
        except ValueError as ve:
            raise ValueError(f"Radius cannot be provided as {type(radius)}, "
                             f"values must be provided in units with compatible dimensions, or as generic numbers")
        if not radius is None:
            self.radius = radius
            self.extended_properties['pl_rade'] = radius

        try:
            orbital_period = self.__ensure_unit(orbital_period, u.d)
        except UnitTypeError as uce:
            raise UnitTypeError(f"Orbital Period cannot be provided as {orbital_period.unit}, "
                                f"values must be provided in units with compatible dimensions, or as generic numbers")
        except ValueError as ve:
            raise ValueError(f"Orbital Period cannot be provided as {type(orbital_period)}, "
                             f"values must be provided in units with compatible dimensions, or as generic numbers")
        if not orbital_period is None:
            self.extended_properties['pl_orbper'] = orbital_period
            self.orbital_period = orbital_period

        if self.system is None and system is not None:
            try:
                self.system = system
            except Exception as e:
                print(f"Could not build system {system}")


        if density is not None:
            try:
                if not isinstance(density, Quantity):
                    density = density * (u.g / u.cm ** 3)
                if not density.unit.is_equivalent(u.g / u.cm ** 3):
                    density = density.value * (u.g / u.cm ** 3)
            except UnitTypeError as uce:
                raise UnitTypeError(f"Density cannot be provided as {density.unit}, "
                                    f"values must be provided in units with compatible dimensions, or as generic numbers")
            except ValueError as ve:
                raise ValueError(f"Density cannot be provided as {type(density)}, "
                                 f"values must be provided in units with compatible dimensions, or as generic numbers")

        if density is not None:
            self.extended_properties['pl_dens'] = density
            self.density = None

        if self.density is None and self.mass is not None and self.radius is not None:
            if 'pl_radeerr1' in self.extended_properties.keys() and 'pl_radeerr2' in self.extended_properties.keys():
                rad_err = [self.extended_properties['pl_radeerr1'], self.extended_properties['pl_radeerr2']]

                if ('pl_bmasseerr1' in self.extended_properties.keys() and 'pl_bmasseerr2' in self.extended_properties.keys()):
                    mass_err = [self.extended_properties['pl_bmasseerr1'], self.extended_properties['pl_bmasseerr2']]

                    self.density, density_err = self.__pl_density_cgs(self.radius, self.mass, rad_err=rad_err, mass_err=mass_err)
                    self.extended_properties['pl_dens'] = self.density
                    self.extended_properties['pl_denserr1'] = density_err[0]
                    self.extended_properties['pl_denserr2'] = density_err[1]
            else:
                self.density = self.__pl_density_cgs(self.radius, self.mass)
                self.extended_properties['pl_dens'] = self.density

        self.extended_properties["base_class"] = self.__get_base_class()
        spec_attr = self.__get_special_attributes()
        if spec_attr is not None:
            self.spec_attributes = spec_attr

    def __str__(self):
        """
        Overrides default tostring
        """
        out = (
                f"Name: {self.name}\n"
                f"Host/System name: {self.system}\n"
                f"Type: {self.extended_properties["base_class"]} \n"
                f"Radius: {self.radius}\n"
                f"Mass: {self.mass}\n"
                f"Orbital Period: {self.orbital_period}\n"
                f"Density: {self.density}" + "\n"
        )

        if 'discoverymethod' in self.extended_properties.keys():
            out += f"Detected by: {self.extended_properties['discoverymethod']}"
        if self.spec_attributes is not None:
            for attr, value in self.spec_attributes.items():
                if value:
                    out += f"-> {attr}\n"

        return out

    def to_dict(self, include='all'):
        """Output planet object as dict.
        :param include: ('all', 'basic') If set to basic, only returns the planet name and extended_properties.
        """
        if include == 'basic':
            row = self.extended_properties
            row['pl_name'] = self.name
            return row

        else:
            row = dict(self.extended_properties, **self.spec_attributes)
            if self.custom_properties:
                row = dict(row, **self.custom_properties)
            row['pl_name'] = self.name
            return row


    def get_references(self):
        """Returns a table mapping observed/calculated values to their source/reference"""
        table = pd.DataFrame(columns=['Name', 'Reference'])
        table = table.set_index('Name')
        for key, value in self.reference_map.items():
            table.loc[key, 'Reference'] = value

        return table

    def __get_special_attributes(self):
        """
        Internal method.
        Used to check if this planet meets the criteria for any sort of special classification.
        Criteria for super-puff is (ρ<0.3gcm3, M<=30Mearth),
        and near-super-puff (ρ<0.3gcm3, 30<=M<55Mearth). Returns None if no special classifications match.

        :returns spec_attributes: Dictionary
        """
        spec_attributes = {'Super-puff':False,
                            'near-Super-puff':False,
                            'Hot Jupiter':False,
                            'USP':False
                            }

        sp_density_threshold = 0.3 * (u.g / u.cm**3)
        sp_mass_threshold = 30 * u.earthMass #not in accordance with [1]
        if not self.density is None:
            if self.density <= sp_density_threshold:
                if self.mass < sp_mass_threshold:
                    spec_attributes["Super-puff"] = True
                elif self.mass < (55 * u.earthMass):
                    spec_attributes["near-Super-puff"] = True

        if not self.orbital_period is None:
            if self.orbital_period.value < 1:
                spec_attributes["USP"] = True
            if self.orbital_period.value < 10 and self.extended_properties['base_class'] == "Jovian":
                spec_attributes["Hot Jupiter"] = True

        return spec_attributes

    def __get_base_class(self):
        """
        Internal method, classifies planets based on radius/mass into categories  vaguely based on the scheme used
        in [1]
        """
        #Prefers radius classification because it is more reliable, follows framework of [1]

        if (self.radius is None) and (self.mass is None):
            raise Exception("Cannot get base class for planet with no mass and no radius")
        base_class = None

        if isinstance(self.radius, u.Quantity):
            R_p = self.radius.value
        else:
            R_p = self.radius
        if isinstance(self.mass, u.Quantity):
            M_p = self.mass.value
        else:
            M_p = self.mass

        if not (self.radius is None):
            if R_p < 1:
                base_class = "sub-Earth"
            if 1 <= R_p < 1.75:
                base_class = "super-Earth"
            if 1.75 <= R_p < 3.5:
                base_class = "sub-Neptune"
            if 3.5 <= R_p < 6:
                base_class = "Neptune"
            if R_p >= 6:
                base_class = "Jovian"
                if not (self.mass is None):
                    if M_p >= 4133:
                        base_class = "Brown Dwarf"

            return base_class

    def get_system(self):
        """Gets the system object associated with the self.system attribute
        :returns system:"""
        system = Queries.build_system(system_name=self.system)

        return system

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
        V = (4 / 3) * pi * radius_cgs ** 3
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
    def __ensure_unit(x, unit: u.Unit):
        """Internal method to ensure input units are correct
        :param x: Value to check
        :param unit: Desired astropy.units instance
        """

        if x is None:
            return x
        if not isinstance(x, Quantity):
            x = x * unit
        elif x.unit != unit:
            try:
                x = x.to(unit)
            except u.UnitConversionError as uce:
                raise u.UnitTypeError(f"{x} cannot be converted to {unit}")
        return x

class Star:

    def __init__(self, name: str, properties: dict = None, spec_type=None, t_eff=None, radius=None, mass=None,
                 luminosity=None, age=None,
                 force_stellar_params=True):
        """
        Constructor for Star object. Allows manual assignment for mass, radius, spectral type, effective temperature,
        luminosity, age.
        Prefers parameters to be passed as a dictionary where entries correspond to Exoplanet Archive ps/pscomppars table column names.
        Expected parameters are st_teff, st_rad, st_mass, st_lum, st_age, st_refname. These and any associated
        errors are stored in the extended_properties attribute.
        Properties dict entries which do not have keys matching one of the preceding values will be stored in
        the custom_properties attribute. Any keys matching the pattern '<property>_reflink' will be stored in the
        reference_map attribute as the reference entry for '<property>'.
        :param spec_type: Morgan-Keenan spectral type (String)
        :param t_eff: Effective temperature (K)
        :param radius: Radius in solar radii
        :param mass: Mass in solar masses
        :param luminosity: In units of log10(solar) to match exoplanet archive format. Converted to non-logarithmic
        :param age: Age in Gyr
        value once stored.
        :param name: Star name
        :param force_stellar_params: (Default True) Query SIMBAD to fill in missing mass/luminosity values
        """
        self.extended_properties = {}
        self.custom_properties = {}
        self.reference_map = {}
        self.spec_type = None #default value required for using Simbad as a fallback
        if properties is None:
            properties = {}

        for key, value in properties.items():
            match key:
                case 'st_teff':
                    try:
                        value = self.__ensure_unit(value, u.K)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                            f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.t_eff = value
                    self.extended_properties[key] = value
                case 'st_rad':
                    try:
                        value = self.__ensure_unit(value, u.R_sun)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                            f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.radius = value
                    self.extended_properties[key] = value
                case 'st_mass':
                    try:
                        value = self.__ensure_unit(value, u.M_sun)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                            f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.mass = value
                    self.extended_properties[key] = value

                case 'st_lum':
                    try:
                        value = (10 ** value) * u.solLum

                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                            f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.luminosity = value
                    self.extended_properties[key] = value

                case 'st_age':
                    try:
                        value = self.__ensure_unit(value, u.Gyr)
                    except UnitTypeError as uce:
                        raise UnitTypeError(f"{key} cannot be provided as {value.unit}, "
                                            f"values must be provided in units with compatible dimensions, or as generic numbers")
                    except ValueError as ve:
                        raise ValueError(f"{key} cannot be provided as {type(value)}, "
                                         f"values must be provided in units with compatible dimensions, or as generic numbers")
                    self.age = value
                    self.extended_properties[key] = value

                case 'st_refname':
                    html_ref = value
                    href_match = re.search(r'href=([^\s>]+)', html_ref)
                    text_match = re.search(r'>([^<]+)<', html_ref)
                    if href_match and text_match:
                        link = href_match.group(1)
                        text = text_match.group(1).strip()
                        ref_string = f"{text} ({link})"
                    else:
                        ref_string = value
                    self.extended_properties[key] = ref_string
                    self.reference_map['primary'] = ref_string

                case 'st_spectype':
                    if isinstance(value, str): #Empty string is given when value is null in IPAC database
                        if len(value) == 0:
                            continue
                        self.extended_properties[key] = value
                        self.spec_type = value



                case 'st_spectype_simbad':
                    self.extended_properties[key] = value
                    if self.spec_type is None: #prefer IPAC spectype
                        self.spec_type = value
                        self.extended_properties['st_spectype_reflink'] = 'Simbad'

                case _:
                    if "reflink" in key:
                        html_ref = value
                        href_match = re.search(r'href=([^\s>]+)', html_ref)
                        text_match = re.search(r'>([^<]+)<', html_ref)
                        if href_match and text_match:
                            link = href_match.group(1)
                            text = text_match.group(1).strip()
                            ref_string = f"{text} ({link})"
                        else:
                            ref_string = value
                        self.reference_map[key.replace("_reflink", "")] = ref_string
                    else:
                        self.custom_properties[key] = value

        self.name = name
        if spec_type is not None:
            self.spec_type = spec_type  # Morgan-Keenan system
        if t_eff is not None:
            self.t_eff = self.__ensure_unit(t_eff, u.K)
        if radius is not None:
            self.radius = self.__ensure_unit(radius, u.R_sun)
        if mass is not None:
            self.mass = self.__ensure_unit(mass, u.M_sun)
        if luminosity is not None:
            self.luminosity = (10 ** luminosity) * u.solLum
        if age is not None:
            try:
                self.age = self.__ensure_unit((age * 1e9), u.Gyr)
            except TypeError as e:
                raise UnitTypeError(f"Age was given in as an incompatible type or unit: {type(age)}")


        self.id_map = pd.DataFrame()
        try:
            self.id_map = Queries.crossref_stellar_ids(id = self.custom_properties['gaia_id'], input_id_type='')
        except Exception as e:
            self.id_map = pd.DataFrame()

        if luminosity is None and force_stellar_params:
            try:
                table = NasaExoplanetArchive.query_criteria(table="stellarhosts",
                                                            where=f"hostname='{self.name}' and st_lum IS NOT null")
                l = table['st_lum'][0]

                self.luminosity = (10 ** l) * u.solLum

                table = NasaExoplanetArchive.query_criteria(table="stellarhosts",
                                                            where=f"hostname='{self.name}' and st_mass IS NOT null")
                m = table['st_mass'][0]

                self.mass = self.__ensure_unit(m, u.M_sun)
            except Exception as e:
                print(f"At least one parameter has no entries for this object: {e}")
                self.luminosity = None

    def __str__(self):
        """
        Overrides default tostring
        """
        return (f"Name: {self.name}\n"
        f"Host/Effective Temp: {self.t_eff}\n"
        f"Type: {self.spec_type}\n" 
        f"Luminosity: {self.luminosity}\n" 
        f"Radius: {self.radius}\n" 
        f"Mass: {self.mass}\n" 
        f"age: {self.age}\n"
        "Use <Star>.get_references() to view sources/references\n")

    def get_plot_color(self):
        """Internal method to determine what color to plot the star as"""
        if self.t_eff >= 33000 * u.K: #O
            return '#92B5FF' #Blue
        if self.t_eff >= 10000 * u.K: #B
            return '#A2C0FF' #Blueish white
        if self.t_eff >= 7300 * u.K: #A
            return 'ivory' #White
        if self.t_eff >= 6000 * u.K: #F
            return 'lemonchiffon' #Yellowish white
        if self.t_eff >= 5300 * u.K: #G
            return 'gold'
        if self.t_eff >= 3900 * u.K: #K
            return '#FFDAB5' #pale orange
        if self.t_eff >= 2300 * u.K: #M
            return 'orangered'
        else:
            return 'lightgrey'

    def get_references(self):
        table = pd.DataFrame(columns=['Name', 'Reference'])
        table = table.set_index('Name')
        for key, value in self.reference_map.items():
            table.loc[key, 'Reference'] = value

        return table

    @staticmethod
    def __ensure_unit(x, unit: u.Unit):
        if x is None:
            return x
        if not isinstance(x, Quantity):
            x = x * unit
        elif x.unit != unit:
            try:
                x = x.to(unit)
            except u.UnitConversionError as uce:
                raise u.UnitTypeError(f"{x} cannot be converted to {unit}")
        return x

class System:
    """A collection of star and planet objects. Intended primarily to be used with Queries.build_system()"""

    def __init__(self, name, star=None, planets=None, binary=False, custom_properties: dict = None):

        if planets is None:
            planets = []
        self.name = name
        self.star = star
        self.planets = sorted(planets, key=lambda p: p.orbital_period)
        self.num_planets = len(planets)
        self.binary = binary

        #Set default
        for p in self.planets:
            p.extended_properties['is_resonant'] = False
            p.is_resonant = False
        #Flip to True where resonances are found
        self.resonances = self.find_resonant_pairs()

        self.custom_properties = custom_properties if custom_properties is not None else {}

        if (not self.star is None) and (not self.star.luminosity is None) and (not self.star.mass is None) and (
        not self.star.t_eff is None):
            self.HZ_bounds = {0.1: self.find_hz(0.1), 1: self.find_hz(1), 5: self.find_hz(5)}
        else:
            self.HZ_bounds = None


    def __str__(self):
        """Overrides default tostring method"""
        out = (f"{self.name}\n-------------------\n"
               f"Spectral type: {self.star.spec_type}\n"
               f"Planets({self.num_planets}):\n")
        for p in self.planets:
            out += f"{p.name} ({p.extended_properties["base_class"]}): R={p.radius}, M={p.mass}, P={p.orbital_period}\n"
            if p.spec_attributes is not None:
                for attr, value in p.spec_attributes.items():
                    if value:
                        out += f"-> {attr}\n"
        out += "To view planetary parameter sources use <Planet Obj>.get_references()\n"

        out += "\nExtended Analytics\n"
        out += "---------------------\n"
        out += "Mean Motion Resonance (MMR) Information:\n"
        for i, resonance_dict in self.resonances.items():
            for pq, within in resonance_dict.items():
                out += f"Planet {i} and {i - 1} are within {(within * 100):.3f}% of {pq} resonance\n(index starts at 0)\n"
        if not self.HZ_bounds is None:
            out += (f"Estimated Habitable Zone Bounds:\n"
                    f"For 0.1 earthMass planet: [{str(self.HZ_bounds[0.1][0])}d - {str(self.HZ_bounds[0.1][1])}]\n"
                    f"For 1 earthMass planet: [{str(self.HZ_bounds[1][0])}d - {str(self.HZ_bounds[1][1])}]\n"
                    f"For 5 earthMass planet: [{str(self.HZ_bounds[5][0])}d - {str(self.HZ_bounds[5][1])}]\n"
                    f"Bounds are between point of runaway greenhouse and maximum greenhouse\n"
                    f"Calculations based on Kopparapu 2014 (https://arxiv.org/pdf/1404.5292)\n"
                    f"Code adapted from https://github.com/Eelt/HabitableZoneCalculator")

        out += "\n Additional/Custom properties can be accessed directly with <system_name>.custom_properties"
        return  out


    def find_resonant_pairs(self):
        """
        Identifies j:j-k mean motion resonance (MMR) pairs within the system. Only considers nearest-neighbor pairs.
        Will report resonant pairs up to 8:1. Resonant pairs are flagged if they are within 5% of a j:j-k MMR,
        defined as those which satisfy the condition:

        |Delta|=|(j - k) / j * (P_outer / P_inner) - 1| < 0.05


        :param self: System
        :returns: A nested dictionary, the outer having keys corresponding to planet pairs from shortest to longest
        orbital period, and the inner being a dictionary of resonances(keys) within 5% and the corresponding value of |Delta|
        """
        if self.planets is None:
            return None

        planets_sorted = sorted(self.planets, key=lambda p: p.orbital_period)
        resonant_pairs = {}
        for i in range(1, len(planets_sorted)):
            P_inner = planets_sorted[i - 1].orbital_period
            P_outer = planets_sorted[i].orbital_period
            resonance_matches = {}
            for j in range(1, 10):
                for k in range(1, j):
                    if gcd(j, k) >= 2: # Stop registering non-relatively-prime pairs
                        continue
                    DeltA = np.abs((j - k) / j * (P_outer / P_inner) - 1)
                    if DeltA < 0.05:
                        resonance_matches[f"{j}:{j - k}"] = float(DeltA)
                        if j <= 7:
                            planets_sorted[i].is_resonant = True
                            planets_sorted[i].extended_properties['is_resonant'] = True
                            planets_sorted[i-1].is_resonant = True
                            planets_sorted[i-1].extended_properties['is_resonant'] = True

            resonant_pairs[i] = resonance_matches

        return resonant_pairs


    def orb_per_from_au(self, au):
        a = au if isinstance(au, u.Quantity) else au * u.AU
        M = self.star.mass
        T = 2 * pi * np.sqrt((a.to(u.m) ** 3) / (G * M))

        return T.to(u.d)

    def au_from_orb_per(self, orb_per):
        if orb_per is None or self.star.mass is None:
            return None
        M = self.star.mass
        T = orb_per if isinstance(orb_per, u.Quantity) else orb_per * u.d

        a =  (( (T / (2*pi) ) ** 2) * G * M) ** (1/3)

        return a.to(u.AU)

    def find_hz(self, pl_mass):
        """Code adapted from https://github.com/Eelt/HabitableZoneCalculator
        pre-condition: 2600-7200K, pl_masse in (0.1, 5)


        L: Luminosity (Lsun)
        a: Semi-major axis (AU)
        T_eff: Star temperature

        """
        L = self.star.luminosity.value
        T_eff = self.star.t_eff.value
        T_s = (T_eff - 5780)

        # A. Find insolation flux from semi-major axis
        def __insolation_flux_from_a(a):
            return ((1 / a) ** 2) * L

        # B. Find semimajor axis from effective solar flux
        # From Kopparapu et al. 2014. Equation 5, Section 3.1, Page 9
        def __dist_from_Seff(Seff):
            au = (L / Seff) ** 0.5
            return au




        # directly from Eelt
        def Kopparapu2014(SeffSUN, a, b, c, d, tS):
            return SeffSUN + a * tS + b * ((tS) ** 2) + c * ((tS) ** 3) + d * ((tS) ** 4)

        def __find_greenhouse_bounds(temp, zone, pl_mass=5):
            a, b, c, d, Seff_solar = 0, 0, 0, 0, 0
            match zone:
                case "rg":  # runaway greenhouse
                    match pl_mass:
                        case 0.1:
                            Seff_solar = 0.99
                        case 1:
                            Seff_solar = 1.107
                        case 5:
                            Seff_solar = 1.188
                    a = 1.332 * (10 ** -4)
                    b = 1.580 * (10 ** -8)
                    c = -8.308 * (10 ** -12)
                    d = -1.931 * (10 ** -15)
                    return Kopparapu2014(Seff_solar, a, b, c, d, T_s)

                case "mg":
                    Seff_solar = 0.356
                    a = 6.171 * (10 ** -5)
                    b = 1.689 * (10 ** -9)
                    c = -3.198 * (10 ** -12)
                    d = -5.575 * (10 ** -16)
                    return Kopparapu2014(Seff_solar, a, b, c, d, T_s)

                case _:
                    raise ValueError("Invalid Zone")

        runawayGreenhouse = __find_greenhouse_bounds(T_eff, "rg", pl_mass)
        maximumGreenhouse = __find_greenhouse_bounds(T_eff, "mg", pl_mass)

        bounds_au = [__dist_from_Seff(runawayGreenhouse), __dist_from_Seff(maximumGreenhouse)]
        bounds_orbper = [self.orb_per_from_au(bounds_au[0]).value, self.orb_per_from_au(bounds_au[1]).value]

        return bounds_orbper

    #---Plotting methods-----#
    def __set_up_axes(self, y_pos, x_bounds):
        label = self.name
        fig, ax = plt.subplots(figsize=(10, 1.5))
        ax.axhspan(0.999 * y_pos, 1.001 * y_pos, xmin=-0.5, xmax=10, color="slategray")

        ax.set_yticks([y_pos])
        ax.set_yticklabels([label])
        ax.set_ylim(0.5, 1.5)
        ax.set_xlim(x_bounds[0], x_bounds[1])
        ax.set_xscale("log")

        return fig, ax

    def __plt_mark_HZ(self, ax, pl_masse=5, y_pos=1):
        lower_bound = self.HZ_bounds[5][0]
        upper_bound = self.HZ_bounds[5][1]
        ax.plot([lower_bound, upper_bound], [0.5, 0.5], color="springgreen",
                label=f"{5}earthMass HZ", linewidth=10)
        ax.annotate("HZ Estimate", [lower_bound, 0.5], textcoords='offset pixels', xytext=(0, 0), fontsize=6)

    def plot(self, ax: matplotlib.axes.Axes = None, options=None):
        """
        Plots a representation of the system using matplotlib. Each base class of planet (e.g. sub-Earth, super-Earth...)
        is plotted as a different size and different color. Super-puffs are plotted in the style of [1].


        :param options: Dictionary containing plot options. Available options are currently:
        show_labels: (Default False) If True, writes the pl_letter onto each plotted planet.
        show_resonance: (Default False) If True, annotates resonances
        show_system_name: (Default True) Display the system name as the y-label
        show_HZ: (Default False) Display the Kopparapu+2014 habitable zone estimate
        show_star: (Default True) Plot star on the left of axis
        use_au: Not yet implemented
        :param ax: (optional) If provided, system will be plotted to the provided matplotlib.axes.Axes object
        :returns: fig, ax or None
        """
        params = {'show_labels': False, 'show_resonance': False, 'show_system_name': True, 'show_HZ': False, 'show_star': True, 'use_au': False}
        if options is not None:
            for key, value in options.items():
                if key in params.keys():
                    if isinstance(value, bool):
                        params[key] = value
                    else:
                        raise TypeError("Options must be boolean values!")
                else:
                    raise ValueError(f"Unrecognized option entry: {key}")


        show_labels = params['show_labels']
        show_resonance = params['show_resonance']
        show_system_name = params['show_system_name']
        show_HZ = params['show_HZ']
        show_star = params['show_star']
        use_au = params['use_au']

        if ax is None:
            ax_provided = False
            fig, ax = plt.subplots(figsize=(10,2))
        else:
            ax_provided = True
        system = list(self.planets)
        removed = []
        for p in system:
            if p.radius is None:
                print(f"Cannot plot {p.name}: no radius")
                removed.append(p.name)
                system.remove(p)

        #Sort by radius to avoid covering smaller planets
        system = sorted(system, key=lambda p: p.radius, reverse=True)

        periods = [p.orbital_period.value for p in system]
        xmin = min(periods) * 0.8
        xmax = max(periods) * 1.2

        y_pos = 1
        system_label = self.name
        ax.axhspan(0.999 * y_pos, 1.001 * y_pos, xmin=-0.5, xmax=10, color="slategray", zorder=0)

        ax.set_yticks([y_pos])
        if show_system_name:
            ax.set_yticklabels([system_label])
        ax.set_ylim(0.5, 1.5)
        ax.set_xlim(xmin, xmax)
        ax.set_xscale("log")
        ax.tick_params(axis='y', labelrotation=90)
        #Plot star and HZ if stellar parameters available
        if show_star:
            if not self.star is None:
                ax.scatter(xmin, 1, s=5000, color=self.star.get_plot_color())
                if (not self.HZ_bounds is None) and show_HZ:
                    self.__plt_mark_HZ(ax, pl_masse=5, y_pos=y_pos)

        if show_resonance:
            system_psorted = list(self.planets)
            for p in system_psorted:
                if p.orbital_period is None:
                    print(f"NO ORBITAL PERIOD FOR {p.name}")
                    system_psorted.remove(p)
            system_psorted = sorted(self.planets, key=lambda p: p.orbital_period)
            for p in system_psorted:
                if p.radius is None:
                    continue
                system_pl_names = [p.name for p in self.planets]
                pl_system_index = system_pl_names.index(p.name)

                if pl_system_index != 0:
                    lesser_resonance = self.resonances[pl_system_index]
                    if bool(lesser_resonance):
                        # planet is resonant with interior nearest neighbor
                        P_outer = p.orbital_period
                        P_inner = system_psorted[pl_system_index - 1].orbital_period
                        ax.plot([P_inner.value, P_outer.value], [y_pos, y_pos], color='red', zorder=0, linewidth=2)
                        ax.text(x=(np.average([P_inner.value, P_outer.value])-((P_outer.value-P_inner.value)*0.2)), y=y_pos*1.05, s=f"{list(lesser_resonance.keys())[0]}", fontsize=8)



        #plot planets
        for p in system:
            if p.orbital_period.value is None:
                print(f"Cannot plot {p.name}: No valid orbital period")

            planet_label = p.name.strip()[-1] #gets the planet letter for labeling
            if self.name == "Solar System":
                planet_label = p.name[0]

            #Defaults to get rid of "may not have been initialized" warning
            plot_size = 10
            plot_color = 'red'
            #Match case to determine plot appearance
            match p.extended_properties["base_class"]:
                case "sub-Earth":
                    plot_size = 100
                    plot_color = 'skyblue'
                case "super-Earth":
                    plot_size = 300
                    plot_color = 'darkgreen'
                case "sub-Neptune":
                    plot_size = 400
                    plot_color = 'blue'
                case "Neptune":
                    plot_size = 500
                    plot_color = 'cadetblue'
                case "Jovian":
                    plot_size = 1100
                    plot_color = 'goldenrod'
                case "Brown Dwarf":
                    plot_size = 4900
                    plot_color = 'saddlebrown'
                case None:
                    print(f"Default parameter set does not include mass or radius measurements for {p.name}!\n Future updates will hopefully handle this!")
                    if show_labels:
                        ax.scatter(p.orbital_period, y_pos, s=500, c='darkgrey')
                        ax.scatter(p.orbital_period, y_pos, s=150, c='maroon', marker=f"${planet_label}$")
                    else:
                        ax.scatter(p.orbital_period, y_pos, s=500, c='darkgrey')
                        ax.scatter(p.orbital_period, y_pos, s=190, c='maroon', marker="$?$")

                    continue

            #add planet
            ax.scatter(p.orbital_period, y_pos, s=plot_size, c=plot_color)


            # Over plot special planets
            if not p.spec_attributes is None:
                if p.spec_attributes["near-Super-puff"] == True or p.spec_attributes["Super-puff"] == True:
                    ax.scatter(p.orbital_period, y_pos, s=plot_size * 0.7, c='white')
                    ax.scatter(p.orbital_period, y_pos, s=plot_size * 0.33, c=plot_color)

            if show_labels:
                ax.scatter(p.orbital_period, y_pos, s=0.1*plot_size, c='black', marker=f"${planet_label}$")
            if len(removed) > 0:
                ax.text(xmin*1.01,1.43, s=f'Not shown: {removed}', fontsize=7)

        if not ax_provided:
            return fig, ax

    def to_pandas(self):
        df = pd.DataFrame()
        for p in self.planets:
            p_dict = p.to_dict()
            df = pd.concat([df, pd.DataFrame(p_dict, index=[p.name])])
        return df

class Queries:

    global system_index
    system_index = {}


    def __init__(self):
        """Does not do anything"""
        pass


    @staticmethod
    def get_planet(planet_name):
        """Builds Planet object by querying NASA Exoplanet Archive, uses default parameter set if
        either pl_rade or pl_bmasse are not null

        :param planet_name: The planet name, as listed on Exoplanet Archive (pl_name)
        :returns Planet: Planet object
        """


        composite_set = NasaExoplanetArchive.query_criteria(table="pscomppars", where=f"pl_name='{planet_name}'")
        comp_df = composite_set.to_pandas()
        comp_dict = comp_df.to_dict(orient='records')[0]

        pl = Planet(name=comp_dict['pl_name'], properties=comp_dict)
        return pl

    @staticmethod
    def get_star(star_name, table="stellarhosts"):
        #gets spectype from simbad because IPAC often does not have an entry


        table = NasaExoplanetArchive.query_criteria(table=table,
                                                    where=f"hostname='{star_name}' and st_teff IS NOT null")
        try:
            t_dict = table.to_pandas().to_dict(orient='records')[0]
        except IndexError:
            raise ValueError(f"No IPAC stellar entries found for {star_name}")
        try:
            spec_type = Simbad.query_tap(f"SELECT main_id, sp_type FROM basic WHERE main_id = '{star_name}'")['sp_type'][0]
            t_dict['st_spectype_simbad'] = spec_type
        except Exception as e:
            spec_type = None
            t_dict['st_spectype_simbad'] = spec_type

        star = Star(name=star_name, properties=t_dict)
        return star

    @staticmethod
    def build_system(system_name):
        """Builds System object by querying NASA Exoplanet Archive for the default parameter set.

        :param system_name: The system name, as listed on Exoplanet Archive ('hostname')
        :returns System: System object
        """
        planet_array = []
        default_columns = ["pl_name",
                           "hostname",
                           "pl_letter",
                           "pl_bmasse",
                           "pl_bmasseerr1",
                           "pl_bmasseerr2",
                           "pl_bmassprov",
                           "pl_bmasse_reflink",
                           "pl_rade",
                           "pl_radeerr1",
                           "pl_radeerr2",
                           "pl_rade_reflink",
                           "pl_dens",
                           "pl_denserr1",
                           "pl_denserr2",
                           "pl_dens_reflink",
                           "pl_orbper",
                           "pl_orbpererr1",
                           "pl_orbpererr2",
                           "pl_orbper_reflink",
                           "discoverymethod",
                           "rv_flag",
                           "tran_flag",
                           "pul_flag",
                           "ptv_flag",
                           "ast_flag",
                           "obm_flag",
                           "micro_flag",
                           "etv_flag",
                           "ima_flag",
                           "dkin_flag",
                           "pl_orbsmax",
                           "pl_orbsmax_reflink",
                           "pl_angsep",
                           "pl_angsep_reflink",
                           "pl_orbeccen",
                           "pl_orbeccen_reflink",
                           "pl_insol",
                           "pl_insol_reflink",
                           "pl_eqt",
                           "pl_eqt_reflink",
                           "ttv_flag",
                           "cb_flag"]
        binary_flag = False
        result = NasaExoplanetArchive.query_criteria(table="pscomppars", where=f"hostname='{system_name}'")
        result = result.to_pandas()
        result = result.to_dict(orient='records')
        for index in range(0,len(result)):
            row = result[index]
            pl = Planet(name=row['pl_name'], properties=row)
            planet_array.append(pl)

            if result[index]['cb_flag'] > 0:
                binary_flag = True
        try:
            star = Queries.get_star(system_name)
        except ValueError as ve:
            print(ve.with_traceback(None))
            star = None
        # stellar parameters
        system = System(system_name, planets=planet_array, star=star, binary=binary_flag)
        return system

    @staticmethod
    def demo_plot(x, y, ax=None, where=None, xscale='log',yscale='log', *args, **kwargs):

        if ax is None:
            fig, ax = plt.subplots()
            ax_provided = False
        else:
            ax_provided = True

        #this is not sanitized, I do not care
        where_str = f"{x} IS NOT null AND {y} IS NOT null"
        if where is not None:
            where_str += (" AND " + where)
        result = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select=f"{x},{y}",
            where=where_str
        )
        result = result.to_pandas()
        ax.scatter(result[x], result[y], *args, **kwargs)
        ax.set_xlabel(f"{x}")
        ax.set_ylabel(f"{y}")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if not ax_provided:
            return fig, ax

    @staticmethod
    def stacked_sysplot(systems: list = None):
        """Generates stacked system plots to visualize orbital period distributions of multiple systems
        Currently very slow. Will be run more efficiently in future updates.
        :param systems: list of systems to include

        """

        num_systems = len(systems)
        vertsize = math.floor(0.5*num_systems) if num_systems > 1 else 1
        fig, ax = plt.subplots(figsize=(10, vertsize))
        highest_period = 0
        lowest_period = 1e9
        sys_dfs = {}
        for sys in systems:
            df = sys.to_pandas().dropna(subset=['pl_orbper'])
            sys_dfs[sys.name] = df

        y_pos = 0
        y_step = vertsize/num_systems
        yticks = []
        ticklabels = []
        for label, df in sys_dfs.items():
            yticks.append(y_pos)
            ticklabels.append(label)
            ax.axhspan(0.9999 * y_pos, 1.0001 * y_pos, xmin=-0.5, xmax=10, color="slategray", zorder=0)
            ax.scatter(df['pl_orbper'], [y_pos]*len(df), marker="o", s=10, color='black')
            if max(df['pl_orbper']) > highest_period:
                highest_period = max(df['pl_orbper'])
            if min(df['pl_orbper']) < lowest_period:
                lowest_period = min(df['pl_orbper'])
            y_pos += y_step

        ax.set_yticks(yticks)
        ax.set_yticklabels(ticklabels)
        ax.set_xlim([0.9*lowest_period, 1.2*highest_period])
        ax.set_xscale("log")

        return fig, ax

    @staticmethod
    def crossref_stellar_ids(id, input_id_type: str = '', to="all"):
        """Cross-matches object identifiers by querying SIMBAD. Takes either the full ID (e.g. Gaia DR3 X...) or the id sans
        the prefix (e.g. just the number in the case of Gaia)

        Parameters:
        :param id: the identifier or list of identifiers. Lists must be passed as lists, now series or anything else
        :param input_id_type: (Optional) the ID prefix (e.g. Gaia DR3). Default is empty string. If passed, assumes that
        ID type precedes the id string.
        :param to: (optional) the identifier prefix you want returned. If not specified, returns all available.

        Returns: dict or String

       """
        identifiers = ['KIC', 'TIC', '2MASS', 'K2', 'HIP', 'Gaia DR2', 'Gaia DR3', 'Kepler', 'KOI', 'TOI', 'WASP', 'KELT',
                       'HATS', 'GALAH', 'NGC', '']

        if input_id_type not in identifiers:
            raise ValueError(
                f"Identifier type not in list of supported identifiers. Please use one of the following: \n {identifiers}")
        if not to == "all":
            if to not in identifiers:
                raise ValueError(
                    f"Output identifier specified not in list of supported identifiers. Please use one of the following: \n {identifiers}")

        identifiers.remove('')
        def simbad_id_translate_query(idstr: str, to: str):

            if not idstr.startswith(input_id_type) and not len(input_id_type) == 0:
                idstr = input_id_type.strip() + " " + idstr
            try:
                result_id = Simbad.query_objectids(f"{idstr}", criteria=f"ident.id LIKE '{to}%'")[0][0]
            except Exception as e:
                raise Exception("Query is broken")

            return result_id

        if isinstance(id, list):

            if len(id) < 1:
                raise TypeError("You're passing something as a list that should not be passed as a list")
            output = pd.DataFrame(columns=identifiers, index=id)
            for i in id:

                if to == "all":
                    for ident in identifiers:
                        if ident == input_id_type:
                            output.at[i, ident] = i
                        elif len(ident) < 1:
                            continue
                        else:
                            try:
                                output.at[i, ident] = simbad_id_translate_query(i, ident)
                            except Exception as e:
                                continue
                else:
                    output.at[i, to] = simbad_id_translate_query(i, to)

            output = output.set_index(input_id_type)
            return output


        else:

            output = pd.DataFrame(columns=identifiers, index=[0])
            if to == "all":
                for ident in identifiers:
                    try:
                        output.at[0, ident] = simbad_id_translate_query(id, ident)
                    except Exception as e:
                        continue
                return output.dropna(axis=1, how='all')
            else:
                return simbad_id_translate_query(id, to)

    @staticmethod
    def get_random_system():
        """Gets a random system with N>2 planets. Stores an index of system names and planet counts in a global var when
         run for the first time to speed up successive runs. This is reset when kernel is restarted. """
        global system_index

        if not system_index:
            result = NasaExoplanetArchive.query_criteria(table="pscomppars", select="hostname,sy_pnum")
            for index, row in result.to_pandas().iterrows():
                hostname = row['hostname']
                if not hostname in system_index:
                    system_index[row['hostname']] = int(row['sy_pnum'])

        valid_keys = [key for key, value in system_index.items() if value > 2]
        if valid_keys:
            return random.choice(valid_keys)
        else:
            raise NotImplementedError()


    @staticmethod
    def build_dataframe(col_names, where=None, table='pscomppars', local="", allow_na = []):
        """Gets full-length (all rows) columns from specified table. Default table is pscomppars.
        
        :param col_names: string column name or list of column names to get
        :param local: (Optional) if specified, will save results to this filepath. Will attempt to load data from this path before querying. 
        """
        if len(local) > 0:
            try:
                df = pd.read_csv(local)
                print(f"LOCAL BACKUP FOUND: LOADING FROM SAVED FILE {local}")
                return df
            except:
                print("Local backup not found, proceeding to query...")

        # this is not sanitized, I do not care
        where_str = f"{col_names[0]} IS NOT null"
        #add additional columns
        if isinstance(col_names, list):
            if len(col_names) > 1:
                for i in range(1, len(col_names)):
                    if not col_names[i] in allow_na:
                        where_str += f" AND {col_names[i]} IS NOT null"

        if table == 'ps':
            where_str += " AND default_flag=1"
        if where is not None:
            where_str += (" AND " + where)

        select_str = f""
        if isinstance(col_names, list):
            for c in col_names:
                select_str += f"{c},"
            select_str = select_str[:-1]
        else:
            select_str = f"{col_names}"
        #print(f"where: {where_str} select = {select_str} table = {table}")
        result = NasaExoplanetArchive.query_criteria(
            table=table,
            select=select_str,
            where=where_str
        )
        result = result.to_pandas()
        return result

class Calc:
    def __init__(self):
        pass

    @staticmethod
    def bulk_density(radius, mass, rad_err = None, mass_err = None, units="cgs"):
        """
        Calculate the bulk density of a planet (default in cgs units (g/cm³)) with error propagation.

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
        
        units : str, specifies the density units to be returned. Accepted are "cgs" and "si". 
            Additionally, "cgsv" can be passed to return the numerical value in cgs without the attached astropy units.

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
            if units=="cgsv":
                return density.value
            elif units=="si":
                return density.si
            elif units=="cgs":
                return density
            else:
                print("units argument not recognized, output is in cgs...")
                return density
        else:
            return density, density_err
    
#PREFABS
#-Solar System
Sun = Star(name="Sun", properties={'st_dens':1.41, 'st_spectype':"G2V", 'st_mass':1, 'st_rad':1, 'st_teff':5778, 'st_met':0, 'st_lum':0, 'st_age':4.603}, force_stellar_params=False)
Mercury = Planet(name = "Mercury", mass=0.0553, radius=0.383, orbital_period=88.0, density=5.427)
Venus = Planet(name = "Venus", mass=0.815, radius=0.949, orbital_period=224.7, density=5.243)
Earth = Planet(name = "Earth", mass=1, radius=1, orbital_period=365.2, density=5.52)
Mars = Planet(name = "Mars", mass=0.107, radius=0.532, orbital_period=687.0, density=3.9335)
Jupiter = Planet(name = "Jupiter", mass=317.8, radius=11.21, orbital_period=4331, density=1.326)
Saturn = Planet(name = "Saturn", mass=95.2, radius=9.45, orbital_period=10747, density=0.69)
Uranus = Planet(name = "Uranus", mass=14.5, radius=4.01, orbital_period=30589, density=1.27)
Neptune = Planet(name = "Neptune", mass=17.1, radius=3.88/2, orbital_period=59800, density=1.638)
Pluto = Planet(name = "Pluto?", mass=0.0022, radius=0.187/2, orbital_period=90560, density=1.853)
solar_system = System("Solar System", star=Sun, planets=[Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto], binary=False)
Pluto = Planet(name = "Pluto?", mass=0.0022, radius=0.187/2, orbital_period=90560, density=1.853)
solar_system = System("Solar System", star=Sun, planets=[Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto], binary=False)
