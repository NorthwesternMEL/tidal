"""
TIDAL : Thermally Induced Deformation Analysis Library
======================================================

Data subpackage
Python 2
Scipy version 1.2.1
Numpy version 1.16.2
Jibril B. Coulibaly, jibril.coulibaly at gmail.com


Provides
  1. Tabulated Raw data of the literature
  2. TBD
  3. TBD

How to use the documentation
----------------------------
Documentation is available through docstrings provided with the code

Copyright (C) 2021 Mechanics and Energy Laboratory, Northwestern University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

See the README file in the top-level TIDAL directory.
UNDER CONSTRUCTION
"""

from os.path import normpath as nrmp

path_IAPWS95_1atm = nrmp(__path__[0] + "/dat_IAPWS95_1atm_10-90-0.5degC")
path_IAPWS95_50kPa = nrmp(__path__[0] + "/dat_IAPWS95_50kPa_10-90-0.5degC")
path_IAPWS95_100kPa = nrmp(__path__[0] + "/dat_IAPWS95_100kPa_10-90-0.5degC")
path_IAPWS95_200kPa = nrmp(__path__[0] + "/dat_IAPWS95_200kPa_10-90-0.5degC")
path_IAPWS95_300kPa = nrmp(__path__[0] + "/dat_IAPWS95_300kPa_10-90-0.5degC")
path_IAPWS95_400kPa = nrmp(__path__[0] + "/dat_IAPWS95_400kPa_10-90-0.5degC")
path_IAPWS95_800kPa = nrmp(__path__[0] + "/dat_IAPWS95_800kPa_10-90-0.5degC")
path_IAPWS95_1MPa = nrmp(__path__[0] + "/dat_IAPWS95_1MPa_10-90-0.5degC")
