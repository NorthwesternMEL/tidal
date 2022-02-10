"""
TIDAL : Thermally Induced Deformation Analysis Library
======================================================

Python 2
Scipy version 1.2.1
Numpy version 1.16.2
Jibril B. Coulibaly, jibril.coulibaly at gmail.com

Data sub-package:
  1. Tabulated IAPWS-95 water properties for select pressure/temperature ranges
  2. Module `rdNg2016` : Available raw data of Ng et al., 2016
  3. Module `rdLiu2018` : Available raw data of Liu et al., 2018

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

import os

datapath = os.path.abspath(os.path.dirname(__file__))

path_IAPWS95_1atm = os.path.join(datapath,"dat_IAPWS95_1atm_10-90-0.5degC")
path_IAPWS95_50kPa = os.path.join(datapath,"dat_IAPWS95_50kPa_10-90-0.5degC")
path_IAPWS95_100kPa = os.path.join(datapath,"dat_IAPWS95_100kPa_10-90-0.5degC")
path_IAPWS95_200kPa = os.path.join(datapath,"dat_IAPWS95_200kPa_10-90-0.5degC")
path_IAPWS95_300kPa = os.path.join(datapath,"dat_IAPWS95_300kPa_10-90-0.5degC")
path_IAPWS95_400kPa = os.path.join(datapath,"dat_IAPWS95_400kPa_10-90-0.5degC")
path_IAPWS95_800kPa = os.path.join(datapath,"dat_IAPWS95_800kPa_10-90-0.5degC")
path_IAPWS95_1MPa = os.path.join(datapath,"dat_IAPWS95_1MPa_10-90-0.5degC")
