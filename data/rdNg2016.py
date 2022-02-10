"""
Raw data from the article of Ng et al., 2016: Volume change behaviour of
saturated sand, Geotechnique Letters. 6:124-131
DOI: https://doi.org/10.1680/jgele.15.00148

General information and data recovered for test D70S200TC, chosen because most
of the data is tabulated and available throughout the original paper

Python 2
Scipy version 1.2.1
Numpy version 1.16.2
Jibril B. Coulibaly, jibril.coulibaly at gmail.com

Units and dimensions must be consistent between all input variables.
No exceptions checked for invalid inputs. Users responsability.

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
"""

import numpy as np

### (1) General material properties

# Initial specific density of the solid grains of Toyoura sand [-]
gsi = 2.65 # data from Verdugo and Ishihara, 1996

# Initial density of the solid grains of Toyoura sand [g/mm3]
rhosi = gsi*1e-3

# Volumetric thermal expansion coefficient of the solid grains [1/degC]
#   p. 127 "linear thermal expansion [...] of sand [...] is 1e-5 1/degC"
bs = 3e-5

### (2) Test D70S200TC initial conditions and results

# Initial void ratio [-] (Table 1)
ei = 0.71

# Initial porosity [-]
ni = ei/(1.0 + ei)

# Initial packing fraction (complement to 1 of porosity) [-]
pfi = 1.0 - ni

# Stress state [Pa]
#   The confining and back pressures used by Ng et al., 2016 are not detailed in
#   the paper. A cell pressure of p = 400 kPa and a back pressure of u = 200 kPa
#   are mentioned in the caption of Figure 1 for the calibration test and
#   correspond to an effective stress of p' = 200 kPa as used in test D70S200TC
#   and is therefore used in this database
# Confining pressure [Pa]
p = 400e3

# Back pressure [Pa]
u = 200e3

# Effective stress [Pa]
peff = p - u

# Temperature [degC] (Table 3)
temp = np.array([23, 30, 40, 50, 40, 30, 23], dtype=float)

# Measured water volume [mm3] (Table 3)
dvme = np.array([0, 132, 337, 527, 464, 342, 291], dtype=float)

# Volume correction due to thermal expansion of drainage system [mm3] (Table 3)
dvde = np.array([0, 15, 41, 76, 55, 25, 7], dtype=float)

# Volume change of water relative to initial sample volume [%] (Table 3)
dvw_vi = np.array([0, 0.08, 0.224, 0.399, 0.233, 0.08, 0])

# Volume change of solid grains relative to initial sample volume [%] (Table 3)
dvs_vi = np.array([0, 0.003, 0.006, 0.01, 0.006, 0.003, 0])

# Volume of water leakage relative to initial sample volume [%] (Table 3)
dvmu_vi = np.array([0, 0.04, 0.088, 0.129, 0.225, 0.265, 0.31])

# Thermally induced volumetric strain [%] (Table 3)
epsv = np.array([0, 0.014, 0.028, -0.016, 0.014, 0.022, 0.022])

# Initial volume of the sample [mm3]
#   The initial volume of the sample is not provided in the original paper. It
#   is calculated using data from Table 3 and equation (1) of Ng et al., 2016.
#   The function vi_compute() can compute the volume for reproducibility.
#   This computed value is also hardcoded in the local variable vi
def vi_compute():
  # Average value calculated from Table 3 except for the first row of zeros
  return np.average((dvme[1:] - dvde[1:])/
                    (epsv[1:] + dvw_vi[1:] + dvs_vi[1:] + dvmu_vi[1:]))*1e2

vi = 85688.62635696691 # Volume obtained by calling `vi_compute()` [mm3]

# Initial volume of water inside the sample [mm3]
vwi = ni*vi

# Initial volume of solid grains inside the sample [mm3]
vsi = pfi*vi

# Volume correction for the measured water volume [mm3]
dvcal = dvde + dvmu_vi*vi*1e-2 # expansion of drainage system and leakage

# Initial mass of solid grains inside the sample [g]
ms = rhosi*vsi

# Volume change of water (absolute value) [mm3]
dvw = dvw_vi*vi*1e-2

# Volume change of solid grains (absolute value) [mm3]
dvs = dvs_vi*vi*1e-2
