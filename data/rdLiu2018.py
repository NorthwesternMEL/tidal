"""
Raw data from the article of Liu et al., 2018: Influence of temperature on
the volume change behavior of saturated sand, Geotechnical Testing Journal.
41(4). DOI: https://www.astm.org/gtj20160308.html

General information and data recovered for test at p' = 50 kPa, chosen because
most of the data is tabulated and available throughout the original paper

Python 2
Scipy version 1.2.1
Numpy version 1.16.2
Jibril B. Coulibaly, jibril.coulibaly at gmail.com


Supplemental material to Coulibaly and Rotta Loria, 2022: Thermally induced
deformation of soils: a critical revision of experimental methods. GETE
DOI: TBD

except for temperature in Celsius degrees (noted degC)
Degrees Celsius [degC]
Degrees Farenheit [degF]
Kelvin [K]
Degrees Rankine [degR]

No exceptions checked for invalid inputs. Users responsability

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

# Initial specific density of the solid grains of Fujian sand [-]
#   data from From Xiao et al., 2017 DOI: https://doi.org/10.1680/jgele.16.00144
gsi = 2.69

# Initial density of the solid grains of Fujian sand [g/mm3]
rhosi = gsi*1e-3

# Volumetric thermal expansion coefficient of the solid grains [1/degC]
#   p. 6 "a value of 3.5e-3 %/degC is assumed", i.e. 3.5e-5 1/degC
bs = 3.5e-5

### (2) Test at p' = 50 kPa initial conditions and results

# Initial void ratio [-] (Table 1)
ei = 0.372

# Initial porosity [-]
ni = ei/(1.0 + ei)

# Initial packing fraction (complement to 1 of porosity) [-]
pfi = 1.0 - ni

# Initial volume of water inside the sample [cm3] (Figure 7)
vwi = 272.4

# Initial volume of solid grains inside the sample [mm3] (Figure 7)
vsi = 732.4

# Initial volume of the sample [mm3]
vi = vsi + vwi

# Initial mass of solid grains inside the sample [g]
ms = rhosi*vsi

# Stress state [Pa]
#   p. 4 "mean effective stresses of p' = 50 [...] kPa by raising the outer and
#   inner stresses to different values (po 350 kPa) while keeping the back
#   stress uw constant at 300 kPa
# Confining pressure [Pa]
p = 350e3

# Back pressure [Pa]
u = 300e3

# Effective stress [Pa]
peff = p - u

# Temperature [degC] (Table 3)
temp = np.array([25, 35, 45, 55], dtype=float)

# Measured water volume [cm3] (Table 3)
dvme = np.array([0.000, 0.430, 1.020, 1.740])

# Volume correction due to thermal expansion of drainage system [cm3] (Table 3)
dvde = np.array([0.000, 0.090, 0.070, 0.040])

# Volume correction for the measured water volume [cm3]
dvcal = dvde # Expansion of drainage system only

# Volume change of water [cm3] (Table 3)
dvw = np.array([0.000, 0.816, 1.852, 3.146])

# Volume change of solid grains [cm3] (Table 3)
dvs = np.array([0.000, 0.254, 0.511, 0.777])

# Total volume change of the sample [cm3] (Table 3)
dv = np.array([0.000, -0.730, -1.413, -2.223])

# Thermally induced volumetric strain [%] (Table 3)
epsv = np.array([0.000, -0.073, -0.141, -0.221])
