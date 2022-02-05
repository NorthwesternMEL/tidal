"""
Verification of the thermal expansion calculations of Ng et al., 2016: Volume
change behaviour of saturated sand, Geotechnique Letters. 6:124-131
DOI: https://doi.org/10.1680/jgele.15.00148
Provides data for Appendix B (Table X) of Coulibaly and Rotta Loria 2022

General information and data recovered for test D70S200TC, chosen because most
of the data is tabulated and available throughout the original paper

Python 2
Scipy version 1.2.1
Numpy version 1.16.2
Jibril B. Coulibaly, jibril.coulibaly at gmail.com

SI units unless indicated otherwise
No exceptions checked for invalid inputs. Users responsability
The top-level TIDAL directory must be accessible to the PYTHONPATH

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
import matplotlib.pyplot as plt

from tidal.core import thexp
from tidal.core import inteq
from tidal.data import rdNg2016

# ------------------------------------------------------------------------------
# General information and data recovered from the paper for test D70S200TC
# Test D70S200TC chosen because most of the data is available through the paper
# ------------------------------------------------------------------------------

# Initial values of volume and density
vwi = rdNg2016.vwi # Initial water volume [cm3]
vsi = rdNg2016.vsi # Initial solid volume [cm3]
vi = rdNg2016.vi # Initial sample volume [cm3]
ei = rdNg2016.ei # Initial void ratio [-]
ni = rdNg2016.ni # Initial porosity [-]
pfi = rdNg2016.pfi # Initial packing fraction [-]

# Values of volume change of water and solid grains due to thermal expansion
temp = rdNg2016.temp # Temperature [degC]
dvw_vi = rdNg2016.dvw_vi # Relative volume change of water [%]
dvs_vi = rdNg2016.dvs_vi # Relative volume change of solid grains [%]

### Verification 1: integration of the thermal expansion of solid grains
bs = rdNg2016.bs*np.ones(len(temp))
# Exact integration, equation (20) in Coulibaly et al., 2022
dvs_vi_exact = inteq.deltaVth('beta', pfi, bs, temp)*1e2
# Small thermal expansion integration, equation (22) in Coulibaly et al., 2022
dvs_vi_small = inteq.deltaVth('small', pfi, bs, temp)*1e2
# Linear thermal expansion formula, equation (11) in Coulibaly et al., 2022
dvs_vi_lin = inteq.deltaVth('linear', pfi, bs, temp)*1e2

plt.figure(1)
plt.plot(temp, dvs_vi, 'ko', label=r"Ng et al., 2016 (Table 3)")
plt.plot(temp, dvs_vi*4.7, 'ro', label=r"Ng et al., 2016 (x4.7)")
plt.plot(temp, dvs_vi_exact, label=r"Exact integration")
plt.plot(temp, dvs_vi_small, label=r"Small coefficient")
plt.plot(temp, dvs_vi_lin, label=r"Linear")
plt.plot(temp, dvs_vi - dvs_vi_lin, label="Correction")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel(r'Relative volume change of solid, $\Delta V_s/V_i$ [%]')
plt.title("Integration of thermal expansion of solid grains")
plt.legend()

np.savetxt("tab_verif_Ng2016_integration_solid.csv",
           np.concatenate((temp[:,np.newaxis], dvs_vi[:,np.newaxis],
                           dvs_vi_exact[:,np.newaxis],
                           dvs_vi_small[:,np.newaxis],
                           dvs_vi_lin[:,np.newaxis]), axis=1),
           header=("temp_degC,dVs_Vi_Ng2016_pct,dVs_Vi_exact_pct,"+
                   "dVs_Vi_small_pct,dVs_Vi_linear_pct"),
           delimiter=',')
# The resuls show a very important mismatch between the data reported in Table 3
# and the results computed here based on the information available in the paper.
# The values in Table 3 are about 4.7 times smaller than what they should be !
# The fact that this factor is consistent show that there must be a mistake in
# the calculation of the thermal expansion of the grains by Ng et al., 2016 !


### Verification 2: integration of the thermal expansion of water

# The text mentions that thermal expansion of water is computed using the linear
# Equation $\Delta V_w = \alpha_w V_w \Delta T$ (Table 2). The data of Table 3
# is compared to the different integration formulas proposed by Coulibaly and
# Rotta Loria, 2022 to verify which one is actually used by Ng et al., 2016.

u = rdNg2016.u # Back pressure [Pa]
bw = thexp.vcte_w_Baldi88(u,temp) # Baldi et al., 1988
# Exact integration, equation (21) in Coulibaly et al., 2022
dvw_vi_exact = inteq.deltaVth('beta', ni, bw, temp)*1e2
# Small thermal expansion integration, equation (23) in Coulibaly et al., 2022
dvw_vi_small = inteq.deltaVth('small', ni, bw, temp)*1e2
# Linear thermal expansion formula, equation (12) in Coulibaly et al., 2022
dvw_vi_lin = inteq.deltaVth('linear', ni, bw, temp)*1e2

plt.figure(2)
plt.plot(temp, dvw_vi, 'ko', label=r"Ng et al., 2016 (Table 3)")
plt.plot(temp, dvw_vi_exact, label=r"Exact integration")
plt.plot(temp, dvw_vi_small, label=r"Small coefficient")
plt.plot(temp, dvw_vi_lin, label=r"Linear")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel(r'Relative volume change of water, $\Delta V_w/V_i$ [%]')
plt.title("Integration of thermal expansion of water")
plt.legend()

np.savetxt("tab_verif_Ng2016_integration_water.csv",
           np.concatenate((temp[:,np.newaxis], dvw_vi[:,np.newaxis],
                           dvw_vi_exact[:,np.newaxis],
                           dvw_vi_small[:,np.newaxis],
                           dvw_vi_lin[:,np.newaxis]), axis=1),
           header=("temp_degC,dVw_Vi_Ng2016_pct,dVw_Vi_exact_pct,"+
                   "dVw_Vi_small_pct,dVw_Vi_linear_pct"),
           delimiter=',')
# The results seem to show that Ng et al., 2016 actually integrated the thermal
# expansion of water adequately, i.e. using equation (21) or (23) of Coulibaly
# and Rotta Loria 2022, but wrongly wrote the linearized formula in text
