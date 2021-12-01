"""
Comparison of integration formulas for the thermal expansion of solid grains 
Provides data for Figure 3 of Coulibaly and Rotta Loria 2022

Python 2
Scipy version 1.2.1
Numpy version 1.16.2
Jibril B. Coulibaly, jibril.coulibaly at gmail.com

Everything in SI units

except for temperature in Celsius degrees (noted degC)
Degrees Celsius [degC]
Degrees Farenheit [degF]
Kelvin [K]
Degrees Rankine [degR]

No exceptions checked for invalid inputs. Users responsability

Copyright (C) 2021 Mechanics and Energly Laboratory, Northwestern University

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
import thexp

# ------------------------------------------------------------------------------
# Comparison of thermal expansion integration formulas for solid grains
# ------------------------------------------------------------------------------

# Temperature range for the calculation [25 - 55] degC
ti = 25 # Initial temperature [degC]
tf = 55 # Final temperature [degC]
tincr = 1 # Temperature increment [degC]
temp = np.arange(ti, tf + tincr, tincr, dtype=float)

# Volumetric thermal expansion coefficient of water
as_lo = 3e-5 # Value used by Ng et al., 2016 [1/degC]
as_hi = 3.5e-5 # Value used by Liu et al., 2018 [1/degC]
as_kos = thexp.coef_s_Kosinski91(temp, 5) # 5th order value used 
                                          # by Kosinski et al., 1991 [1/degC]

# Relative volume change, use unit initial volume (vsi = 1) to make relative
# Exact integration, equation (13) in Coulibaly et al., 2022
dVs_kos_exact = thexp.deltaVs(1.0, as_kos, temp, 'exact')
# Small expansion integration, equation (14) in Coulibaly et al., 2022
dVs_kos_small = thexp.deltaVs(1.0, as_kos, temp, 'small')
# Linear formula, equation (15) in Coulibaly et al., 2022
dVs_kos_lin = thexp.deltaVs(1.0, as_kos, temp, 'linear') # Kosinski et al., 1991
dVs_lo_lin = thexp.deltaVs(1.0, as_lo, temp, 'linear') # as = 3e-5 [1/degC]
dVs_hi_lin = thexp.deltaVs(1.0, as_hi, temp, 'linear') # as = 3.5e-5 [1/degC]

# Plot results and export to comma-separated tables
plt.figure(1)
plt.plot(temp, dVs_kos_exact*1e2, label=r"Exact (Kosinski et al., 1991)")
plt.plot(temp, dVs_kos_small*1e2, label=r"Small (Kosinski et al., 1991)")
plt.plot(temp, dVs_kos_lin*1e2, label=r"Linear (Kosinski et al., 1991)")
plt.plot(temp, dVs_lo_lin*1e2, label=r"Linear ($\alpha=$"+str(as_lo)+" 1/degC)")
plt.plot(temp, dVs_hi_lin*1e2, label=r"Linear ($\alpha=$"+str(as_hi)+" 1/degC)")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel(r'Relative volume change of solid grains $\Delta V_s/V_{s,i}$ [%]')
plt.title("Figure 3 of Coulibaly and Rotta Loria 2022")
plt.legend()

np.savetxt("tab_integration_solid.csv",
           np.concatenate((temp[:,np.newaxis], dVs_kos_exact[:,np.newaxis]*1e2,
                           dVs_kos_small[:,np.newaxis]*1e2,
                           dVs_kos_lin[:,np.newaxis]*1e2,
                           dVs_lo_lin[:,np.newaxis]*1e2,
                           dVs_hi_lin[:,np.newaxis]*1e2), axis=1),
           header=("temp_degC,dVs_Vs_kos91_exact_pct,dVs_Vs_kos91_small_pct,"+
                   "dVs_Vs_kos91_linear_pct,dVs_Vs_as=3e-5_lin_pct,"+
                   "dVs_Vs_as=3.5e-5_lin_pct"),
           delimiter=',')