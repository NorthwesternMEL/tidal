"""
Comparison of integration formulas for the thermal expansion of water
Provides data for Figure 4 of Coulibaly and Rotta Loria 2022

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
# Comparison of volume change integration formulas for water
# Based on the test performed by Liu et al., 2018 at p' = 50 kPa
# ------------------------------------------------------------------------------

# Initial volume of water vwi = 272.4 cm^3 ! (Figure 7 of Liu et al., 2018)
vwi = 272.4*1e3 # Initial volume of water [mm^3]
vinvpct = 1e2/vwi

# Temperature and expelled water time series digitized from Figure 7
temp_in = np.genfromtxt("Liu2018_temp_in.csv", delimiter=',', names=True)
temp_out = np.genfromtxt("Liu2018_temp_out.csv", delimiter=',', names=True)
vexp = np.genfromtxt("Liu2018_expelled_volume.csv", delimiter=',', names=True)

# Linear interpolation of the digitized values from 0 to 9h
ti = 0. # Initial time [h]
tf = 9. # Final time [h]
tincr = 1./60 # Timestep [h] (choen as 1 minute)
time = np.arange(ti, tf + 0.5*tincr, tincr, dtype=float) # Time [h]

# Temperature computed as the average of the outer and inner temperature
temp_in_interp = np.interp(time,temp_in["time_h"],temp_in["temp_C"])
temp_out_interp = np.interp(time,temp_out["time_h"],temp_out["temp_C"])
temp = 0.5*(temp_in_interp + temp_out_interp) # Temperature of the sample [degC]

# Volume of water expelled
vdr = np.interp(time,vexp["time_h"],vexp["vol_mm3"]) # [mm^3]

# Thermal expansion coefficient of water from IAPWS-95 at 300 kPa
# Add 1 increment of padding to the temperature for the IAPWS-95 so that thermal
# expansion is calculated with 2nd order central differences at first/last value
tempad = np.concatenate([[2*temp[0]-temp[1]],temp,[2*temp[-1]-temp[-2]]])
aw = thexp.coef_w_IAPWS95_tab("dat_IAPWS95_300kPa_10-90-0.5degC", tempad)[1:-1]

## First term: thermal expansion of the initial volume of water

# Volume of expelled water discarded (vdr = 0) to get first term only
# Exact integration, equation (16) in Coulibaly et al., 2022
dVw_term1_exact = thexp.deltaVw(vwi, aw, np.zeros(len(temp)), temp, 'exact')
# Small expansion integration, equation (17) in Coulibaly et al., 2022
dVw_term1_small = thexp.deltaVw(vwi, aw, 0.0, temp, 'small')
# Linear formula, equation (18) in Coulibaly et al., 2022
dVw_term1_lin = thexp.deltaVw(vwi, aw, 0.0, temp, 'linear')

# Plot results and export to comma-separated tables
plt.figure(1)
plt.plot(temp, dVw_term1_exact*vinvpct, label=r"Term 1, exact (IAPWS-95)")
plt.plot(temp, dVw_term1_small*vinvpct, label=r"Term 1, small (IAPWS-95)")
plt.plot(temp, dVw_term1_lin*vinvpct, label=r"Term 1, linear (IAPWS-95)")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel('Relative thermal expansion of initial water volume (first term)'+
           r'$\Delta V_w^{th}/V_{w,i}$ [%]')
plt.title("Figure 4a of Coulibaly and Rotta Loria 2022")
plt.legend()

np.savetxt("tab_integration_water_term1_expansion_initial_volume_only.csv",
           np.concatenate((temp[:,np.newaxis],
                           dVw_term1_exact[:,np.newaxis]*vinvpct,
                           dVw_term1_small[:,np.newaxis]*vinvpct,
                           dVw_term1_lin[:,np.newaxis]*vinvpct), axis=1),
           header=("temp_degC,dVw_Vwi_term1_exact_pct,dVw_Vwi_term1_small_pct,"+
                   "dVw_Vwi_term1_lin_pct"),
           delimiter=',')

## Second term: volume of expelled water coupled with thermal expansion

# Exact integration, equation (16) in Coulibaly et al., 2022
dVw_term2_exact = dVw_term1_exact - thexp.deltaVw(vwi, aw, vdr, temp, 'exact')
# Small expansion integration, equation (17) in Coulibaly et al., 2022
# The second term is identical for small expansion (17) and linear equation (18)
# and is actually directly equal to -Vdr (allows double checking implementation)
dVw_term2_small = dVw_term1_small - thexp.deltaVw(vwi, aw, vdr, temp, 'small')


# Plot results and export to comma-separated tables
plt.figure(2)
plt.plot(temp, dVw_term2_exact*vinvpct, label=r"Term 2, coupled (IAPWS-95)")
plt.plot(temp, dVw_term2_small*vinvpct, label=r"Term 2, decoupled (IAPWS-95)")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel('Relative expelled water coupling term (second term)'+
           r'$\Delta V_w^{dr}/V_{w,i}$ [%]')
plt.title("Figure 4b of Coulibaly and Rotta Loria 2022")
plt.legend()

np.savetxt("tab_integration_water_term2_expelled_water_coupling_only.csv",
           np.concatenate((temp[:,np.newaxis],
                           dVw_term2_exact[:,np.newaxis]*vinvpct,
                           dVw_term2_small[:,np.newaxis]*vinvpct), axis=1),
           header=("temp_degC,dVw_Vwi_term2_exact_pct,dVw_Vwi_term2_small_pct"),
           delimiter=',')
