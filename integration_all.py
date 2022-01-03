"""
Comparison of integration formulas for the thermal expansion of solid grains and
water. Provides data for Figure 3 of Coulibaly and Rotta Loria 2022

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
import thexp

# Temperature range for the calculation [25 - 55] degC
ti = 25 # Initial temperature [degC]
tf = 55 # Final temperature [degC]
tincr = 1 # Temperature increment [degC]
temp = np.arange(ti, tf + 0.5*tincr, tincr, dtype=float)

# ------------------------------------------------------------------------------
# Comparison of thermal expansion integration formulas for solid grains
# ------------------------------------------------------------------------------

# Volumetric thermal expansion coefficient of the solid grains
bs_lo = 3e-5 # Value used by Ng et al., 2016 [1/degC]
bs_hi = 3.5e-5 # Value used by Liu et al., 2018 [1/degC]
bs_kos = thexp.coef_s_Kosinski91(temp, 5) # 5th order value used
                                          # by Kosinski et al., 1991 [1/degC]

# Relative volume change, use unit initial volume (vsi = 1) to make relative
# Exact integration, equation (20) in Coulibaly et al., 2022
dVs_kos_exact = thexp.deltaVth(1.0, bs_kos, temp, 'exact')*1e2
# Small expansion integration, equation (22) in Coulibaly et al., 2022
dVs_kos_small = thexp.deltaVth(1.0, bs_kos, temp, 'small')*1e2
# Linear formula, equation (11) in Coulibaly et al., 2022
dVs_kos_lin = thexp.deltaVth(1.0, bs_kos, temp, 'linear')*1e2
dVs_lo_lin = thexp.deltaVth(1.0, bs_lo, temp, 'linear')*1e2
dVs_hi_lin = thexp.deltaVth(1.0, bs_hi, temp, 'linear')*1e2

# Plot results and export to comma-separated tables
plt.figure(1)
plt.plot(temp, dVs_kos_exact, label=r"Exact (Kosinski et al., 1991)")
plt.plot(temp, dVs_kos_small, label=r"Small (Kosinski et al., 1991)")
plt.plot(temp, dVs_kos_lin, label=r"Linear (Kosinski et al., 1991)")
plt.plot(temp, dVs_lo_lin, label=r"Linear ($\beta_s=$"+str(bs_lo)+" 1/degC)")
plt.plot(temp, dVs_hi_lin, label=r"Linear ($\beta_s=$"+str(bs_hi)+" 1/degC)")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel(r'Thermal expansion of solid grains $\Delta V_s^{th}/V_{s,i}$ [%]')
plt.title("Figure 3 of Coulibaly and Rotta Loria 2022")
plt.legend()

np.savetxt("tab_integration_solid_expansion.csv",
           np.concatenate((temp[:,np.newaxis], dVs_kos_exact[:,np.newaxis],
                           dVs_kos_small[:,np.newaxis],
                           dVs_kos_lin[:,np.newaxis],
                           dVs_lo_lin[:,np.newaxis],
                           dVs_hi_lin[:,np.newaxis]), axis=1),
           header=("temp_degC,dVs_Vsi_kos91_exact_pct,dVs_Vsi_kos91_small_pct,"+
                   "dVs_Vsi_kos91_linear_pct,dVs_Vsi_as=3e-5_lin_pct,"+
                   "dVs_Vsi_as=3.5e-5_lin_pct"),
           delimiter=',')

# ------------------------------------------------------------------------------
# Comparison of thermal expansion integration formulas for water
# ------------------------------------------------------------------------------

# Volumetric thermal expansion coefficient of water from IAPWS-95 at atmospheric
# pressure.
# Add 1 increment of padding to the temperature for the IAPWS-95 so that thermal
# expansion is calculated with 2nd order central differences at first/last value
tempad = np.concatenate([[2*temp[0]-temp[1]],temp,[2*temp[-1]-temp[-2]]])
bw = thexp.coef_w_IAPWS95_tab("dat_IAPWS95_1atm_10-90-0.5degC", tempad)[1:-1]

# Relative volume change, use unit initial volume (vwi = 1) to make relative
# Exact integration, equation (21) in Coulibaly et al., 2022
dVwth_exact = thexp.deltaVth(1.0, bw, temp, 'exact')*1e2
# Small expansion integration, equation (23) in Coulibaly et al., 2022
dVwth_small = thexp.deltaVth(1.0, bw, temp, 'small')*1e2
# Linear formula, equation (12) in Coulibaly et al., 2022
dVwth_lin = thexp.deltaVth(1.0, bw, temp, 'linear')*1e2

# Plot results and export to comma-separated tables
plt.figure(2)
plt.plot(temp, dVwth_exact, label=r"Exact (IAPWS-95)")
plt.plot(temp, dVwth_small, label=r"Small (IAPWS-95)")
plt.plot(temp, dVwth_lin, label=r"Linear (IAPWS-95)")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel('Thermal expansion of initial water volume '+
           r'$\Delta V_w^{th}/V_{w,i}$ [%]')
plt.title("Figure 4a of Coulibaly and Rotta Loria 2022")
plt.legend()

np.savetxt("tab_integration_water_expansion.csv",
           np.concatenate((temp[:,np.newaxis],
                           dVwth_exact[:,np.newaxis],
                           dVwth_small[:,np.newaxis],
                           dVwth_lin[:,np.newaxis]), axis=1),
           header=("temp_degC,dVwth_Vwi_exact_pct,dVwth_Vwi_small_pct,"+
                   "dVwth_Vwi_lin_pct"),
           delimiter=',')

# ------------------------------------------------------------------------------
# Coupling between expelled water and thermal expansion
# ------------------------------------------------------------------------------

### (1) Expelled water volume data from Liu et al., 2018, test at p' = 50 kPa

# Initial volume (Fig. 7: vsi = 732.4 cm^3, vwi = 272.4 cm^3, vi = 1004.8 cm^3)
vi = 1004.8e3 # [mm^3]

# Temperature and expelled water time series digitized from Figure 7
temp_in = np.genfromtxt("Liu2018_temp_in.csv", delimiter=',', names=True)
temp_out = np.genfromtxt("Liu2018_temp_out.csv", delimiter=',', names=True)
vexp = np.genfromtxt("Liu2018_expelled_volume.csv", delimiter=',', names=True)

# Linear interpolation of the digitized values from 0 to 9h
ti = 0. # Initial time [h]
tf = 9. # Final time [h]
tincr = 1./60 # Timestep [h] (choen as 1 minute)
time = np.arange(ti, tf + 0.5*tincr, tincr, dtype=float) # Time [h]

# Temperature of sample: average of the outer and inner temperatures [degC]
temp = 0.5*(np.interp(time,temp_in["time_h"],temp_in["temp_C"]) +
            np.interp(time,temp_out["time_h"],temp_out["temp_C"]))

# Volume of water expelled [mm^3]
vdr = np.interp(time,vexp["time_h"],vexp["vol_mm3"])

# Volumetric thermal expansion coefficient of water from IAPWS-95 at 300 kPa
# Add 1 increment of padding to the temperature for the IAPWS-95 so that thermal
# expansion is calculated with 2nd order central differences at first/last value
tempad = np.concatenate([[2*temp[0]-temp[1]],temp,[2*temp[-1]-temp[-2]]])
bw = thexp.coef_w_IAPWS95_tab("dat_IAPWS95_300kPa_10-90-0.5degC", tempad)[1:-1]

# Coupled drainage-expansion volume change of water [mm^3]
# Equation (24) in Coulibaly et al., 2022
dVw_dr = thexp.deltaVw_dr(bw, vdr, temp)

# Plot results and export to comma-separated tables
plt.figure(3)
plt.plot(temp, (dVw_dr-vdr)/vi*1e2, label="Liu et al., 2018")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel('Relative difference between coupled/uncoupled drainage-expansion '+
           r'$\Delta V_w^{dr}/V_i$ [%]')
plt.title("Figure 4 of Coulibaly and Rotta Loria 2022")
plt.legend()

np.savetxt("tab_integration_water_drainage_coupling_Liu2018.csv",
           np.concatenate((time[:,np.newaxis],
                           temp[:,np.newaxis],
                           dVw_dr[:,np.newaxis],
                           vdr[:,np.newaxis],
                           (dVw_dr-vdr)[:,np.newaxis]/vi*1e2), axis=1),
           header=("time_h,temp_degC,dVw_dr_mm3,dVdr_mm3,(dVw_dr-dVdr)/Vi_pct"),
           delimiter=',')

### (2) Expelled water volume data from Ng et al., 2016, test DR70S200TC

# Initial volume vi = 85689 mm^3 back-calculated from Table 3. Calculations
# available in file `analysis_Ng_et_al_2016.py`
vi = 85689 # [mm^3]

# Temperature and expelled water time series digitized from Figure 3
texp = np.genfromtxt("Ng2016_temp.csv", delimiter=',', names=True)
vexp = np.genfromtxt("Ng2016_expelled_volume.csv", delimiter=',', names=True)

# Linear interpolation of the digitized values from 0 to 31h
ti = 0. # Initial time [h]
tf = 31. # Final time [h]
tincr = 1./60 # Timestep [h] (choen as 1 minute)
time = np.arange(ti, tf + 0.5*tincr, tincr, dtype=float) # Time [h]

# Temperature (interpolated)
temp = np.interp(time,texp["time_h"],texp["temp_degC"])
# Volume of water expelled (interpolated)
vdr = np.interp(time,vexp["time_h"],vexp["vol_mm3"]) # [mm^3]

# Volumetric thermal expansion coefficient of water from IAPWS-95 at 200 kPa
# (a back pressure of 200 kPa only appears once in the caption of Figure 1)
# Add 1 increment of padding to the temperature for the IAPWS-95 so that thermal
# expansion is calculated with 2nd order central differences at first/last value
tempad = np.concatenate([[2*temp[0]-temp[1]],temp,[2*temp[-1]-temp[-2]]])
bw = thexp.coef_w_IAPWS95_tab("dat_IAPWS95_200kPa_10-90-0.5degC", tempad)[1:-1]

# Coupled drainage-expansion volume change of water [mm^3]
# Equation (24) in Coulibaly et al., 2022
dVw_dr = thexp.deltaVw_dr(bw, vdr, temp)

# Plot results and export to comma-separated tables
plt.figure(3)
plt.plot(temp, (dVw_dr-vdr)/vi*1e2, label="Ng et al., 2016")
plt.legend()

np.savetxt("tab_integration_water_drainage_coupling_Ng2016.csv",
           np.concatenate((temp[:,np.newaxis],
                           dVw_dr[:,np.newaxis],
                           vdr[:,np.newaxis],
                           (dVw_dr-vdr)[:,np.newaxis]/vi*1e2), axis=1),
           header=("temp_degC,dVw_dr_mm3,dVdr_mm3,(dVw_dr-dVdr)/Vi_pct"),
           delimiter=',')