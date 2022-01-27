"""
Verification of the thermal expansion calculations of Liu et al., 2018:
Influence of temperature on the volume change behavior of saturated sand,
Geotechnical Testing Journal. 41(4). DOI: https://www.astm.org/gtj20160308.html
Provides data for Appendix B (Table X) of Coulibaly and Rotta Loria 2022

General information and data recovered for test at p' = 50 kPa, chosen because
most of the data is tabulated and available throughout the original paper

Python 2
Scipy version 1.2.1
Numpy version 1.16.2
Jibril B. Coulibaly, jibril.coulibaly at gmail.com

SI units unless indicated otherwise
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
import rdLiu2018
import scipy.integrate as integrate
import thexp

# ------------------------------------------------------------------------------
# General information and data recovered from the paper for test at p' = 50 kPa
# ------------------------------------------------------------------------------

# Initial volumes
vwi = rdLiu2018.vwi # Initial water volume [cm3]
vsi = rdLiu2018.vsi # Initial solid volume [cm3]
vi = rdLiu2018.vi # Initial sample volume [cm3]

# Values of volume change of water and solid grains due to thermal expansion
temp = rdLiu2018.temp # Temperature [degC]
dvw = rdLiu2018.dvw # Volume change of water [cm3]
dvs = rdLiu2018.dvs # Volume change of solid grains [cm3]


### Verification 1: integration of the thermal expansion of solid grains
bs = rdLiu2018.bs*np.ones(len(temp))
# Exact integration, equation (20) in Coulibaly et al., 2022
dvs_exact = thexp.deltaVth('beta', vsi, bs, temp)
# Small thermal expansion integration, equation (22) in Coulibaly et al., 2022
dvs_small = thexp.deltaVth('small', vsi, bs, temp)
# Linear thermal expansion formula, equation (11) in Coulibaly et al., 2022
dvs_lin = thexp.deltaVth('linear', vsi, bs, temp)

plt.figure(1)
plt.plot(temp, dvs, 'ko', label="Liu et al., 2018 (Table 3)")
plt.plot(temp, dvs_exact, label="Exact integration")
plt.plot(temp, dvs_small, label="Small coefficient")
plt.plot(temp, dvs_lin, label="Linear")
#plt.plot(temp, vs_v - vs_v_lin, label="Correction")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel(r'Volume change of solid grains, $\Delta V_s$ [cm^3]')
plt.title("Integration of thermal expansion of solid grains")
plt.legend()

np.savetxt("tab_verif_Liu2018_integration_solid.csv",
           np.concatenate((temp[:,np.newaxis], dvs[:,np.newaxis],
                           dvs_exact[:,np.newaxis], dvs_small[:,np.newaxis],
                           dvs_lin[:,np.newaxis]), axis=1),
           header=("temp_degC,dVs_Liu2018_cm3,dVs_exact_cm3,"+
                   "dVs_small_cm3,dVs_linear_cm3"),
           delimiter=',')
# Good agreement between results of Liu et al., 2018 and formulas for the
# thermal expansion of the grains. As expected.


### Verification 2: integration of the thermal expansion of water

# The text mentions that thermal expansion of water is computed using the linear
# Equation $\Delta V_w = \beta_w V_w \Delta T$ (equation (4) and (5)). The data
# of Table 3 is compared to the different integration formulas proposed by
# Coulibaly and Rotta Loria, 2022 to verify which one is actually used by
# Liu et al., 2018

# Different integrations of the thermal expansion of CRC 40th edition
bw = thexp.coef_w_CRC40ed(temp) # CRC Handbook 40th ed, 1958
# Exact integration, equation (21) in Coulibaly et al., 2022
dvw_exact_CRC = thexp.deltaVth('beta', vwi, bw, temp)
# Small thermal expansion integration, equation (23) in Coulibaly et al., 2022
dvw_small_CRC = thexp.deltaVth('small', vwi, bw, temp)
# Linear thermal expansion formula, equation (12) in Coulibaly et al., 2022
dvw_lin_CRC = thexp.deltaVth('linear', vwi, bw, temp)

plt.figure(2)
plt.plot(temp, dvw, 'ko', label="Liu et al., 2018 (Table 3)")
plt.plot(temp, dvw_exact_CRC, label="Exact integration (CRC)")
plt.plot(temp, dvw_small_CRC, label="Small coefficient (CRC)")
plt.plot(temp, dvw_lin_CRC, label="Linear (CRC)")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel(r'Volume change of water, $\Delta V_w$ [cm^3]')
plt.title("Integration of thermal expansion of water")
plt.legend()

# The results of Liu et al., 2018 could not be retrieved by any integration of
# the thermal expansion coefficient of water from the CRC Handbook 40th edition
# Let's try the formula of Baldi et al., 1988 for back pressure of 300 kPa
u = rdLiu2018.u # Back pressure [Pa]
bw = thexp.coef_w_Baldi88(u,temp) # Baldi et al., 1988
# Exact integration, equation (21) in Coulibaly et al., 2022
dvw_exact_Baldi88 = thexp.deltaVth('beta', vwi, bw, temp)
# Small thermal expansion integration, equation (23) in Coulibaly et al., 2022
dvw_small_Baldi88 = thexp.deltaVth('small', vwi, bw, temp)
# Linear thermal expansion formula, equation (12) in Coulibaly et al., 2022
dvw_lin_Baldi88 = thexp.deltaVth('linear', vwi, bw, temp)

plt.figure(2)
plt.plot(temp, dvw_exact_Baldi88, '--', label="Exact (Baldi et al., 300 kPa)")
plt.plot(temp, dvw_small_Baldi88, '--', label="Small (Baldi et al., 300 kPa)")
plt.plot(temp, dvw_lin_Baldi88, '--', label="Linear (Baldi et al., 300 kPa)")
plt.legend()

np.savetxt("tab_verif_Liu2018_integration_water.csv",
           np.concatenate((temp[:,np.newaxis], dvw[:,np.newaxis],
                           dvw_exact_CRC[:,np.newaxis],
                           dvw_small_CRC[:,np.newaxis],
                           dvw_lin_CRC[:,np.newaxis],
                           dvw_exact_Baldi88[:,np.newaxis],
                           dvw_small_Baldi88[:,np.newaxis],
                           dvw_lin_Baldi88[:,np.newaxis]),
                          axis=1),
           header=("temp_degC,dVw_Liu2016_cm3,dVw_CRC_exact_cm3,"+
                   "dVw_CRC_small_cm3,dVw_CRC_linear_cm3,"+
                   "dVw_Baldi88_exact_cm3,dVw_Baldi88_small_cm3,"+
                   "dVw_Baldi88_linear_cm3"),
           delimiter=',')
# The results are in better agreement with the formula of Baldi et al., 1988 !
# It seems likely that Liu et al., 2018 actually used the formula of Baldi et
# al., 1988 but mistakenly credited it to Campanella and Mitchell, 1968!

# //////////////////////////////
### EVERYTHING BELOW THIS LINE IS TEMPORARY MEMO TO BE DELETED LATER ###


# Data digitized from Figure 7 and linearly interpolated from 0 to 9h
ti = 0.
tf = 9. # 9h total test
nstep = (int)((tf-ti)*60) # 1 minute time steps
time = np.linspace(ti, tf, nstep, dtype=float) # Time [h]

# Temperature computed as the average of the outer and inner temperature
temp_in = np.genfromtxt("Liu2018_temp_in.csv", delimiter=',', names=True)
temp_out = np.genfromtxt("Liu2018_temp_out.csv", delimiter=',', names=True)
temp_in_interp = np.interp(time,temp_in["time_h"],temp_in["temp_C"])
temp_out_interp = np.interp(time,temp_out["time_h"],temp_out["temp_C"])
temp = 0.5*(temp_in_interp + temp_out_interp) # Temperature of the sample [degC]

# Volume of water expelled 
vexp = np.genfromtxt("Liu2018_expelled_volume.csv", delimiter=',', names=True)
vdr = np.interp(time,vexp["time_h"],vexp["vol_mm3"])

# Thermal expansion coefficient of water from IAPWS-95 at 300 kPa
alpha_w = thexp.w_IAPWS95("water_IAPWS95_300kPa_20-80-0.5degC", temp)



# Exact integration, equation (15) in Coulibaly et al., 2022
# TO CHANGE: $\Delta V / V_i = \exp(\int_{T_i}^{T_f} \alpha(T) dT) - 1$
int_alphadt = integrate.cumtrapz(alpha_w, temp, initial=0) # \int \alpha(T) dT
exp_int = np.exp(int_alphadt)
exp_minus_int = np.exp(-int_alphadt)
v_w_exact = exp_int*(v_w_i - integrate.cumtrapz(exp_minus_int, vdr, initial=0))
# - Finite (forward) difference explicit integration to back-up exact formula
v_w_fd = np.zeros(len(time))
v_w_fd[0] = v_w_i
for i in range(len(time)-1):
  v_w_fd[i+1] = (v_w_fd[i] + 
                 v_w_fd[i]*alpha_w[i]*(temp[i+1] - temp[i]) -
                 (vdr[i+1] - vdr[i]))

