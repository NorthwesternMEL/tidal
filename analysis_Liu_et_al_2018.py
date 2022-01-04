"""
Critical analysis the results of Liu et al., 2018: Influence of temperature on
the volume change behavior of saturated sand, Geotechnical Testing Journal.
41(4). DOI: https://www.astm.org/gtj20160308.html

Python 2

Comparison of different volume conservation equations

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
import scipy.integrate as integrate
import thexp

ONETHIRD = 1./3

# ------------------------------------------------------------------------------
# General information and data recovered from the paper for test at p' = 50 kPa
# Test at 50 kPa chosen because most of the data is available through the paper
# ------------------------------------------------------------------------------

vwi = 272.4 # Initial water volume (Figure 7) [cm^3]
vsi = 732.4 # Initial solid volume (Figure 7) [cm^3]
# Initial volume from sum: vi = vwi+vsi = 1004.8 cm^3
# Initial volume from dimension: D_out = 100 mm, D_in = 60 mm, H = 200 mm
# vi = H*pi*(D_out**2 - D_in**2)/4 = 1005.3 cm^3 (slight difference 0.5 cm^3)
vi = vwi + vsi
ei = vwi/vsi # Initial void ratio
ni = vwi/vi # Initial porosity
pfi = 1.0 - ni # Initial packing fraction (complement to 1 of porosity)

### Values of volume change from Table 3, p' = 50 kPa
# Could potentially reverse-engineer initial volume for other pressures

temp = np.array([25, 35, 45, 55], dtype=float) # Temperature [degC]
vme = np.array([0.000, 0.430, 1.020, 1.740]) # Expelled volume [cm^3]
vde = np.array([0.000, 0.090, 0.070, 0.040]) # Volume calibration (vcal) [cm^3]
vdr = vme - vde # Corrected expelled volume [cm^3], MAYBE UNNECESSARY
vw = np.array([0.000, 0.816, 1.852, 3.146]) # Water volume variation (vw) [cm^3]
vs = np.array([0.000, 0.254, 0.511, 0.777]) # Solid volume variation (vs) [cm^3]


### Analysis 1: integration of the thermal expansion of water and grains
# The text mentions that thermal expansion of water is computed using the linear
# Equation $\Delta V_w = \beta_w V_w \Delta T$ (equation (4) and (5)). The data
# of Table 3 is compared to the different integration formulas proposed by
# Coulibaly and Rotta Loria, 2022 to verify which one is actually used by
# Liu et al., 2018

# Different integrations of the thermal expansion of CRC 40th edition used by
# Liu et al., 2018. No pressure dependence in the CRC formula.

bw = thexp.coef_w_CRC40ed(temp) # CRC Handbook 40th ed, 1958
# Exact integration, equation (9) in Coulibaly et al., 2022
vw_exact_CRC = thexp.deltaVth(vwi, bw, temp, 'exact')
# Small thermal expansion integration, equation (10) in Coulibaly et al., 2022
vw_small_CRC = thexp.deltaVth(vwi, bw, temp, 'small')
# Linear thermal expansion formula, equation (12) in Coulibaly et al., 2022
vw_lin_CRC = thexp.deltaVth(vwi, bw, temp, 'linear')

plt.figure(1)
plt.plot(temp, vw, 'ko', label=r"Liu et al., 2018 (Table 3)")
plt.plot(temp, vw_exact_CRC, label=r"Exact integration (CRC)")
plt.plot(temp, vw_small_CRC, label=r"Small coefficient (CRC)")
plt.plot(temp, vw_lin_CRC, label=r"Linear (CRC)")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel(r'Volume change of water, $\Delta V_w$ [cm^3]')
plt.title("Integration of thermal expansion of water")
plt.legend()

# The results of Liu et al., 2018 could not be retrieved by any integration of
# the thermal expansion coefficient of water from the CRC Handbook 40th edition
# Let's try the formula of Baldi et al., 1988 for back pressure of 300 kPa
u = 300e3 # Back pressure [Pa]
bw = thexp.coef_w_Baldi88(u,temp) # Baldi et al., 1988
# Exact integration, equation () in Coulibaly et al., 2022
vw_exact_Baldi88 = thexp.deltaVth(vwi, bw, temp, 'exact')
# Small thermal expansion integration, equation () in Coulibaly et al., 2022
vw_small_Baldi88 = thexp.deltaVth(vwi, bw, temp, 'small')
# Linear thermal expansion formula, equation () in Coulibaly et al., 2022
vw_lin_Baldi88 = thexp.deltaVth(vwi, bw, temp, 'linear')

plt.figure(1)
plt.plot(temp, vw_exact_Baldi88, '--', label=r"Exact (Baldi et al., 300 kPa)")
plt.plot(temp, vw_small_Baldi88, '--', label=r"Small (Baldi et al., 300 kPa)")
plt.plot(temp, vw_lin_Baldi88, '--', label=r"Linear (Baldi et al., 300 kPa)")
plt.legend()

np.savetxt("tab_Liu2018_integration_water.csv",
           np.concatenate((temp[:,np.newaxis], vw[:,np.newaxis],
                           vw_exact_CRC[:,np.newaxis],
                           vw_small_CRC[:,np.newaxis],
                           vw_lin_CRC[:,np.newaxis],
                           vw_exact_Baldi88[:,np.newaxis],
                           vw_small_Baldi88[:,np.newaxis],
                           vw_lin_Baldi88[:,np.newaxis]),
                          axis=1),
           header=("temp_degC,dVw_Liu2016_cm3,dVw_CRC_exact_cm3,"+
                   "dVw_CRC_small_cm3,dVw_CRC_linear_cm3,"+
                   "dVw_Baldi88_exact_cm3,dVw_Baldi88_small_cm3,"+
                   "dVw_Baldi88_linear_cm3"),
           delimiter=',')
# The results are in good agreement with the formula of Baldi et al., 1988 !
# It seems likely that Liu et al., 2018 actually used the formula of Baldi et
# al., 1988 but mistakenly credited it to Campanella and Mitchell, 1968...

# We verify the correctness of the thermal expansion of the grains in Table 3
# p. 6 "a value of 3.5e-3 %/degC is assumed", i.e. 3.5e-5 1/degC
bs = 3.5e-5*np.ones(len(temp))
# Exact integration, equation () in Coulibaly et al., 2022
vs_exact = thexp.deltaVth(vsi, bs, temp, 'exact')
# Small thermal expansion integration, equation () in Coulibaly et al., 2022
vs_small = thexp.deltaVth(vsi, bs, temp, 'small')
# Linear thermal expansion formula, equation () in Coulibaly et al., 2022
vs_lin = thexp.deltaVth(vsi, bs, temp, 'linear')

plt.figure(2)
plt.plot(temp, vs, 'ko', label=r"Liu et al., 2018 (Table 3)")
plt.plot(temp, vs_exact, label=r"Exact integration")
plt.plot(temp, vs_small, label=r"Small coefficient")
plt.plot(temp, vs_lin, label=r"Linear")
#plt.plot(temp, vs_v - vs_v_lin, label="Correction")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel(r'Volume change of grains, $\Delta V_s$ [cm^3]')
plt.title("Integration of thermal expansion of grains")
plt.legend()

np.savetxt("tab_Liu2018_integration_solid.csv",
           np.concatenate((temp[:,np.newaxis], vs[:,np.newaxis],
                           vs_exact[:,np.newaxis], vs_small[:,np.newaxis],
                           vs_lin[:,np.newaxis]), axis=1),
           header=("temp_degC,dVs_Liu2018_cm3,dVs_exact_cm3,"+
                   "dVs_small_cm3,dVs_linear_cm3"),
           delimiter=',')
# Good agreement between results of Liu et al., 2018 and formulas for the
# thermal expansion of the grains. As expected.



### Analysis 2: Uncertainty quantification
# Conservative assumption: accuracy considered equal to 3 standard deviations
# When no values of accuracy or standard deviations are given, conservative
# estimates suggested by Coulibaly and Rotta Loria 2022 are used.

# Although the paper refers to Campanella and Mitchell 1968 for the thermal
# expansion coefficient of water, it is shown above that the authors likely used
# the formula of Baldi et al., 1988. The formula of Baldi et al., 1988 is used
# in the sequel for all computations

# Initial density of solid [g/mm^3]
# From Xiao et al., 2017 DOI: https://doi.org/10.1680/jgele.16.00144
rhosi = 2.69*1e-3 # Fujian sand
s_rhosi = ONETHIRD*0.01*1e-3 # Standard deviation of initial density (estimate, Coulibaly Rotta Loria 2022)

# Mass of solid grains [g]
ms = rhosi*vsi # Computed from initial solid volume and density
s_ms = ONETHIRD*0.01 # Standard deviation of solid grains mass (estimated, Coulibaly Rotta Loria 2022)

# Measured expelled volume of water [mm^3]
# Vme given above in the general data section
s_vme = ONETHIRD*9 # accuracy of 9 mm^3 (bottom of Table 2)

# Calibration of the expelled volume measurement [mm^3]
vcal = vde
s_vcal = ONETHIRD*9*np.sqrt(2) # accuracy of 9 mm^3, same as Vme

# Thermal expansion coefficient of the solid grains [1/degC]
# bs computed above during Analysis 1
s_bs = ONETHIRD*3.7e-5 # Accuracy estimate from Campanella and Mitchell 1968

# Thermal expansion coefficient of water [1/degC]
# bw computed above during Analysis 1. Formula of Baldi et al., 1988 is used
s_bw = 0.0 # Systematic error using Baldi et al., 1988, but no random error

# Temperature [degC]
# temp obtained above in the general data section
s_temp = ONETHIRD*0.1 # Accuracy estimate, e.g., Cekerevac et al., 2005

# Initial sample volume [mm3]
# vi computed above during Analysis 1
s_vi = ONETHIRD*1134 # Standard deviation of initial volume, estimated from 1mm height accuracy and D~38mm for 2:1 aspect ratio: 1134 mm^3, if height, vi*dh/H also works

dic = {"vi":0, "vme":1, "vcal":2, "ms":3, "rhosi":4, "bs":5, "bw":6, "t":7}
val = [vi, vme, vcal, ms, rhosi, bs, bw, temp]
std = [s_vi, s_vme, s_vcal, s_ms, s_rhosi, s_bs, s_bw, s_temp]
uq = thexp.propagUQ(dic, val, std)

plt.figure(3)
plt.plot(temp,uq[1][dic["vi"]], label='factor V, Liu et al., 2018')
plt.plot(temp,uq[1][dic["vme"]], label='factor Vme, Liu et al., 2018')
plt.plot(temp,uq[1][dic["vcal"]], label='factor Vcal, Liu et al., 2018')
plt.plot(temp,uq[1][dic["ms"]], label='factor ms and rhosi, Liu et al., 2018')
plt.plot(temp,uq[1][dic["bs"]], label='factor bs, Liu et al., 2018')
plt.plot(temp,uq[1][dic["bw"]], label='factor bw, Liu et al., 2018')
plt.plot(temp,uq[1][dic["t"]], label='factor t, Liu et al., 2018')
plt.xlabel(' Temperature [degC]')
plt.ylabel('Error factors [-]')
plt.title("Uncertainty quantification for Liu et al., 2018")
plt.legend()

# plt.figure(4)
# plt.plot(ev, temp, 'k', label='p\' = 50 kPa')
# plt.plot(ev + 1*uq[0]*1e2, temp, '--', label='DR70S200TC+1std')
# plt.plot(ev + 2*uq[0]*1e2, temp, '--', label='DR70S200TC+2std')
# plt.plot(ev + 3*uq[0]*1e2, temp, '--', label='DR70S200TC+3std')
# plt.plot(ev - 1*uq[0]*1e2, temp, '-.', label='DR70S200TC-1std')
# plt.plot(ev - 2*uq[0]*1e2, temp, '-.', label='DR70S200TC-2std')
# plt.plot(ev - 3*uq[0]*1e2, temp, '-.', label='DR70S200TC-3std')
# plt.xlabel('Volumetric strain [%]')
# plt.ylabel(' Temperature [degC]')
# plt.title("Uncertainty quantification for Liu et al., 2018")
# plt.legend()

np.savetxt("tab_Liu2018_UQ.csv",
           np.concatenate((temp[:,np.newaxis], uq[1][dic["vi"]][:,np.newaxis],
                           uq[1][dic["vme"]][:,np.newaxis],
                           uq[1][dic["vcal"]][:,np.newaxis],
                           uq[1][dic["ms"]][:,np.newaxis],
                           uq[1][dic["bs"]][:,np.newaxis],
                           uq[1][dic["bw"]][:,np.newaxis],
                           uq[1][dic["t"]][:,np.newaxis]), axis=1),
           header=("temp_degC,F_Vi_Liu2018,F_Vme_Liu2018,F_Vcal_Liu2018,"+
                   "F_ms_rhosi_Liu2018,F_bs_Liu2018,F_bw_Liu2018,F_temp_Liu2018"),
           delimiter=',')


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



# Exact integration of water thermal expansion, but no coupling with expelled water inside the bracket
v_w_exp1 = exp_int*(v_w_i - vdr)
# Exact integration of water thermal expansion, but no full linearization of expelled water on its own
v_w_lin = exp_int*(v_w_i) - vdr

plt.figure(1)
plt.plot(time, v_w_exact, label="exact ODE integration")
plt.plot(time, v_w_fd, linestyle='--', label="explicit integration")
plt.plot(time, v_w_exp1, label="no coupling ODE integration")
plt.plot(time, v_w_lin, label="linear ODE integration")
plt.xlabel(' Time [h]')
plt.ylabel('Volume of water inside the sample [mm^3]')
plt.title("Volume of water inside cell for Liu et al., 2018")
plt.legend()

# Export the data to comma-separated tables



# Solid grains: compare the different values of thermal expansion coefficient
# used in the literature for quartz sands
# 
alpha_s_lo = 3e-5 # As used by Ng et al., 2016
alpha_s_hi = 3.5e-5 # As used by Liu et al., 2018
alpha_s_Kosinski91 = thexp.s_Kosinski91(temp, 5) # 5th order Kosinski et al., 1991

# Linear thermal expansion integration, equation (12) in Coulibaly et al., 2022
# $\Delta V / V_i = \alpha(T) \Delta T$
dVs_Kosinski91_V_lin = alpha_s_Kosinski91*dtemp
dVs_lo_V_lin = alpha_s_lo*dtemp
dVs_hi_V_lin = alpha_s_hi*dtemp

# Small thermal expansion integration, equation (11) in Coulibaly et al., 2022
# $\Delta V / V_i = \int_{T_i}^{T_f} \alpha(T) dT$
dVs_Kosinski91_V_small = integrate.cumtrapz(alpha_s_Kosinski91, temp, initial=0)


plt.figure(2)
plt.plot(temp, dVs_Kosinski91_V_small*1e2, label=r"$\int_{T_i}^{T_f} \alpha(T) dT$ (Kosinski et al., 1991)")
plt.plot(temp, dVs_Kosinski91_V_lin*1e2, label=r"$\alpha \Delta T$ (Kosinski et al., 1991)")
plt.plot(temp, dVs_lo_V_lin*1e2, label=r"$\alpha \Delta T, \alpha=3\cdot 10^{-5}$ 1/degC")
plt.plot(temp, dVs_hi_V_lin*1e2, label=r"$\alpha \Delta T, \alpha=3.5\cdot 10^{-5}$ 1/degC")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel(r'Quartz relative volumetric thermal expansion $\Delta V_s/V_{s,i}$ [%]')
plt.title("Relative variation of Quartz volume for T=[25;55] degC")
plt.legend()

# Export the data to comma-separated tables
