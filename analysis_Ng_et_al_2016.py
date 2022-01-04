"""
Critical analysis the results of Ng et al., 2016: Volume change behaviour
of saturated sand, Geotechnique Letters. 6:124-131
DOI: https://doi.org/10.1680/jgele.15.00148

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
import matplotlib.pyplot as plt
import thexp

ONETHIRD = 1./3

# ------------------------------------------------------------------------------
# General information and data recovered from the paper for test D70S200TC
# Test D70S200TC chosen because most of the data is available through the paper
# ------------------------------------------------------------------------------

ei = 0.71 # Initial void ratio (Table 1)
ni = ei/(1.0 + ei) # Initial porosity
pfi = 1.0 - ni # Initial packing fraction (complement to 1 of porosity)

### Values of volume change from Table 3
# Temperature [degC]
temp = np.array([23, 30, 40, 50, 40, 30, 23], dtype=float)
# Expelled volume [mm^3]
vme = np.array([0, 132, 337, 527, 464, 342, 291], dtype=float)
# Volume calibration (vde) [mm^3]
vde = np.array([0, 15, 41, 76, 55, 25, 7], dtype=float)
# Water volume variation / total volume (vw/vi) [%]
vw_v = np.array([0, 0.08, 0.224, 0.399, 0.233, 0.08, 0])
# Solid volume variation / total volume (vs/vi) [%]
vs_v = np.array([0, 0.003, 0.006, 0.01, 0.006, 0.003, 0])
# Leaked volume / total volume (mu*t/vi) [%]
mut_v = np.array([0, 0.04, 0.088, 0.129, 0.225, 0.265, 0.31])
# Volumetric strain [%]
ev = np.array([0, 0.014, 0.028, -0.016, 0.014, 0.022, 0.022])

# The initial volume, not provided the paper, is back-calculated using the data
# from Table 3 and equation (1). An average value of vi = 85689 mm^3 is
# calculated from all the rows in Table 3 except for the first row which
# corresponds to the initial state and is only zeros.
vi = np.average((vme[1:]-vde[1:])/
                (ev[1:] + vw_v[1:] + vs_v[1:] + mut_v[1:]))*1e2

vw = vw_v*vi*1e-2 # Variation of water volume [mm3]
vs = vs_v*vi*1e-2 # Variation of solid volume [mm3]
vwi = vi*ni # Initial water volume [mm^3]
vsi = vi - vwi # Initial solid volume [mm^3]

### Analysis 1: integration of the thermal expansion of water and grains
# The text mentions that thermal expansion of water is computed using the linear
# Equation $\Delta V_w = \alpha_w V_w \Delta T$ (Table 2). The data of Table 3
# is compared to the different integration formulas proposed by Coulibaly and
# Rotta Loria, 2022 to verify which one is actually used by Ng et al., 2016.

# Different integrations of the thermal expansion of Baldi et al., 1988 used by
# Ng et al., 2016. The confining and back pressures used by Ng et al., 2016
# are not detailed in the paper. A back pressure of u = 200 kPa is mentionned
# only once in the caption of Figure 1 and is considered in this calculation.

u = 200e3 # Back pressure [Pa]
bw = thexp.coef_w_Baldi88(u,temp) # Baldi et al., 1988
# Exact integration, equation (9) in Coulibaly et al., 2022
vw_v_exact = thexp.deltaVth(ni, bw, temp, 'exact')*1e2
# Small thermal expansion integration, equation (10) in Coulibaly et al., 2022
vw_v_small = thexp.deltaVth(ni, bw, temp, 'small')*1e2
# Linear thermal expansion formula, equation (12) in Coulibaly et al., 2022
vw_v_lin = thexp.deltaVth(ni, bw, temp, 'linear')*1e2

plt.figure(1)
plt.plot(temp, vw_v, 'ko', label=r"Ng et al., 2016 (Table 3)")
plt.plot(temp, vw_v_exact, label=r"Exact integration")
plt.plot(temp, vw_v_small, label=r"Small coefficient")
plt.plot(temp, vw_v_lin, label=r"Linear")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel('Volume change of water relative to sample volume, '+
           r'$\Delta V_w/V_i$ [%]')
plt.title("Integration of thermal expansion of water")
plt.legend()

np.savetxt("tab_Ng2016_integration_water.csv",
           np.concatenate((temp[:,np.newaxis], vw_v[:,np.newaxis],
                           vw_v_exact[:,np.newaxis], vw_v_small[:,np.newaxis],
                           vw_v_lin[:,np.newaxis]), axis=1),
           header=("temp_degC,dVw_Vi_Ng2016_pct,dVw_Vi_exact_pct,"+
                   "dVw_Vi_small_pct,dVw_Vi_linear_pct"),
           delimiter=',')
# The results seem to show that Ng et al., 2016 actually integrated the thermal
# expansion of water adequately, i.e. using equation () by Coulibaly and Rotta
# Loria 2022, but wrongly wrote the linearized formula in text

# We verify the correctness of the thermal expansion of the grains in Table 3
# p. 127 "the linear thermal expansion [...] of sand [...] is 1e-5 1/degC"
bs = 3*1e-5*np.ones(len(temp))
# Exact integration, equation (9) in Coulibaly et al., 2022
vs_v_exact = thexp.deltaVth(pfi, bs, temp, 'exact')*1e2
# Small thermal expansion integration, equation (11) in Coulibaly et al., 2022
vs_v_small = thexp.deltaVth(pfi, bs, temp, 'small')*1e2
# Linear thermal expansion formula, equation (12) in Coulibaly et al., 2022
vs_v_lin = thexp.deltaVth(pfi, bs, temp, 'linear')*1e2

plt.figure(2)
plt.plot(temp, vs_v, 'ko', label=r"Ng et al., 2016 (Table 3)")
plt.plot(temp, vs_v*4.7, 'ro', label=r"Ng et al., 2016 (x4.7)")
plt.plot(temp, vs_v_exact, label=r"Exact integration")
plt.plot(temp, vs_v_small, label=r"Small coefficient")
plt.plot(temp, vs_v_lin, label=r"Linear")
plt.plot(temp, vs_v - vs_v_lin, label="Correction")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel('Volume change of grains relative to sample volume, '
           r'$\Delta V_s/V_i$ [%]')
plt.title("Integration of thermal expansion of grains")
plt.legend()

np.savetxt("tab_Ng2016_integration_solid.csv",
           np.concatenate((temp[:,np.newaxis], vs_v[:,np.newaxis],
                           vs_v_exact[:,np.newaxis], vs_v_small[:,np.newaxis],
                           vs_v_lin[:,np.newaxis]), axis=1),
           header=("temp_degC,dVs_Vi_Ng2016_pct,dVs_Vi_exact_pct,"+
                   "dVs_Vi_small_pct,dVs_Vi_linear_pct"),
           delimiter=',')
# The resuls show a very important mismatch between the data reported in Table 3
# and the results computed here based on the information available in the paper.
# The values in Table 3 are about 4.7 times smaller than what they should be !
# The fact that this factor is consistent show that there must be a mistake in
# the calculation of the thermal expansion of the grains by Ng et al., 2016




### Analysis 2: Uncertainty quantification
# Conservative assumption: accuracy considered equal to 3 standard deviations
# When no values of accuracy or standard deviations are given, conservative
# estimates suggested by Coulibaly and Rotta Loria 2022 are used.

# In the work of Ng et al., 2016, the leakage is added to the formula for the
# calculation of the thermally induced strain equation (1). This term is not
# present in the formula of Coulibaly and Rotta Loria 2022 and in the functions
# developped ehrein. To account for it, the volume of leakage will be added as
# part of the calibration volume, i.e. vcal = vde + mut. Because the volume of
# leakage is obtained by measurment using the same volume-pressure controller
# that measures all water volumes, the standard deviation of the calibration
# volume will be affected by a factor of sqrt(2). This factor comes from the
# fact that variances on both measurements are considered equal and independent
# so that var(vcal) = 2*var(vde), and s_vcal = sqrt(2)*s(vde)

# Initial density of solid [g/mm^3]
rhosi = 2.65*1e-3 # Toyoura sand, from Verdugo and Ishihara, 1996
s_rhosi = ONETHIRD*0.01*1e-3 # Standard deviation of initial density (estimate, Coulibaly Rotta Loria 2022)

# Mass of solid grains [g]
ms = rhosi*vsi # Computed from initial solid volume and density
s_ms = ONETHIRD*0.01 # Standard deviation of solid grains mass (estimated, Coulibaly Rotta Loria 2022)

# Measured expelled volume of water [mm^3]
# vme given above in the general data section
s_vme = ONETHIRD*9 # accuracy of 9 mm^3 (bottom of Table 2)

# Calibration of the expelled volume measurement [mm^3]
vcal = vde + mut_v*vi*1e-2 # Leakage and expansion of drainage, both calibration
s_vcal = ONETHIRD*9*np.sqrt(2) # accuracy of 9 mm^3, same as vme

# Thermal expansion coefficient of the solid grains [1/degC]
# bs computed above during Analysis 1
s_bs = ONETHIRD*3.7e-5 # Accuracy estimate from Campanella and Mitchell 1968

# Thermal expansion coefficient of water [1/degC]
# bw computed above during Analysis 1
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
plt.plot(temp,uq[1][dic["vi"]], label='factor V, Ng et al., 2016')
plt.plot(temp,uq[1][dic["vme"]], label='factor Vme, Ng et al., 2016')
plt.plot(temp,uq[1][dic["vcal"]], label='factor Vcal, Ng et al., 2016')
plt.plot(temp,uq[1][dic["ms"]], label='factor ms and rhosi, Ng et al., 2016')
plt.plot(temp,uq[1][dic["bs"]], label='factor bs, Ng et al., 2016')
plt.plot(temp,uq[1][dic["bw"]], label='factor bw, Ng et al., 2016')
plt.plot(temp,uq[1][dic["t"]], label='factor t, Ng et al., 2016')
plt.xlabel(' Temperature [degC]')
plt.ylabel('Error factors [-]')
plt.title("Uncertainty quantification for Ng et al., 2016")
plt.legend()

plt.figure(4)
plt.plot(ev, temp, 'k', label='D70S200TC')
plt.plot(ev + 1*uq[0]*1e2, temp, '--', label='D70S200TC+1std')
plt.plot(ev + 2*uq[0]*1e2, temp, '--', label='D70S200TC+2std')
plt.plot(ev + 3*uq[0]*1e2, temp, '--', label='D70S200TC+3std')
plt.plot(ev - 1*uq[0]*1e2, temp, '-.', label='D70S200TC-1std')
plt.plot(ev - 2*uq[0]*1e2, temp, '-.', label='D70S200TC-2std')
plt.plot(ev - 3*uq[0]*1e2, temp, '-.', label='D70S200TC-3std')
plt.xlabel('Volumetric strain [%]')
plt.ylabel('Temperature [degC]')
plt.title("Uncertainty quantification for Ng et al., 2016")
plt.legend()

np.savetxt("tab_Ng2016_UQ.csv",
           np.concatenate((temp[:,np.newaxis], uq[1][dic["vi"]][:,np.newaxis],
                           uq[1][dic["vme"]][:,np.newaxis],
                           uq[1][dic["vcal"]][:,np.newaxis],
                           uq[1][dic["ms"]][:,np.newaxis],
                           uq[1][dic["bs"]][:,np.newaxis],
                           uq[1][dic["bw"]][:,np.newaxis],
                           uq[1][dic["t"]][:,np.newaxis]), axis=1),
           header=("temp_degC,F_Vi_Ng2016,F_Vme_Ng2016,F_Vcal_Ng2016,"+
                   "F_ms_rhosi_Ng2016,F_bs_Ng2016,F_bw_Ng2016,F_temp_Ng2016"),
           delimiter=',')














### EVERYTHING BELOW THIS LINE IS TEMPORARY MEMO TO BE DELETED LATER ###







# Initial porosity, void ratio
n = 1.0 - ms/rhos/vi
e = (1.0 - n) / n

# Temperature
ti = 20 # Initial tmperature [degC]
tf = 55 # Finale [degC]
dt = 1 # Temperature increment [degC]
t = np.arange(ti,tf+dt,dt)
delt = t-t[0] # temperature variation
s_t = 0.1 # Standard deviation of current temperature [degC]
cov_t = s_t / t # # Coefficient of variation of temperature [m^3]

# Expelled water volume
vdr = time_series() # Volume of water expelled, corrected [m^3]
s_vcal = 1 # Standard deviation of volume measurement [m^3]
cov_vcal = s_vcal / vdr # Coefficient of variation of volume measurement [m^3]

# Thermal expansion of water
# Thermal expansion computed using the small variations assumption
a_w = thexp.coef_w_IAPWS95_tab("water_IAPWS95_100kPa_10-90-0.5degC", temp) # Thermal expansion of water [1 / degC]
int_aw = thexp.deltaV_thexp(1.0, a_w, t, 'small') # Relative change based on vi = 1
s_aw = 0.0 # Standard deviation of Thermal expansion of water [1 / degC] # Considered negligible
cov_aw = s_aw / a_w # Coefficient of variation of Thermal expansion of water [1 / degC]

# Thermal expansion of solid grains
bs = 3.5e-5 # Thermal expansion of solid grains [1 / degC]
int_as = thexp.deltaV_thexp(1.0, bs, t, 'small') # Relative change based on vi = 1
s_as = 1e-5 # Standard deviation of Thermal expansion of solid grains [1 / degC]
cov_as = s_as / bs # Coefficient of variation of Thermal expansion of solid grains [1 / degC]
































































# Based on the results of Ng et al., 2016 (Figure 6)
# It is not known what formula they used to integrate the thermal expansion
# WARNING: the test at p'=200 kPa is made with u=200 kPa, they don't tell for p' = 50 kPa what is the value of "u", so assuming it might be wrong, cannot really prove it until we know
# Do we have their water volume? total volume ? no need if we have initial porosity
# Initial porosity n = V_{w,i}/V_i, considering an initial water volume of 1 since all is relative

# Toyoura sand (Verdugo and Ishihara, 1996)
# Maximum void ratio: emax = 0.977
# Minimum void ratio: emin = 0.597
# Initial relative density: Dr = 20 %
# Initial void ratio: ei = emax - Dr*(emax - emin) = 0.901
# Initial porosity: n = e/(1+e) = 0.4739

e = 0.901 # Initial void ratio
n = 0.4739 # Initial porosity

# Average initial volume back-calculated from the strain






int_as = vs/vsi # true value of integral (at least true to how they computed it, which is unclear)
int_aw = vw/vwi # true value of integral (at least true to how they computed it, which is unclear)




# Assume integral equal alpha*delta T. how big is the gap? results is smaller than 1 for increasing values, and goes down to 0.77
# It is not conservative to assume it is equal to 1 if it has not been done that way, but we can't track it for Liu, assume 1

# We could also reverse engineer Liu to see if they computed the thermal expansion according to CRC (Campanella and Mitchell)






















# Based on the results of Ng et al., 2016 (Figure 6)
# It is not known what formula they used to integrate the thermal expansion
# WARNING: the test at p'=200 kPa is made with u=200 kPa, they don't tell for p' = 50 kPa what is the value of "u", so assuming it might be wrong, cannot really prove it until we know
# Do we have their water volume? total volume ? no need if we have initial porosity
# Initial porosity n = V_{w,i}/V_i, considering an initial water volume of 1 since all is relative

# Toyoura sand (Verdugo and Ishihara, 1996)
# Maximum void ratio: emax = 0.977
# Minimum void ratio: emin = 0.597
# Initial relative density: Dr = 20 %
# Initial void ratio: ei = emax - Dr*(emax - emin) = 0.901
# Initial porosity: n = e/(1+e) = 0.4739

v_w_i = 0.4739 # Initial volume of water, chosen as initial porosity (relative to 1)
dt = 0.5 # Temperature increment for integration [degC]

# Data digitized from Figure 6
# 3-column: olumetric strain [%]: "ev", Temperature [degC]: "T", temperature variation [degC]: "dT"
Ng_200kPa = np.genfromtxt("Ng_2016_Dr20_P200kPa.csv", delimiter=',', names=True)
Ng_50kPa = np.genfromtxt("Ng_2016_Dr20_P50kPa.csv", delimiter=',', names=True)

temp = np.arange(Ng_200kPa["T"][0],Ng_200kPa["T"][-1]+dt,dt)

# Thermal expansion according to Baldi et al., 1988 for given pressures
alpha_Baldi88_50kPa = thexp.coef_w_Baldi88(50e3,temp) # 50 kPa
alpha_Baldi88_200kPa = thexp.coef_w_Baldi88(200e3,temp) # 200 kPa

# Thermal expansion according to IAPWS-95 for given pressures
alpha_IAPWS95_50kPa = thexp.coef_w_IAPWS95_tab("water_IAPWS95_50kPa_10-90-0.5degC", temp)
alpha_IAPWS95_200kPa = thexp.coef_w_IAPWS95_tab("water_IAPWS95_200kPa_10-90-0.5degC", temp)

# Correction of volumetric strain due to thermal expansion of water, exact integration
dVw_cor_50kPa = thexp.deltaV_thexp(v_w_i, alpha_IAPWS95_50kPa, temp, 'exact') - thexp.deltaV_thexp(v_w_i, alpha_Baldi88_50kPa, temp, 'exact')
dVw_cor_200kPa = thexp.deltaV_thexp(v_w_i, alpha_IAPWS95_200kPa, temp, 'exact') - thexp.deltaV_thexp(v_w_i, alpha_Baldi88_200kPa, temp, 'exact')


temp_plot_200kPa = np.in1d(temp,Ng_200kPa["T"])
temp_plot_50kPa = np.in1d(temp,Ng_50kPa["T"])

plt.figure(1)
plt.plot(Ng_200kPa["ev"], Ng_200kPa["T"], label="Ng et al., 2016, 200 kPa")
plt.plot(Ng_50kPa["ev"], Ng_50kPa["T"], label="Ng et al., 2016, 50 kPa")
plt.plot(Ng_200kPa["ev"] - dVw_cor_200kPa[temp_plot_200kPa]*1e2, Ng_200kPa["T"], linestyle='--', label="corrected, 200 kPa")
plt.plot(Ng_50kPa["ev"] - dVw_cor_50kPa[temp_plot_50kPa]*1e2, Ng_50kPa["T"], linestyle='--', label="corrected, 50 kPa")
plt.xlabel('Volumetric strain [%]')
plt.ylabel(' Temperature [degC]')
plt.title("Correction of Toyoura sand at Dr=20% for Ng et al., 2016")
plt.legend()

