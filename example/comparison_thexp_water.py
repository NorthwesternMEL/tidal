"""
Comparison of different formulas of the literature used to compute the
volumetric thermal expansion of liquid water at atmospheric pressure
Provides data for Figure 2 of Coulibaly and Rotta Loria 2022

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
import matplotlib.pyplot as plt

from tidal import data
from tidal.core import thexp

# ------------------------------------------------------------------------------
# Comparison of different formulas of the literature for the thermal expansion
# of water at atmospheric pressure
# ------------------------------------------------------------------------------

patm = 101325. # Atmospheric pressure [Pa]
tcw = thexp.k2c(647.096) # Critical temperature of water [degC] (from IAPWS-95)

tlo = 20 # Lower bound temperature [degC]
thi = 80 # Upper bound temperature [degC]
tincr = 0.5 # Temperature increment [degC]
temp = np.arange(tlo, thi + 0.5*tincr, tincr, dtype=float)
# Add 1 increment of padding to the temperature for the IAPWS-95 so that thermal
# expansion is calculated with 2nd order central differences at tlo and thi
tempad = np.arange(tlo-tincr, thi + 0.5*tincr + tincr, tincr, dtype=float)

bw_Baldi88 = thexp.vcte_w_Baldi88(patm,temp) # Baldi et al., 1988
bw_Cekerevac05 = thexp.vcte_w_Cekerevac05(patm,temp) # Cekerevac et al., 2005
bw_Smith54 = thexp.vcte_w_Smith54(tcw, temp) # Smith et al., 2005
bw_Chapman67 = thexp.vcte_w_Chapman67(temp) # Chapman, 1967
bw_IAPWS95 = thexp.vcte_w_IAPWS95_tab(data.path_IAPWS95_1atm, tempad)[0][1:-1]
bw_CRC40ed = thexp.vcte_w_CRC40ed(temp) # CRC Handbook 40th ed, 1958-1959

# Plot results and export to comma-separated tables
plt.figure(1)
plt.plot(temp, bw_Baldi88*1e4, label="Baldi et al., 1988 (p=1 atm)")
plt.plot(temp, bw_Cekerevac05*1e4, label="Cekerevac et al., 2005 (p=1 atm)")
plt.plot(temp, bw_Smith54*1e4, label="Smith et al., 1954")
plt.plot(temp, bw_Chapman67*1e4, label="Chapman, 1967")
plt.plot(temp, bw_IAPWS95*1e4, label="IAPWS-95 (p=1 atm)")
plt.plot(temp, bw_CRC40ed*1e4, label="CRC Handbook, 40th ed.")
plt.xlim(20, 80) # Limit plot to [ 20 ; 80] degC
plt.title("Figure 2a of Coulibaly and Rotta Loria 2022")
plt.xlabel('Temperature [degC]')
plt.ylabel('Volumetric thermal expansion of water [$10^{-4}$ /degC]')
plt.legend()

np.savetxt("tab_comparison_thermal_expansion_formulas_water.csv",
           np.concatenate((temp[:,np.newaxis], bw_Baldi88[:,np.newaxis],
                           bw_Cekerevac05[:,np.newaxis],
                           bw_Smith54[:,np.newaxis], bw_Chapman67[:,np.newaxis],
                           bw_IAPWS95[:,np.newaxis], bw_CRC40ed[:,np.newaxis]),
                          axis=1),
           header=("temperature_degC,bw_Baldi88_1atm_1perdegC,"+
                   "bw_Cekerevac05_1atm_1perdegC,bw_Smith54_1perdegC,"+
                   "bw_Chapman67_1perdegC,bw_IAPWS95_1atm_1perdegC,"+
                   "bw_CRC40ed_1perdegC"),
           delimiter=',')

# ------------------------------------------------------------------------------
# Comparison of Baldi et al., 1988 and IAPWS-95 formulas for different levels of
# pressure: 50, 200, 400, and 1000 kPa
# ------------------------------------------------------------------------------

# Baldi et al., 1988
bw_Baldi88_50kPa = thexp.vcte_w_Baldi88(50e3,temp)
bw_Baldi88_200kPa = thexp.vcte_w_Baldi88(200e3,temp)
bw_Baldi88_400kPa = thexp.vcte_w_Baldi88(400e3,temp)
bw_Baldi88_1000kPa = thexp.vcte_w_Baldi88(1000e3,temp)

# IAPWS-95: tabulated values
bw_IAPWS95_p = [
  thexp.vcte_w_IAPWS95_tab(data.path_IAPWS95_50kPa, tempad)[0][1:-1],
  thexp.vcte_w_IAPWS95_tab(data.path_IAPWS95_200kPa, tempad)[0][1:-1],
  thexp.vcte_w_IAPWS95_tab(data.path_IAPWS95_400kPa, tempad)[0][1:-1],
  thexp.vcte_w_IAPWS95_tab(data.path_IAPWS95_1MPa, tempad)[0][1:-1]]

# IAPWS-95: 3rd degree polynomial fitting of the IAPWS-95 values. Temperature
# range T=[20 ; 80] degC and pressure range p=[50 ; 1000] kPa.
# The coefficients obtained in the variable `bw_IAPWS95_coef` below are used to
# inform the preset 'd3_t20-80_p50-1000' of the fitting function named
# 'vcte_w_IAPWS95_fit' in the 'thexp' module
bw_IAPWS95_fit, bw_IAPWS95_coef = thexp.vcte_w_IAPWS95_fit('compute', temp,
                                                           bw_IAPWS95_p, 3)

# Plot results and export to comma-separated tables
plt.figure(2)
plt.plot(temp, bw_Baldi88_50kPa*1e4, label="Baldi et al., 1988 (p=50 kPa)")
plt.plot(temp, bw_Baldi88_200kPa*1e4, label="Baldi et al., 1988 (p=200 kPa)")
plt.plot(temp, bw_Baldi88_400kPa*1e4, label="Baldi et al., 1988 (p=400 kPa)")
plt.plot(temp, bw_Baldi88_1000kPa*1e4, label="Baldi et al., 1988 (p=1000 kPa)")
plt.plot(temp, bw_IAPWS95_p[0]*1e4, 'x', label="IAPWS-95 (p=50 kPa)")
plt.plot(temp, bw_IAPWS95_p[1]*1e4, '+', label="IAPWS-95 (p=200 kPa)")
plt.plot(temp, bw_IAPWS95_p[2]*1e4, 's', label="IAPWS-95 (p=400 kPa)")
plt.plot(temp, bw_IAPWS95_p[3]*1e4, 'v', label="IAPWS-95 (p=10000 kPa)")
plt.plot(temp, bw_IAPWS95_fit*1e4, 'k-', label="IAPWS-95 (3rd degree poly fit)")
plt.title("Figure 2b of Coulibaly and Rotta Loria 2022")
plt.xlabel('Temperature [degC]')
plt.ylabel('Volumetric thermal expansion of water [$10^{-4}$ 1/degC]')
plt.legend()

np.savetxt("tab_comparison_pressure_IAPWS95_Baldi88.csv",
           np.concatenate((temp[:,np.newaxis], bw_Baldi88_50kPa[:,np.newaxis],
                           bw_Baldi88_200kPa[:,np.newaxis],
                           bw_Baldi88_400kPa[:,np.newaxis],
                           bw_Baldi88_1000kPa[:,np.newaxis],
                           bw_IAPWS95_p[0][:,np.newaxis],
                           bw_IAPWS95_p[1][:,np.newaxis],
                           bw_IAPWS95_p[2][:,np.newaxis],
                           bw_IAPWS95_p[3][:,np.newaxis],
                           bw_IAPWS95_fit[:,np.newaxis]),
                          axis=1),
           header=("temperature_degC,bw_Baldi88_50kPa_1perdegC,"+
                   "bw_Baldi88_200kPa_1perdegC,bw_Baldi88_400kPa_1perdegC,"+
                   "bw_Baldi88_1000kPa_1perdegC,bw_IAPWS95_50kPa_1perdegC,"+
                   "bw_IAPWS95_200kPa_1perdegC,bw_IAPWS95_400kPa_1perdegC,"+
                   "bw_IAPWS95_1000kPa_1perdegC,"+
                   "bw_IAPWS95_fit_d3_t20-80_p50_1000_1perdegC"),
           delimiter=',')
