"""
Comparison of integration formulas for the thermal expansion of solid grains and
water, and for conservation equations. Provides data for section 3.4 (Figures 5
and 6) of Coulibaly and Rotta Loria 2022

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

from tidal.core import thexp
from tidal.core import inteq
from tidal import data
from tidal.data import rdNg2016
from tidal.data import rdLiu2018


# Temperature range for the calculation [25 - 55] degC
ti = 25 # Initial temperature [degC]
tf = 55 # Final temperature [degC]
tincr = 1 # Temperature increment [degC]
temp = np.arange(ti, tf + 0.5*tincr, tincr, dtype=float)

# ------------------------------------------------------------------------------
# [1] Comparison of thermal expansion integration formulas for solid grains
# ------------------------------------------------------------------------------

# Volumetric thermal expansion coefficient of the solid grains
bs_lo = 3e-5 # Value used by Ng et al., 2016 [1/degC]
bs_hi = 3.5e-5 # Value used by Liu et al., 2018 [1/degC]
bs_kos = thexp.vcte_s_Kosinski91(temp, 5) # 5th order value used
                                          # by Kosinski et al., 1991 [1/degC]

# Relative volume change, use unit initial volume (vsi = 1) to make relative
# Exact integration, equation (20) in Coulibaly et al., 2022
dvs_kos_exact = inteq.deltaVth('beta', 1.0, bs_kos, temp)*1e2
# Small expansion integration, equation (22) in Coulibaly et al., 2022
dvs_kos_small = inteq.deltaVth('small', 1.0, bs_kos, temp)*1e2
# Linear formula, equation (11) in Coulibaly et al., 2022
dvs_kos_lin = inteq.deltaVth('linear', 1.0, bs_kos, temp)*1e2
dvs_lo_lin = inteq.deltaVth('linear', 1.0, bs_lo, temp)*1e2
dvs_hi_lin = inteq.deltaVth('linear', 1.0, bs_hi, temp)*1e2

# Plot results and export to comma-separated tables
plt.figure(1)
plt.plot(temp, dvs_kos_exact, label="Exact (Kosinski et al., 1991)")
plt.plot(temp, dvs_kos_small, label="Small (Kosinski et al., 1991)")
plt.plot(temp, dvs_kos_lin, label="Linear (Kosinski et al., 1991)")
plt.plot(temp, dvs_lo_lin, label=r"Linear ($\beta_s=$"+str(bs_lo)+" 1/degC)")
plt.plot(temp, dvs_hi_lin, label=r"Linear ($\beta_s=$"+str(bs_hi)+" 1/degC)")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel(r'Thermal expansion of solid grains $\Delta V_s^{th}/V_{s,i}$ [%]')
plt.title("Figure 3 of Coulibaly and Rotta Loria 2022")
plt.legend()

np.savetxt("tab_integration_solid_expansion.csv",
           np.concatenate((temp[:,np.newaxis], dvs_kos_exact[:,np.newaxis],
                           dvs_kos_small[:,np.newaxis],
                           dvs_kos_lin[:,np.newaxis],
                           dvs_lo_lin[:,np.newaxis],
                           dvs_hi_lin[:,np.newaxis]), axis=1),
           header=("temp_degC,dVs_Vsi_kos91_exact_pct,dVs_Vsi_kos91_small_pct,"+
                   "dVs_Vsi_kos91_linear_pct,dVs_Vsi_bs=3e-5_lin_pct,"+
                   "dVs_Vsi_bs=3.5e-5_lin_pct"),
           delimiter=',')

# ------------------------------------------------------------------------------
# [2] Comparison of thermal expansion integration formulas for water
# ------------------------------------------------------------------------------

# Volumetric thermal expansion coefficient of water from IAPWS-95 at atmospheric
# pressure.
# Add 1 increment of padding to the temperature for the IAPWS-95 so that thermal
# expansion is calculated with 2nd order central differences at first/last value
tempad = np.concatenate([[2*temp[0]-temp[1]],temp,[2*temp[-1]-temp[-2]]])
bw, rhow = thexp.vcte_w_IAPWS95_tab(data.path_IAPWS95_1atm, tempad)
bw = bw[1:-1]
rhow = rhow[1:-1] # Density of water
rhow0 = rhow[0] # Density of water at room temperature T0 assuming T0=Ti

# Relative volume change, use unit initial volume (vwi = 1) to make relative
# Exact integration, equation (21) in Coulibaly et al., 2022
dvwth_exact = inteq.deltaVth('beta', 1.0, bw, temp)*1e2
#           = inteq.deltaVth('rho', 1.0, rhow)*1e2 # Alternative using density
# Small expansion integration, equation (23) in Coulibaly et al., 2022
dvwth_small = inteq.deltaVth('small', 1.0, bw, temp)*1e2
# Linear formula, equation (12) in Coulibaly et al., 2022
dvwth_lin = inteq.deltaVth('linear', 1.0, bw, temp)*1e2

# Plot results and export to comma-separated tables
plt.figure(2)
plt.plot(temp, dvwth_exact, label="Exact (IAPWS-95)")
plt.plot(temp, dvwth_small, label="Small (IAPWS-95)")
plt.plot(temp, dvwth_lin, label="Linear (IAPWS-95)")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel('Thermal expansion of initial water volume '+
           r'$\Delta V_w^{th}/V_{w,i}$ [%]')
plt.title("Figure 4a of Coulibaly and Rotta Loria 2022")
plt.legend()

np.savetxt("tab_integration_water_expansion.csv",
           np.concatenate((temp[:,np.newaxis],
                           dvwth_exact[:,np.newaxis],
                           dvwth_small[:,np.newaxis],
                           dvwth_lin[:,np.newaxis]), axis=1),
           header=("temp_degC,dVwth_Vwi_exact_pct,dVwth_Vwi_small_pct,"+
                   "dVwth_Vwi_lin_pct"),
           delimiter=',')

# ------------------------------------------------------------------------------
# [3] Volume correction for porous dummy sample: TO BE COMPLETED !!!
# ------------------------------------------------------------------------------

bm = 3e-5*np.ones(temp.size) # Get realistic value from some material, e.g., stainless steel
# Volume correction for porous dummy sample [mm3]. Use vme_por = 0, and unit
# initial volume (vwi = 1) to make relative
# Exact integration, equation (26) in Coulibaly et al., 2022
dvcal_exact = inteq.deltaVcal_por('exact', 0.0, 1.0, bw, bm, temp, rhow, rhow0)
# Simple integration, equation (27) in Coulibaly et al., 2022
dvcal_simple = inteq.deltaVcal_por('simple', 0.0, 1.0, bw, bm, temp)
errdvcal = (dvcal_exact - dvcal_simple)*1e2

plt.figure(10)
plt.plot(temp, errdvcal)
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel('Relative error on porous dummy volume correction '+
           r'$(\Delta V_{cal}^{exact} - \Delta V_{cal}^{simple})/V_{w,i}$ [%]')
plt.title("Figure 4a of Coulibaly and Rotta Loria 2022")
plt.legend()


# ------------------------------------------------------------------------------
# [4] Volume of expelled water and drainage - expansion coupling
# ------------------------------------------------------------------------------

for (ref, study, vu, fnameIAPWS95) in zip(['Ng2016', 'Liu2018'],
                                          [rdNg2016, rdLiu2018],
                                          ['mm3', 'cm3'],
                                          [data.path_IAPWS95_200kPa,
                                           data.path_IAPWS95_200kPa]):
  # (1) Expelled water volume data from Ng et al., 2016, test D70S200TC
  # (2) Expelled water volume data from Liu et al., 2018, test at p' = 50 kPa
  t = study.temp # Temperature [degC]
  dvme = study.dvme # Measured water volume [mm3/cm3]
  dvcal = study.dvcal # Volume correction [mm3/cm3]
  vi = study.vi # Initial volume [mm3/cm3]

  # Linear interpolation (non-monotonic)
  npt = 500 # Number of interpolation points
  temp = np.interp(np.linspace(0,t.size-1,npt), np.arange(t.size), t)
  dvme_interp = np.interp(np.linspace(0,dvme.size-1,npt),
                         np.arange(dvme.size), dvme)
  dvcal_interp = np.interp(np.linspace(0,dvcal.size-1,npt),
                           np.arange(dvcal.size), dvcal)

  # Volumetric thermal expansion coefficient of water from IAPWS-95
  # Add padding using linear extrapolation of temperature so thermal expansion
  # is calculated with 2nd order central differences at first/last value
  tempad = np.concatenate([[2*temp[0]-temp[1]],temp,[2*temp[-1]-temp[-2]]])
  bw, rhow = thexp.vcte_w_IAPWS95_tab(fnameIAPWS95, tempad)
  bw = bw[1:-1]
  rhow = rhow[1:-1] # Density of water
  rhow0 = rhow[0] # Density of water at room temperature T0 assuming T0=Ti

  # Volume of expelled water [mm3]. Equation (8) and (24) of Coulibaly and
  # Rotta Loria 2022
  # Neglect density ratio
  dvdr_vc = inteq.deltaVdr('vc', dvme_interp, dvcal_interp)
  # Exact integration
  dvdr_mc = inteq.deltaVdr('mc', dvme_interp, dvcal_interp, rhow, rhow0)
  # Relative error between volume/mass conservation expressions [%]
  errvdr = (dvdr_mc - dvdr_vc)/vi*1e2

  # Coupled drainage-expansion volume change of water [mm3]
  # Coupled term (25) always obtained using Vdr with density ratio from (24)
  dvw_dr = inteq.deltaVw_dr('beta', bw, dvdr_mc, temp) # Vdr from Equation (24)
  #      = inteq.deltaVw_dr('rho', rhow, dvdr_mc) # Using density
  #      = inteq.deltaVw_dr('ratio', rhow, dvdr_vc, rhow0) # Using density ratio
  # Relative error between coupled/uncoupled expressions [%]
  # Uncoupled term obtained using either (24), to isolate effects of coupling
  # or using (8) to highlight the effects of both density ratio and coupling in
  # comparison to usual method, i.e. equation (8) and vdw_dr = vdr (uncoupled)
  errvwdrvc = (dvw_dr - dvdr_vc)/vi*1e2 # Decoupled Vdr from Equation (8)
  errvwdrmc = (dvw_dr - dvdr_mc)/vi*1e2 # Decoupled Vdr from Equation (24)

  # Plot results and export to comma-separated tables
  plt.figure(3)
  plt.plot(temp, errvdr, label=ref)
  plt.xlabel(r'Temperature $T$ [degC]')
  plt.ylabel('Relative error on mass/volume conservation equation '+
             r'$(\Delta V_{dr}^{mc}-\Delta V_{dr}^{vc})/V_i$ [%]')
  plt.title("Figure 6a of Coulibaly and Rotta Loria 2022")
  plt.legend()

  plt.figure(4)
  plt.plot(temp, errvwdrvc, label=ref+" (decoupled Vdr from (8))")
  plt.plot(temp, errvwdrmc, label=ref+" (decoupled Vdr from (24))")
  plt.xlabel(r'Temperature $T$ [degC]')
  plt.ylabel('Relative difference between coupled/uncoupled drainage-expansion'+
             r' $\Delta V_w^{dr}/V_i$ [%]')
  plt.title("Figure 6b of Coulibaly and Rotta Loria 2022")
  plt.legend()

  np.savetxt("tab_integration_water_expelled_and_coupling_"+ref+".csv",
             np.concatenate((temp[:,np.newaxis],
                             dvdr_vc[:,np.newaxis],
                             dvdr_mc[:,np.newaxis],
                             errvdr[:,np.newaxis],
                             dvw_dr[:,np.newaxis],
                             errvwdrvc[:,np.newaxis],
                             errvwdrmc[:,np.newaxis]), axis=1),
             header=("temp_degC,dVdr_vc_"+vu+",dVdr_mc_"+vu+
                     ",(dVdr_mc-dVdr_vc)/Vi_pct,dVw_dr_"+vu+
                     ",(dVw_dr-dVdr_vc)/Vi_pct,(dVw_dr-dVdr_mc)/Vi_pct"),
             delimiter=',')

