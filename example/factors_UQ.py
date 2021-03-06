"""
Determination of the error factors for uncertainty quantification based on the
measurements of Ng et al., 2016 and Liu et al., 2018. Provides data for section
3.5 (Figure 7) of Coulibaly and Rotta Loria 2022

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
from tidal.core import uq
from tidal import data
from tidal.data import rdNg2016
from tidal.data import rdLiu2018


# ------------------------------------------------------------------------------
# Determination of the error factors for the propagation of ucnertainty
# ------------------------------------------------------------------------------

for i, (ref, study, fname) in enumerate(zip(['Ng2016', 'Liu2018'],
                                            [rdNg2016, rdLiu2018],
                                            [data.path_IAPWS95_200kPa,
                                             data.path_IAPWS95_300kPa])):
  # (1) Measurements data from Ng et al., 2016, test D70S200TC
  # (2) Measurements data from Liu et al., 2018, test at p' = 50 kPa

  # Linear interpolation (non-monotonic)
  npt = 500 # Number of interpolation points
  # Temperature [degC]
  temp = np.interp(np.linspace(0,study.temp.size-1,npt),
                   np.arange(study.temp.size), study.temp)
  # Measured water volume [mm3/cm3]
  dvme = np.interp(np.linspace(0,study.dvme.size-1,npt),
                   np.arange(study.dvme.size), study.dvme)
  # Volume correction [mm3/cm3]
  dvcal = np.interp(np.linspace(0,study.dvcal.size-1,npt),
                   np.arange(study.dvcal.size), study.dvcal)

  # Volumetric thermal expansion coefficient of solid grains, must be given as a
  # numpy array for integration
  bs = study.bs*np.ones(temp.size)
  # Volumetric thermal expansion coefficient of water from IAPWS-95.
  # Add padding using linear extrapolation of temperature so thermal expansion
  # is calculated with 2nd order central differences at first/last value
  tempad = np.concatenate([[2*temp[0]-temp[1]],temp,[2*temp[-1]-temp[-2]]])
  bw = thexp.vcte_w_IAPWS95_tab(fname, tempad)[0][1:-1]

  # Single test: mean value = test value
  val = [study.vi, # Initial volume [mm3/cm3]
         dvme, # Measured water volume [mm3/cm3]
         dvcal, # Volume correction [mm3/cm3]
         study.vsi, # Initial volume of the solid grains [mm3/cm3]
         bs, # Thermal expansion coefficient of solid grains [1/degC]
         bw, # Thermal expansion coefficient of water [1/degC]
         temp - temp[0]] # Temperature varation [degC]
  # Unknown standard deviation for the tests replicated: set all to zero
  # Use dedicated functions for vi, vsi and dt for illustration purposes
  std = [uq.std_vi(0.0, 0.0, 0.0),
         0.0,
         0.0,
         uq.std_vsi(1.0, 1.0, 0.0, 0.0), # ms=rhosi=1 to avoid division by 0
         0.0,
         0.0,
         uq.std_dt(0.0)]

  # Compute factors only (unknown standard deviations, single test)
  f = uq.propagUQ(val, std, 1)[0]

  plt.figure(i)
  plt.plot(temp, f[0], label='factor V, '+ref)
  plt.plot(temp, f[1], label='factor Vme, '+ref)
  plt.plot(temp, f[2], label='factor Vcal, '+ref)
  plt.plot(temp, f[3], label='factor Vsi, '+ref)
  plt.plot(temp, f[4], label='factor bs, '+ref)
  plt.plot(temp, f[5], label='factor bw, '+ref)
  plt.plot(temp, f[6], label='factor dT, '+ref)
  plt.xlabel(' Temperature [degC]')
  plt.ylabel('Error factors [-]')
  plt.title("Figure 7"+chr(97+i)+" of Coulibaly and Rotta Loria 2022")
  plt.legend()

  np.savetxt("tab_factors_UQ_"+ref+".csv",
             np.concatenate((temp[:,np.newaxis], f[0][:,np.newaxis],
                             f[1][:,np.newaxis], f[2][:,np.newaxis],
                             f[3][:,np.newaxis], f[4][:,np.newaxis],
                             f[5][:,np.newaxis], f[6][:,np.newaxis]), axis=1),
             header=("temp_degC,F_Vi,F_Vme,F_Vcal,F_Vsi,F_bs,F_bw,F_temp"),
             delimiter=',')
