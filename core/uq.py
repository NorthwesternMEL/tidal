"""
Module that performs uncertainty quantification for the volumetric strain.

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
from scipy.stats import t as studt
from . import inteq

# Exnternally (publically) accessed methods
__all__ = ['propagUQ', 'std_vi', 'std_vsi', 'std_dt']


def propagUQ(mean, std, m, c=0.95):
  """
  Compute the propagation of uncertainty for the volumetric strain.

  Arguments:
  ----------
    mean : list of scalars and numpy arrays
      Mean values of the variables
    std : list of scalars and numpy arrays
      Standard deviation of the variables
    m : int
      Number of tests performed
    c : float
      Confidence level, default: 95%
  Input lists `mean` and `std` index variables using the following fixed order:
  [0] Initial volume of the sample: "vi"
  [1] Measured (not corrected) volume change of water: "vme"
  [2] Correction for the volume of expelled water: "vcal"
  [3] Initial volume of solid grains: "vsi"
  [4] Volumetric thermal expansion coefficient of solid grains: "bs"
  [5] Volumetric thermal expansion coefficient of water: "bw"
  [6] Temperature variation: "delt"

  Returns:
  --------
    out[0] : list of numpy arrays
      Error factors ordered with the reference indexing
    out[1] : float
      Variance of the thermally induced volumetric strain
    out[2] : float
      Critical value of Student's t-distribution
    out[3] : float
      Half width of the confidence interval
  """
  ### Unpack variables (for clarity)
  # Mean value
  vi = mean[0]
  vme = np.copy(mean[1]) # Copy for potential modification to avoid
  vcal = np.copy(mean[2]) # division by zero in the COV calculations
  vsi = mean[3]
  bs = mean[4]
  bw = mean[5]
  delt = mean[6]
  # Standard deviations
  s_vi = std[0] # Determined from function `std_vi()`
  s_vme = std[1]
  s_vcal = std[2]
  s_vsi = std[3] # Determined from function `std_vsi()`
  s_bs = std[4]
  s_bw = std[5]
  s_dt = std[6] # Determined from function `std_dt()`

  ### Useful quantities
  ni = 1.0 - vsi/vi # Initial porosity
  pfi = 1.0 - ni # Initial packing fraction (complement to 1 of porosity)
  # Relative thermal expansion computed using the small variations assumption
  # Compute integral of the difference between coefficients of water and grains
  # Integral coordinate "dx" can be temperature variation, no change in value
  int_bwbs = inteq.deltaVth('small', 1.0, bw - bs, delt)

  ### Factors that multiply squared coefficient of variation of each variables
  f = [None]*len(mean)
  # Total volume. Equation (34) of Coulibaly and Rotta Loria, 2022
  f[0] = ((vme - vcal)/vi + pfi*int_bwbs)**2
  # Measured expelled volume. Equation (35) of Coulibaly and Rotta Loria, 2022
  f[1] = (vme/vi)**2
  # Volume correction. Equation (36) of Coulibaly and Rotta Loria, 2022
  f[2] = (vcal/vi)**2
  # Volume of solid grains. Equation (37) of Coulibaly and Rotta Loria, 2022
  f[3] = (pfi*int_bwbs)**2
  # Thermal expansion of solid. Equation (38) of Coulibaly and Rotta Loria, 2022
  f[4] = (pfi*bs*delt)**2
  # Thermal expansion of water. Equation (39) of Coulibaly and Rotta Loria, 2022
  f[5] = (ni*bw*delt)**2
  # Temperature variation. Equation (40) of Coulibaly and Rotta Loria, 2022
  f[6] = (pfi*bs*delt + ni*bw*delt)**2

  # Compute coefficients of variations
  # Avoid division by zero for volumes and temperature variation
  # Zero volume are changed to infinite so that division in COV is zero
  vme[np.logical_and(vme<=0, vme>=0)] = np.inf
  vcal[np.logical_and(vcal<=0, vcal>=0)] = np.inf
  delt[np.logical_and(delt<=0, delt>=0)] = np.inf
  cov_vi = s_vi/vi
  cov_vme = s_vme/vme
  cov_vcal = s_vcal/vcal
  cov_vsi = s_vsi/vsi
  cov_bs = s_bs/bs
  cov_bw = s_bw/bw
  cov_dt = s_dt/delt

  v_ev = (f[0]*cov_vi**2 + f[1]*cov_vme**2 + f[2]*cov_vcal**2 +
          f[3]*cov_vsi**2 + f[4]*cov_bs**2 + f[5]*cov_bw**2 + f[6]*cov_dt**2)

  tstar = studt.ppf(.5*(1+c),5*(m-1)) # Safe for m=1, returns nan
  return f, v_ev, tstar, tstar*np.sqrt(v_ev/m)


def std_vi(s_viprep, s_vmec, s_vmuc):
  """
  Compute the standard deviation of the initial volume of the sample.

  This is a simple calculation that could be implemented directly by the users.
  It is defined as a standalone function to provide standardized and centralized
  procedures and avoid possible implementation mistakes by users.

  Arguments:
  ----------
    s_viprep : float
      Standard deviation of sample volume after preparation
    s_vmec : float
      Standard deviation of measured volume change during consolidation
    s_vmuc : float
      Standard deviation of leakage volume change during consolidation

  Returns:
  --------
    out : float
      Standard deviation of the initial sample volume before thermal loading
  """
  return np.sqrt(s_viprep**2 + s_vmec**2 + s_vmuc**2)


def std_vsi(ms, rhosi, s_ms, s_rhosi):
  """
  Compute the standard deviation of the initial volume of the solid grains.

  This is a simple calculation that could be implemented directly by the users.
  It is defined as a standalone function to provide standardized and centralized
  procedures and avoid possible implementation mistakes by users.

  Arguments:
  ----------
    ms : float
      Mean initial mass of solid grains
    rhosi : float
      Mean initial density of solid grains
    s_ms : float
      Standard deviation of the initial mass of solid grains
    s_rhosi : float
      Standard deviation of the initial density of solid grains

  Returns:
  --------
    out : float
      Standard deviation of the initial volume of solid grains
  """
  return np.sqrt((s_ms/rhosi)**2 + (s_rhosi/ms)**2)


def std_dt(s_t):
  """
  Compute the standard deviation of the temperature variation.

  This is a simple calculation that could be implemented directly by the users.
  It is defined as a standalone function to provide standardized and centralized
  procedures and avoid possible implementation mistakes by users.

  Arguments:
  ----------
    s_t: float
      Standard deviation of the temperature measurement / sensor

  Returns:
  --------
    out : float
      Standard deviation of the temperature variation
  """
  return np.sqrt(2)*s_t
