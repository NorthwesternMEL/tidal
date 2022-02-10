"""
Module that performs integration of the differential and conservation equations.

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
import scipy.integrate as integrate

# Exnternally (publically) accessed methods
__all__ = ['deltaVth', 'deltaVw_dr', 'deltaVdr', 'deltaVcal_por',
           'resid_sol_por']


def deltaVth(f, vxi, x, t=None):
  """
  Compute the variation of volume of water / grains due to thermal expansion.

  Arguments:
  ----------
    f : {'rho','beta', 'small', 'linear'}
      Integration method. Exact, small coefficient or linear formula available
      after equation (21-24) and (11-12) of Coulibaly and Rotta Loria, 2022.
      Exact integration available using density 'rho' or thermal expansion
      coefficient 'beta'. Simplified integrations 'small' and 'linear' only
      available using thermal expansion coefficient
    vxi : float
      Initial volume of solid grains / water inside the sample
    x : numpy array
      Density or volumetric thermal expansion coefficient of grains / water
    t : numpy array, optional
      Temperature, only used for 'beta', 'small', and 'linear'

  Returns:
  --------
    out : numpy array
      Volume variation of solid grains / water due to thermal expansion only
  """
  if (f == 'rho'):
    # Exact integration: equations (21-22) of Coulibaly and Rotta Loria, 2022
    # Volume variation given in terms of the density
    return vxi*(x[0]/x - 1.0)
  elif (f == 'beta'):
    # Exact integration: equations (21-22) of Coulibaly and Rotta Loria, 2022
    return vxi*(np.exp(integrate.cumtrapz(x, t, initial=0)) - 1.0)
  elif (f == 'small'):
    # Small expansion: equation (23-24) of Coulibaly and Rotta Loria, 2022
    return vxi*integrate.cumtrapz(x, t, initial=0)
  elif (f == 'linear'):
    # Linear formula: equation (11-12) of Coulibaly and Rotta Loria, 2022
    return vxi*x*(t-t[0])


def deltaVw_dr(f, x, vdr, t=None):
  """
  Compute the exact variation of the volume of water inside the sample.

  Coupled effects of expelled water and thermal expansion according to equation
  (26) of Coulibaly and Rotta Loria, 2022.

  Arguments:
  ----------
    f : {'rho', 'beta'}
      Integration method. Density 'rho' or thermal expansion coefficient 'beta'
      available
    x : numpy array
      Density or volumetric thermal expansion coefficient of water
    vdr : numpy array
      Volume of water expelled from the sample
    t : numpy array, optional
      Temperature, only used for 'beta'

  Returns:
  --------
    out : numpy array
      Exact volume change of water due to coupled drainage-expansion
  """
  if (f == 'rho'):
    # Volume variation given in terms of the density
    return integrate.cumtrapz(x, vdr, initial=0)/x
  elif (f == 'beta'):
    # Volume variation given in terms of thermal expansion coefficient
    pbw = integrate.cumtrapz(x, t, initial=0) # Primitive of thermal expansion
    expp = np.exp(pbw)
    exppinv = np.exp(-pbw)
    return expp*integrate.cumtrapz(exppinv,vdr,initial=0)


def deltaVdr(f, vme, vcal, rho=None, rho0=None):
  """
  Compute the volume of water expelled outside of the sample bounds.

  Arguments:
  ----------
    f : {'vc', 'mc'}
      Integration method. Mass conservation (exact) 'mc' with density ratio, and
      volume conservation 'vc' (simple difference) available after equations
      (25) and (8) of Coulibaly and Rotta Loria, 2022, respectively
    vme : numpy array
      Measured volume change of water by the PVC controller at room temperature
    vcal : numpy array
      Correction for the volume of expelled water obtained during calibration
    rho : numpy array, optional
      Density of water, only used for 'mc'
    rho0 : float, optional
      Density of water at room temperature, only used for 'mc'

  Returns:
  --------
    out : numpy array
      Volume of water expelled outside of the sample bounds
  """
  vdr = vme - vcal # Simple difference (volume conservation), default for 'vc'
  if (f == 'mc'):
    # Exact integration (mass conservation)
    vdr = rho0*integrate.cumtrapz(1.0/rho, vdr, initial=0)
  return vdr


def deltaVcal_por(f, vme_por, vwi, bw, bm, t, rho=None, rho0=None):
  """
  Compute the volume correction for calibration tests with porous dummy sample.

  Arguments:
  ----------
    f : {'exact', 'simple'}
      Integration method. Exact or simplified integration after equations (26)
      and (27) of Coulibaly and Rotta Loria, 2022, respectively
    vme_por : numpy array
      Measured volume change of water by the PVC controller at room temperature
      during the calibration test on porous dummy sample
    vwi : float
      Initial volume of water inside the sample
    bw : numpy array
      Volumetric thermal expansion coefficient of water
    bm : numpy array
      Volumetric thermal expansion coefficient of dummy sample material
    t : numpy array
      Temperature
    rho : numpy array, optional
      Density of water, only used for 'mc'
    rho0 : float, optional
      Density of water at room temperature, only used for 'mc'

  Returns:
  --------
    out : numpy array
      Volume correction for calibration tests on porous dummy sample
  """
  if (f == 'exact'):
    vw = vwi*np.exp(integrate.cumtrapz(bm, t, initial=0))
    return vme_por - integrate.cumtrapz(rho*vw*(bw - bm), t, initial=0)/rho0
  elif (f == 'simple'):
    return vme_por - vwi*integrate.cumtrapz(bw - bm, t, initial=0)


def resid_sol_por(f, vme_sol, vme_por, vwi, bw, bm, t, rho=None, rho0=None):
  """
  Compute the residual between porous / solid dummy sample calibration tests.

  Arguments:
  ----------
    f : {'exact', 'simple'}
      Integration method. Exact or simplified integration after equations (28)
      and (29) of Coulibaly and Rotta Loria, 2022, respectively
    vme_sol : numpay array
      Measured volume change of water by the PVC controller at room temperature
      during the calibration test on solid dummy sample
    vme_por : numpy array
      Measured volume change of water by the PVC controller at room temperature
      during the calibration test on porous dummy sample
    vwi : float
      Initial volume of water inside the sample
    bw : numpy array
      Volumetric thermal expansion coefficient of water
    bm : numpy array
      Volumetric thermal expansion coefficient of dummy sample material
    t : numpy array
      Temperature
    rho : numpy array, optional
      Density of water, only used for 'mc'
    rho0 : float, optional
      Density of water at room temperature, only used for 'mc'

  Returns:
  --------
    out : numpy array
      Residual volume change between porous and solid dummy sample
  """
  return vme_sol - deltaVcal_por(f, vme_por, vwi, bw, bm, t, rho, rho0)
