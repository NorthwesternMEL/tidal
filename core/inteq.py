"""
Module that performs integration of the differential and conservation equations

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
import scipy.integrate as integrate

# Exnternally (publically) accessed methods
__all__ = ['deltaVth', 'deltaVw_dr', 'deltaVdr', 'deltaVcal_por',
           'resid_sol_por']

# ------------------------------------------------------------------------------
# Computes and returns the variation of volume only due to thermal expansion
# using exact integration, small values of the thermal expansion coefficient, or
# linear formula according to equation (21) to (24), (11) and (12) of Coulibaly
# and Rotta Loria, 2022. Function valid for water and solid grains. Integration
# is available using the density of the material (always exact integration when
# using density). All integrations are available using the thermal expansion
# coefficient, in this case, the additional temperature argument is required.
# Arguments:
#   f: integration formula, string {'rho','beta', 'small', 'linear'}
#   vi: initial volume of solid grains / water inside the sample [mm^3]
#   x: volumetric thermal expansion coefficient [1/degC], numpy array, or,
#       density [kg/m3], numpy array
#   t: temperature [degC], numpy array, dtype=float
# Return value:
#   Volume variation of solid grains / water [mm3], numpy array
# ------------------------------------------------------------------------------
def deltaVth(f, vi, x, t=None):
  delv = 0.0
  if (f == 'rho'):
    # Exact integration: equations (21-22) of Coulibaly and Rotta Loria, 2022
    # Volume variation given in terms of the density
    delv = vi*(x[0]/x - 1.0)
  elif (f == 'beta'):
    # Exact integration: equations (21-22) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i[\exp(\int_{T_i}^T \alpha(T) dT) - 1]$
    delv = vi*(np.exp(integrate.cumtrapz(x, t, initial=0)) - 1.0)
  elif (f == 'small'):
    # Small expansion: equation (23-24) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i \int_{T_i}^T \alpha(T) dT$
    delv = vi*integrate.cumtrapz(x, t, initial=0)
  elif (f == 'linear'):
    # Linear formula: equation (11-12) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i \alpha(T) \Delta T
    delv = vi*x*(t-t[0])
  return delv

# ------------------------------------------------------------------------------
# Computes and returns the variation of water volume inside the sample due to
# the coupled effects of expelled water and thermal expansion according to
# equation (26) of Coulibaly and Rotta Loria, 2022. Integrations available using
# the density, or, the volumetric thermal expansion coefficient of water. When
# using the thermal expansion coefficient, the additional temperature argument
# is required.
# Arguments:
#   f: integration formula, string {'rho', 'beta'}
#   x: volumetric thermal expansion coefficient of water [1/degC], numpy array,
#      or, density of water [kg/m3], numpy array
#   vdr: volume of water expelled from the sample [mm3], numpy array
#   t: temperature [degC], numpy array, dtype=float
# Return value:
#   Volume change of water due to coupled drainage-expansion [mm3], numpy array
# ------------------------------------------------------------------------------
def deltaVw_dr(f, x, vdr, t=None):
  delv = 0.0
  if (f == 'rho'):
    # Volume variation given in terms of the density
    delv = integrate.cumtrapz(x, vdr, initial=0)/x
  elif (f == 'beta'):
    # Volume variation given in terms of thermal expansion coefficient
    pbw = integrate.cumtrapz(x, t, initial=0) # Primitive of thermal expansion
    expp = np.exp(pbw)
    exppinv = np.exp(-pbw)
    delv = expp*integrate.cumtrapz(exppinv,vdr,initial=0)
  return delv

# ------------------------------------------------------------------------------
# Computes and returns the volume of expelled water using exact integration
# accounting for the density ratio (mass conservation), or, using simple
# difference (volume conservation) according to equations (25) and (8) of
# Coulibaly and Rotta Loria, 2022, respectively. When computing exact
# integration, additional density arguments are required. When computing the
# volume difference, only the integration flag, measured volume and volume
# correction should be specified.
# Arguments:
#   f: integration formula, string {'vc', 'mc'}
#   vme: measured volume change of water [mm3]
#   vcal: correction for the volume of expelled water [mm^3], numpy array
#   rho: density of water [kg/m3], numpy array, dtype=float
#   rho0: density of water at room temperature T0 [kg/m3], float
# Return value:
#   Volume of water expelled out of the sample [mm3], numpy array
# ------------------------------------------------------------------------------
def deltaVdr(f, vme, vcal, rho=None, rho0=None):
  vdr = vme - vcal # Simple difference (volume conservation), default for 'vc'
  if (f == 'mc'):
    # Exact integration (mass conservation)
    vdr = rho0*integrate.cumtrapz(1.0/rho, vdr, initial=0)
  return vdr

# ------------------------------------------------------------------------------
# Computes and returns the volume correction for the calibration tests using
# a porous dummy sample, according to equations (26) and (27) of Coulibaly and
# Rotta Loria, 2022, respectively. When computing exact integration, additional
# density arguments are required.
# Arguments:
#   f: integration formula, string {'exact', 'simple'}
#   vme_por: measured water volume for calibration test on porous dummy [mm^3]
#   vwi: initial volume of water inside the sample [mm3]
#   bw: volumetric thermal expansion coefficient of water [1/degC], numpy array
#   bm: volumetric thermal expansion coefficient of dummy [1/degC], numpy array
#   t: temperature [degC], numpy array, dtype=float
#   rho: density of water [kg/m3], numpy array, dtype=float
#   rho0: density of water at room temperature T0 [kg/m3], float
# Return value:
#   Volume correction for calibration test on porous dummy [mm3], numpy array
# ------------------------------------------------------------------------------
def deltaVcal_por(f, vme_por, vwi, bw, bm, t, rho=None, rho0=None):
  if (f == 'exact'):
    vw = vwi*np.exp(integrate.cumtrapz(bm, t, initial=0))
    vcal = vme_por - integrate.cumtrapz(rho*vw*(bw - bm), t, initial=0)/rho0
  elif (f == 'simple'):
    vcal = vme_por - vwi*integrate.cumtrapz(bw - bm, t, initial=0)
  return vcal

# ------------------------------------------------------------------------------
# Computes and returns the residual between calibration tests using a porous
# dummy sample and calibration tests using a solid dummy sample according to
# equations (28) and (29) of Coulibaly and Rotta Loria, 2022, respectively.
# When computing exact integration, additional density arguments are required.
# Arguments:
#   f: integration formula, string {'exact', 'simple'}
#   vme_sol: measured water volume for calibration test on solid dummy [mm^3]
#   vme_por: measured water volume for calibration test on porous dummy [mm^3]
#   vwi: initial volume of water inside the sample [mm^3]
#   bw: volumetric thermal expansion coefficient of water [1/degC], numpy array
#   bm: volumetric thermal expansion coefficient of dummy [1/degC], numpy array
#   t: temperature [degC], numpy array, dtype=float
#   rho: density of water [kg/m3], numpy array, dtype=float
#   rho0: density of water at room temperature T0 [kg/m3], float
# Return value:
#   Residual volume change [mm^3], numpy array
# ------------------------------------------------------------------------------
def resid_sol_por(f, vme_sol, vme_por, vwi, bw, bm, t, rho=None, rho0=None):
  return vme_sol - deltaVcal_por(f, vme_por, vwi, bw, bm, t, rho, rho0)
