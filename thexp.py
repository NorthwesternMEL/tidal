"""

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

# Utility functions for temperature conversion
FIVENINTH = 5./9
# Converts degrees Celsius to degrees Farenheit
def c2f(t_degC):
  return t_degC*1.8 + 32.0
  
# Converts degrees Celsius to Kelvin
def c2k(t_degC):
  return t_degC + 273.15

# Converts Kelvin to degrees Celsius
def k2c(t_K):
  return t_K - 273.15

# Converts degrees Farenheit to degrees Celsius
def f2c(t_degF):
  return (t_degF - 32.0)*FIVENINTH

# Converts Kelvin to degrees Rankine
def k2r(t_K):
  return t_K*1.8
  
# Converts degrees Rankine to Kelvin
def r2k(t_degR):
  return t_degR*FIVENINTH


# ------------------------------------------------------------------------------
# Computes and returns volumetric thermal expansion coefficient of water from
# Baldi et al., 1988: Thermal volume changes of the mineral-water system in
# low-porosity clay soils. Canadian Geotechnical Journal
# DOI: https://doi.org/10.1139/t88-089
# Arguments:
#   p: water pressure [Pa], scalar
#   t: temperature [degC], numpy array
# Return value:
#   volumetric thermal expansion coefficient of water [1/degC], numpy array
# ------------------------------------------------------------------------------
def coef_w_Baldi88(p, t):
  # Equation [8], p. 815
  a0 = 4.505e-4 # [1/degC]
  a1 = 9.156e-5 # [1/degC]
  a2 = 6.381e-6 # [1/degC]
  b1 = -1.2e-6 # [1/degC^2]
  b2 = -5.766e-8 # [1/degC^2]
  m = 1.5e-9 # [1/Pa], given as 0.15 [1/kbar] in the original paper
  return a0 + (a1 + b1*t)*np.log(m*p) + (a2 + b2*t)*(np.log(m*p)**2)

# ------------------------------------------------------------------------------
# Computes and returns volumetric thermal expansion coefficient of water from
# Cekerevac et al., 2005: A Novel Triaxial Apparatus for Thermo-Mechanical
# Testing of Soils. Geotechnical Testing Journal
# DOI: https://doi.org/10.1520/GTJ12311
# Identical to Baldi et al. 1988, but with m = 15 MPa^-1
# Arguments:
#   p: water pressure [Pa], scalar
#   t: temperature [degC], numpy array
# Return value:
#   volumetric thermal expansion coefficient of water [1/degC], numpy array
# ------------------------------------------------------------------------------
def coef_w_Cekerevac05(p, t):
  # Equation (7), p. 8
  a0 = 4.505e-4 # [1/degC]
  a1 = 9.156e-5 # [1/degC]
  a2 = 6.381e-6 # [1/degC]
  b1 = -1.2e-6 # [1/degC^2]
  b2 = -5.766e-8 # [1/degC^2]
  m = 1.5e-5 # [1/Pa], given as 15 [1/MPa] in the original paper
  return a0 + (a1 + b1*t)*np.log(m*p) + (a2 + b2*t)*(np.log(m*p)**2)

# ------------------------------------------------------------------------------
# Computes and returns volumetric thermal expansion coefficient of organic
# liquids from Smith et al., 1954: Correlation of critical temperatures with
# thermal# expansion coefficients of organic liquids. The Journal of Physical
# Chemistry. DOI: https://doi.org/10.1021/J150515A016
# As discussed in Chapman 1974, Heat Transfer, 3rd edition. Macmillan
# ISBN: 0023214503
# Equation (27), p. 33 of Chapman 1974 is given for temperature in deg Rankine
# so the prefactor is 0.06284 = 0.04314*(5/9)**(-0.641), instead of 0.04314. The
# factor 5/9 comes from the conversion from degR variations to degC variations
# Arguments:
#   tc: critical temperature [degC], scalar
#   t: temperature [degC], numpy array
# Return value:
#   volumetric thermal expansion coefficient of liquids [1/degC], numpy array
# ------------------------------------------------------------------------------
def coef_w_Smith54(tc, t):
  # Equation (7A), p. 445 of Smith et al., 1954
  return 0.04314*(tc-t)**(-0.641)

# ------------------------------------------------------------------------------
# Computes and returns volumetric thermal expansion coefficient of saturated
# water from Chapman 1974, Heat Transfer, 3rd edition. Macmillan
# ISBN: 0023214503
# The tabulated values are linearly interpolated
# Arguments:
#   t: temperature in range [0 ; 287] [degC], numpy array dtype=float
# Return value:
#   volumetric thermal expansion coefficient of water [1/degC], numpy array
# ------------------------------------------------------------------------------
def coef_w_Chapman74(t):
  # Temperature [degF] and thermal expansion [1/degR] from Table A.4 p. 586
  t_tab_degF = np.array([32,40,50,60,70,80,90,100,110,120,130,140,150,160,170,
  180,190,200,210,220,230,240,250,260,270,280,290,300,350,400,450,500,550])
  b_tab_perdegR = np.array([0.03,0.045,0.07,0.1,0.13,0.15,0.18,0.2,0.22,0.24,
  0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.48,0.5,0.51,0.53,
  0.55,0.56,0.58,0.62,0.72,0.93,1.18,1.63])*1e-3
  # Conversion of temperatures to degC and expansion to 1/degC (same as 1/K)
  t_tab_degC = f2c(t_tab_degF)
  b_tab_perdegC = k2r(b_tab_perdegR) # K = 5/9 * degR --> 1/K = 9/5 * 1/degR
  # Linear interpolation of tabulated values
  return np.interp(t,t_tab_degC,b_tab_perdegC)

# ------------------------------------------------------------------------------
# Computes and returns volumetric thermal expansion coefficient of water from
# CRC Handbook of Chemistry and Physics, 40th edition. 1958-1959
# Arguments:
#   t: temperature [degC], numpy array dtype=float
# Return value:
#   volumetric thermal expansion coefficient of water [1/degC], numpy array
# ------------------------------------------------------------------------------
def coef_w_CRC40ed(t):
  # Cubic polynomial for volume change: V = Vi*(1 + a*t + b*t^2 + c*t^3) p. 2247
  a = -0.06427e-3 # [1/degC]
  b = 8.5053e-6 # [1/degC^2]
  c = -6.7900e-8  # [1/degC^3]
  return (a + 2*b*t + 3*c*t**2)/(1 + a*t + b*t**2 + c*t**3)

# ------------------------------------------------------------------------------
# Computes and return volumetric thermal expansion coefficient of liquid water
# from IAPWS-95: International Association for the Properties of Water and Steam
# Equation of State (EOS): Wagner and Pruss, 2002. The IAPWS Formulation 1995
# for the Thermodynamic Properties of Ordinary Water Substance for General and
# Scientific Use. DOI: https://doi.org/10.1063/1.1461829.
# Arguments: 
#   ifile: path to file with tabulated properties from the National Institute of
#          Standards and Technology (NIST) Chemistry WebBook Standard Reference
#          Database 69 (SRD 69): https://webbook.nist.gov/chemistry/fluid/
#          (accessed 15Oct2021) (temperature [degC], density [kg/m3], isobaric)
#          Recommanded temperature increment for adequate computations: 0.5 degC
#   ofile: (optional) path to output CSV file with temperature and expansion
#   t: temperature [degC], numpy array
# Return value:
#   volumetric thermal expansion coefficient of water [1/degC], numpy array
#   density of water [kg/m^3], numpy array
# ------------------------------------------------------------------------------
def coef_w_IAPWS95_tab(ifile, t, ofile=None):
  data = np.genfromtxt(ifile, delimiter='\t', names=True)
  # Thermal expansion computed from specific volume v and density \rho=1/v as:
  # $\alpha = \frac{1}{v} \frac{\partial v}{\partial T}$
  # $\alpha = \rho \frac{\partial v}{\partial T}$
  # All values computed using 2nd order central difference except for the
  # first / last values computed using 1st order forward / backward difference
  # Do not use first/last values: less precise, only here to maintain size

  thexp = np.zeros(len(data))
  thexp[0] = ((data["Volume_m3kg"][1] - data["Volume_m3kg"][0])/
              (data["Temperature_C"][1] - data["Temperature_C"][0]))
  thexp[-1] = ((data["Volume_m3kg"][-1] - data["Volume_m3kg"][-2])/
               (data["Temperature_C"][-1] - data["Temperature_C"][-2]))
  thexp[1:-1] = ((data["Volume_m3kg"][2:] - data["Volume_m3kg"][0:-2])/
                 (data["Temperature_C"][2:] - data["Temperature_C"][0:-2]))
  thexp *= data["Density_kgm3"]
  
  if (ofile is not None):
    np.savetxt(ofile,
               np.concatenate((data["Temperature_C"][:,np.newaxis],
                               thexp[:,np.newaxis]),axis=1),
               fmt = ['%.2f', '%.3e'],
               header="temperature_degC,thermal_expansion_1perdegC",
               delimiter=",")
  return (np.interp(t,data["Temperature_C"],thexp),
          np.interp(t,data["Temperature_C"],data["Density_kgm3"]))

# ------------------------------------------------------------------------------
# Computes and returns best-fit polynomial for the volumetric thermal expansion
# coefficient of liquid water from IAPWS-95 on chosen temperature and pressure
# range. The function can compute the polynomial fit, or return existing
# hard-coded values previously computed using this function. When computing new
# values, additional arguments are required. When accessing existing values,
# only the preset value sought and the temperature range should be specified.
# Arguments:
#   preset: determines how the function is operated, string. Must be 'compute'
#   when new value is computed, or an existing preset string when accessing
#   existing value
#   t: temperature [degC], numpy array
#   bwl: thermal expansion coefficients at various pressure, [1/degC],
#        list of numpy arrays (all arrays must be the same length as t)
#   deg: degree of polynomial fit
#   ofile: (optional) path to output CSV file with temperature and expansion
# Return value:
#   volumetric thermal expansion coefficient of water [1/degC], numpy array
#   coefficient of the polynomial fit
# ------------------------------------------------------------------------------
def coef_w_IAPWS95_fit(preset, t, bwl=None, deg=None, ofile=None):
  if (preset == 'compute'):
    n = len(bwl)
    coef = np.polyfit(np.tile(t,n), np.concatenate(bwl), deg)
  elif (preset == 'd3_t20-80_p50-1000'):
    # Preset values obtained for 3rd degree polynomial,
    # temperature range T=[20;80] degC, pressure range p=[50;1000] kPa
    # in example file "comparison_thermal_expansion_formulas_water.py"
    # Equation (11) in Coulibaly and Rotta Loria, 2022
    coef = [4.416308e-10, -1.030440e-07, 1.381959e-05, -3.054531e-05]

  fit = np.polyval(coef, t)
  if (ofile is not None):
    np.savetxt(ofile,
               np.concatenate((t[:,np.newaxis],fit[:,np.newaxis]),axis=1),
               fmt = ['%.2f', '%.3e'],
               header="temperature_degC,thermal_expansion_1perdegC",
               delimiter=",")
  return fit, coef

# ------------------------------------------------------------------------------
# Computes and returns volumetric thermal expansion coefficient of crystalline
# quartz from Kosinski et al., 1991: Thermal Expansion of Alpha Quartz,
# Proceedings of the 45th Annual Symposium on Frequency Control 1991, IEEE
# DOI: https://doi.org/10.1109/FREQ.1991.145883
# The 3rd, 4th and 5th-order fit of the values of linear thermal expansion (CTE)
# coefficient along the a-axis and c-axis, recommended by the High Temperature
# Materials, Mechanical, Electronic and Thermophysical Properties Information
# Analysis Center (HTMIAC) are used. The volumetric thermal expansion is defined
# as: \beta = 2*CTE_{a-axis} + CTE_{c-axis}
# Arguments: 
#   t: temperature in range [-50 ; 150] [degC], numpy array, dtype=float
# Return value:
#   volumetric thermal expansion coefficient of quartz [1/degC], numpy array
# ------------------------------------------------------------------------------
def coef_s_Kosinski91(t, order):
  #
  # Best fit coefficients for HTMIAC data, from Table I p. 24
  # Linear coefficient of thermal expansion along the a-axis
  coef_a = {3:[13.1e-6, 25.0e-9, -64.7e-12, 45.1e-15, 0, 0],
            4:[13.1e-6, 25.8e-9, -54.3e-12, -320.3e-15, 1827e-18, 0],
            5:[13.1e-6, 26.1e-9, -60.6e-12, -466.2e-15, 4645e-18, -11270e-21]}
  # Linear coefficient of thermal expansion along the c-axis
  # There is a sign error in the original paper of Kosinksi et al. for the
  # 4-th order coefficient a3. Table I says a3 = -200.0, fitted data show that
  # it should be a plus sign. a3 = +200.0 is used in the present function
  coef_c = {3:[7.1e-6, 17.7e-9, -48.4e-12, 111.7e-15, 0, 0],
            4:[7.1e-6, 17.5e-9, -50.9e-12, +200.0e-15, -441.6e-18, 0],
            5:[7.1e-6, 17.2e-9, -45.5e-12, 323.0e-15, -2817e-18, 9503e-21]}
  # Volumetric coefficient of thermal expansion
  coef_vol = 2*np.array(coef_a[order]) + np.array(coef_c[order])

  return (coef_vol[0] +
          coef_vol[1]*t +
          coef_vol[2]*t**2 +
          coef_vol[3]*t**3 +
          coef_vol[4]*t**4 +
          coef_vol[5]*t**5)

# ------------------------------------------------------------------------------
# Computes and returns the variation of volume only due to thermal expansion
# using exact integration, small values of the thermal expansion coefficient, or
# linear formula according to equation (X), (Y), and (Z), respectively, of
# Coulibaly and Rotta Loria, 2022. Function valid for water and solid grains
# Arguments:
#   vi: initial volume of solid grains / water inside the sample [mm^3]
#   b: volumetric thermal expansion coefficient[1/degC], numpy array
#   t: temperature [degC], numpy array, dtype=float
#   f: integration formula, string {'exact', 'const', 'small', 'linear'}
# Return value:
#   Volume variation of solid grains / water [mm^3], numpy array
# ------------------------------------------------------------------------------
def deltaVth(vi, b, t, f):
  delv = 0.0
  if (f == 'exact'):
    # Exact integration: equation (X) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i[\exp(\int_{T_i}^T \alpha(T) dT) - 1]$
    delv = vi*(np.exp(integrate.cumtrapz(b, t, initial=0)) - 1.0)
  elif (f == 'small'):
    # Small expansion: equation (X2) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i \int_{T_i}^T \alpha(T) dT$
    delv = vi*integrate.cumtrapz(b, t, initial=0)
  elif (f == 'linear'):
    # Linear formula: equation (Y) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i \alpha(T) \Delta T
    delv = vi*b*(t-t[0])
  return delv

# ------------------------------------------------------------------------------
# Computes and returns the variation of water volume inside the sample due to
# the coupled effects of expelled water and thermal expansion according to
# equation (24) of Coulibaly and Rotta Loria, 2022.
# Arguments:
#   bw: volumetric thermal expansion coefficient of water [1/degC], numpy array
#   vdr: volume of water expelled from the sample [mm^3], numpy array
#   t: temperature [degC], numpy array, dtype=float
# Return value:
#   Volume change of water due to coupled drainage-expansion [mm^3], numpy array
# ------------------------------------------------------------------------------
def deltaVw_dr(bw, vdr, t):
  primbw = integrate.cumtrapz(bw, t, initial=0) # Primitive of thermal expansion
  expp = np.exp(primbw)
  exppinv = np.exp(-primbw)
  return expp*integrate.cumtrapz(exppinv,vdr,initial=0)

# ------------------------------------------------------------------------------
# Computes and returns the volume of expelled water using exact integration
# accounting for the density ratio (mass conservation), or, using simple
# difference (volume conservation) according to equations (25) and (8) of
# Coulibaly and Rotta Loria, 2022, respectively. When computing exact
# integration, additional density arguments are required. When computing the
# volume difference, only the measured volume, volume correction and integration
# flag should be specified.
# Arguments:
#   vme: measured volume change of water [mm^3]
#   vcal: correction for the volume of expelled water [mm^3], numpy array
#   f: integration formula, string {'vc', 'mc'}
#   rho: density of water [kg/m3], numpy array, dtype=float
#   rho0: density of water at room temperature T0 [kg/m3], float
# Return value:
#   Volume of water expelled out of the sample [mm^3], numpy array
# ------------------------------------------------------------------------------
def deltaVdr(vme, vcal, f, rho=None, rho0=None):
  vdr = vme - vcal # Simple difference (volume conservation, default for 'vc')
  if (f == 'mc'):
    # Exact integration with mass conservation
    vdr = rho0*integrate.cumtrapz(1.0/rho, vdr, initial=0)
  return vdr

# ------------------------------------------------------------------------------
# Computes and returns the volume correction for the calibration tests using
# a porous dummy sample, according to equations (26) and (27) of Coulibaly and
# Rotta Loria, 2022, respectively. When computing exact integration, additional
# density arguments are required.
# Arguments:
#   vme_por: measured water volume for calibration test on porous dummy [mm^3]
#   vwi: initial volume of water inside the sample [mm^3]
#   bw: volumetric thermal expansion coefficient of water [1/degC], numpy array
#   bm: volumetric thermal expansion coefficient of dummy [1/degC], numpy array
#   t: temperature [degC], numpy array, dtype=float
#   f: integration formula, string {'exact', 'simple'}
#   rho: density of water [kg/m3], numpy array, dtype=float
#   rho0: density of water at room temperature T0 [kg/m3], float
# Return value:
#   Volume correction for calibration test on porous dummy [mm^3], numpy array
# ------------------------------------------------------------------------------
def deltaVcal_por(vme_por, vwi, bw, bm, t, f, rho=None, rho0=None):
  if (f == 'exact'):
    vw = vwi*np.exp(integrate.cumtrapz(bm, t, initial=0))
    vcal = vme_por - integrate.cumtrapz(rho*vw*(bw - bm), t, initial=0)/rho0
  elif (f == 'simple'):
    vcal = vme_por - vwi*integrate.cumtrapz(bw - bm, t, initial=0)
  return vcal

# ------------------------------------------------------------------------------
# Computes and returns the residual between calibration tests using a porous
# dummy sample and calibration tests using a solid dummy sample according to
#  equations (28) and (29) of Coulibaly and Rotta Loria, 2022, respectively.
# When computing exact integration, additional density arguments are required.
# Arguments:
#   vme_sol: measured water volume for calibration test on solid dummy [mm^3]
#   vme_por: measured water volume for calibration test on porous dummy [mm^3]
#   vwi: initial volume of water inside the sample [mm^3]
#   bw: volumetric thermal expansion coefficient of water [1/degC], numpy array
#   bm: volumetric thermal expansion coefficient of dummy [1/degC], numpy array
#   t: temperature [degC], numpy array, dtype=float
#   f: integration formula, string {'exact', 'simple'}
#   rho: density of water [kg/m3], numpy array, dtype=float
#   rho0: density of water at room temperature T0 [kg/m3], float
# Return value:
#   Residual volume change [mm^3], numpy array
# ------------------------------------------------------------------------------
def resid_sol_por(vme_sol, vme_por, vwi, bw, bm, t, f, rho=None, rho0=None):
  return vme_sol - deltaVcal_por(vme_por, vwi, bw, bm, t, f, rho, rho0)

# ------------------------------------------------------------------------------
# Computes the propagation of uncertainty on the volumetric strain formula
# Returns error factors for all input variables
# returns variance on the thermally induced volumetric strain
# Lists indexing variables according to the following fixed order:
# [0] Initial volume of the sample: "vi"
# [1] Measured (not corrected) volume change of water: "vme"
# [2] Correction for the volume of expelled water: "vcal"
# [3] Initial volume of solid grains: "vsi"
# [4] Volumetric thermal expansion coefficient of solid grains: "bs"
# [5] Volumetric thermal expansion coefficient of water: "bw"
# [6] Temperature variation: "dt"
# Units and dimensions must be consistent between all input variables
#
# Arguments:
#   mean: mean values of the variables, list of scalars/numpy arrays. Note: the
#         input for the "dt" entry may either be the absolute temperature or the
#         temperature variation since either choice does not alter the value of
#         the integrals (simple translation variable)
#   std: standard deviation of the variables, list of scalars/numpy arrays.
#        Note: the input for the "dt" entry must be the standard deviation of
#        the temperature variation, i.e. s_dt = sqrt(2)*s_t, with s_t the
#        standard deviation of the temperature measurement
# Return value:
#   list of error factors ordered with the reference indexing
#   standard deviation of the thermally induced volumetric strain
# ------------------------------------------------------------------------------
def propagUQ(mean, std):
  ### Unpack variables (for clarity)
  vi = mean[0]
  vme = np.copy(mean[1]) # Copy for potential modification to avoid
  vcal = np.copy(mean[2]) # division by zero in the cov calculation
  vsi = mean[3]
  bs = mean[4]
  bw = mean[5]
  t = mean[6] # Either absolute temperature or temperature variation
  # Unpack standard deviations (for clarity)
  s_vi = std[0] # Must be determined from function `std_vi()`
  s_vme = std[1]
  s_vcal = std[2]
  s_vsi = std[3] # Must be determined from function `std_vsi()`
  s_bs = std[4]
  s_bw = std[5]
  s_dt = std[6] # Must be determined from function `std_dt()`

  ### Useful quantities
  ni = 1.0 - vsi/vi # Initial porosity
  pfi = 1.0 - ni # Initial packing fraction (complement to 1 of porosity)
  delt = t - t[0] # Temperature variation
  # Relative thermal expansion computed using the small variations assumption
  # Compute integral of the difference between coefficients of water and grains
  int_bwbs = deltaVth(1.0, bw - bs, t, 'small')

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

  s_ev = np.sqrt(f[0]*cov_vi**2 + f[1]*cov_vme**2 + f[2]*cov_vcal**2 +
                 f[3]*cov_vsi**2 + f[4]*cov_bs**2 + f[5]*cov_bw**2 +
                 f[6]*cov_dt**2)

  return f, s_ev

# ------------------------------------------------------------------------------
# Computes the standard deviation of the initial volume of the sample. This is a
# simple calculation that could be implemented directly by the users. It is
# defined as a standalone function here to provide standardized and centralized
# procedures and avoid possible mistakes
# Arguments:
#   s_viprep: standard deviation of the sample volume after preparation
#   s_vc: standard deviation of the volume change during consolidation
#   s_mu: standard deviation of the flowrate of leakage during consolidation
#   tc: duration of the consolidation
# Return value:
#   standard deviation of the initial sample volume before thermal loading
# ------------------------------------------------------------------------------
def std_vi(s_viprep, s_vc, s_mu, tc):
  return np.sqrt(s_viprep**2 + s_vc**2 + (s_mu*tc)**2)

# ------------------------------------------------------------------------------
# Computes the standard deviation of the initial volume of solid grains. This is
# a simple calculation that could be implemented directly by the users. It is
# defined as a standalone function here to provide standardized and centralized
# procedures and avoid possible mistakes
# Arguments:
#   ms: initial mass of solid grains
#   rhosi: initial density of solid grains
#   s_ms: standard deviation of the initial mass of solid grains
#   s_rhosi: standard deviation of the initial density of solid grains
# Return value:
#   standard deviation of the initial volume of solid grains
# ------------------------------------------------------------------------------
def std_vsi(ms, rhosi, s_ms, s_rhosi):
  return np.sqrt((s_ms/rhosi)**2 + (s_rhosi/ms)**2)

# ------------------------------------------------------------------------------
# Computes the standard deviation of the temperature *variation* from the
# standard deviation of the temperature measurement. This is a very simple
# calculation that could be implemented directly by the users. It is defined as
# a standalone function here to provide standardized and centralized procedures
# and avoid possible mistakes
# Arguments:
#   s_t: standard deviation of the temperature measurement
# Return value:
#   standard deviation of the temperature variation
# ------------------------------------------------------------------------------
def std_dt(s_t):
  return np.sqrt(2)*s_t
