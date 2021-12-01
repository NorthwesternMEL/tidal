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

Copyright (C) 2021 Mechanics and Energly Laboratory, Northwestern University

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
  return np.interp(t,data["Temperature_C"],thexp)

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
#   awl: thermal expansion coefficients at various pressure, [1/degC],
#        list of numpy arrays (all arrays must be the same length as t)
#   deg: degree of polynomial fit
#   ofile: (optional) path to output CSV file with temperature and expansion
# Return value:
#   volumetric thermal expansion coefficient of water [1/degC], numpy array
#   coefficient of the polynomial fit
# ------------------------------------------------------------------------------
def coef_w_IAPWS95_fit(preset, t, awl=None, deg=None, ofile=None):
  if (preset == 'compute'):
    n = len(awl)
    coef = np.polyfit(np.tile(t,n), np.concatenate(awl), deg)
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
# as: \alpha = 2*CTE_{a-axis} + CTE_{c-axis}
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
# Computes and returns the variation of volume of the solid grains due to
# thermal expansion using either exact integration, small values of the thermal
# expansion coefficient, or linear formula according to equation (X) to (Y),
# respectively, of Coulibaly and Rotta Loria, 2022.
# Arguments:
#   vsi: initial volume of solid inside the sample [mm^3]
#   a_s: volumetric thermal expansion coefficient of solid [1/degC], numpy array
#   t: temperature [degC], numpy array, dtype=float
#   f: integration formula, string {'exact', 'const', 'small', 'linear'}
# Return value:
#   Volume variation of solid grains [mm^3], numpy array
# ------------------------------------------------------------------------------
def deltaVs(vsi, a_s, t, f):
  delvs = 0.0
  if (f == 'exact'):
    # Exact integration: equation (X) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i[\exp(\int_{T_i}^T \alpha(T) dT) - 1]$
    delvs = vsi*(np.exp(integrate.cumtrapz(a_s, t, initial=0)) - 1.0)
  elif (f == 'small'):
    # Small expansion: equation (X2) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i \int_{T_i}^T \alpha(T) dT$
    delvs = vsi*integrate.cumtrapz(a_s, t, initial=0)
  elif (f == 'linear'):
    # Linear formula: equation (Y) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i \alpha(T) \Delta T
    delvs = vsi*a_s*(t-t[0])
  return delvs

# ------------------------------------------------------------------------------
# Computes and returns the variation of water volume inside the sample due to
# the combined effects of thermal expansion and expelled water using either
# exact integration, small values of the thermal expansion coefficient, or
# linear formula according to equations (XXXX) to (Y), respectively, of
# Coulibaly and Rotta Loria, 2022.
# Arguments:
#   vwi: initial volume of water inside the sample [mm^3]
#   aw: volumetric thermal expansion coefficient of water [1/degC], numpy array
#   vdr: volume of water expelled from the sample [mm^3], numpy array
#   t: temperature [degC], numpy array, dtype=float
#   f: integration formula, string {'exact', 'const', 'small', 'linear'}
# Return value:
#   time series of the volume of water inside the sample [mm^3], numpy array
# ------------------------------------------------------------------------------
def deltaVw(vwi, aw, vdr, t, f):
  delvw = 0.0
  if (f == 'exact'):
    # Exact integration: equation (X) of Coulibaly and Rotta Loria, 2022
    intat = integrate.cumtrapz(aw, t, initial=0)
    expintat = np.exp(intat)
    expintatinv = np.exp(-intat)
    delvw = vwi*expintat - expintat*integrate.cumtrapz(expintatinv,
                                                       vdr,initial=0)
  elif (f == 'small'):
    # Small expansion: equation (X2) of Coulibaly and Rotta Loria, 2022
    delvw = vwi*integrate.cumtrapz(aw, t, initial=0) - vdr
  elif (f == 'linear'):
    # Small expansion: equation (X2) of Coulibaly and Rotta Loria, 2022
    delvw = vwi*aw*(t-t[0]) - vdr
  return delvw

# ------------------------------------------------------------------------------
# Computes the propagation of uncertainty on the volumetric strain formula
# Returns variance factors for all input variables
# returns variance on the thermally induced volumetric strain
# Dictionary indexing variables according to the following fixed naming:
# Initial volume of the sample: "vi"
# Initial mass of solid grains: "ms"
# Initial density of solid grains: "rhos"
# Volumetric thermal expansion coefficient of solid grains: "as"
# Volumetric thermal expansion coefficient of water: "aw"
# Temperature: "t"
# Measured (not corrected) volume of expelled water: "vdrm"
# Calibration / correction for the volume of expelled water: "vcal"
# Units and dimensions must be consistent between all input variables
#
#
# Arguments:
#   dic: dictionary mapping variables to list index
#   val: values of the variables, list of scalars/numpy arrays
#   std: standard deviation of the variables, list of scalars/numpy arrays
# Return value:
#   list[0]: standard deviation of the thermally induced volumetric strain
#   list[1]: list of variance factors with same indexing as input dictionary
# ------------------------------------------------------------------------------
def propagationUQ(dic, val, std):
  ### Unpack variables
  vi = val[dic["vi"]]
  ms = val[dic["ms"]]
  rhos = val[dic["rhos"]]
  a_s = val[dic["as"]]
  a_w = val[dic["aw"]]
  t = val[dic["t"]]
  vdrm = val[dic["vdrm"]]
  vcal = val[dic["vcal"]]
  # Unpack standard deviations
  s_vi = std[dic["vi"]]
  s_ms = std[dic["ms"]]
  s_rhos = std[dic["rhos"]]
  s_as = std[dic["as"]]
  s_aw = std[dic["aw"]]
  s_t = std[dic["t"]]
  s_vdrm = std[dic["vdrm"]]
  s_vcal = std[dic["vcal"]]

  ### Useful quantities
  vs = ms/rhos # Initial volume of solid
  # s_vs = see if and how needed after we clarify the paper
  # s2_vdr = s_vdrm**2 + s_vcal**2 # Variance of corrected expelled water volume, see if and how needed after we clarify the paper
  vdr = vdrm - vcal # Corrected volume of expelled water
  n = 1.0 - vs/vi # Initial porosity
  delt = t - t[0] # Temperature variation
  # Relative thermal expansion computed using the small variations assumption
  int_as = deltaVs(1.0, a_s, t, 'small') # Solid grains
  int_aw = deltaVw(1.0, a_w, 0.0, t, 'small') # Water (decoupled formula, vdr=0)

  ### Factors that multiply coefficient of variation of each variables
  f = [None]*len(dic)
  # Total volume. Equation () of Coulibaly and Rotta Loria, 2022
  f[dic["vi"]] = (vdr/vi)**2 + ((1.0-n)*int_as)**2 + ((1.0-n)*int_aw)**2
  # Solid mass. Equation () of Coulibaly and Rotta Loria, 2022
  f[dic["ms"]] = ((1.0-n)*int_as)**2 + ((1.0-n)*int_aw)**2
  # Solid density. Equation () of Coulibaly and Rotta Loria, 2022
  f[dic["rhos"]] = ((1.0-n)*int_as)**2 + ((1.0-n)*int_aw)**2 # Identical to ms
  # Thermal expansion of solid. Equation () of Coulibaly and Rotta Loria, 2022
  f[dic["as"]] = ((1.0-n)*a_s*delt)**2
  # Thermal expansion of water. Equation () of Coulibaly and Rotta Loria, 2022
  f[dic["aw"]] = (n*a_w*delt)**2
  # Temperature. Equation () of Coulibaly and Rotta Loria, 2022
  f[dic["t"]] = ((1.0-n)*a_s*delt)**2 + (n*a_w*delt)**2
  # Measured expelled volume. Equation () of Coulibaly and Rotta Loria, 2022
  f[dic["vdrm"]] = (vdr/vi)**2
  # Correction of expelled volume. Equation () of Coulibaly and Rotta Loria, 2022
  f[dic["vcal"]] = (vdr/vi)**2 # Identical to vdrm

  # Compute coefficients of variations
  cov_vi = s_vi/vi
  cov_ms = s_ms/ms
  cov_rhos = s_rhos/rhos
  cov_as = s_as/a_s
  cov_aw = s_aw/a_w
  cov_t = s_t/t
  cov_vdrm = s_vdrm/vdr # Divivded by corrected volume vdr, maybe clarify on final larticle draft
  cov_vcal = s_vcal/vdr # Divivded by corrected volume vdr, maybe clarify on final larticle draft

  s_ev = np.sqrt(f[dic["vi"]]*cov_vi**2 + f[dic["ms"]]*cov_ms**2 +
                 f[dic["rhos"]]*cov_rhos**2 + f[dic["as"]]*cov_as**2 +
                 f[dic["aw"]]*cov_aw**2 + f[dic["t"]]*cov_t**2 +
                 f[dic["vdrm"]]*cov_vdrm**2 + f[dic["vcal"]]*cov_vcal**2)

  return [s_ev,f]

