"""
Licensing information TBD

Python 2
Scipy version 1.2.1
Numpy version 1.16.2
Jibril B. Coulibaly, jibril.coulibaly at gmail.com

Volumetric Thermal expansion of Water, formula from Baldi et al., 1988, taken from Juza 1966

Everything in SI units

except for temperature in Celsius degrees (noted degC)
Degrees Celsius [degC]
Degrees Farenheit [degF]
Kelvin [K]
Degrees Rankine [degR]

No exceptions checked for invalid inputs. Users responsability

See the README file in the top-level XXX directory.
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
def coef_w_IAPWS95(ifile, t, ofile=None):
  data = np.genfromtxt(ifile, delimiter='\t', names=True)
  # Thermal expansion computed from specific volume v and density \rho=1/v as:
  # $\alpha = \frac{1}{v} \frac{\partial v}{\partial T}$
  # $\alpha = \rho \frac{\partial v}{\partial T}$
  # First / last values computed using 1st order forward / backward difference
  # Other values computed using 2nd order central difference
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
# Computes and returns the variation of volume due to thermal expansion using
# either exact integration, constant value of the thermal expansion coefficient,
# small values of the thermal expansion coefficient, or linear formula according
# to equation (X) to (Y), respectively of Coulibaly and Rotta Loria, 2022:
# $\Delta V / V_i = \exp(\int_{T_i}^T \alpha(T) dT) - 1$
# Arguments:
#   vi: initial volume of water/solid inside the sample [mm^3]
#   a: volumetric thermal expansion coefficient [1/degC], numpy array
#   t: temperature [degC], numpy array, dtype=float
#   f: integration formula, string {'exact', 'const', 'small', 'linear'}
# Return value:
#   relative volume variation [-], numpy array
# ------------------------------------------------------------------------------
def deltaV_thexp(vi, a, t, f):
  vdiff = 0.0
  if (f is 'exact'):
    # Exact integration: equation (X) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i[\exp(\int_{T_i}^T \alpha(T) dT) - 1]$
    vdiff = vi*(np.exp(integrate.cumtrapz(a, t, initial=0)) - 1.0)
  elif (f is 'const'):
    # Constant coefficient: equation (X1) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i[\exp(\alpha(T) \Delta T) - 1]$
    vdiff = vi*(np.exp(a*t) - 1.0)
  elif (f is 'small'):
    # Small expansion: equation (X2) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i \int_{T_i}^T \alpha(T) dT$
    vdiff = vi*integrate.cumtrapz(a, t, initial=0)
  elif (f is 'linear'):
    # Linear formula: equation (Y) of Coulibaly and Rotta Loria, 2022
    # $\Delta V = V_i \alpha(T) \Delta T
    vdiff = vi*a*t
  return vdiff

# ------------------------------------------------------------------------------
# Computes and returns the water volume inside the sample during heating using
# the exact differential equation (XXXX) of Coulibaly and Rotta Loria, 2022:
# $\Delta V_w = \exp(\int_{T_i}^T \alpha(T)dT) [V_{w,i} - 
#               \int_{t_i}^t \exp(-\int_{T_i}^T \alpha(T)dT) dV_{dr}] - V_{w,i}$
# Arguments:
#   vwi: initial volume of water inside the sample [mm^3]
#   aw: volumetric thermal expansion coefficient of water [1/degC], numpy array
#   vdr: volume of water expelled from the sample [mm^3], numpy array
#   t: temperature [degC], numpy array, dtype=float
# Return value:
#   time series of the volume of water inside the sample [mm^3], numpy array
# ------------------------------------------------------------------------------
def deltaVw_exact(vwi, aw, vdr, t):
  intat = integrate.cumtrapz(aw, t, initial=0)
  expintat = np.exp(intat)
  expintatinv = np.exp(-intat)
  return expintat*(vwi - integrate.cumtrapz(expintatinv, vdr, initial=0))
