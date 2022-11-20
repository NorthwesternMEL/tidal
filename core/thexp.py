"""
Module that computes the thermal expansion coefficients of substances.

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

# Exnternally (publically) accessed methods
__all__ = ['vcte_w_Baldi88', 'vcte_w_Cekerevac05', 'vcte_w_Smith54',
           'vcte_w_Chapman67', 'vcte_w_CRC40ed', 'vcte_w_IAPWS95_tab',
           'vcte_w_IAPWS95_fit', 'vcte_s_Kosinski91']

# Utility functions for temperature conversion
FIVENINTH = 5./9
#
def c2f(t_degC):
  """Converts degrees Celsius to degrees Farenheit."""
  return t_degC*1.8 + 32.0
  

def c2k(t_degC):
  """Converts degrees Celsius to Kelvin."""
  return t_degC + 273.15


def k2c(t_K):
  """Converts Kelvin to degrees Celsius."""
  return t_K - 273.15


def f2c(t_degF):
  """Converts degrees Farenheit to degrees Celsius."""
  return (t_degF - 32.0)*FIVENINTH


def k2r(t_K):
  """Converts Kelvin to degrees Rankine."""
  return t_K*1.8
  

def r2k(t_degR):
  """Converts degrees Rankine to Kelvin."""
  return t_degR*FIVENINTH


def vcte_w_Baldi88(u, t):
  """
  Compute thermal expansion coefficient of water from Baldi et al., 1988.

  Thermal volume changes of the mineral-water system in low-porosity clay soils.
  Canadian Geotechnical Journal. Baldi et al., 1988.
  DOI: https://doi.org/10.1139/t88-089
  Equation [8], p. 815

  Arguments:
  ----------
    u : float
      Pore water pressure [Pa]
    t : numpy array
      Temperature [degC]

  Returns:
  --------
    out : numpy array
      Volumetric thermal expansion coefficient of water [1/degC]
  """
  a0 = 4.505e-4 # [1/degC]
  a1 = 9.156e-5 # [1/degC]
  a2 = 6.381e-6 # [1/degC]
  b1 = -1.2e-6 # [1/degC^2]
  b2 = -5.766e-8 # [1/degC^2]
  m = 1.5e-9 # [1/Pa], given as 0.15 [1/kbar] in the original paper
  return a0 + (a1 + b1*t)*np.log(m*u) + (a2 + b2*t)*(np.log(m*u)**2)


def vcte_w_Cekerevac05(u, t):
  """
  Compute thermal expansion coefficient of water from Cekerevac et al., 2005.

  A Novel Triaxial Apparatus for Thermo-Mechanical Testing of Soils.
  Geotechnical Testing Journal. Cekereval et al., 2005.
  DOI: https://doi.org/10.1520/GTJ12311
  Equation (7), p. 8. Identical to Baldi et al. 1988, but with m = 15 MPa^-1

  Arguments:
  ----------
    u : float
      Pore water pressure [Pa]
    t : numpy array
      Temperature [degC]

  Returns:
  --------
    out : numpy array
      Volumetric thermal expansion coefficient of water [1/degC]
  """
  a0 = 4.505e-4 # [1/degC]
  a1 = 9.156e-5 # [1/degC]
  a2 = 6.381e-6 # [1/degC]
  b1 = -1.2e-6 # [1/degC^2]
  b2 = -5.766e-8 # [1/degC^2]
  m = 1.5e-5 # [1/Pa], given as 15 [1/MPa] in the original paper
  return a0 + (a1 + b1*t)*np.log(m*u) + (a2 + b2*t)*(np.log(m*u)**2)


def vcte_w_Smith54(tc, t):
  """
  Compute thermal expansion coefficient of org. liquids from Smith et al., 1954.

  Correlation of critical temperatures with thermal expansion coefficients of
  organic liquids. The Journal of Physical Chemistry. Smith et al., 1954.
  DOI: https://doi.org/10.1021/J150515A016
  Equation (7A), p. 445.
  Also discussed in Chapman 1967, Heat Transfer, 3rd edition. Macmillan
  ISBN: 9780023214400
  Equation (2.7), p. 33 of Chapman 1967 is given for temperature in Rankine
  degrees so the prefactor is 0.06284 = 0.04314*(5/9)**(-0.641), instead of the
  original value of 0.04314 from Smith et al., 1954. The factor 5/9 comes from
  the conversion from Rankine degree to Celsius degree.

  Arguments:
  ----------
    tc : float
      Critical temperature [degC]
    t : numpy array
      Temperature [degC]

  Returns:
  --------
    out : numpy array
      Volumetric thermal expansion coefficient of organic liquids [1/degC]
  """
  return 0.04314*(tc-t)**(-0.641)


def vcte_w_Chapman67(t):
  """
  Compute thermal expansion coefficient of saturated water from Chapman 1967.

  Heat Transfer, 2nd edition. Macmillan. Chapman, 1967. ISBN: 9780023214400
  Tabulated values from Table A.4 p. 558 are linearly interpolated.

  Arguments:
  ----------
    t : numpy array
      Temperature in range [0 ; 287] [degC]

  Returns:
  --------
    out : numpy array
      Volumetric thermal expansion coefficient of water [1/degC]
  """
  # Temperature [degF]
  t_tab_degF = np.array([32,40,50,60,70,80,90,100,110,120,130,140,150,160,170,
  180,190,200,210,220,230,240,250,260,270,280,290,300,350,400,450,500,550])
  # Volumetric thermal expansion coefficient [1/degR]
  b_tab_perdegR = np.array([0.03,0.045,0.07,0.1,0.13,0.15,0.18,0.2,0.22,0.24,
  0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.48,0.5,0.51,0.53,
  0.55,0.56,0.58,0.62,0.72,0.93,1.18,1.63])*1e-3
  # Conversion of temperatures to degC and expansion to 1/degC (same as 1/K)
  t_tab_degC = f2c(t_tab_degF)
  b_tab_perdegC = k2r(b_tab_perdegR) # K = 5/9 * degR --> 1/K = 9/5 * 1/degR
  # Linear interpolation of tabulated values
  return np.interp(t,t_tab_degC,b_tab_perdegC)


def vcte_w_CRC40ed(t):
  """
  Compute thermal expansion coefficient of water from CRC Handbook 40th ed.

  Handbook of Chemistry and Physics, 40th edition. 1958-1959.
  Chemical Rubber Publishing Company.
  Polynomial volume from table p. 2247: V = Vi*(1 + a*t + b*t^2 + c*t^3) used to
  derive thermal expansion coefficient after equation (3) of Coulibaly and Rotta
  Loria 2022.

  Arguments:
  ----------
    t : numpy array
      Temperature [degC]

  Returns:
  --------
    out : numpy array
      Volumetric thermal expansion coefficient of water [1/degC]
  """
  a = -0.06427e-3 # [1/degC]
  b = 8.5053e-6 # [1/degC^2]
  c = -6.7900e-8  # [1/degC^3]
  return (a + 2*b*t + 3*c*t**2)/(1 + a*t + b*t**2 + c*t**3)


def vcte_w_IAPWS95_tab(ifile, t, ofile=None):
  """
  Compute thermal expansion coefficient of water from IAPWS-95 property tables.

  The IAPWS Formulation 1995 for the Thermodynamic Properties of Ordinary Water
  Substance for General and Scientific Use. Wagner and Pruss, 2002.
  DOI: https://doi.org/10.1063/1.1461829
  International Association for the Properties of Water and Steam (IAPWS)
  Tabulated isobaric properties of water from the National Institute of
  Standards and Technology (NIST) Chemistry WebBook Standard Reference Database
  69 (SRD 69): https://webbook.nist.gov/chemistry/fluid/ (accessed 8 Jan 2022)
  used to derive thermal expansion coefficient after equation (3) of Coulibaly
  and Rotta Loria 2022. Units: temperature [degC], density [kg/m3].
  Recommendations: temperature increment: 0.5 degC, number of digits: 8.
  Values from NIST SRD69 Database are linearly interpolated.

  Arguments:
  ----------
    ifile: string
      Path to file with tab-separated isobaric properties from SRD69 database
    t : numpy array
      Temperature [degC]
    ofile: string, optional
      Path to output CSV file with computed thermal expansion coefficient

  Returns:
  --------
    out[0] : numpy array
      Volumetric thermal expansion coefficient of water [1/degC]
    out[1] : numpy array
      Density of water [kg/m3]
  """
  data = np.genfromtxt(ifile, delimiter='\t', names=True, encoding=None,
                       dtype=None)
  # All values computed using 2nd order central difference except for the
  # first / last values computed using 1st order forward / backward difference
  # Users should not use first/last values, only here to maintain size.
  ndata = data.size
  thexp = np.zeros(ndata)
  temp = data["Temperature_C"]
  vol = data["Volume_m3kg"]
  dens = data["Density_kgm3"]

  # Verify if water is in liquid phase in the input temperature range t
  # Only check highest/lowest value, i.e., first value below min and first value
  # above max, within range of the data.
  idmin = max((np.abs(temp - np.min(t))).argmin() - 1, 0)
  idmax = min((np.abs(temp - np.max(t))).argmin() + 1, ndata - 1)
  if ((data["Phase"][idmin] != 'liquid') or
      (data["Phase"][idmax] != 'liquid')):
    raise ValueError("Water is not in liquid phase for this temperature range")

  thexp[0] = ((vol[1] - vol[0])/(temp[1] - temp[0]))
  thexp[-1] = ((vol[-1] - vol[-2])/(temp[-1] - temp[-2]))
  thexp[1:-1] = ((vol[2:] - vol[0:-2])/(temp[2:] - temp[0:-2]))
  thexp *= dens

  if (ofile is not None):
    np.savetxt(ofile,
               np.concatenate((temp[:,np.newaxis],
                               thexp[:,np.newaxis]), axis=1),
               fmt = ['%.2f', '%.3e'],
               header="temperature_degC,thermal_expansion_1perdegC",
               delimiter=",")
  return (np.interp(t, temp, thexp), np.interp(t, temp, dens))


def vcte_w_IAPWS95_fit(preset, t, bwl=None, deg=None, ofile=None):
  """
  Compute polynomial thermal expansion coefficient of water from IAPWS-95.

  Arguments:
  ----------
    preset : {'compute', 'd3_t20-80_p50-1000'} ( can be extended !)
      Calculation behavior. 'compute' from thermal expansion coefficients or use
      previously computed and hardcoded polynomial with other available presets.
      Example: 'd3_t20-80_p50-1000' for 3rd degree polynomial in the temperature
      range T=[20;80] degC and pressure range p=[50;1000] kPa.
    t : numpy array
      Temperature [degC]
    bwl : list of numpy arrays, optional
      Thermal expansion coefficients [1/degC]. Only used for 'compute'. Each
      array in the list corresponds to a given pore water pressure. Each array
      must be same size as the input temperature array t.
    deg : int, optional
      Degree of polynomial fit. Only used for 'compute'.
    ofile : string, optional
      Path to output CSV file with computed thermal expansion coefficient

  Returns:
  --------
    out[0] : numpy array
      Volumetric thermal expansion coefficient of water [1/degC]
    out[1] : numpy array
      Coefficients of the polynomial fit, highest power first
  """
  if (preset == 'compute'):
    n = len(bwl)
    coef = np.polyfit(np.tile(t,n), np.concatenate(bwl), deg)
  elif (preset == 'd3_t20-80_p50-1000'):
    # Preset values obtained for 3rd degree polynomial,
    # temperature range T=[20;80] degC, pressure range p=[50;1000] kPa
    # in example file "comparison_thexp_water.py"
    # Equation (19) in Coulibaly and Rotta Loria, 2022
    coef = np.array([4.416308e-10, -1.030440e-07, 1.381959e-05, -3.054531e-05])

  fit = np.polyval(coef, t)
  if (ofile is not None):
    np.savetxt(ofile,
               np.concatenate((t[:,np.newaxis],fit[:,np.newaxis]),axis=1),
               fmt = ['%.2f', '%.3e'],
               header="temperature_degC,thermal_expansion_1perdegC",
               delimiter=",")
  return fit, coef


def vcte_s_Kosinski91(t, order):
  """
  Compute thermal expansion coefficient of quartz from Kosinski et al., 1991.

  Thermal Expansion of Alpha Quartz. Proceedings of the 45th Annual Symposium on
  Frequency Control 1991, IEEE. DOI: https://doi.org/10.1109/FREQ.1991.145883
  The 3rd, 4th and 5th-order fit for the linear thermal expansion coefficient of
  alpha-quartz along the a-axis and c-axis (HTMIAC data Table 1 p. 24) are used
  to define the volumetric thermal expansion coefficient after equation (20) of
  Coulibaly and Rotta Loria 2022.

  Arguments:
  ----------
    t : numpy array
      Temperature in range [-50 ; 150] [degC]
    order : {3, 4, 5}
      Order of polynomial fit

  Returns:
  --------
    out : numpy array
      Volumetric thermal expansion coefficient of alpha quartz [1/degC]
  """
  # Linear coefficient of thermal expansion along the a-axis
  coef_a = [[13.1e-6, 25.0e-9, -64.7e-12, 45.1e-15],
            [13.1e-6, 25.8e-9, -54.3e-12, -320.3e-15, 1827e-18],
            [13.1e-6, 26.1e-9, -60.6e-12, -466.2e-15, 4645e-18, -11270e-21]]
  # Linear coefficient of thermal expansion along the c-axis
  # There is a sign error in the original paper of Kosinksi et al. for the
  # 4-th order coefficient a3. Table I says a3 = -200.0, fitted data show that
  # it should be a plus sign. a3 = +200.0 is used in the present function
  coef_c = [[7.1e-6, 17.7e-9, -48.4e-12, 111.7e-15],
            [7.1e-6, 17.5e-9, -50.9e-12, +200.0e-15, -441.6e-18],
            [7.1e-6, 17.2e-9, -45.5e-12, 323.0e-15, -2817e-18, 9503e-21]]
  # Volumetric coefficient of thermal expansion
  coef_vol = 2*np.array(coef_a[order-3]) + np.array(coef_c[order-3])
  return np.polyval(np.flip(coef_vol), t)
