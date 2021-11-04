"""
Licensing information TBD

Python 2
Scipy version 1.2.1
Numpy version 1.16.2
Jibril B. Coulibaly, jibril.coulibaly at gmail.com

Critical analysis the results of Ng et al., 2016: Volume change behaviour
of saturated sand, Geotechnique Letters. 6:124-131
DOI: https://doi.org/10.1680/jgele.15.00148

Supplemental material to Coulibaly and Rotta Loria, 2022: Thermally induced
deformation of soils: a critical revision of experimental methods. GETE
DOI: TBD

except for temperature in Celsius degrees (noted degC)
Degrees Celsius [degC]
Degrees Farenheit [degF]
Kelvin [K]
Degrees Rankine [degR]

No exceptions checked for invalid inputs. Users responsability
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import thexp

# ------------------------------------------------------------------------------
# General information and data recovered from the paper for test DR70S200TC
# Test DR70S200TC chosen because most of the data is available through the paper
# ------------------------------------------------------------------------------

e = 0.71 # Initial void ratio (Table 1)
n = e/(1.0 + e) # Initial porosity

### Values of volume change from Table 3
# Temperature [degC]
temp = np.array([23, 30, 40, 50, 40, 30, 23], dtype=float)
# Expelled volume [mm^3]
vdr_measured = np.array([0, 132, 337, 527, 464, 342, 291], dtype=float)
# Volume calibration (vde) [mm^3]
vde = np.array([0, 15, 41, 76, 55, 25, 7], dtype=float)
# Water volume variation / total volume (vw/vi) [%]
vw_v = np.array([0, 0.08, 0.224, 0.399, 0.233, 0.08, 0])
# Solid volume variation / total volume (vs/vi) [%]
vs_v = np.array([0, 0.003, 0.006, 0.001, 0.006, 0.003, 0])
# Leaked volume / total volume (mu*t/vi) [%]
mut_v = np.array([0, 0.04, 0.088, 0.129, 0.225, 0.265, 0.31])
# Volumetric strain [%]
ev = np.array([0, 0.014, 0.028, -0.016, 0.014, 0.022, 0.022])

# Corrected expelled volume of water [mm^3]
vdr = vdr_measured - vde

# The initial volume, not provided the paper, is back-calculated using the data
# from Table 3 and equation (1). An average value of vi = 85941.25 mm^3 is
# calculated from all the rows in Table 3 except for the first row which
# corresponds to the initial state and is only zeros.
vi = np.average(vdr[1:]/(ev[1:] + vw_v[1:] + vs_v[1:] + mut_v[1:]))*1e2

vw = vw_v*vi*1e-2 # Variation of water volume [mm3]
vs = vs_v*vi*1e-2 # Variation of solid volume [mm3]
vwi = vi*n # Initial water volume [mm^3]
vsi = vi - vwi # Initial solid volume [mm^3]

### Analysis 1: integration of the thermal expansion of water
# The text mentions that thermal expansion of water is computed using the linear
# Equation $\Delta V_w = \alpha_w V_w \Delta T$ (Table 2). The data of Table 3
# is compared to the different integration formulas proposed by Coulibaly and
# Rotta Loria, 2022 to verify which one is actually used by Ng et al., 2016.



# Different integrations of the thermal expansion of Baldi et al., 1988 used by
# Ng et al., 2016. The cell pressures and back pressures used by Ng et al., 2016
# are not detailed in the paper. A back pressure of u = 200 kPa is mentionned
# only once in the caption of Figure 1 and is considered in this calculation.

u = 200e3 # Back pressure [Pa]
aw_Baldi88 = thexp.coef_w_Baldi88(u,temp) # Baldi et al., 1988
# Exact integration, equation (9) in Coulibaly et al., 2022
vw_v_exact = thexp.deltaV_thexp(n, aw_Baldi88, temp, 'exact')*1e2
# Constant coefficient integration, equation (10) in Coulibaly et al., 2022
vw_v_const = thexp.deltaV_thexp(n, aw_Baldi88, temp, 'const')*1e2
# Small thermal expansion integration, equation (11) in Coulibaly et al., 2022
vw_v_small = thexp.deltaV_thexp(n, aw_Baldi88, temp, 'small')*1e2
# Linear thermal expansion integration, equation (12) in Coulibaly et al., 2022
vw_v_lin = thexp.deltaV_thexp(n, aw_Baldi88, temp, 'linear')*1e2

plt.figure(1)
plt.plot(temp, vw_v, 'ko', label=r"Ng et al., 2016 (Table 3)")
plt.plot(temp, vw_v_exact, label=r"Exact integration")
plt.plot(temp, vw_v_const, label=r"Constant coefficient")
plt.plot(temp, vw_v_small, label=r"Small coefficient")
plt.plot(temp, vw_v_lin, label=r"Linear")
plt.xlabel(r'Temperature $T$ [degC]')
plt.ylabel(r'Volume change of water relative to total volume $\Delta V_w/V_i$ [%]')
plt.title("Integration of thermal expansion of water")
plt.legend()

np.savetxt("tab_water_expansion_integration_Ng2016.csv",
           np.concatenate((temp[:,np.newaxis], vw_v[:,np.newaxis],
                           vw_v_exact[:,np.newaxis], vw_v_const[:,np.newaxis],
                           vw_v_small[:,np.newaxis], vw_v_lin[:,np.newaxis]),
                          axis=1),
           header=("Vw_V_pct_Ng2016,Vw_V_pct_exact,Vw_V_pct_const,"+
                   "Vw_V_pct_small,Vw_V_pct_linear"),
           delimiter=',')
# The results seem to show that Ng et al., 2016 actually integrated the thermal
# expansion of water adequately, i.e. using equation () by Coulibaly and Rotta
# Loria 2022, but wrongly wrote the linearized formula in text




### Analysis X: Uncertainty quantification


































































# Based on the results of Ng et al., 2016 (Figure 6)
# It is not known what formula they used to integrate the thermal expansion
# WARNING: the test at p'=200 kPa is made with u=200 kPa, they don't tell for p' = 50 kPa what is the value of "u", so assuming it might be wrong, cannot really prove it until we know
# Do we have their water volume? total volume ? no need if we have initial porosity
# Initial porosity n = V_{w,i}/V_i, considering an initial water volume of 1 since all is relative

# Toyoura sand (Verdugo and Ishihara, 1996)
# Maximum void ratio: emax = 0.977
# Minimum void ratio: emin = 0.597
# Initial relative density: Dr = 20 %
# Initial void ratio: ei = emax - Dr*(emax - emin) = 0.901
# Initial porosity: n = e/(1+e) = 0.4739

e = 0.901 # Initial void ratio
n = 0.4739 # Initial porosity

# Average initial volume back-calculated from the strain






int_as = vs/vsi # true value of integral (at least true to how they computed it, which is unclear)
int_aw = vw/vwi # true value of integral (at least true to how they computed it, which is unclear)




# Assume integral equal alpha*delta T. how big is the gap? results is smaller than 1 for increasing values, and goes down to 0.77
# It is not conservative to assume it is equal to 1 if it has not been done that way, but we can't track it for Liu, assume 1

# We could also reverse engineer Liu to see if they computed the thermal expansion according to CRC (Campanella and Mitchell)






















# Based on the results of Ng et al., 2016 (Figure 6)
# It is not known what formula they used to integrate the thermal expansion
# WARNING: the test at p'=200 kPa is made with u=200 kPa, they don't tell for p' = 50 kPa what is the value of "u", so assuming it might be wrong, cannot really prove it until we know
# Do we have their water volume? total volume ? no need if we have initial porosity
# Initial porosity n = V_{w,i}/V_i, considering an initial water volume of 1 since all is relative

# Toyoura sand (Verdugo and Ishihara, 1996)
# Maximum void ratio: emax = 0.977
# Minimum void ratio: emin = 0.597
# Initial relative density: Dr = 20 %
# Initial void ratio: ei = emax - Dr*(emax - emin) = 0.901
# Initial porosity: n = e/(1+e) = 0.4739

v_w_i = 0.4739 # Initial volume of water, chosen as initial porosity (relative to 1)
dt = 0.5 # Temperature increment for integration [degC]

# Data digitized from Figure 6
# 3-column: olumetric strain [%]: "ev", Temperature [degC]: "T", temperature variation [degC]: "dT"
Ng_200kPa = np.genfromtxt("Ng_2016_Dr20_P200kPa.csv", delimiter=',', names=True)
Ng_50kPa = np.genfromtxt("Ng_2016_Dr20_P50kPa.csv", delimiter=',', names=True)

temp = np.arange(Ng_200kPa["T"][0],Ng_200kPa["T"][-1]+dt,dt)

# Thermal expansion according to Baldi et al., 1988 for given pressures
alpha_Baldi88_50kPa = thexp.coef_w_Baldi88(50e3,temp) # 50 kPa
alpha_Baldi88_200kPa = thexp.coef_w_Baldi88(200e3,temp) # 200 kPa

# Thermal expansion according to IAPWS-95 for given pressures
alpha_IAPWS95_50kPa = thexp.coef_w_IAPWS95_tab("water_IAPWS95_50kPa_10-90-0.5degC", temp)
alpha_IAPWS95_200kPa = thexp.coef_w_IAPWS95_tab("water_IAPWS95_200kPa_10-90-0.5degC", temp)

# Correction of volumetric strain due to thermal expansion of water, exact integration
dVw_cor_50kPa = thexp.deltaV_thexp(v_w_i, alpha_IAPWS95_50kPa, temp, 'exact') - thexp.deltaV_thexp(v_w_i, alpha_Baldi88_50kPa, temp, 'exact')
dVw_cor_200kPa = thexp.deltaV_thexp(v_w_i, alpha_IAPWS95_200kPa, temp, 'exact') - thexp.deltaV_thexp(v_w_i, alpha_Baldi88_200kPa, temp, 'exact')


temp_plot_200kPa = np.in1d(temp,Ng_200kPa["T"])
temp_plot_50kPa = np.in1d(temp,Ng_50kPa["T"])

plt.figure(1)
plt.plot(Ng_200kPa["ev"], Ng_200kPa["T"], label="Ng et al., 2016, 200 kPa")
plt.plot(Ng_50kPa["ev"], Ng_50kPa["T"], label="Ng et al., 2016, 50 kPa")
plt.plot(Ng_200kPa["ev"] - dVw_cor_200kPa[temp_plot_200kPa]*1e2, Ng_200kPa["T"], linestyle='--', label="corrected, 200 kPa")
plt.plot(Ng_50kPa["ev"] - dVw_cor_50kPa[temp_plot_50kPa]*1e2, Ng_50kPa["T"], linestyle='--', label="corrected, 50 kPa")
plt.xlabel('Volumetric strain [%]')
plt.ylabel(' Temperature [degC]')
plt.title("Correction of Toyoura sand at Dr=20% for Ng et al., 2016")
plt.legend()

