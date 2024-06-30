import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('data-fopt-fit.csv')

# Assign each column to a separate variable
vw = df['vw'].to_numpy()
alpha = df['alpha'].to_numpy()
betaH = df['betaH'].to_numpy()
Tn = df['Tn'].to_numpy()
f_p = df['f_p'].to_numpy()
Om_p = df['Om_p'].to_numpy()
at = df['at'].to_numpy()
bt = df['bt'].to_numpy()
rbt = df['rbt'].to_numpy()
s0t = df['s0t'].to_numpy()
omega_0t = df['omega_0t'].to_numpy()
K = df['K'].to_numpy()

# Interpolatoin over data (t : tilde)
int_f_p = LinearNDInterpolator(list(zip(vw, alpha)), f_p)
int_Om_p = LinearNDInterpolator(list(zip(vw, alpha)), Om_p)
int_at = LinearNDInterpolator(list(zip(vw, alpha)), at)
int_bt = LinearNDInterpolator(list(zip(vw, alpha)), bt)
int_rbt = LinearNDInterpolator(list(zip(vw, alpha)), rbt)
int_s0t = LinearNDInterpolator(list(zip(vw, alpha)), s0t)
int_omega_0t = LinearNDInterpolator(list(zip(vw, alpha)), omega_0t)
int_K = LinearNDInterpolator(list(zip(vw, alpha)), K)

vw0 = 0.5
alpha0 = 0.5 

# Generatng fit parameters from the interpolated function
f_p0 = int_f_p(vw0, alpha0)
at0 = int_at(vw0, alpha0)
bt0 = int_bt(vw0, alpha0)
rbt0 = int_rbt(vw0, alpha0)
s0t0 = int_s0t(vw0, alpha0)
omega_p0 = int_omega_0t(vw0, alpha0)
K0 = int_K(vw0, alpha0)

# Degrees of freedom
# arXiv: https://arxiv.org/pdf/1503.03513
filename = "gstar.txt"
data_g = np.loadtxt(filename, skiprows=0)
g_high_temp = np.array([[1.e14 * data_g[0, 0], data_g[0, 1]]])
data_g1 = np.concatenate((g_high_temp, data_g))
T_data = data_g1[:, 0]  # [GeV]
gstar_data = data_g1[:, 1]  # [GeV]
gstar_fun = interp1d(T_data, gstar_data)
#
f_b0 = rbt0 * f_p0
# This part can be modified based on different nucleation tamperatures and bubble nucleation rates.
# beta over H
betaH_default = 1
betaH_new = 10
# Nucleation temperature
Tn_default = 100 # GeV
Tn_new = 1000 # GeV
# 
rstar_default = (8*np.pi)**(1/3)*vw0/betaH_default
rstar_new = (8*np.pi)**(1/3)*vw0/betaH_new
# Degrees of Freedom
gstar_default = gstar_fun(Tn_default)
gstar_new = gstar_fun(Tn_new)
#
x_default = rstar_default/(K0**(1/2))
J_default = rstar_default*(1-1/(1+2*x_default)**(1/2))
Fgw_default = 3.57*1e-5*(100/gstar_default)**(1/3)
x_new = rstar_new/(K0**(1/2))
J_new = rstar_new*(1-1/(1+2*x_new)**(1/2))
Fgw_new = 3.57*1e-5*(100/gstar_new)**(1/3)

# Frequency
fdomain_default = 10**np.linspace(-24,24,1000) 

f0_default = 2.6*1.e-6*(Tn_default/100)*(gstar_default/100)**(1/6)
omega_p0_default = omega_p0
kR = fdomain_default * (rstar_default/f0_default) 

# GW spectrum - default
omega_fit_default = omega_p0_default * (fdomain_default/s0t0)**9 * (
    (2 + rbt0**(-12 + bt0)) / 
    ((fdomain_default/s0t0)**at0 + (fdomain_default/s0t0)**bt0 + 
     rbt0**(-12 + bt0) * (fdomain_default/s0t0)**12))

f0_new = 2.6*1.e-6*(Tn_new/100)*(gstar_new/100)**(1/6)
fdomain_new = kR / (rstar_new/f0_new) 
omega_fit_new = (omega_fit_default  / (Fgw_default*J_default) ) * (Fgw_new*J_new)


# Plotting
plt.figure(figsize=(10, 6))
plt.loglog(fdomain_default, omega_fit_default, label='FOPT - Sound Waves - GW - $T_n = 100$ GeV and $\\beta /H = 1$')
plt.loglog(fdomain_new, omega_fit_new, label='FOPT - Sound Waves - GW')

# Set limits for frequency and GW relic axes
plt.xlim(1.e-12, 1.e4)  
plt.ylim(1.e-22, 1.e-2)  
plt.legend()
plt.xlabel('f (Hz)')
plt.ylabel('$\Omega_{GW}h^2$(f)')
plt.grid(True)
plt.show()