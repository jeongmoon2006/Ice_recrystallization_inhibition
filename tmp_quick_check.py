import numpy as np
from iri_model import run_simulation, critical_radii, get_growth_velocity
NA = 6.02e23
molL_to_num_per_um3 = NA / 1e15
delT = 5.0
T_eq = 273.0
T = 268.0
R_gas = 8.314
Dh = 6000.0
gamma = 30.0
Pm = 0.005
Pf = 0.01
L = 1
n_water_nm3 = 33.0
n_water_cm3 = n_water_nm3 * 1e21
v_m = NA / n_water_cm3
D = 1120.0
rho_ice = 910/18*1000*6.02*100_000
c_flat = np.exp(-Dh/T_eq*delT/R_gas/T) * 55.0 * molL_to_num_per_um3
alpha = 2.0 * (gamma * 1e-3) / (R_gas * T) * v_m * c_flat
invL2 = 1.0
kf = alpha * Pf / 2
km = alpha * Pm / 2
R = np.linspace(0, 40.0, 800)
n = 100
V = 170**3
mean_R = 10.0
std_R = 2.0
A = n / (std_R * V * np.sqrt(2.0 * np.pi))
f_init = A * np.exp(-0.5 * ((R - mean_R) / std_R) ** 2)
c_bulk_init = (c_flat + alpha / mean_R + kf * invL2) * 1.001
params = {'D': D,'rho_ice': rho_ice,'c_flat': c_flat,'alpha': alpha,'k_f': kf,'k_m': km,'invL2': invL2,'time_unit': 'ms','solver_method': 'BDF','solver_rtol': 1e-6,'solver_atol': 1e-9}

rm, rf = critical_radii(c_bulk_init, params)
print('initial rm,rf', rm, rf, 'rm<rf=', rm<rf)
v = get_growth_velocity(R, c_bulk_init, params, mode='double')
print('velocity finite?', np.all(np.isfinite(v)), 'v@R0', v[0], 'v@R1', v[1])
print('stay count', int(np.sum(np.isclose(v,0.0))))

# short run for stability
sol = run_simulation(f_init, c_bulk_init, R, (0.0, 5.0e4), params, mode='double', t_eval=np.linspace(0,5.0e4,40))
print('success', sol.success)
print('message', sol.message)
f = sol.y[:-1,:]
print('neg entries', int(np.sum(f<0)))
print('c range', float(np.min(sol.y[-1,:])), float(np.max(sol.y[-1,:])))
