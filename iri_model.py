"""
IRI Research Module - UPenn Ph.D. Project
Modular ODE solver for Ice Recrystallization Inhibition (IRI) kinetics.
"""
import numpy as np
from scipy.integrate import solve_ivp

def get_growth_velocity(R, c_bulk, params, mode='single'):
    """Calculates v_R based on current bulk concentration."""
    D = params['D']
    rho_ice = params['rho_ice']
    c_flat = params['c_flat']
    alpha = params['alpha']
    k1 = params.get('k1', params.get('k', 0))
    k2 = params.get('k2', 0)
    invL2 = params['invL2']

    if mode == 'single':
        return D / (R * rho_ice) * (c_bulk - c_flat - alpha / R - k1 * invL2) / 1000
    
    elif mode == 'double':
        # Condition for Two Critical Radii (Freezing Hysteresis)
        c1 = R > alpha / (c_bulk - c_flat - k1 * invL2)
        c2 = R < alpha / (c_bulk - c_flat + k2 * invL2)
        v_vals = [
            D / (R * rho_ice) * (c_bulk - c_flat - alpha / R - k1 * invL2) / 1000,
            D / (R * rho_ice) * (c_bulk - c_flat - alpha / R + k2 * invL2) / 1000
        ]
        return np.select([c1, c2], v_vals, default=0)

def ode_system(t, y, R, params, mode):
    """The system of ODEs: returns [df/dt, dc_bulk/dt]."""
    f = y[:-1]
    c_bulk = y[-1]
    rho_ice = params['rho_ice']
    
    v_R = get_growth_velocity(R, c_bulk, params, mode=mode)
    
    # Continuity Equation (Advection of PSD)
    df_dt = -np.gradient(f * v_R, R)
    
    # Conservation Equation (Mass balance of solution)
    # Using Simpson's rule for better integration accuracy than trapz
    from scipy.integrate import simpson
    dc_bulk_dt = -4 * np.pi * rho_ice * simpson(y=R * R * v_R * f, x=R)
    
    return np.concatenate([df_dt, [dc_bulk_dt]])

def run_simulation(f_init, c_bulk_init, R, t_span, params, mode='single'):
    """Professional BDF solver (equivalent to MATLAB ode15s)."""
    y0 = np.concatenate([f_init, [c_bulk_init]])
    
    # Solve with BDF method for stiff systems
    sol = solve_ivp(
        fun=ode_system,
        t_span=t_span,
        y0=y0,
        args=(R, params, mode),
        method='BDF', 
        rtol=1e-6, 
        atol=1e-9
    )
    return sol