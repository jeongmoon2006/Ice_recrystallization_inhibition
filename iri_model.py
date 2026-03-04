"""Minimal IRI model with single or double critical radii kinetics."""

import numpy as np
from scipy.integrate import simpson, solve_ivp


def _safe_radius(radius):
    return np.maximum(np.asarray(radius, dtype=float), 1e-12)


def _velocity_time_scale(params):
    """Return scale factor to convert velocities from µm/s to requested time unit.

    Model equations naturally yield dR/dt in µm/s when:
    - D: µm^2/s
    - R: µm
    - rho_ice: number/µm^3
    - c terms: number/µm^3

    If params['time_unit'] == 'ms', divide by 1000 to get µm/ms.
    """
    time_unit = str(params.get("time_unit", "ms")).lower()
    if time_unit == "s":
        return 1.0
    if time_unit == "ms":
        return 1000.0
    raise ValueError("params['time_unit'] must be either 'ms' or 's'.")


def _flux_divergence_upwind(f, v_r, radius):
    """Return d(f*v)/dR using first-order upwind finite-volume fluxes.

    Boundary condition is "open" with zero inflow:
    - Left boundary (smallest R): if v>0 (inflow), flux=0; if v<0 (outflow), flux=v*f.
    - Right boundary (largest R): if v<0 (inflow), flux=0; if v>0 (outflow), flux=v*f.
    """
    f = np.asarray(f, dtype=float)
    v_r = np.asarray(v_r, dtype=float)
    radius = np.asarray(radius, dtype=float)

    n = len(radius)
    if n < 2:
        return np.zeros_like(f)

    face_flux = np.zeros(n + 1, dtype=float)

    left_v = v_r[0]
    right_v = v_r[-1]
    face_flux[0] = left_v * f[0] if left_v < 0.0 else 0.0
    face_flux[-1] = right_v * f[-1] if right_v > 0.0 else 0.0

    face_velocity = 0.5 * (v_r[:-1] + v_r[1:])
    upwind_left = np.where(face_velocity >= 0.0, f[:-1], f[1:])
    face_flux[1:n] = face_velocity * upwind_left

    cell_width = np.empty(n, dtype=float)
    cell_width[0] = radius[1] - radius[0]
    cell_width[-1] = radius[-1] - radius[-2]
    if n > 2:
        cell_width[1:-1] = 0.5 * (radius[2:] - radius[:-2])
    cell_width = np.maximum(cell_width, 1e-12)

    return (face_flux[1:] - face_flux[:-1]) / cell_width


def critical_radii(c_bulk, params):
    """Return (R_melt, R_freeze) for the two-threshold model.

    R_freeze = alpha / (c_bulk - c_flat - k_f * L^-2)
    R_melt   = alpha / (c_bulk - c_flat + k_m * L^-2)

    Unit consistency:
    - alpha: number/µm^2
    - k_f, k_m: number/µm
    - L^-2 (invL2): 1/µm^2
    so k_*L^-2 has units number/µm^3, matching concentration terms.
    """
    c_flat = params["c_flat"]
    alpha = params["alpha"]
    k_f = params.get("k_f", 0.0)
    k_m = params.get("k_m", 0.0)
    invL2 = params["invL2"]

    denom_freeze = c_bulk - c_flat - k_f * invL2
    denom_melt = c_bulk - c_flat + k_m * invL2

    r_melt = np.inf if denom_melt <= 0 else alpha / denom_melt
    r_freeze = np.inf if denom_freeze <= 0 else alpha / denom_freeze
    return r_melt, r_freeze


def get_growth_velocity(R, c_bulk, params, mode="single"):
    """Compute dR/dt over radius grid R.

    `single`: one-threshold (LSW-like) dynamics.
        `double`: two thresholds with a stay region where velocity is zero.

        Optional smoothing for `double` mode:
        - params['switch_width'] in µm (default 0.0) controls smooth blending
            near critical radii.
        - switch_width <= 0 keeps the original hard piecewise switching.
    """
    radius = _safe_radius(R)

    D = params["D"]
    rho_ice = params["rho_ice"]
    c_flat = params["c_flat"]
    alpha = params["alpha"]
    k_f = params.get("k_f", params.get("k1", params.get("k", 0.0)))
    k_m = params.get("k_m", params.get("k2", 0.0))
    invL2 = params["invL2"]

    time_scale = _velocity_time_scale(params)
    prefactor = D / (radius * rho_ice) / time_scale
    
    # Compute the baseline physical branch
    v_small_r = prefactor * (c_bulk - c_flat - alpha / radius + k_m * invL2)

    if mode == "single":
        return v_small_r

    if mode != "double":
        raise ValueError("mode must be 'single' or 'double'.")

    # Compute second branch and critical radii for double mode
    v_large_r = prefactor * (c_bulk - c_flat - alpha / radius - k_f * invL2)
    r_melt, r_freeze = critical_radii(c_bulk, params)

    # Keep a true three-region piecewise structure:
    #   R < r_melt   -> +k_m branch (small-radius branch)
    #   r_melt <= R <= r_freeze -> stay region (v = 0)
    #   R > r_freeze -> -k_f branch (large-radius branch)
    # This mirrors the legacy notebook logic and preserves a finite stay window
    # whenever r_melt < r_freeze.
    switch_width = float(params.get("switch_width", 0.0))

    if switch_width <= 0.0:
        in_large_r_region = radius > r_freeze
        in_small_r_region = radius < r_melt
        return np.select([in_large_r_region, in_small_r_region], [v_large_r, v_small_r], default=0.0)

    eps = max(switch_width, 1e-12)
    w_small = 0.5 * (1.0 - np.tanh((radius - r_melt) / eps))
    w_large = 0.5 * (1.0 + np.tanh((radius - r_freeze) / eps))
    w_stay = np.clip(1.0 - w_small - w_large, 0.0, 1.0)

    return w_small * v_small_r + w_large * v_large_r + w_stay * 0.0


def ode_system(t, y, R, params, mode):
    """ODE system for [f(R,t), c_bulk(t)]."""
    f = y[:-1]
    c_bulk = y[-1]
    rho_ice = params["rho_ice"]
    grad_edge_order = int(params.get("gradient_edge_order", 1))
    use_simpson = bool(params.get("use_simpson_integral", False))
    flux_scheme = str(params.get("flux_scheme", "upwind")).lower()
    enforce_nonnegative_f = bool(params.get("enforce_nonnegative_f", True))

    v_r = get_growth_velocity(R, c_bulk, params, mode=mode)
    flux = f * v_r
    if flux_scheme == "upwind":
        df_dt = -_flux_divergence_upwind(f=f, v_r=v_r, radius=R)
    elif flux_scheme == "gradient":
        df_dt = -np.gradient(flux, R, edge_order=grad_edge_order)
    else:
        raise ValueError("params['flux_scheme'] must be 'upwind' or 'gradient'.")

    if enforce_nonnegative_f:
        df_dt = np.where((f <= 0.0) & (df_dt < 0.0), 0.0, df_dt)

    integrand = R * R * v_r * f
    if use_simpson:
        integral_val = simpson(y=integrand, x=R)
    else:
        integral_val = np.trapezoid(integrand, R)
    dc_bulk_dt = -4.0 * np.pi * rho_ice * integral_val

    return np.concatenate([df_dt, [dc_bulk_dt]])


def run_simulation(
    f_init,
    c_bulk_init,
    R,
    t_span,
    params,
    mode="double",
    t_eval=None,
    rtol=1e-6,
    atol=1e-9,
    max_step=np.inf,
    method="BDF",
):
    """Run IRI PSD simulation with a stiff solver.

    Units (recommended):
    - R in µm
    - f(R) in 1/µm^4
    - time in milliseconds if params['time_unit'] is omitted or set to 'ms'
    - c_bulk, c_flat in number/µm^3
    - alpha in number/µm^2
    - D in µm^2/s
    - invL2 in 1/µm^2
    - k_f, k_m in number/µm (so k_*invL2 is number/µm^3)

    Solver controls:
    - method: any scipy.integrate.solve_ivp method (e.g., 'BDF', 'Radau', 'LSODA')
    - rtol: relative tolerance
    - atol: absolute tolerance
    - max_step: maximum internal time step (same unit as t_span)
    """
    radius = np.asarray(R, dtype=float)
    f_init = np.asarray(f_init, dtype=float)

    if radius.ndim != 1 or f_init.ndim != 1 or len(radius) != len(f_init):
        raise ValueError("R and f_init must be one-dimensional arrays with equal length.")

    y0 = np.concatenate([f_init, [float(c_bulk_init)]])

    method = params.get("solver_method", "BDF")
    rtol = float(params.get("solver_rtol", 1e-4))
    atol = float(params.get("solver_atol", 1e-7))

    return solve_ivp(
        fun=ode_system,
        t_span=t_span,
        y0=y0,
        args=(radius, params, mode),
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )