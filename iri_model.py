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
    time_unit = str(params.get("time_unit", "s")).lower()
    if time_unit == "s":
        return 1.0
    if time_unit == "ms":
        return 1000.0
    raise ValueError("params['time_unit'] must be either 's' or 'ms'.")


def critical_radii(c_bulk, params):
    """Return (R_melt, R_freeze) for the two-threshold model.

    R_freeze = alpha / (c_bulk - c_flat - k_f/L^2)
    R_melt   = alpha / (c_bulk - c_flat + k_m/L^2)
    """
    c_flat = params["c_flat"]
    alpha = params["alpha"]
    k_f = params.get("k_f", params.get("k1", params.get("k", 0.0)))
    k_m = params.get("k_m", params.get("k2", 0.0))
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
    v_melt = prefactor * (c_bulk - c_flat - alpha / radius + k_m * invL2)

    if mode == "single":
        return v_melt

    if mode != "double":
        raise ValueError("mode must be 'single' or 'double'.")

    v_freeze = prefactor * (c_bulk - c_flat - alpha / radius - k_f * invL2)
    r_melt, r_freeze = critical_radii(c_bulk, params)

    in_melt_region = radius > r_melt
    in_freeze_region = radius < r_freeze

    return np.select([in_melt_region, in_freeze_region], [v_melt, v_freeze], default=0.0)


def ode_system(t, y, R, params, mode):
    """ODE system for [f(R,t), c_bulk(t)]."""
    f = y[:-1]
    c_bulk = y[-1]
    rho_ice = params["rho_ice"]

    v_r = get_growth_velocity(R, c_bulk, params, mode=mode)
    flux = f * v_r
    df_dt = -np.gradient(flux, R, edge_order=2)
    dc_bulk_dt = -4.0 * np.pi * rho_ice * simpson(y=R * R * v_r * f, x=R)

    return np.concatenate([df_dt, [dc_bulk_dt]])


def run_simulation(f_init, c_bulk_init, R, t_span, params, mode="double", t_eval=None):
    """Run IRI PSD simulation with a stiff solver.

    Units (recommended):
    - R in µm
    - f(R) in 1/µm^4
    - time in seconds if params['time_unit'] is omitted or set to 's'
    - c_bulk, c_flat in number/µm^3
    - alpha in number/µm^2
    - D in µm^2/s
    - invL2 in 1/µm
    - k_f, k_m in number/µm^2 (so k_*invL2 is number/µm^3)
    """
    radius = np.asarray(R, dtype=float)
    f_init = np.asarray(f_init, dtype=float)

    if radius.ndim != 1 or f_init.ndim != 1 or len(radius) != len(f_init):
        raise ValueError("R and f_init must be one-dimensional arrays with equal length.")

    y0 = np.concatenate([f_init, [float(c_bulk_init)]])

    return solve_ivp(
        fun=ode_system,
        t_span=t_span,
        y0=y0,
        args=(radius, params, mode),
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
    )