"""
Utilities for extracting critical radii from particle size distributions (PSD).

This module provides functions to identify critical radii by detecting discontinuities
in the PSD derivatives, which represent transitions between different kinetic regimes.
"""

import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import uniform_filter1d


def extract_critical_radii_from_psd(
    R_um,
    f_R,
    radius_range=(2.0, 15.0),
    smoothing_window=5,
    order=3,
    verbose=True
):
    """
    Extract critical radii from a particle size distribution by detecting
    discontinuities in the second derivative.
    
    The algorithm identifies where df/dR has sharp changes (kinks), which correspond
    to transitions between freezing-dominated and melting-dominated growth regimes.
    
    Parameters
    ----------
    R_um : array-like
        Radius values [µm]
    f_R : array-like
        PSD values f(R) [arbitrary units, typically number/µm³]
    radius_range : tuple, optional
        (min, max) radius range [µm] to search for critical radii. 
        Default: (2.0, 15.0)
    smoothing_window : int, optional
        Window size for smoothing the third derivative. Default: 5
    order : int, optional
        Order for local extrema detection. Default: 3
    verbose : bool, optional
        Print diagnostic information. Default: True
    
    Returns
    -------
    r_c_melt : float
        Critical radius for melting [µm] (smaller value, first discontinuity)
    r_c_freeze : float
        Critical radius for freezing [µm] (larger value, second discontinuity)
    derivatives : dict
        Dictionary containing computed derivatives:
        - 'dF_dR': first derivative
        - 'd2F_dR2': second derivative
        - 'd3F_dR3': third derivative
    
    Notes
    -----
    The physical interpretation:
    - r_c,melt marks the boundary where the melting inhibition effect begins
    - r_c,freeze marks the boundary where the freezing inhibition starts to dominate
    - Particles with r < r_c,melt: primarily subjected to freezing
    - Particles with r_c,melt < r < r_c,freeze: mixed regime
    - Particles with r > r_c,freeze: primarily subjected to melting
    """
    
    R_um = np.asarray(R_um)
    f_R = np.asarray(f_R)
    
    if verbose:
        print(f"PSD data: {len(R_um)} radius points")
        print(f"Radius range: {R_um[0]:.6f} to {R_um[-1]:.6f} µm")
        print(f"PSD range: {f_R.min():.6e} to {f_R.max():.6e}")
    
    # Compute derivatives
    dF_dR = np.gradient(f_R, R_um)      # First derivative
    d2F_dR2 = np.gradient(dF_dR, R_um)  # Second derivative (curvature)
    d3F_dR3 = np.gradient(d2F_dR2, R_um)  # Third derivative (discontinuities)
    
    if verbose:
        print(f"\nMax |d²f/dR²| = {np.max(np.abs(d2F_dR2)):.6e}")
        print(f"Max |d³f/dR³| = {np.max(np.abs(d3F_dR3)):.6e}")
    
    # Create search mask for reasonable critical radii range
    mask = (R_um >= radius_range[0]) & (R_um <= radius_range[1])
    search_indices = np.where(mask)[0]
    
    if len(search_indices) == 0:
        raise ValueError(
            f"No data points found in radius range {radius_range}. "
            f"Available range: ({R_um[0]:.2f}, {R_um[-1]:.2f}) µm"
        )
    
    # Find peaks in |d³f/dR³| (discontinuities in the second derivative)
    d3F_search = np.abs(d3F_dR3[search_indices])
    
    # Smooth to reduce noise
    d3F_smoothed = uniform_filter1d(d3F_search, size=smoothing_window, mode='nearest')
    
    # Find local maxima
    local_max_idx_search = argrelextrema(d3F_smoothed, np.greater, order=order)[0]
    
    if verbose:
        print(f"\nFound {len(local_max_idx_search)} local maxima in |d³f/dR³|")
    
    if len(local_max_idx_search) < 2:
        raise ValueError(
            f"Could not find 2 distinct critical radii. "
            f"Found only {len(local_max_idx_search)} local maxima. "
            f"Try adjusting radius_range, smoothing_window, or order parameters."
        )
    
    # Get the two strongest maxima
    strongest = np.argsort(d3F_smoothed[local_max_idx_search])[-2:]
    strongest_indices = local_max_idx_search[strongest]
    strongest_indices = np.sort(strongest_indices)
    
    r_candidates = R_um[search_indices[strongest_indices]]
    
    if verbose:
        print(f"Using the two strongest discontinuities:")
        for i, (idx, r) in enumerate(zip(strongest_indices, r_candidates)):
            abs_idx = search_indices[idx]
            print(f"  {i+1}. R = {r:.6f} µm, |d³f/dR³| = {np.abs(d3F_dR3[abs_idx]):.6e}")
    
    r_c_melt = r_candidates[0]    # First discontinuity (smaller)
    r_c_freeze = r_candidates[1]  # Second discontinuity (larger)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CRITICAL RADII EXTRACTED")
        print(f"{'='*60}")
        print(f"r_c,melt   = {r_c_melt:.6f} µm (melting inhibition boundary)")
        print(f"r_c,freeze = {r_c_freeze:.6f} µm (freezing inhibition boundary)")
    
    # Package derivatives for optional return
    derivatives = {
        'dF_dR': dF_dR,
        'd2F_dR2': d2F_dR2,
        'd3F_dR3': d3F_dR3,
        'search_indices': search_indices,
        'strongest_indices': search_indices[strongest_indices],
    }
    
    return r_c_melt, r_c_freeze, derivatives


def plot_psd_with_critical_radii(
    R_um,
    f_R,
    r_c_melt,
    r_c_freeze,
    derivatives=None,
    figsize=(14, 10),
    time_label="t=10min",
):
    """
    Plot the PSD and its derivatives with critical radii marked.
    
    Parameters
    ----------
    R_um : array-like
        Radius values [µm]
    f_R : array-like
        PSD values f(R)
    r_c_melt : float
        Critical radius for melting [µm]
    r_c_freeze : float
        Critical radius for freezing [µm]
    derivatives : dict, optional
        Dictionary with 'dF_dR', 'd2F_dR2', 'd3F_dR3' keys. If None, recomputed.
    figsize : tuple, optional
        Figure size (width, height). Default: (14, 10)
    time_label : str, optional
        Label for the time point. Default: "t=10min"
    
    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    
    R_um = np.asarray(R_um)
    f_R = np.asarray(f_R)
    
    # Compute derivatives if not provided
    if derivatives is None:
        dF_dR = np.gradient(f_R, R_um)
        d2F_dR2 = np.gradient(dF_dR, R_um)
        d3F_dR3 = np.gradient(d2F_dR2, R_um)
    else:
        dF_dR = derivatives['dF_dR']
        d2F_dR2 = derivatives['d2F_dR2']
        d3F_dR3 = derivatives['d3F_dR3']
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: PSD
    ax = axes[0]
    ax.plot(R_um, f_R, 'b-', linewidth=2, label=f'f(R) at {time_label}')
    ax.axvline(r_c_freeze, color='red', linestyle='--', linewidth=2.5, 
               label=f'$r_{{c,freeze}}$ = {r_c_freeze:.4f} µm')
    ax.axvline(r_c_melt, color='orange', linestyle='--', linewidth=2.5,
               label=f'$r_{{c,melt}}$ = {r_c_melt:.4f} µm')
    ax.set_xlabel('Radius (µm)', fontsize=12)
    ax.set_ylabel('PSD f(R)', fontsize=12)
    ax.set_title(f'Particle Size Distribution at {time_label} ', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 2: First derivative
    ax = axes[1]
    ax.plot(R_um, dF_dR, 'g-', linewidth=2, label='df/dR')
    ax.axvline(r_c_freeze, color='red', linestyle='--', linewidth=2.5, alpha=0.7, 
               label='Critical radii')
    ax.axvline(r_c_melt, color='orange', linestyle='--', linewidth=2.5, alpha=0.7)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Radius (µm)', fontsize=12)
    ax.set_ylabel('First Derivative df/dR', fontsize=12)
    ax.set_title('First Derivative of PSD (Kinks at Critical Radii)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 3: Third derivative
    # ax = axes[2]
    # ax.plot(R_um, d3F_dR3, 'purple', linewidth=2, label='d³f/dR³')
    # ax.axvline(r_c_freeze, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
    # ax.axvline(r_c_melt, color='orange', linestyle='--', linewidth=2.5, alpha=0.7, 
    #            label='Detected critical radii')
    # ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    # ax.set_xlabel('Radius (µm)', fontsize=12)
    # ax.set_ylabel('Third Derivative d³f/dR³', fontsize=12)
    # ax.set_title('Third Derivative of PSD (Peaks Show Discontinuities in d²f/dR²)',
    #              fontsize=13, fontweight='bold')
    # ax.legend(fontsize=11)
    # ax.grid(True, alpha=0.3)
    # ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    return fig, axes
