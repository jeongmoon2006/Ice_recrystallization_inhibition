# Ice Recrystallization Inhibition Model using Two Critical Radii Framework

## Objective
Extract AFP-related constants (k_m*L^-2 and k_f*L^-2) from experimental particle size distribution data using a two critical radii framework for ice recrystallization.

## Model Equations
The critical radii for melting and freezing are defined as:

$$
r_{c,\mathrm{melt}}=\frac{\alpha}{c_{\mathrm{bulk}}(t)-c_{\mathrm{flat}}+k_mL^{-2}},
\qquad
r_{c,\mathrm{freeze}}=\frac{\alpha}{c_{\mathrm{bulk}}(t)-c_{\mathrm{flat}}-k_fL^{-2}}
$$

where both critical radii are time-dependent through c_bulk(t) and AFP-related terms.

## Algorithm Workflow

### Input Parameters (Known from experiments)
- c_flat: flat interface concentration
- D: diffusion coefficient
- rho_ice: ice density

### Step 1: Extract Critical Radii
From particle size distribution f(r,t), identify r_c,melt and r_c,freeze as inflection points

### Step 2: Initialize Simulation
- Start with initial distribution f(r,t_0)
- Use initial guess for c_bulk(t)
- Run forward simulation

### Step 3: Parameter Optimization
Iteratively adjust parameters to match evolved distribution:
- c_bulk(t)
- k_m*L^-2
- k_f*L^-2

## Implementation Strategy
For initial distribution at t=10 min as a starting point:
1. Identify r_c,melt and r_c,freeze from the distribution
2. With initial guess of c_bulk(t=10min), extract k_m*L^-2 and k_f*L^-2 using the model equations
3. Use these values as initial estimates for the optimization

## Code Requirements
Implement functions to:
- [ ] Extract inflection points from f(r,t) to find critical radii
- [ ] Solve for AFP constants given r_c and c_bulk(t)
- [ ] Simulate particle size evolution
- [ ] Optimize parameters by comparing simulated vs experimental distributions