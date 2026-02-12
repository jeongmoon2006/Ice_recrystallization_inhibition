# Ice Recrystallization Inhibition (IRI) Analysis Code

This repository contains numerical simulation tools for analyzing **Ice Recrystallization Inhibition (IRI)** of **Antifreeze Protein (AFP)**, developed as part of Ph.D. research at the **University of Pennsylvania**.

## 🔬 Research Background
The code simulates the evolution of ice particle size distribution (PSD) over time, considering the governing equations for particle growth and conservation of bulk concentration.

### Key Features
* **Particle Growth Simulation:** Implementation of $v_R$ equations considering curvature and AFP-related parameters ($L$, $k$).
* **Dimerization Analysis:** Tools to calculate binding free energy and its impact on dimerization.
* **Visualization:** Generation of PDF evolution plots and animated GIFs (Lifshitz-Slyozov-Wagner theory analysis).

## 📂 Structure
- `iri_model.py`: Core numerical integration and governing equations.
- `2CR_D.ipynb`: Analysis of two critical radii and ice crystal growth inhibition.
- `SBW/`: Subfolder for specific simulation batches.

## 🚀 Getting Started
```python
from iri_model import time_integration
# Define your parameters and run the simulation
results = run_simulation(f_init, params, time_arr)
