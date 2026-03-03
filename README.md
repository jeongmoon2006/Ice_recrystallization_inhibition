# Ice Recrystallization Inhibition (IRI)

Minimal model for ice recrystallization with **two critical radii**:
- melting critical radius,
- freezing critical radius,
- stay region between them where growth velocity is zero.

## Files
- `iri_model.py`: core ODE model (`run_simulation`, `critical_radii`).
- `IRI_ODE.ipynb`: example notebook with initial configuration and plotting.
- `requirements.txt`: pip dependency list.
- `environment.yml`: Conda environment specification.

## Quick start
### Option A: Conda (recommended if you use Anaconda)
1. `conda env create -f environment.yml`
2. `conda activate iri-afp`
3. `jupyter notebook`

### Option B: pip/venv
1. `python -m venv .venv`
2. `# Windows: .venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `jupyter notebook`

Then open `IRI_ODE.ipynb` and run all cells.

## API (core)
- `critical_radii(c_bulk, params)` → `(R_melt, R_freeze)`
- `run_simulation(f_init, c_bulk_init, R, t_span, params, mode="double", t_eval=None)`

`mode="double"` enables the two-threshold IRI mechanism.
