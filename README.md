# Testing PINNs

This repository supports the following study with code, configurations, trained
models and numerical reference solutions:

Demir, K. T.; Logemann, K.; Greenberg, D. S. (2024). Closed-Boundary
Reflections of Shallow Water Waves as an Open Challenge for Physics-Informed
Neural Networks. *Mathematics*, **12**(21), 3315.
[doi:10.3390/math12213315](https://doi.org/10.3390/math12213315).

Heat diffusion and viscous Burgers are additional test cases.

## Publication data and figures

The accompanying [Zenodo dataset](https://zenodo.org/records/23168993)
(DOI: [10.5281/zenodo.23168993](https://doi.org/10.5281/zenodo.23168993))
provides `pinn_closed_boundary_swe_mathematics2024_data_code.zip`, containing:

- Processed NetCDF datasets in `datasets/`, with descriptive metadata and
  units.
- `generate_publication_figures_and_tables.ipynb`, which generates the
  publication figures and tables from those datasets.

Extract the archive and open the notebook inside the extracted folder. Run its
imports cell, then the desired figure or table section. The trained parameter
states and their corresponding model configurations are in this repository
under [`publication_cases/`](publication_cases/README.md).

## Repository structure

- [`publication_cases/`](publication_cases/README.md): Published SWE cases,
  each with configurations and trained models
- [`training_cases/`](training_cases/README.md): Default SWE configuration and
  the two optional PDE test cases, ready to use or copy for new runs
- [`reference_solutions/`](reference_solutions/README.md): Numerical solutions
  and generation code, including convergence studies
- [`src/swe_pinn/`](src/swe_pinn/README.md): Shared model and its components
- [`training_results/`](training_results/README.md): New outputs, separated by
  case and run ID
- [`cluster_jobs/`](cluster_jobs/README.md): Levante and Strand launchers
- [`software_tests/`](software_tests/README.md): Short checks of equations,
  training and saved outputs
- [`scripts/`](scripts/README.md): Environment, compiler and memory utilities
- [`docs/`](docs/README.md): Setup, running, model and optional-method guides

## Getting started

Follow the [environment setup](docs/environment.md), then run the default SWE
case from the repository root:

```bash
python -m swe_pinn train
```

See [running and outputs](docs/running.md) for equation selection,
configuration, resuming, plotting and monitoring. For a published case, start
with the [publication cases guide](publication_cases/README.md).
