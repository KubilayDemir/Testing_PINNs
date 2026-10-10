# Repository for testing physics-informed neural network applications for the one-dimensional shallow water equations

This repository supports the study _Closed-Boundary Reflections of Shallow
Water Waves as an Open Challenge for Physics-Informed Neural Networks_ with
code, configurations, trained models and numerical reference solutions:

Demir, K. T.; Logemann, K.; Greenberg, D. S. (2024). Closed-Boundary
Reflections of Shallow Water Waves as an Open Challenge for Physics-Informed
Neural Networks. *Mathematics*, **12**(21), 3315.
[doi:10.3390/math12213315](https://doi.org/10.3390/math12213315).

The heat diffusion equation and the viscous Burgers equation are provided as
additional test cases for checking the implementation. These cases are not
part of the shallow water study.

## Publication data and figures

The data and notebook used to generate the publication figures and tables are
available on Zenodo:
[doi:10.5281/zenodo.23168993](https://doi.org/10.5281/zenodo.23168993).

The archive `pinn_closed_boundary_swe_mathematics2024_data_code.zip` contains:

- `datasets/`: Processed datasets in Network Common Data Form (NetCDF),
  including descriptions of the stored variables and their units.
- `generate_publication_figures_and_tables.ipynb`: A Jupyter notebook that
  reads these datasets and generates the publication figures and tables.

To generate a figure or table, extract the archive and open the notebook in
the extracted folder. First run the cell that imports the required Python
packages, then run the section for the figure or table you need. The trained
model parameters and their corresponding configurations are stored in this
repository under [`publication_cases/`](publication_cases/README.md).

## Repository structure

- [`publication_cases/`](publication_cases/README.md) contains the shallow
  water configurations and trained models associated with the publication.
  Each stored model has a corresponding configuration.
- [`training_cases/`](training_cases/README.md) contains the default shallow
  water configuration and the configurations for the heat diffusion and
  viscous Burgers test cases. These files can be used directly or copied and
  modified to define new training cases.
- [`reference_solutions/`](reference_solutions/README.md) contains numerical
  reference solutions used to evaluate model accuracy. It also contains code
  for generating the shallow water references and studying their convergence.
- [`src/swe_pinn/`](src/swe_pinn/README.md) contains the Python implementation
  of the shared neural network, equation residuals, training procedures,
  evaluation and output handling.
- [`training_results/`](training_results/README.md) stores the results of new
  training runs. Results are organized by case and run identifier and include
  the configuration, trained model parameters, training histories and
  evaluation outputs.
- [`cluster_jobs/`](cluster_jobs/README.md) contains scripts for submitting
  and running jobs on the Levante and Strand computing clusters.
- [`software_tests/`](software_tests/README.md) contains tests of the equation
  residuals, training procedures, checkpoint recovery and saved outputs. These
  tests use small networks and short runs to check the implementation.
- [`scripts/`](scripts/README.md) contains utilities for checking the Python
  environment, installing a local Fortran compiler on Windows, measuring
  graphics processing unit memory use and converting Fortran solver output.
- [`docs/`](docs/README.md) contains instructions for environment setup,
  training, model configuration, monitoring and optional training methods.

## Getting started

Follow the [environment setup instructions](docs/environment.md) to install
the package and its dependencies. Then run the following command from the
repository root to train the default shallow water case:

```bash
python -m swe_pinn train
```

This command uses the configuration in
`training_cases/shallow_water/config.yaml` and the included viscous numerical
reference solution. New training results are saved under `training_results/`.

The [running and outputs guide](docs/running.md) explains how to select an
equation, use a different configuration, resume a training run, generate plots
and monitor training losses. The
[publication cases guide](publication_cases/README.md) provides instructions
for training the published shallow water cases and evaluating the stored
models.
