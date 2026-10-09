# Testing PINNs

PINNs for the published one-dimensional shallow-water study, plus two optional
PDE test cases: heat diffusion and viscous Burgers.

The SWE study is described in the [paper](https://doi.org/10.3390/math12213315).
Published data and figure notebooks are on [Zenodo](https://zenodo.org/records/23168993).

## Run

From the repository root, using a CPU/CUDA PyTorch environment:

```bash
python -m pip install -e ".[test]"
python -m swe_pinn train
python -m swe_pinn train --equation heat
python -m swe_pinn train --equation burgers
```

Alternatively, create the supplied environment once, then activate it:

```bash
conda env create -f environment.yaml
conda activate testing-pinns
python scripts/check_environment.py --notebooks
```

It installs the training, plotting, test and notebook packages. Activation
selects the environment; it does not install missing packages. See
[environment setup](docs/environment.md) for pip, CUDA and Fortran instructions.

The default SWE run uses the included viscous numerical reference.
Append `--device cuda:0` to use a GPU.

## Configure

Choose a configuration under [training_cases/](training_cases/README.md), or
copy one and pass `--config your_case.yaml`. Each file includes its model, physics,
sampling, optimizer and output settings. Adam defaults to learning rate 0.0005.

All cases use minibatches. For full-batch updates, set each batch size to its
network's complete sampling-pool size and `iterations_per_epoch: 1`.
Hard conditions use transition functions; time-decomposition handoffs are soft ICs.

## Layout

| Directory | Purpose |
|---|---|
| [`training_cases/`](training_cases/README.md) | Default SWE setup and the two optional PDE test cases |
| [`publication_cases/`](publication_cases/README.md) | Published SWE cases, each with configurations and trained models |
| [`reference_solutions/`](reference_solutions/README.md) | Numerical solutions and generation code |
| [`src/swe_pinn/`](src/swe_pinn/README.md) | Shared model and its components |
| [`training_results/`](training_results/README.md) | New outputs, separated by case and run ID |
| [`cluster_jobs/`](cluster_jobs/README.md) | Levante and Strand launchers |
| [`software_tests/`](software_tests/README.md) | Short checks of equations, training and saved outputs |

## Results

Each run stores its configuration, environment, source, checkpoints and NetCDF
outputs. `best.pt` selects the smallest reference error; `final.pt` saves the
last iterate. The reference only enters the loss when explicitly enabled.

```bash
python -m swe_pinn resume --run-dir training_results/<case>/<run-id>
python -m swe_pinn plot --run-dir training_results/<case>/<run-id>
python -m swe_pinn monitor --run-dir training_results/<case>/<run-id>
python -m pytest
```

SWE outputs are under `segments/000/`, `001/`, etc.; benchmark outputs are
at the run root. Spatial histories are streamed and diagnostics are batched.
See [model settings](docs/model.md) for equations, scaling and memory controls.
The optional [live loss viewer](docs/live_losses.md) updates in a browser.

Local `.venv/`, `.tools/`, `__pycache__/` and `*.egg-info/` directories are
generated environment/tooling files, ignored by Git. Editable installation
creates `egg-info` package metadata; it is not a model or an experiment.
