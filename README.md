# Testing PINNs

PINNs for one-dimensional shallow-water equations (SWE), with optional heat
diffusion and viscous Burgers benchmarks.

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

The default SWE run uses the included viscous numerical reference.
Append `--device cuda:0` to use a GPU.

## Configure

Edit `cases/swe/config.yaml`, `cases/heat/config.yaml` or `cases/burgers/config.yaml`, or copy
one and pass `--config your_case.yaml`. Each file includes its model, physics,
sampling, optimizer and output settings. Adam defaults to learning rate 0.0005.

All cases use minibatches. For full-batch updates, set each batch size to its
network's complete sampling-pool size and `iterations_per_epoch: 1`.
Hard conditions use transition functions; time-decomposition handoffs are soft ICs.

## Layout

| Directory | Purpose |
|---|---|
| [`cases/`](cases/README.md) | New-run configurations and optional benchmark weights |
| [`paper/`](paper/README.md) | SWE paper configurations and trained models |
| [`numerical_reference/`](numerical_reference/swe/README.md) | Numerical solutions and Fortran generator |
| `src/swe_pinn/` | Shared PINN class, equation physics and training workflows |
| `runs/` | New results, separated by case and run ID |
| [`slurm/`](slurm/README.md) | Levante and Strand launchers |

## Results

Each run stores its configuration, environment, source, checkpoints and NetCDF
outputs. `best.pt` selects the smallest reference error; `final.pt` saves the
last iterate. The reference only enters the loss when explicitly enabled.

```bash
python -m swe_pinn resume --run-dir runs/<case>/<run-id>
python -m swe_pinn plot --run-dir runs/<case>/<run-id>
python -m pytest
```

SWE outputs are under `segments/000/`, `001/`, etc.; benchmark outputs are
at the run root. Spatial histories are streamed and diagnostics are batched.
See [model settings](docs/model.md) for equations, scaling and memory controls.

Local `.venv/`, `.tools/`, `__pycache__/` and `*.egg-info/` directories are
generated environment/tooling files, ignored by Git. Editable installation
creates `egg-info` package metadata; it is not a model or an experiment.
