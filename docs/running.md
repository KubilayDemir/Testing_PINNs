# Running and outputs

Run commands from the repository root after following the
[environment setup](environment.md). The default SWE case uses the included
viscous numerical reference. Heat and Burgers are optional test cases:

```bash
python -m swe_pinn train
python -m swe_pinn train --equation heat
python -m swe_pinn train --equation burgers
```

Append `--device cuda:0` to use a GPU, or `--device cpu` for CPU execution.
For SLURM submission, see [cluster jobs](../cluster_jobs/README.md).

## Configuring a run

Choose a configuration under [training cases](../training_cases/README.md), or
copy one and pass `--config your_case.yaml`. Each file includes model, physics,
sampling, optimizer and output settings. Adam defaults to learning rate 0.0005.

All cases use minibatches. For full-batch updates, set each batch size to its
network's complete sampling-pool size and `iterations_per_epoch: 1`. Hard
conditions use transition functions; time-decomposition handoffs are soft ICs.
See [model settings](model.md) for equations, scaling and memory controls.

Optional [causal training and adaptive sampling](causal_training.md) use the
existing `training_methods` block. They are disabled by default and for paper
reproduction. Published configurations and evaluation commands are documented
under [publication cases](../publication_cases/README.md).

## Saved results and monitoring

Each run stores its configuration, environment, source, checkpoints and NetCDF
outputs. `best.pt` selects the smallest reference error; `final.pt` saves the
last iterate. The reference only enters the loss when explicitly enabled.

```bash
python -m swe_pinn resume --run-dir training_results/<case>/<run-id>
python -m swe_pinn plot --run-dir training_results/<case>/<run-id>
python -m swe_pinn monitor --run-dir training_results/<case>/<run-id>
```

SWE outputs are under `segments/000/`, `001/`, etc.; heat and Burgers outputs
are at the run root. Spatial histories are streamed and diagnostics are
batched. See [training results](../training_results/README.md) for the stored
files and [live losses](live_losses.md) for browser monitoring and SSH access.

The plot command creates figures from new outputs. The publication's NetCDF
datasets and figure/table notebook are available on
[Zenodo](https://zenodo.org/records/23168993).

## Software checks and generated folders

Run `python -m pytest` for the default software checks; see
[software tests](../software_tests/README.md) for the optional equation checks.

Local `.venv/`, `.tools/`, `__pycache__/` and `*.egg-info/` directories are
generated environment/tooling files, ignored by Git. Editable installation
creates `egg-info` package metadata.
