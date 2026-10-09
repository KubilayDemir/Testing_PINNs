# Published shallow-water cases

These are the SWE configurations and trained models associated with the
[paper](https://doi.org/10.3390/math12213315). Heat and Burgers are separate
optional cases under `training_cases/`.

Each case has a runnable `config.yaml` beside `trained_models/`. Each stored
model retains its compatible configuration and available training metadata.
`models.json` lists all archived model/configuration paths.

| Case or group | Study setting |
|---|---|
| `baseline_viscous/`, `baseline_inviscid/` | Baselines with repeated initializations |
| `network_sizes/` | Alternative hidden-layer sizes |
| `collocation_points/` | Alternative sampling-pool sizes |
| `gradient_methods/` | Plain Adam and adaptive weights with PCGrad |
| `transition_functions/` | TF1 boundary enforcement; TF2/TF3 boundary and initial enforcement |
| `pde_loss_training/` | Supplied models trained with PDE/IC/BC losses |
| `reference_supervised/` | Numerical-reference supervision with repeated initializations |
| `reference_loss_training/` | Supplied reference-loss training schedules |

```bash
python -m swe_pinn train --config publication_cases/baseline_viscous/config.yaml --seed 0 --device cuda:0
python -m swe_pinn evaluate \
  --config publication_cases/baseline_viscous/trained_models/viscous_initialization_1/config.yaml \
  --checkpoint publication_cases/baseline_viscous/trained_models/viscous_initialization_1/trained.pt \
  --output training_results/publication_evaluation/viscous_initialization_1 --device cpu
```

Published data and figure notebooks are on [Zenodo](https://zenodo.org/records/23168993).
New-run figures use `python -m swe_pinn plot --run-dir <run-directory>`.
