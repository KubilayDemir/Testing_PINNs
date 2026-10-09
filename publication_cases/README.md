# Published shallow-water cases

These are the SWE configurations and trained models associated with the
[paper](https://doi.org/10.3390/math12213315). Heat and Burgers are separate
optional cases under `training_cases/`.

Each case has a runnable `config.yaml` beside `trained_models/`. Each stored
model retains its compatible configuration and available training metadata.
`models.json` lists all archived model/configuration paths.

- `baseline_viscous/`, `baseline_inviscid/`: Baselines with repeated
  initializations
- `network_sizes/`: Alternative hidden-layer sizes
- `collocation_points/`: Alternative sampling-pool sizes
- `gradient_methods/`: Plain Adam and adaptive weights with PCGrad
- `transition_functions/`: TF1 boundary enforcement; TF2/TF3 boundary and
  initial enforcement
- `pde_loss_training/`: Supplied models trained with PDE/IC/BC losses
- `reference_supervised/`: Numerical-reference supervision with repeated
  initializations
- `reference_loss_training/`: Supplied reference-loss training schedules

```bash
case=publication_cases/baseline_viscous
model="$case/trained_models/viscous_initialization_1"
python -m swe_pinn train --config "$case/config.yaml" \
  --seed 0 --device cuda:0
python -m swe_pinn evaluate \
  --config "$model/config.yaml" \
  --checkpoint "$model/trained.pt" \
  --output training_results/publication_evaluation/viscous_initialization_1 \
  --device cpu
```

The [Zenodo dataset](https://zenodo.org/records/23168993) provides
`pinn_closed_boundary_swe_mathematics2024_data_code.zip`, with processed NetCDF
files in `datasets/` and `generate_publication_figures_and_tables.ipynb`.
Extract the archive, open the notebook in the extracted folder, and run its
imports cell before a figure or table section. New-run figures use `python -m
swe_pinn plot --run-dir <run-directory>`.
