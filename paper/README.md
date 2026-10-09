# SWE paper cases

`configurations/` contains runnable study presets. `trained_models/` groups
archived weights and settings by case; each run has a compatible `config.yaml`.
`models.json` indexes their paths. See the [case list](trained_models/README.md).

```bash
python -m swe_pinn train --config paper/configurations/baseline_viscous.yaml --seed 0
python -m swe_pinn evaluate \
  --config paper/trained_models/baseline/viscous/viscous_initialization_1/config.yaml \
  --checkpoint paper/trained_models/baseline/viscous/viscous_initialization_1/trained.pt \
  --output runs/paper_evaluation/viscous_initialization_1 --device cpu
```

TF1 imposes boundary conditions; TF2 and TF3 impose boundary and first-segment
initial conditions using different elevation transitions. Published data and
figures are supplied by the [Zenodo notebook](https://zenodo.org/records/23168993).
