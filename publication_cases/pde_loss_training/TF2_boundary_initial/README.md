# TF2 boundary initial

SWE publication case. `config.yaml` defines a new training run;
`trained_models/` contains supplied weights and their compatible settings.

```bash
python -m swe_pinn train --config publication_cases/pde_loss_training/TF2_boundary_initial/config.yaml --device cuda:0
python -m swe_pinn evaluate \
  --config publication_cases/pde_loss_training/TF2_boundary_initial/trained_models/config.yaml \
  --checkpoint publication_cases/pde_loss_training/TF2_boundary_initial/trained_models/trained.pt \
  --output training_results/publication_evaluation/TF2_boundary_initial --device cpu
```

The evaluation example uses the first stored model. Select another model
with its adjacent configuration to evaluate a different initialization.
