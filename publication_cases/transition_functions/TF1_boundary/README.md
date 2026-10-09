# TF1 boundary

SWE publication case. `config.yaml` defines a new training run;
`trained_models/` contains supplied weights and their compatible settings.

```bash
python -m swe_pinn train --config publication_cases/transition_functions/TF1_boundary/config.yaml --device cuda:0
python -m swe_pinn evaluate \
  --config publication_cases/transition_functions/TF1_boundary/trained_models/config.yaml \
  --checkpoint publication_cases/transition_functions/TF1_boundary/trained_models/trained.pt \
  --output training_results/publication_evaluation/TF1_boundary --device cpu
```

The evaluation example uses the first stored model. Select another model
with its adjacent configuration to evaluate a different initialization.
