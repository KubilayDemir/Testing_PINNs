# baseline inviscid

SWE publication case. `config.yaml` defines a new training run;
`trained_models/` contains supplied weights and their compatible settings.

```bash
python -m swe_pinn train --config publication_cases/baseline_inviscid/config.yaml --device cuda:0
python -m swe_pinn evaluate \
  --config publication_cases/baseline_inviscid/trained_models/inviscid_initialization_1/config.yaml \
  --checkpoint publication_cases/baseline_inviscid/trained_models/inviscid_initialization_1/trained.pt \
  --output training_results/publication_evaluation/baseline_inviscid --device cpu
```

The evaluation example uses the first stored model. Select another model
with its adjacent configuration to evaluate a different initialization.
