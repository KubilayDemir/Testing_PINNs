# plain adam

SWE publication case. `config.yaml` defines a new training run;
`trained_models/` contains supplied weights and their compatible settings.

```bash
python -m swe_pinn train --config publication_cases/gradient_methods/plain_adam/config.yaml --device cuda:0
python -m swe_pinn evaluate \
  --config publication_cases/gradient_methods/plain_adam/trained_models/viscous/LRA_OFF_PCGRAD_OFF_Init_1/config.yaml \
  --checkpoint publication_cases/gradient_methods/plain_adam/trained_models/viscous/LRA_OFF_PCGRAD_OFF_Init_1/trained.pt \
  --output training_results/publication_evaluation/plain_adam --device cpu
```

The evaluation example uses the first stored model. Select another model
with its adjacent configuration to evaluate a different initialization.
