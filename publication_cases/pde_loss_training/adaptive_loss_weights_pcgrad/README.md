# Adaptive loss weights with PCGrad

SWE publication case. `config.yaml` defines a new training run;
`trained_models/` contains supplied weights and their compatible settings.

```bash
python -m swe_pinn train --config publication_cases/pde_loss_training/adaptive_loss_weights_pcgrad/config.yaml --device cuda:0
python -m swe_pinn evaluate \
  --config publication_cases/pde_loss_training/adaptive_loss_weights_pcgrad/trained_models/config.yaml \
  --checkpoint publication_cases/pde_loss_training/adaptive_loss_weights_pcgrad/trained_models/trained.pt \
  --output training_results/publication_evaluation/adaptive_weights_pcgrad --device cpu
```

The evaluation example uses the first stored model. Select another model
with its adjacent configuration to evaluate a different initialization.
