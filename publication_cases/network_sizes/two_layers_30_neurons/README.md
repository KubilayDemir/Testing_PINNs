# Two hidden layers with 30 neurons each

SWE publication case. `config.yaml` defines a new training run;
`trained_models/` contains supplied weights and their compatible settings.

```bash
python -m swe_pinn train --config publication_cases/network_sizes/two_layers_30_neurons/config.yaml --device cuda:0
python -m swe_pinn evaluate \
  --config publication_cases/network_sizes/two_layers_30_neurons/trained_models/2x30_10k_25k_Init_1/config.yaml \
  --checkpoint publication_cases/network_sizes/two_layers_30_neurons/trained_models/2x30_10k_25k_Init_1/trained.pt \
  --output training_results/publication_evaluation/2x30 --device cpu
```

The evaluation example uses the first stored model. Select another model
with its adjacent configuration to evaluate a different initialization.
