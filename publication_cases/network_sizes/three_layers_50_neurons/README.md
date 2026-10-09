# Three hidden layers with 50 neurons each

SWE publication case. `config.yaml` defines a new training run;
`trained_models/` contains supplied weights and their compatible settings.

```bash
python -m swe_pinn train --config publication_cases/network_sizes/three_layers_50_neurons/config.yaml --device cuda:0
python -m swe_pinn evaluate \
  --config publication_cases/network_sizes/three_layers_50_neurons/trained_models/3x50_10k_25k_Init_1/config.yaml \
  --checkpoint publication_cases/network_sizes/three_layers_50_neurons/trained_models/3x50_10k_25k_Init_1/trained.pt \
  --output training_results/publication_evaluation/3x50 --device cpu
```

The evaluation example uses the first stored model. Select another model
with its adjacent configuration to evaluate a different initialization.
