# Three hidden layers with 50 neurons each

SWE publication case. `config.yaml` defines a new training run;
`trained_models/` contains supplied weights and their compatible settings.

From the repository root (Bash):

```bash
case=publication_cases/network_sizes/three_layers_50_neurons
model="$case/trained_models/3x50_10k_25k_Init_1"
python -m swe_pinn train --config "$case/config.yaml" --device cuda:0
python -m swe_pinn evaluate \
  --config "$model/config.yaml" \
  --checkpoint "$model/trained.pt" \
  --output "training_results/publication_evaluation/$case" \
  --device cpu
```
