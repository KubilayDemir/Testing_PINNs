# Two hidden layers with 30 neurons each

SWE publication case. `config.yaml` defines a new training run;
`trained_models/` contains supplied weights and their compatible settings.

From the repository root (Bash):

```bash
case=publication_cases/network_sizes/two_layers_30_neurons
model="$case/trained_models/2x30_10k_25k_Init_1"
python -m swe_pinn train --config "$case/config.yaml" --device cuda:0
python -m swe_pinn evaluate \
  --config "$model/config.yaml" \
  --checkpoint "$model/trained.pt" \
  --output "training_results/publication_evaluation/$case" \
  --device cpu
```
