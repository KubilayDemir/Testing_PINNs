# TF1 boundary

SWE publication case. `config.yaml` defines a new training run;
`trained_models/` contains supplied weights and their compatible settings.

From the repository root (Bash):

```bash
case=publication_cases/transition_functions/TF1_boundary
model="$case/trained_models"
python -m swe_pinn train --config "$case/config.yaml" --device cuda:0
python -m swe_pinn evaluate \
  --config "$model/config.yaml" \
  --checkpoint "$model/trained.pt" \
  --output "training_results/publication_evaluation/$case" \
  --device cpu
```
