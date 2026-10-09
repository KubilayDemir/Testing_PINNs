# 20,000 condition points and 50,000 PDE points

The initial-condition pool and each boundary pool use 20,000 points. The PDE
sampling pool uses 50,000 points.

SWE publication case. `config.yaml` defines a new training run;
`trained_models/` contains supplied weights and their compatible settings.

From the repository root (Bash):

```bash
case=publication_cases/collocation_points
case="$case/20000_condition_points_50000_pde_points"
model="$case/trained_models/20k_50k_CP_Init_1/624799/0"
python -m swe_pinn train --config "$case/config.yaml" --device cuda:0
python -m swe_pinn evaluate \
  --config "$model/config.yaml" \
  --checkpoint "$model/trained.pt" \
  --output "training_results/publication_evaluation/$case" \
  --device cpu
```
