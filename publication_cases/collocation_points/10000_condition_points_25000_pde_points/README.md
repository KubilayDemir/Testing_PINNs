# 10,000 condition points and 25,000 PDE points

The initial-condition pool and each boundary pool use 10,000 points.
The PDE sampling pool uses 25,000 points.

SWE publication case. `config.yaml` defines a new training run;
`trained_models/` contains supplied weights and their compatible settings.

```bash
python -m swe_pinn train --config publication_cases/collocation_points/10000_condition_points_25000_pde_points/config.yaml --device cuda:0
python -m swe_pinn evaluate \
  --config publication_cases/collocation_points/10000_condition_points_25000_pde_points/trained_models/10k_25k_CP_Init_1/config.yaml \
  --checkpoint publication_cases/collocation_points/10000_condition_points_25000_pde_points/trained_models/10k_25k_CP_Init_1/trained.pt \
  --output training_results/publication_evaluation/10k_25k --device cpu
```

The evaluation example uses the first stored model. Select another model
with its adjacent configuration to evaluate a different initialization.
