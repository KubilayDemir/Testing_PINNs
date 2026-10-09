# Archived study groups

| Folder | Contents |
|---|---|
| [`baseline/`](baseline/) | Viscous/inviscid default runs, five initializations each |
| [`network_sizes/`](network_sizes/) | Alternative hidden-layer sizes |
| [`collocation_points/`](collocation_points/) | Alternative collocation pool sizes |
| [`gradient_methods/`](gradient_methods/) | Plain Adam and adaptive weights with PCGrad |
| [`transition_functions/`](transition_functions/) | TF1, TF2 and TF3 |
| [`pinn_loss/`](pinn_loss/) | PDE/IC/BC loss training |
| [`reference_supervised/`](reference_supervised/) | Numerical-reference supervision, repeated initializations |
| [`reference_loss_training/`](reference_loss_training/) | Reference-loss training with its archived schedule |

Open a run directory for its `trained.pt`, `best.pt` where supplied,
`training_metadata.json` and runnable `config.yaml`. Training/evaluation commands
are in [`../README.md`](../README.md); [`../models.json`](../models.json) lists
every configuration/state-dict path.
