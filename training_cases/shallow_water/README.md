# Shallow-water training

`config.yaml` is the complete default for new SWE runs. It uses the included
viscous numerical solution. Run `python -m swe_pinn train` from the repository
root. Use `--device cuda:0` for a GPU, or copy the configuration for another
experiment.

The network predicts velocity and surface elevation. PDE losses enforce
momentum and continuity; boundary/initial losses or transition functions
enforce conditions. Time decomposition uses the previous network's final output
as a soft IC target. See [model settings](../../docs/model.md) and [publication
cases](../../publication_cases/README.md).
