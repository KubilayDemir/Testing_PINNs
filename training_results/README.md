# Training results

New runs are stored as `<experiment>/<run-id>/` to avoid overwriting results.
This directory contains generated output, outside the published model archive.

- `config.yaml`, `config.json`: Complete settings used for this run
- `metadata.json`, `environment-freeze.txt`, `source.zip`: Environment, source
  and reference information
- `checkpoints/best.pt`, `final.pt`: Best reference-error model and final model
- `checkpoints/resume.pt`: Model, optimizer and random state for recovery, when
  enabled
- `history.jsonl`, `history.nc`: Training losses and reference errors
- `evaluation.nc`, `figures/`: Predictions, comparisons and plots
- `snapshots/`: Streamed fields at selected epochs
- `training_strategy.json`: Optional causal weighting and sampling diagnostics

SWE stores per-network outputs under `segments/000/`, `001/`, etc. Heat and
Burgers store them at the run root. Resume and plot commands take the run root.
`python -m swe_pinn monitor --run-dir <run-directory>` shows live loss curves.
These generated results are ignored by Git; only this README is tracked.
