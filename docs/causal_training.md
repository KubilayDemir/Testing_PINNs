# Optional causal training and adaptive sampling

All three equations use the same `training_methods` settings. Edit the block in
a copied case configuration and run `python -m swe_pinn train --config` with
that file. Every option is disabled in the default and publication runs. These
methods change training, while preserving the network formulation, equation
residuals, transition functions and soft time-segment handoffs.

## Choose a method

- **Ordinary training:** `causal_weighting: none`, `sampling: lhs`.
- **Causal residual weighting:** `causal_weighting: residual`, `sampling: lhs`.
  Fixed probes estimate losses in ordered time slices. Later slices receive
  lower weights until earlier residuals and soft conditions are learned.
- **Causal adaptive sampling:** `causal_weighting: residual`, `sampling:
  causal_as`. Selects candidate points by squared residual times temporal
  weight raised to `sampling_power`.
- **Residual refinement:** `causal_weighting: none`, `sampling: residual`.
  Selects candidates with the largest squared PDE residuals.
- **Retain/resample/release:** `sampling: r3`. Retains above-mean residual
  points and replaces others with uniformly sampled points.
- **Causal R3:** `causal_weighting: none`, `sampling: causal_r3`. Applies its
  own smooth temporal gate to the PDE loss and retention scores. The gate
  progresses toward later times as its weighted PDE loss falls.
- **Causal attention:** `causal_weighting: attention`, `sampling: lhs` or
  `residual`. Uses relative IC error to control exponential temporal weights.
  Requires soft ICs; `network.enforce_ic: true` is incompatible.

Start with causal residual weighting, then compare causal adaptive sampling.
For example, change these two entries in the existing configuration:

```yaml
training_methods:
  causal_weighting: residual
  sampling: causal_as
```

A partial block inherits other settings from the package defaults. Complete
training-case YAMLs show every available parameter.

## Parameters and output

`time_bins` sets temporal resolution; `probe_points` must be at least that
large. Probes cover every slice and stay fixed through training and recovery.
`epsilon` controls residual weighting strength. Soft IC and time-binned BC
errors enter the temporal statistics; hard conditions add no fitting penalty.

For Causal AS, `power_schedule: temporal_alignment` enables TADU (temporal
alignment driven update): the sampling exponent adapts to align selected
points with the temporal training frontier.
`alignment_rate`, `alignment_steepness` and `alignment_threshold` control it.
The default `power_schedule: fixed` keeps `sampling_power` constant.

IC and BC statistics also use fixed subsets of at most `probe_points` per
condition pool. This bounds their evaluation cost for large SWE pools.

`candidate_points` controls the candidate budget, with a minimum equal to the
number of adaptive points selected. `update_period` gives the
number of epochs between adaptive sampling updates, starting at epoch one.
`uniform_fraction` reserves coverage points and limits retained populations.
PDE pool size stays constant. Existing `sampling.resample_period` independently
controls ordinary pool regeneration, including IC and BC points.

`attention_scale` controls attention strength. Causal R3 uses `gate_initial`,
`gate_steepness`, `gate_learning_rate`, `gate_tolerance` and `gate_max_step`.
Time is normalized within each network's own interval. For SWE, residual scores
sum squared momentum and continuity residuals in the configured units.

Weights are detached and refreshed before each optimizer step. They stay fixed
during L-BFGS line searches. Candidate residuals and probes are evaluated in
batches using `output.evaluation_batch_size`. SWE requires
`output.batch_diagnostics: true` with these methods.

`training_strategy.json` reports temporal weights or gate position, IC error,
the last weighted PDE objective, sampling updates and residual evaluation
counts. SWE writes one file per
segment. Recovery checkpoints preserve probes, temporal state, sampling pools
and the separate sampler RNG. Reported PDE losses use unweighted fixed probes;
BC and IC diagnostics use their condition pools. Compare those diagnostics and
full-interval reference errors across methods, rather than gated losses.

## Scientific references

- [Causal PINN weighting](https://arxiv.org/abs/2203.07404).
- [Causal R3](https://proceedings.mlr.press/v202/daw23a.html).
- [Causal AS](https://doi.org/10.1016/j.physd.2025.134878).
- [Causal attention](https://doi.org/10.1016/j.jcp.2026.115071), with the
  [author implementation](https://github.com/Chenrui-Z/Causal-Attention).

The implementation uses fixed diagnostic bins for arbitrary minibatches,
configurable sampling cadence and an explicit uniform coverage fraction.
Causal AS preserves its background population between ordinary pool resets.
TADU uses normalized segment time and bin centers for the frontier. These are
method options for comparison, rather than reproductions of the cited authors'
architectures and benchmark schedules.
