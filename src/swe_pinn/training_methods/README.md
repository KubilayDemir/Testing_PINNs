# Optional training strategies

`controller.py` manages causal residual weighting, causal attention, greedy
residual refinement, causal adaptive sampling and retain/resample/release. The
same controller serves SWE, heat diffusion and viscous Burgers. `prepare_swe`
supplies the correct IC target for each SWE time segment.

The controller is separate from network layers and equation residuals. Disabled
strategies consume no random numbers and leave ordinary losses unchanged.
Enabled strategies use fixed probes and a separate CPU sampler generator, both
saved in recovery checkpoints.

See [causal training settings](../../../docs/causal_training.md) for usage,
compatible combinations, diagnostics and scientific references.
