# Models and settings

All three equations use `swe_pinn.model.PINN`. SWE predicts velocity `u` and
surface elevation `zeta` with a linear output layer. Heat and Burgers predict
one field `u` and activate their output layer with the configured activation.
Equation-specific physics and conditions retain these formulations.

`train` loads the selected case configuration, constructs this class, and
samples PDE, boundary and initial-condition points. SWE training handles its
two fields and optional time segments; heat/Burgers share a one-field training
workflow. Both use the same model class and optimizer utilities.

- SWE: momentum: `u_t + u*u_x + g*zeta_x - viscosity*u_xx + drag*u*abs(u)`;
  continuity: `zeta_t + (depth+zeta)*u_x + u*zeta_x` / zero velocity, Gaussian
  elevation; zero boundary velocities
- Heat diffusion: `u_t - diffusivity*u_xx` / `sin(pi*x)`; zero at both
  boundaries
- Viscous Burgers: `u_t + u*u_x - diffusivity*u_xx` / `-sin(pi*x)`; zero at
  both boundaries

SWE physics/domain YAML values use SI units. Nondimensionalization uses `t/T`,
`x/L`, `u*T/L`, `zeta/H`, with `H = vertical_factor*L**2/(g*T**2)`. The scaled
viscosity is `viscosity*T/L**2`. Saved fields use physical units. Heat and
Burgers use `t in [0,1]`, `x in [-1,1]`; default diffusivities are 0.25 and
0.01/pi respectively. A different Burgers diffusivity needs a matching
reference.

`network.enforce_bc` selects the boundary transition; `enforce_ic` selects an
initial transition. SWE TF2 blends a decaying initial elevation; TF3 adds the
initial elevation. Hard ICs apply to the first time segment; later segments use
the preceding network's final output as a soft IC target. The switches select
hard enforcement of the prescribed conditions; arbitrary boundary values or
types are not configurable through these switches.

`network.hidden_layers` sets network size, e.g. `[200,200,200,200,200]`. GPU
memory depends mainly on batch size and derivative graphs. CPU reference
storage, streamed histories and batched diagnostics are enabled by default.
Reduce batch sizes or `output.evaluation_batch_size` if needed.
`network.activation_checkpointing` optionally trades recomputation for memory.
No mixed precision is applied to PDE derivatives.

On the allocated GPU, measure a proposed size before long training: `python
scripts/check_gpu_memory.py --layers 5 --width 200 --checkpointing`.

The complete equation YAMLs expose PCGrad, adaptive weights, transition
functions, output frequencies and DataLoader settings. SWE also supports split
networks and time decomposition. Heat/Burgers expose optional Adam
learning-rate schedules and `initial_checkpoint` warm starts; recovery restores
optimizer/RNG state.
