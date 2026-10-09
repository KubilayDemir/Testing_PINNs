# PINN components

The command loads a configuration, builds `PINN`, samples points, optimizes its
losses, and saves predictions and checkpoints. All three PDEs share `PINN`.

| File or folder | Responsibility |
|---|---|
| `cli.py`, `__main__.py` | Training, evaluation, plotting and monitoring commands |
| `config.py`, `defaults.yaml` | Configuration validation, defaults and physical scaling |
| `model.py` | Shared neural network, transition functions and loss assembly |
| `physics.py` | Momentum/continuity, heat and Burgers PDE residuals |
| `conditions.py` | Prescribed SWE initial and boundary values |
| `sampling.py` | PDE, initial-condition and boundary sampling pools |
| `training.py`, `training_loop.py` | SWE run setup, optimization and time-segment handoffs |
| `heat_and_burgers/` | One-field training workflow, using the shared model |
| `optimizers/` | PCGrad and the supplied L-BFGS implementation |
| `diagnostics.py`, `evaluation.py` | Batched loss reports and reference comparisons |
| `reference.py`, `output_formats.py` | Reference loading and dataset variable conventions |
| `output_storage.py` | Configuration/environment records and NetCDF output |
| `training_checkpoints.py` | Streamed histories, recovery checkpoints and RNG state |
| `plotting.py`, `live_losses.py` | Saved figures and a read-only live loss viewer |
| `synthetic_test_data.py`, `equation_checks.py` | Artificial software-test inputs and optional known-field checks |

Equation-specific physics and conditions remain distinct. See
[model settings](../../docs/model.md) for residuals, hard conditions and scaling.
