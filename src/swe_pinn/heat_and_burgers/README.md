# Heat and Burgers training

`training.py` handles the optional one-field PDE cases: sampling, optimization,
reference comparison, checkpoints and plots. It constructs the shared `PINN`
from `model.py`; it does not define a separate network class. `defaults.yaml`
supplies fallback settings for partial custom configurations. Use the complete
files under `training_cases/` for new experiments.
