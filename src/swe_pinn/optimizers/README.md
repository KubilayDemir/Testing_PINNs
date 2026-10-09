# Optimizers

Adam and standard L-BFGS come from PyTorch. `pcgrad.py` projects conflicting
gradients from separate PDE, initial and boundary losses when enabled.
`lbfgs.py` contains the supplied full-batch L-BFGS implementation.
These methods are selected in the training configuration; the default is Adam.
`PCGrad_LICENSE` retains the upstream code's license.
