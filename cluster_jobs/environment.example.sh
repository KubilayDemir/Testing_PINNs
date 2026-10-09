#!/bin/bash
# Select modules appropriate to your cluster, then activate your environment.
# module load <your Python/CUDA module>
# module load <your gfortran module>  # for numerical reference jobs
source /path/to/your/venv/bin/activate
export PINN_PYTHON="$(command -v python)"
