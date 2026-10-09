#!/bin/bash
set -euo pipefail
manifest=$1
environment=$2
if [[ "$environment" != "-" ]]; then source "$environment"; fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
srun "${PINN_PYTHON:-python}" slurm/run.py "$manifest"
