# Cluster runs

Install the package in the cluster's CPU/CUDA environment. Copy
`environment.example.sh` to `environment.sh` and set modules/activation paths.
Load gfortran for reference generation.

```bash
python slurm/submit.py --cluster levante --account YOUR_ACCOUNT --env-file slurm/environment.sh
python slurm/submit.py --cluster strand --partition YOUR_GPU_PARTITION --env-file slurm/environment.sh
python slurm/submit.py --cluster levante --account YOUR_ACCOUNT --equation heat
python slurm/submit.py --cluster levante --account YOUR_ACCOUNT --task reference --case inviscid
```

Use `--config` for any paper/new configuration, `--seeds 0 1 2` for independent
array jobs, and `--time`, `--cpus`, `--memory` for resources. Strand needs its
partition name; Levante needs your project account. Training uses one GPU.
`--resume runs/<case>/<run-id>` restores a run. `--dry-run` previews submission.
Outputs are written to the shared `runs/` directory; job logs are in `logs/`.
