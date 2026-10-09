# Cluster runs

Install the package in the cluster's CPU/CUDA environment. Copy
`environment.example.sh` to `environment.sh` and set modules/activation paths.
Load gfortran for reference generation.
Run `python scripts/check_environment.py --gpu` inside a GPU allocation to
verify the selected environment. See [environment setup](../docs/environment.md).

```bash
python cluster_jobs/submit.py --cluster levante --account YOUR_ACCOUNT --env-file cluster_jobs/environment.sh
python cluster_jobs/submit.py --cluster strand --partition YOUR_GPU_PARTITION --env-file cluster_jobs/environment.sh
python cluster_jobs/submit.py --cluster levante --account YOUR_ACCOUNT --equation heat
python cluster_jobs/submit.py --cluster levante --account YOUR_ACCOUNT --task reference --case inviscid
```

Use `--config` for any publication or new-run configuration. `--seeds 0 1 2`
creates a SLURM array: independent training runs that can use separate GPUs.
It does not split one network across GPUs. Use `--time`, `--cpus`, `--memory`
for resources. Strand needs its
partition name; Levante needs your project account. Training uses one GPU.
`--resume training_results/<case>/<run-id>` restores a run. `--dry-run` previews submission.
Outputs are written to the shared `training_results/` directory; job logs are in `logs/`.

`submit.py` prepares commands and requests resources; `job.sh` activates the
environment and starts a job; `run.py` executes the selected array member.
`generated/` holds job manifests (lists of commands), and `logs/` holds SLURM output.
Both folders are generated locally and ignored by Git.
See [live losses](../docs/live_losses.md) for monitoring through an SSH tunnel.
