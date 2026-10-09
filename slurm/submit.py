"""Submit training, reference generation or convergence jobs to SLURM."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import uuid

ROOT=Path(__file__).resolve().parents[1]


def build(args):
    if args.cluster=="strand" and not args.partition:
        raise ValueError("Pass --partition with your Strand partition name")
    gpu=args.task=="train"
    partition=args.partition or ("gpu" if gpu else "shared")
    logs=(ROOT/"slurm/logs"); logs.mkdir(parents=True,exist_ok=True)
    generated=ROOT/"slurm/generated"; generated.mkdir(parents=True,exist_ok=True)
    commands=[]
    if gpu:
        if args.resume:
            if len(args.seeds)!=1: raise ValueError("Resume accepts a single run, without a seed array")
            command=["-m","swe_pinn","resume","--run-dir",args.resume]
            commands.append(command)
        else:
            for seed in args.seeds:
                command=["-m","swe_pinn","train","--equation",args.equation,"--config",args.config or f"configs/{args.equation}.yaml",
                         "--device","cuda:0","--cpu-threads",str(args.cpus),"--seed",str(seed),
                         "--run-id",f"slurm-{{job_id}}-{{task_id}}-seed-{seed}"]
                if args.options: command += ["--options",*args.options]
                commands.append(command)
    else:
        command=[f"numerical_reference/swe/{'generate' if args.task=='reference' else 'convergence'}.py","--case",args.case]
        if args.task=="convergence": command += ["--study",args.study,"--workers",str(args.cpus)]
        commands.append(command)
    manifest=generated/f"{args.task}-{uuid.uuid4().hex[:8]}.json"
    manifest.write_text(json.dumps({"commands":commands},indent=2)+"\n")
    command=["sbatch","--parsable",f"--chdir={ROOT}",f"--partition={partition}","--nodes=1","--ntasks=1",
             f"--cpus-per-task={args.cpus}",f"--time={args.time}",f"--mem={args.memory}",
             f"--job-name=pinn-{args.task}",f"--output={logs}/%x-%A_%a.out"]
    if args.account: command.append(f"--account={args.account}")
    if gpu: command.append("--gpus=1" if args.cluster=="levante" else f"--gres={args.gpu_resource}")
    if args.constraint: command.append(f"--constraint={args.constraint}")
    if len(commands)>1: command.append(f"--array=0-{len(commands)-1}%{args.max_parallel}")
    if args.dependency: command.append(f"--dependency=afterok:{args.dependency}")
    environment=str(Path(args.env_file).resolve()) if args.env_file else "-"
    command += [str(ROOT/"slurm/job.sh"),str(manifest),environment]
    return command,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster",choices=("levante","strand"),required=True)
    p.add_argument("--partition"); p.add_argument("--account"); p.add_argument("--env-file")
    p.add_argument("--task",choices=("train","reference","convergence"),default="train")
    p.add_argument("--equation",choices=("swe","heat","burgers"),default="swe")
    p.add_argument("--config"); p.add_argument("--options",nargs="+")
    p.add_argument("--seeds",type=int,nargs="+",default=[0])
    p.add_argument("--resume"); p.add_argument("--case",choices=("viscous","inviscid"),default="viscous")
    p.add_argument("--study",choices=("dx","dt"),default="dx")
    p.add_argument("--cpus",type=int,default=4); p.add_argument("--memory",default="32G")
    p.add_argument("--time",default="08:00:00"); p.add_argument("--max-parallel",type=int,default=5)
    p.add_argument("--constraint"); p.add_argument("--gpu-resource",default="gpu:1")
    p.add_argument("--dependency",help="start after this reference/other job succeeds")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    if args.cpus<1 or args.max_parallel<1: p.error("cpus and max-parallel must be positive")
    if args.cluster=="levante" and not args.account: p.error("Levante requires --account")
    try: command,manifest=build(args)
    except ValueError as error: p.error(str(error))
    print("Manifest:",manifest)
    print(shlex.join(command))
    if not args.dry_run:
        if not shutil.which("sbatch"): p.error("sbatch unavailable; run this on the cluster or use --dry-run")
        subprocess.run(command,check=True)

if __name__=="__main__": main()
