"""Execute one SLURM array member inside the activated Python environment."""
import json
import os
from pathlib import Path
import sys

manifest=json.loads(Path(sys.argv[1]).read_text())
index=int(os.environ.get("SLURM_ARRAY_TASK_ID",0))
command=manifest["commands"][index]
command=[part.replace("{job_id}",os.environ.get("SLURM_ARRAY_JOB_ID",os.environ["SLURM_JOB_ID"]))
             .replace("{task_id}",str(index)) for part in command]
print("Running:",command,flush=True)
os.execv(sys.executable,[sys.executable,*command])
