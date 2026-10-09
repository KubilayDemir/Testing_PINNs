"""Scheduler manifests select the requested experiment and resources."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

path=Path(__file__).resolve().parents[1]/'cluster_jobs/submit.py'
spec=importlib.util.spec_from_file_location('submit',path)
submit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)

def settings(**changes):
    result=dict(cluster='levante',partition=None,task='train',account='project',equation='swe',
        config='publication_cases/transition_functions/TF3_boundary_initial/config.yaml',seeds=[0,1],resume=None,options=None,
        cpus=4,time='08:00:00',memory='32G',constraint=None,gpu_resource='gpu:1',max_parallel=2,
        dependency=None,env_file=None,case='viscous',study='dx')
    return SimpleNamespace(**(result|changes))

def test_seed_array_manifest(tmp_path,monkeypatch):
    monkeypatch.setattr(submit,'ROOT',tmp_path)
    command,path=submit.build(settings())
    assert '--gpus=1' in command and '--array=0-1%2' in command
    commands=json.loads(path.read_text())['commands']
    assert len(commands)==2
    assert commands[1][commands[1].index('--seed')+1]=='1'
    assert 'publication_cases/transition_functions/TF3_boundary_initial/config.yaml' in commands[0]

def test_strand_requires_partition(tmp_path,monkeypatch):
    monkeypatch.setattr(submit,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='partition'): submit.build(settings(cluster='strand'))
    command,_=submit.build(settings(cluster='strand',partition='local_gpu'))
    assert '--gres=gpu:1' in command

def test_reference_uses_cpu(tmp_path,monkeypatch):
    monkeypatch.setattr(submit,'ROOT',tmp_path)
    command,path=submit.build(settings(task='reference',seeds=[0]))
    assert '--partition=shared' in command
    assert '--gpus=1' not in command
    assert json.loads(path.read_text())['commands'][0][0].endswith('generate.py')
