"""Infrastructure options must preserve the training result and recover it."""

from copy import deepcopy
import json
import zipfile

import numpy as np
import pytest
import torch
import xarray as xr

from swe_pinn.training import train, evaluate_checkpoint
from swe_pinn.plotting import plot_study, snapshot_epochs, load_snapshot


@pytest.mark.parametrize(
    "options", [{}, {"split_networks": True}, {"mixed_activations": True}]
)
def test_checkpointed_swe_derivatives(setup_run, options):
    from swe_pinn.config import model_kwargs
    from swe_pinn.model import PINN

    config, reference, _ = setup_run
    config["network"].update(options)
    ordinary = PINN(**model_kwargs(config, reference))
    config["network"]["activation_checkpointing"] = True
    checked = PINN(**model_kwargs(config, reference))
    checked.load_state_dict(ordinary.state_dict())
    points = torch.tensor([[0.001, -0.4], [0.003, 0.6]], requires_grad=True)
    first = ordinary.pde_residual(points)
    second = checked.pde_residual(points)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
    expected = torch.autograd.grad(
        sum(value.square().mean() for value in first),
        tuple(ordinary.parameters()),
    )
    actual = torch.autograd.grad(
        sum(value.square().mean() for value in second),
        tuple(checked.parameters()),
    )
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b)


def test_atomic_write_survives_transient_lock(tmp_path, monkeypatch):
    from pathlib import Path
    from swe_pinn.output_storage import write_json

    target = tmp_path / "status.json"
    target.write_text('{"epoch": 1}')
    original = Path.replace
    calls = []

    def locked_once(self, destination):
        calls.append(self)
        if len(calls) == 1:
            assert json.loads(target.read_text()) == {"epoch": 1}
            raise PermissionError("temporary Windows lock")
        return original(self, destination)

    monkeypatch.setattr(Path, "replace", locked_once)
    write_json(target, {"epoch": 2})
    assert json.loads(target.read_text()) == {"epoch": 2}
    assert not list(tmp_path.glob("*.tmp"))


def assert_same_state(first, second):
    a = torch.load(first, weights_only=True)
    b = torch.load(second, weights_only=True)
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_streaming_and_batched_diagnostics_preserve_training(setup_run):
    config, _, root = setup_run
    baseline = train(config, root, "baseline")
    optional = deepcopy(config)
    optional["output"].update(
        stream_history=True,
        batch_diagnostics=True,
        evaluation_batch_size=7,
        keep_improving_checkpoints=False,
    )
    optional["training"]["reference_on_cpu"] = True
    streamed = train(optional, root, "streamed")
    assert_same_state(
        baseline / "segments/000/checkpoints/final.pt",
        streamed / "segments/000/checkpoints/final.pt",
    )
    segment = streamed / "segments/000"
    assert snapshot_epochs(segment) == [0, 1, 2]
    assert (segment / "history.jsonl").exists()
    assert (segment / "timing.jsonl").exists()
    assert not list((segment / "checkpoints").glob("best_epoch_*.pt"))
    with xr.open_dataset(baseline / "segments/000/history.nc") as ds:
        for epoch in (0, 1, 2):
            snapshot = load_snapshot(segment, epoch)
            np.testing.assert_allclose(
                snapshot.momentum_residual.values[0],
                ds.momentum_residual.values[epoch],
                rtol=1e-5,
                atol=1e-6,
            )
    with zipfile.ZipFile(streamed / "source.zip") as bundle:
        assert "src/swe_pinn/physics.py" in bundle.namelist()
    assert plot_study([segment], epochs=[0, 2]).is_dir()


@pytest.mark.parametrize(
    "pcgrad,adaptive,resample", [(False, False, 10), (True, True, 1)]
)
def test_interrupted_run_resumes_exactly(
    setup_run, monkeypatch, pcgrad, adaptive, resample
):
    from swe_pinn import training_checkpoints as runtime

    config, _, root = setup_run
    config["training"].update(
        resume_period=1, use_pcgrad=pcgrad, adaptive_loss_weights=adaptive
    )
    config["sampling"]["resample_period"] = resample
    config["output"]["stream_history"] = True
    uninterrupted = train(config, root, "uninterrupted")
    original = runtime.save_resume

    def interrupt(model, epoch):
        original(model, epoch)
        if epoch == 1:
            raise KeyboardInterrupt("simulated interruption")

    monkeypatch.setattr(runtime, "save_resume", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train(config, root, "interrupted")
    monkeypatch.setattr(runtime, "save_resume", original)
    interrupted = root / "training_results/quick_software_check/interrupted"
    train(config, root, resume_run=interrupted)
    assert_same_state(
        uninterrupted / "segments/000/checkpoints/final.pt",
        interrupted / "segments/000/checkpoints/final.pt",
    )
    assert_same_state(
        uninterrupted / "segments/000/checkpoints/best.pt",
        interrupted / "segments/000/checkpoints/best.pt",
    )
    epochs = [
        json.loads(line)["epoch"]
        for line in (interrupted / "segments/000/history.jsonl")
        .read_text()
        .splitlines()
    ]
    assert epochs == [0, 1, 2]


def test_soft_initial_handoff_with_time_decomposition(setup_run):
    config, _, root = setup_run
    config["domain"]["n_time_segments"] = 2
    config["network"].update(enforce_bc=True, enforce_ic=True)
    run = train(config, root, "hard-handoff")
    checkpoint = run / "segments/001/checkpoints/best.pt"
    replay = evaluate_checkpoint(
        config,
        root,
        checkpoint,
        root / "replay",
        1,
        run / "segments/000/handoff.nc",
    )
    with (
        xr.open_dataset(run / "segments/001/evaluation.nc") as original,
        xr.open_dataset(replay / "evaluation.nc") as repeated,
    ):
        np.testing.assert_array_equal(original.zeta_pred, repeated.zeta_pred)
        assert not np.allclose(
            original.ic_zeta_pred, original.ic_zeta_target, atol=1e-6
        )
        np.testing.assert_allclose(original.u_bc_left, 0, atol=1e-6)


@pytest.mark.parametrize("siren,adaptive", [(False, True), (True, False)])
def test_split_network_optional_methods(setup_run, siren, adaptive):
    config, _, root = setup_run
    config["network"].update(split_networks=True, siren_init=siren)
    config["training"].update(adaptive_loss_weights=adaptive)
    config["output"].update(save_predictions=False, save_residuals=False)
    run = train(config, root, "split")
    assert (
        json.loads((run / "status.json").read_text())["status"] == "complete"
    )
