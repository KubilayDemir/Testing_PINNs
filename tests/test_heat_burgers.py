"""Heat diffusion and viscous Burgers residuals and training workflows."""

from pathlib import Path

import numpy as np
import pytest
import torch

from swe_pinn.model import PINN
from swe_pinn.benchmarks.training import load_config, train

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["heat", "burgers"])
def test_checkpointed_benchmark_derivatives(kind):
    ordinary = PINN(equation=kind, hidden_layers=[8, 8])
    checked = PINN(equation=kind, hidden_layers=[8, 8], activation_checkpointing=True)
    checked.load_state_dict(ordinary.state_dict())
    points = torch.tensor([[0.1, -0.4], [0.3, 0.6]], requires_grad=True)
    expected = ordinary.pde_residual(points)
    actual = checked.pde_residual(points)
    torch.testing.assert_close(actual, expected)
    first = torch.autograd.grad(expected.square().mean(), tuple(ordinary.parameters()))
    second = torch.autograd.grad(actual.square().mean(), tuple(checked.parameters()))
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("kind", ["heat", "burgers"])
def test_benchmark_residual_for_polynomial_field(kind):
    model = PINN(equation=kind, hidden_layers=[8, 8])
    model.forward = lambda t, x: t + x*x
    points = torch.tensor([[0.1, -0.4], [0.3, 0.6]], requires_grad=True)
    t,x=points[:,:1],points[:,1:2]
    expected=torch.full_like(t,1-2*model.diffusivity) if kind=="heat" else 1+(t+x*x)*2*x-2*.01/model.pi
    torch.testing.assert_close(model.pde_residual(points),expected)


@pytest.mark.parametrize("kind", ["heat", "burgers"])
@pytest.mark.parametrize("optimizer,pcgrad,adaptive", [("LBFGS", False, False), ("Adam", True, True)])
def test_small_benchmark_workflow(tmp_path, kind, optimizer, pcgrad, adaptive):
    c = load_config(ROOT / f"cases/{kind}/config.yaml")
    c["device"] = "cpu"
    c["reference"] = str(ROOT / "numerical_reference/burgers/shock.mat")
    c["output_root"] = str(tmp_path)
    c["network"]["hidden_layers"] = [8, 8]
    c["sampling"].update(n_pde=8, n_bc=4, n_ic=4)
    c["training"].update(epochs=2, optimizer=optimizer, use_pcgrad=pcgrad, adaptive_loss_weights=adaptive,
                         minibatch=True, pde_batch_size=4, bc_batch_size=2, ic_batch_size=2,
                         iterations_per_epoch=1, resume_period=1)
    c["output"].update(stream_history=True, save_residuals=True, batch_diagnostics=True)
    first = train(c, ROOT, "benchmark")
    before = torch.load(first / "checkpoints/final.pt", weights_only=True)
    train(c, ROOT, resume_run=first)
    after = torch.load(first / "checkpoints/final.pt", weights_only=True)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert (first / "figures/solution.png").exists()


def test_schedule_resume_and_sparse_reporting(tmp_path, monkeypatch):
    import copy
    import json
    import swe_pinn.benchmarks.training as runner

    c = load_config(ROOT / "cases/heat/config.yaml")
    c["output_root"] = str(tmp_path)
    c["network"]["hidden_layers"] = [8, 8]
    c["sampling"].update(n_pde=12, n_bc=4, n_ic=4, resample_period=2)
    c["training"].update(epochs=6, resume_period=1, lr_schedule="step", lr_step_size=2,
                         report_period=3, minibatch=True, pde_batch_size=6, bc_batch_size=2,
                         ic_batch_size=2, iterations_per_epoch=1)
    c["output"].update(plots=False, stream_history=True, snapshot_period=2)
    full = train(c, ROOT, "full")
    original = runner.write_json

    def interrupt(path, data):
        if path.name == "status.json" and data.get("epoch") == 4:
            raise RuntimeError("interrupted")
        return original(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "write_json", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            train(c, ROOT, "interrupted")
    recovered = train(c, ROOT, resume_run=tmp_path / c["experiment"] / "interrupted")
    dense_config = copy.deepcopy(c)
    dense_config["training"]["report_period"] = 1
    dense = train(dense_config, ROOT, "dense")
    for checkpoint in ("final.pt", "best.pt"):
        expected = torch.load(full / "checkpoints" / checkpoint, weights_only=True)
        for run in (recovered, dense):
            actual = torch.load(run / "checkpoints" / checkpoint, weights_only=True)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    state = torch.load(recovered / "checkpoints/resume.pt", weights_only=False)
    assert state["optimizer"]["param_groups"][0]["lr"] == c["training"]["learning_rate"] / 8
    assert [row["epoch"] for row in state["history"]] == [0, 3, 6]
    assert json.loads((full / "metrics.json").read_text()) == json.loads((recovered / "metrics.json").read_text())
