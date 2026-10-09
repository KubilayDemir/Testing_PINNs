"""Temporal ordering, bounded sampling and recovery of optional methods."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from swe_pinn.config import default_case_path, load_config
from swe_pinn.model import PINN
from swe_pinn.training_methods import TrainingStrategy

ROOT = Path(__file__).resolve().parents[1]


def settings(weighting="residual", sampling="causal_as"):
    values = load_config(ROOT / default_case_path("swe"))["training_methods"]
    values.update(
        causal_weighting=weighting,
        sampling=sampling,
        time_bins=4,
        probe_points=16,
        candidate_points=32,
        update_period=1,
    )
    if sampling == "causal_as":
        values["power_schedule"] = "temporal_alignment"
    return values


def test_disabled_strategy_preserves_rng_and_loss():
    before = torch.get_rng_state().clone()
    strategy = TrainingStrategy(
        settings("none", "lhs"), [[0, 1], [-1, 1]], 1, 8
    )
    assert not strategy.active
    assert strategy.state_dict() is None
    assert torch.equal(before, torch.get_rng_state())
    model = PINN(
        equation="heat", hidden_layers=[4], training_methods=strategy.settings
    )
    points = [torch.rand(8, 2).requires_grad_(True) for _ in range(4)]
    first = model.losses(points)
    second = model.losses(points, weighted=False)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "sampling", ["residual", "causal_as", "r3", "causal_r3"]
)
def test_bounded_sampling_and_recovery(sampling):
    weighting = "none" if sampling == "causal_r3" else "residual"
    config = settings(weighting, sampling)
    model = PINN(equation="heat", hidden_layers=[4])
    strategy = TrainingStrategy(config, [[2, 4], [-3, 3]], 9, 5)
    pool = strategy.sample(20)
    initial = strategy.sample(8)
    initial[:, 0] = 2
    strategy.prepare(model, initial, torch.ones(8, 1), [], [])
    assert len(strategy.condition_probes[0][0]) <= config["probe_points"]
    assert strategy.weights[1:].le(strategy.weights[:-1]).all()
    assert not strategy.weights.requires_grad
    assert len(torch.unique(strategy.bin_indices(strategy.probes))) == 4
    state = deepcopy(strategy.state_dict())
    result = strategy.adapt(model, pool)
    restored = TrainingStrategy(config, [[2, 4], [-3, 3]], 9, 5)
    restored.load_state_dict(state)
    torch.testing.assert_close(
        result, restored.adapt(model, pool), rtol=0, atol=0
    )
    assert result.shape == pool.shape
    assert result[:, 0].ge(2).all() and result[:, 0].le(4).all()
    assert result[:, 1].ge(-3).all() and result[:, 1].le(3).all()
    values = torch.ones(20, requires_grad=True)
    strategy.pde_loss(result, values).backward()
    assert torch.isfinite(values.grad).all()


def test_causal_weights_control_later_time_and_attention():
    strategy = TrainingStrategy(settings(), [[0, 1], [-1, 1]], 1, 8)
    strategy.weights = torch.tensor([1.0, 0.1, 0.01, 0.001])
    points = torch.tensor([[0.01, 0], [0.99, 0]])
    torch.testing.assert_close(
        strategy.point_weights(points), torch.tensor([1.0, 0.001])
    )
    attention = TrainingStrategy(
        settings("attention", "lhs"), [[0, 1], [-1, 1]], 1, 8
    )
    attention.ic_error = 0.1
    assert (
        attention.point_weights(points)[0] > attention.point_weights(points)[1]
    )
    attention.ic_error = 0
    assert attention.point_weights(points).eq(1).all()


def test_causal_sampling_prefers_early_residuals():
    strategy = TrainingStrategy(settings(), [[0, 1], [-1, 1]], 1, 8)
    strategy.weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
    strategy.scores = lambda model, points: torch.where(
        points[:, 0] < 0.25, 1.0, 20.0
    )
    strategy.settings["candidate_points"] = 1024
    result = strategy.adapt(None, strategy.sample(20))
    assert result[:16, 0].lt(0.25).all()
    ordinary = TrainingStrategy(
        settings("none", "residual"), [[0, 1], [-1, 1]], 1, 8
    )
    ordinary.weights = strategy.weights
    assert ordinary.point_weights(result).eq(1).all()


def test_condition_statistics_have_fixed_bounded_cost():
    strategy = TrainingStrategy(settings(), [[0, 1], [-1, 1]], 1, 8)
    model = PINN(equation="heat", hidden_layers=[4])
    initial = strategy.sample(1000)
    initial[:, 0] = 0
    boundaries = [strategy.sample(1000), strategy.sample(1000)]
    targets = torch.ones(1000, 1)
    zero = [torch.zeros(1000, 1), torch.zeros(1000, 1)]
    calls = []
    forward = model.forward

    def counted(t, x):
        calls.append(len(t))
        return forward(t, x)

    model.forward = counted
    strategy.prepare(model, initial, targets, boundaries, zero)
    assert sum(calls) <= 4 * strategy.settings["probe_points"]
    error = strategy.ic_error
    strategy.prepare(model, initial, targets * 3, boundaries, zero)
    assert strategy.ic_error == error


@pytest.mark.parametrize(
    "changes",
    [
        {"sampling": "causal_as", "causal_weighting": "none"},
        {"sampling": "causal_r3", "causal_weighting": "residual"},
        {"probe_points": 1},
        {"uniform_fraction": 0},
        {"alignment_threshold": 1},
    ],
)
def test_invalid_method_combinations(changes):
    from swe_pinn.config import validate

    config = load_config(ROOT / default_case_path("swe"))
    config["training_methods"].update(changes)
    with pytest.raises(ValueError):
        validate(config)


def test_attention_requires_soft_initial_conditions():
    from swe_pinn.config import validate

    config = load_config(ROOT / default_case_path("swe"))
    config["training_methods"] = settings("attention", "lhs")
    config["network"]["enforce_ic"] = True
    with pytest.raises(ValueError, match="soft IC"):
        validate(config)


@pytest.mark.parametrize("kind", ["heat", "burgers"])
@pytest.mark.parametrize(
    "weighting,sampling,optimizer",
    [
        ("residual", "causal_as", "Adam"),
        ("none", "causal_r3", "Adam"),
        ("attention", "residual", "Adam"),
        ("residual", "lhs", "LBFGS"),
    ],
)
def test_benchmark_methods_and_interrupted_resume(
    tmp_path, monkeypatch, kind, weighting, sampling, optimizer
):
    import swe_pinn.heat_and_burgers.training as runner

    config = runner.load_config(ROOT / default_case_path(kind))
    config["output_root"] = str(tmp_path)
    config["network"]["hidden_layers"] = [4, 4]
    config["sampling"].update(n_pde=12, n_bc=8, n_ic=8, resample_period=2)
    config["training"].update(
        epochs=3,
        resume_period=1,
        optimizer=optimizer,
        report_period=1,
        pde_batch_size=6,
        bc_batch_size=4,
        ic_batch_size=4,
        iterations_per_epoch=1,
    )
    config["output"].update(plots=False, save_predictions=False)
    config["training_methods"] = settings(weighting, sampling)
    complete = runner.train(config, ROOT, "complete")
    original = runner.write_json

    def interrupt(path, values):
        if path.name == "status.json" and values.get("epoch") == 2:
            raise RuntimeError("test interruption")
        return original(path, values)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "write_json", interrupt)
        with pytest.raises(RuntimeError, match="test interruption"):
            runner.train(config, ROOT, "interrupted")
    interrupted = tmp_path / config["experiment"] / "interrupted"
    runner.train(config, ROOT, resume_run=interrupted)
    a = torch.load(complete / "checkpoints/final.pt", weights_only=True)
    b = torch.load(interrupted / "checkpoints/final.pt", weights_only=True)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert json.loads((complete / "training_strategy.json").read_text()) == (
        json.loads((interrupted / "training_strategy.json").read_text())
    )


@pytest.mark.parametrize(
    "weighting,sampling,segments",
    [
        ("residual", "causal_as", 2),
        ("none", "causal_r3", 1),
        ("attention", "residual", 1),
    ],
)
def test_swe_methods(setup_run, monkeypatch, weighting, sampling, segments):
    from swe_pinn.training import train
    from swe_pinn import training_checkpoints as recovery

    config, _, root = setup_run
    config["training_methods"] = settings(weighting, sampling)
    config["output"].update(batch_diagnostics=True, stream_history=True)
    config["training"].update(resume_period=1, epochs=2)
    config["domain"]["n_time_segments"] = segments
    run = train(config, root, "methods")
    original = recovery.save_resume

    def interrupt(model, epoch):
        original(model, epoch)
        if epoch == 1:
            raise KeyboardInterrupt("test interruption")

    with monkeypatch.context() as patch:
        patch.setattr(recovery, "save_resume", interrupt)
        with pytest.raises(KeyboardInterrupt):
            train(config, root, "interrupted-methods")
    interrupted = (
        root / "training_results/quick_software_check/interrupted-methods"
    )
    train(config, root, resume_run=interrupted)
    for index in range(segments):
        segment = run / f"segments/{index:03d}"
        state = torch.load(
            segment / "checkpoints/resume.pt", weights_only=False
        )
        assert state["training_strategy"]["probes"].shape == (16, 2)
        assert (segment / "training_strategy.json").is_file()
        repeated = interrupted / f"segments/{index:03d}"
        first = torch.load(segment / "checkpoints/final.pt", weights_only=True)
        second = torch.load(
            repeated / "checkpoints/final.pt", weights_only=True
        )
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
