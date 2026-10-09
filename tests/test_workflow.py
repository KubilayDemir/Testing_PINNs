from copy import deepcopy
import json

import numpy as np
import pytest
import torch
import xarray as xr

from swe_pinn.config import validate
from swe_pinn.plotting import animate_segment, animate_history, plot_segment
from swe_pinn.training import train, evaluate_checkpoint


def test_train_reload_netcdf_plots_and_no_overwrite(setup_run):
    config, _, root = setup_run
    run = train(config, root, "smoke")
    segment = run / "segments/000"
    checkpoints = segment / "checkpoints"
    for name in (
        "best.pt",
        "final.pt",
        "optimizer_final.pt",
    ):
        assert (checkpoints / name).is_file()
    best = torch.load(checkpoints / "best.pt", weights_only=True)
    assert list(checkpoints.glob("best_epoch_*.pt"))
    assert json.loads((run / "config.json").read_text()) == config
    with xr.open_dataset(segment / "evaluation.nc") as ds:
        assert ds.u_pred.dims == ("x_u", "time")
        assert ds.zeta_pred.dims == ("x_zeta", "time")
        assert ds.u_pred.attrs["units"] == "m s-1"
        assert ds.zeta_pred.attrs["units"] == "m"
        metadata = json.loads(ds.attrs["metadata_json"])
        assert metadata["checkpoint_sha256"]
        assert metadata["dependencies"]["torch"]
        assert ds.attrs["data_role"] == "synthetic_software_test"
        assert np.allclose(ds.u_error, ds.u_pred - ds.u_ref)
        original_u = ds.u_pred.values.copy()
        original_h = ds.zeta_pred.values.copy()
    replay = evaluate_checkpoint(
        config, root, checkpoints / "best.pt", root / "replay"
    )
    with xr.open_dataset(replay / "evaluation.nc") as ds:
        np.testing.assert_array_equal(ds.u_pred, original_u)
        np.testing.assert_array_equal(ds.zeta_pred, original_h)
    with xr.open_dataset(segment / "history.nc") as ds:
        np.testing.assert_array_equal(ds.epoch, [0, 1, 2])
        np.testing.assert_array_equal(ds.snapshot_epoch, [0, 1, 2])
        assert ds.u_pred.shape[0] == 3
    plot_segment(segment)
    assert len(list((segment / "figures").glob("*.png"))) == 6
    assert animate_segment(segment).is_file()
    assert animate_history(segment).is_file()
    assert animate_history(segment, kind="residuals").is_file()
    with pytest.raises(FileExistsError):
        train(config, root, "smoke")


@pytest.mark.parametrize(
    "options",
    [
        {"use_pcgrad": False},
        {
            "use_pcgrad": False,
            "optimizer": "LBFGS",
            "minibatch": False,
            "line_search": "strong_wolfe",
        },
        {
            "use_pcgrad": False,
            "optimizer": "FullBatchLBFGS",
            "minibatch": False,
            "line_search": "Armijo",
        },
        {
            "use_pcgrad": False,
            "optimizer": "FullBatchLBFGS",
            "minibatch": False,
            "line_search": "Wolfe",
        },
        {
            "use_pcgrad": False,
            "optimizer": "FullBatchLBFGS",
            "minibatch": False,
            "line_search": None,
        },
        {"adaptive_loss_weights": True},
    ],
)
def test_optimizer_paths(setup_run, options):
    config, _, root = setup_run
    config["training"].update(options)
    config["training"]["epochs"] = 2
    config["output"].update(save_predictions=False, save_residuals=False)
    run = train(config, root, "optimizer")
    assert (
        json.loads((run / "status.json").read_text())["status"] == "complete"
    )
    checkpoints = run / "segments/000/checkpoints"
    initial = torch.load(
        checkpoints / "best_epoch_000000.pt", weights_only=True
    )
    final = torch.load(checkpoints / "final.pt", weights_only=True)
    assert any(not torch.equal(initial[k], final[k]) for k in initial)


@pytest.mark.parametrize("mode", ["reference", "transitions", "resampling"])
def test_objectives_and_reporting(setup_run, mode):
    config, _, root = setup_run
    if mode == "reference":
        config["loss"].update(
            use_reference=True, use_pde=False, use_bc=False, use_ic=False
        )
        config["training"]["use_pcgrad"] = False
    elif mode == "transitions":
        config["network"].update(enforce_bc=True, enforce_ic=True)
    else:
        config["sampling"]["resample_period"] = 1
        config["training"]["iterations_per_epoch"] = 5
        config["training"]["report_period"] = 2
    run = train(config, root, mode)
    if mode == "transitions":
        with xr.open_dataset(run / "segments/000/evaluation.nc") as ds:
            np.testing.assert_allclose(ds.u_bc_left, 0, atol=1e-6)
            np.testing.assert_allclose(ds.ic_zeta_pred, ds.ic_zeta_target, atol=1e-6)


def test_multisegment_and_deterministic_sampling(setup_run):
    config, _, root = setup_run
    config["domain"]["n_time_segments"] = 2
    first = train(config, root, "first")
    second = train(config, root, "second")
    for segment_id in ("000", "001"):
        with (
            xr.open_dataset(
                first / f"segments/{segment_id}/evaluation.nc"
            ) as a,
            xr.open_dataset(
                second / f"segments/{segment_id}/evaluation.nc"
            ) as b,
        ):
            np.testing.assert_array_equal(a.u_pred, b.u_pred)
    with (
        xr.open_dataset(first / "segments/000/evaluation.nc") as a,
        xr.open_dataset(first / "segments/001/evaluation.nc") as b,
    ):
        assert a.time.values[-1] < b.time.values[0]
        assert b.time.values[0] == 300
    replay = evaluate_checkpoint(
        config,
        root,
        first / "segments/001/checkpoints/best.pt",
        root / "segment_replay",
        1,
        first / "segments/000/handoff.nc",
    )
    with (
        xr.open_dataset(replay / "evaluation.nc") as a,
        xr.open_dataset(first / "segments/001/evaluation.nc") as b,
    ):
        np.testing.assert_array_equal(a.zeta_pred, b.zeta_pred)


def test_dimensional_and_split_networks(setup_run):
    config, _, root = setup_run
    config["network"]["split_networks"] = True
    config["scales"]["nondimensionalize"] = False
    config["training"]["learning_rate"] = 1e-6
    config["training"]["epochs"] = 1
    run = train(config, root, "dimensional")
    with xr.open_dataset(run / "segments/000/evaluation.nc") as ds:
        assert ds.momentum_residual.attrs["units"] == "m s-2"
        assert ds.time.values[-1] == 540


def test_reject_unsupported_settings(setup_run):
    config, _, _ = setup_run
    c = deepcopy(config)
    c["training"]["optimizer"] = "LBFGS"
    with pytest.raises(ValueError, match="PCGrad"):
        validate(c)
