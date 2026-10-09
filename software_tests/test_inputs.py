"""Validate configurations, reference grids, and invalid numerical inputs."""

from copy import deepcopy

import numpy as np
import pytest
import torch
import xarray as xr

from swe_pinn.config import load_config, validate
from swe_pinn.reference import import_arrays, load_reference
from swe_pinn.training import create_model


def test_dimensional_advection_switch(setup_run):
    config, reference, _ = setup_run
    config["scales"]["nondimensionalize"] = False
    config["physics"]["viscosity"] = 0
    model, _ = create_model(config, reference)
    model.forward = lambda t, x: torch.cat((t + x**2, 2 * t + x**3), dim=1)
    t = torch.tensor([[1.0], [2.0]], requires_grad=True)
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    model.pde_points = torch.hstack((t, x))
    model.momentum_advection = True
    model.compute_pde_residuals()
    with_advection = model.momentum_residual.detach().clone()
    model.momentum_advection = False
    model.compute_pde_residuals()
    torch.testing.assert_close(
        with_advection - model.momentum_residual, (t + x**2) * (2 * x)
    )


def test_zero_amplitude_uses_supplied_reference(setup_run):
    config, reference, _ = setup_run
    config["physics"]["amplitude"] = 0
    model, _ = create_model(config, reference)
    with torch.no_grad():
        model.compute_reference_error()
    expected = (
        (
            model.zeta_ref
            - model.network_output_zeta.reshape(
                model.zeta_solution_mesh_grid_shape
            )
        )
        ** 2
    ).mean()
    torch.testing.assert_close(model.mse_ref_zeta, expected, rtol=0, atol=0)


@pytest.mark.parametrize("key", ["time", "x_u", "x_zeta"])
def test_reference_rejects_infinite_coordinates(setup_run, key):
    _, reference, root = setup_run
    with xr.open_dataset(reference) as source:
        ds = source.load()
    values = ds[key].values.copy()
    values[-1] = np.inf
    ds = ds.assign_coords({key: (key, values, dict(ds[key].attrs))})
    invalid = root / "invalid.nc"
    ds.to_netcdf(invalid)
    with pytest.raises(ValueError, match="finite"):
        load_reference(invalid)


def test_reference_rejects_wrong_cell_centres(setup_run):
    _, reference, root = setup_run
    with xr.open_dataset(reference) as source:
        ds = source.load()
    ds = ds.assign_coords(x_zeta=("x_zeta", ds.x_zeta.values + 10, dict(ds.x_zeta.attrs)))
    invalid = root / "wrong_grid.nc"
    ds.to_netcdf(invalid)
    with pytest.raises(ValueError, match="cell centres"):
        load_reference(invalid)


def test_invalid_import_leaves_no_file(tmp_path):
    u, h = np.ones((5, 5)), np.ones((4, 5))
    h[0, 0] = np.nan
    np.save(tmp_path / "u.npy", u)
    np.save(tmp_path / "h.npy", h)
    output = tmp_path / "invalid.nc"
    with pytest.raises(ValueError, match="finite"):
        import_arrays(
            tmp_path / "u.npy", tmp_path / "h.npy", output, 60, 400, -800
        )
    assert not output.exists()


@pytest.mark.parametrize("content", ["false", "0", "[]"])
def test_config_root_must_be_mapping(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_reject_oversized_seed_and_changing_lbfgs_objective(setup_run):
    config, _, _ = setup_run
    invalid = deepcopy(config)
    invalid["seed"] = 2**32
    with pytest.raises(ValueError, match="seed"):
        validate(invalid)
    invalid = deepcopy(config)
    invalid["training"].update(
        optimizer="LBFGS", use_pcgrad=False, adaptive_loss_weights=True
    )
    with pytest.raises(ValueError, match="line search"):
        validate(invalid)


def test_nonfinite_initial_model_creates_no_best_checkpoint(setup_run):
    config, reference, root = setup_run
    checkpoints = root / "checkpoints"
    model, _ = create_model(config, reference, checkpoint_dir=checkpoints)
    with torch.no_grad():
        model.layers[0].weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="epoch 0"):
        model.fit()
    assert not checkpoints.exists()
