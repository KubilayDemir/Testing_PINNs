import numpy as np
import pytest
import xarray as xr

from swe_pinn.reference import import_arrays, load_reference


def test_old_netcdf_elevation_names_remain_readable(setup_run):
    _, reference, root = setup_run
    with xr.open_dataset(reference) as ds:
        old = ds.load().rename({"x_zeta": "x_h", "zeta_ref": "h_ref"})
    path = root / "old_reference.nc"
    old.to_netcdf(path)
    loaded = load_reference(path)
    np.testing.assert_array_equal(loaded.zeta, old.h_ref)
    np.testing.assert_array_equal(loaded.x_zeta, old.x_h)


def test_legacy_import_preserves_values_and_staggering(tmp_path):
    u = np.arange(25.0).reshape(5, 5)
    h = np.arange(20.0).reshape(4, 5)
    np.save(tmp_path / "u.npy", u)
    np.save(tmp_path / "h.npy", h)
    output = tmp_path / "reference.nc"
    import_arrays(
        tmp_path / "u.npy",
        tmp_path / "h.npy",
        output,
        60,
        400,
        -800,
        source="fixture",
    )
    loaded = load_reference(output)
    np.testing.assert_array_equal(loaded.u, u)
    np.testing.assert_array_equal(loaded.zeta, h)
    np.testing.assert_array_equal(loaded.x_zeta, [-600, -200, 200, 600])
    sliced = load_reference(output, (60, 240))
    np.testing.assert_array_equal(sliced.time, [60, 120, 180])
    with pytest.raises(FileExistsError):
        import_arrays(
            tmp_path / "u.npy", tmp_path / "h.npy", output, 60, 400, -800
        )


def test_reject_missing_units(setup_run):
    _, reference, root = setup_run
    with xr.open_dataset(reference) as source:
        ds = source.load()
    ds.u_ref.attrs.pop("units")
    bad = root / "bad.nc"
    ds.to_netcdf(bad)
    with pytest.raises(ValueError, match="units"):
        load_reference(bad)
