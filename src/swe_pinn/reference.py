"""Reference data on separate velocity nodes and elevation cell centres."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from .output_formats import canonical_dataset


@dataclass
class Reference:
    time: np.ndarray
    x_u: np.ndarray
    x_zeta: np.ndarray
    u: np.ndarray
    zeta: np.ndarray


def load_reference(path, time_bounds=None):
    """Read dimensional SI arrays; never infer coordinates from filenames."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Reference dataset not found: {path}. Import real reference arrays "
            "with 'swe-pinn import-reference'; see README.md."
        )
    with xr.open_dataset(path) as dataset:
        ds = canonical_dataset(dataset)
        if time_bounds is not None:
            if "time" not in ds:
                raise ValueError("Reference file requires time")
            start, stop = time_bounds
            ds = ds.isel(
                time=np.flatnonzero(
                    (ds.time.values >= start) & (ds.time.values < stop)
                )
            )
            if ds.sizes["time"] < 2:
                raise ValueError("Requested interval contains fewer than two reference times")
        # Materialize only the selected segment, rather than the entire run.
        ds = ds.load()
    validate_dataset(ds)
    u = ds.u_ref.transpose("x_u", "time").values
    zeta = ds.zeta_ref.transpose("x_zeta", "time").values
    return Reference(ds.time.values, ds.x_u.values, ds.x_zeta.values, u, zeta)


def validate_dataset(ds):
    """Validate the schema and physical coordinates before using or writing a file."""
    ds = canonical_dataset(ds)
    for name, units in (
            ("time", "s"),
            ("x_u", "m"),
            ("x_zeta", "m"),
            ("u_ref", "m s-1"),
            ("zeta_ref", "m"),
    ):
        if name not in ds:
            raise ValueError(f"Reference file requires {name}")
        if ds[name].attrs.get("units") != units:
            raise ValueError(f"{name} must have units={units!r}")
        if (
                ds[name].dtype.kind not in "fiu"
                or not np.isfinite(ds[name].values).all()
        ):
            raise ValueError(f"{name} must contain finite real numbers")
    for name in ("time", "x_u", "x_zeta"):
        values = ds[name].values
        if (
                values.ndim != 1
                or len(values) < 2
                or not np.all(np.diff(values) > 0)
        ):
            raise ValueError(
                f"{name} must be strictly increasing with at least two points"
            )
        if ds[name].dims != (name,):
            raise ValueError(
                f"{name} must be its own one-dimensional coordinate"
            )
    for name, expected in (
            ("u_ref", {"x_u", "time"}),
            ("zeta_ref", {"x_zeta", "time"}),
    ):
        if set(ds[name].dims) != expected:
            raise ValueError(f"{name} requires dimensions {sorted(expected)}")
    if ds.sizes["x_u"] != ds.sizes["x_zeta"] + 1:
        raise ValueError(
            (
                "Reference staggered grid requires one more velocity nod"
                "e than elevation cell"
            )
        )
    centres = 0.5 * (ds.x_u.values[:-1] + ds.x_u.values[1:])
    if not np.allclose(ds.x_zeta.values, centres, rtol=1e-9, atol=1e-6):
        raise ValueError(
            "Elevation coordinates must be cell centres between velocity nodes"
        )


def import_arrays(
        u_path, zeta_path, output, dt, dx, x_min, t_start=0.0, source=""
):
    """Convert legacy (space,time) arrays without changing their values."""
    u, zeta = np.load(u_path, allow_pickle=False, mmap_mode="r"), np.load(
        zeta_path, allow_pickle=False, mmap_mode="r"
    )
    if (
            u.ndim != 2
            or zeta.ndim != 2
            or u.shape[1] != zeta.shape[1]
            or u.shape[0] != zeta.shape[0] + 1
    ):
        raise ValueError(
            "Expected u[n_x+1,n_time] and zeta[n_x,n_time] on a staggered grid"
        )
    if not np.isfinite([dt, dx, x_min, t_start]).all() or dt <= 0 or dx <= 0:
        raise ValueError(
            "dt/dx must be positive and all grid parameters must be finite"
        )
    ds = xr.Dataset(
        {
            "u_ref": (
                ("x_u", "time"),
                u,
                {"units": "m s-1", "long_name": "reference zonal velocity"},
            ),
            "zeta_ref": (
                ("x_zeta", "time"),
                zeta,
                {"units": "m", "long_name": "reference sea level elevation"},
            ),
        },
        coords={
            "time": (
                "time",
                t_start + dt * np.arange(u.shape[1]),
                {"units": "s", "long_name": "elapsed physical time"},
            ),
            "x_u": ("x_u", x_min + dx * np.arange(u.shape[0]), {"units": "m"}),
            "x_zeta": (
                "x_zeta",
                x_min + dx * (np.arange(zeta.shape[0]) + 0.5),
                {"units": "m"},
            ),
        },
        attrs={
            "title": "Shallow-water numerical reference",
            "source": source,
            "schema_version": "2",
            "grid": "staggered: velocity at nodes, elevation at cell centres",
            "data_role": "scientific_reference",
            "history": "Imported losslessly from legacy NumPy arrays",
        },
    )
    validate_dataset(ds)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    from .output_storage import write_netcdf

    write_netcdf(ds, output)
    load_reference(output)
    return output
