"""Small synthetic data for exercising serialization and training software."""

from pathlib import Path
import numpy as np
import xarray as xr

from .artifacts import write_netcdf


def make_fixture(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    t = np.arange(0.0, 600.0, 60.0)
    xu = np.linspace(-1e6, 1e6, 9)
    xh = (xu[:-1] + xu[1:]) / 2
    h = np.exp(-0.5 * (xh[:, None] / 1e5) ** 2) * np.cos(t[None, :] / 600)
    u = (
        0.01
        * np.sin(np.pi * (xu[:, None] + 1e6) / 2e6)
        * np.sin(t[None, :] / 600)
    )
    ds = xr.Dataset(
        {
            "u_ref": (("x_u", "time"), u, {"units": "m s-1"}),
            "zeta_ref": (("x_zeta", "time"), h, {"units": "m"}),
        },
        coords={
            "time": ("time", t, {"units": "s"}),
            "x_u": ("x_u", xu, {"units": "m"}),
            "x_zeta": ("x_zeta", xh, {"units": "m"}),
        },
        attrs={
            "title": "SYNTHETIC software fixture; not a numerical SWE solution",
            "data_role": "synthetic_software_test",
            "schema_version": "1",
        },
    )
    write_netcdf(ds, output)
    return output
