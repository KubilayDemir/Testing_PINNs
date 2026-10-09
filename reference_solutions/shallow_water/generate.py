"""Generate a SWE numerical reference with the semi-implicit Fortran solver."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import os

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent


def validate(dx, dt, viscosity, duration, save_interval):
    for name, value in (
        ("dx", dx),
        ("dt", dt),
        ("duration", duration),
        ("save_interval", save_interval),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(viscosity) or viscosity < 0:
        raise ValueError("viscosity must be nonnegative and finite")
    for value, divisor in (
        (2000000.0, dx),
        (duration, dt),
        (save_interval, dt),
        (duration, save_interval),
    ):
        if not np.isclose(
            value / divisor, round(value / divisor), rtol=0, atol=1e-7
        ):
            raise ValueError(
                "Domain, duration and output interval must be divisible "
                "by grid/timestep"
            )
    if viscosity * dt / dx**2 > 0.5:
        raise ValueError(
            "Explicit diffusion requires viscosity*dt/dx**2 <= 0.5"
        )


def compile_solver(directory, compiler="gfortran"):
    executable = Path(directory) / (
        "swe_solver.exe" if os.name == "nt" else "swe_solver"
    )
    local = HERE.parents[1] / ".tools/compiler/mingw64/bin/gfortran.exe"
    if (
        compiler == "gfortran"
        and not shutil.which(compiler)
        and local.is_file()
    ):
        compiler = str(local)
    compiler = shutil.which(compiler)
    if not compiler:
        raise RuntimeError(
            f"Fortran compiler unavailable: {compiler}; load its cluster module first"
        )
    flags = ["-O2", "-ffree-line-length-none"] + (
        ["-static"] if os.name == "nt" else []
    )
    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(compiler).parent) + os.pathsep + environment.get("PATH", "")
    )
    subprocess.run(
        [compiler, *flags, str(HERE / "solver.f90"), "-o", str(executable)],
        check=True,
        env=environment,
    )
    return executable.resolve()


def generate(
    output,
    dx=400.0,
    dt=1.0,
    viscosity=50000.0,
    duration=270000.0,
    save_interval=60.0,
    output_dx=10000.0,
    compiler="gfortran",
    executable=None,
    initial_center=0.0,
):
    validate(dx, dt, viscosity, duration, save_interval)
    if not np.isfinite(initial_center):
        raise ValueError("initial-center must be finite")
    ratio = output_dx / dx
    if (
        ratio < 1
        or not np.isclose(ratio, round(ratio))
        or int(round(ratio)) % 2 != 1
    ):
        raise ValueError(
            "output-dx/dx must be an odd positive integer for both staggered grids"
        )
    stride = int(round(ratio))
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cells, count = round(2000000.0 / dx), round(duration / save_interval) + 1
    with tempfile.TemporaryDirectory(
        prefix="solver-", dir=output.parent
    ) as directory:
        executable = executable or compile_solver(directory, compiler)
        subprocess.run(
            [
                str(executable),
                *map(
                    str,
                    (
                        dx,
                        dt,
                        viscosity,
                        duration,
                        save_interval,
                        initial_center,
                    ),
                ),
            ],
            cwd=directory,
            check=True,
        )
        u = np.memmap(
            Path(directory) / "u.bin",
            dtype=np.float32,
            mode="r",
            shape=(count, cells + 1),
        )
        zeta = np.memmap(
            Path(directory) / "zeta.bin",
            dtype=np.float32,
            mode="r",
            shape=(count, cells),
        )
        for start in range(0, count, 64):
            if (
                not np.isfinite(u[start : start + 64]).all()
                or not np.isfinite(zeta[start : start + 64]).all()
            ):
                raise RuntimeError(
                    "Nonfinite numerical solution; reduce the timestep"
                )
        ds = xr.Dataset(
            {
                "u_ref": (("x_u", "time"), u[:, ::stride].T),
                "zeta_ref": (
                    ("x_zeta", "time"),
                    zeta[:, stride // 2 :: stride].T,
                ),
            },
            coords={
                "time": np.arange(count) * save_interval,
                "x_u": -1000000.0 + np.arange(cells + 1)[::stride] * dx,
                "x_zeta": -1000000.0
                + (np.arange(cells)[stride // 2 :: stride] + 0.5) * dx,
            },
            attrs={
                "schema_version": "2",
                "data_role": "scientific_reference",
                "solver": "semi-implicit SWE; upstream advection; explicit viscosity",
                "dx_m": dx,
                "dt_s": dt,
                "viscosity_m2_s": viscosity,
                "save_interval_s": save_interval,
                "output_dx_m": output_dx,
                "initial_center_m": initial_center,
                "gravity_m_s2": 9.81,
                "mean_depth_m": 100.0,
                "compiler": compiler,
                "duration_s": duration,
            },
        )
        ds.u_ref.attrs["units"] = "m s-1"
        ds.zeta_ref.attrs["units"] = "m"
        ds.time.attrs["units"] = "s"
        for name in ("x_u", "x_zeta"):
            ds[name].attrs["units"] = "m"
        temporary = output.with_suffix(".nc.tmp")
        try:
            ds.to_netcdf(
                temporary,
                engine="netcdf4",
                encoding={
                    name: {"zlib": True, "complevel": 4}
                    for name in ds.data_vars
                },
            )
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
            ds.close()
            del ds, u, zeta
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case", choices=("viscous", "inviscid"), default="viscous"
    )
    p.add_argument("--output", type=Path)
    for name, value in (
        ("dx", 400.0),
        ("dt", 1.0),
        ("duration", 270000.0),
        ("save-interval", 60.0),
        ("output-dx", 10000.0),
        ("initial-center", 0.0),
    ):
        p.add_argument("--" + name, type=float, default=value)
    p.add_argument("--compiler", default="gfortran")
    a = p.parse_args()
    print(
        generate(
            a.output or HERE / "solutions" / (a.case + ".nc"),
            dx=a.dx,
            dt=a.dt,
            viscosity=50000.0 if a.case == "viscous" else 0.0,
            duration=a.duration,
            save_interval=a.save_interval,
            output_dx=a.output_dx,
            compiler=a.compiler,
            initial_center=a.initial_center,
        )
    )


if __name__ == "__main__":
    main()
