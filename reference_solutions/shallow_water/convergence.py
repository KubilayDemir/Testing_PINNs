"""Repeat grid/timestep convergence studies using the reference solver."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr
from generate import HERE, compile_solver, generate, validate


def cases(study, viscosity, duration, save_interval):
    if study == "dx":
        candidates = [
            (dx, 0.05 if viscosity else 1.0)
            for dx in (10000.0, 2000.0, 400.0, 80.0)
        ]
    else:
        candidates = [(10000.0, dt) for dt in (60.0, 30.0, 10.0, 5.0, 1.0)]
        candidates += [(2000.0, dt) for dt in (60.0, 30.0, 10.0, 5.0, 1.0)]
        candidates += [
            (400.0, dt)
            for dt in (
                (10.0, 5.0, 2.0, 1.0, 0.5)
                if viscosity
                else (20.0, 10.0, 5.0, 2.0, 1.0)
            )
        ]
        candidates += (
            [(80.0, dt) for dt in (0.05, 0.02, 0.01)]
            if viscosity
            else [(80.0, dt) for dt in (5.0, 2.0, 1.0)]
        )
    result = []
    for dx, dt in candidates:
        # The viscous 80 m case needs dt <= 0.064 s for explicit diffusion.
        if viscosity and viscosity * dt / dx**2 > 0.5:
            continue
        validate(dx, dt, viscosity, duration, save_interval)
        if (dx, dt) not in result:
            result.append((dx, dt))
    return result


def run_case(settings):
    output, dx, dt, viscosity, duration, save_interval, executable = settings
    if not output.exists():
        generate(
            output,
            dx,
            dt,
            viscosity,
            duration,
            save_interval,
            dx,
            executable=executable,
        )
    else:
        with xr.open_dataset(output) as ds:
            for key, value in (
                ("dx_m", dx),
                ("dt_s", dt),
                ("viscosity_m2_s", viscosity),
                ("duration_s", duration),
                ("save_interval_s", save_interval),
            ):
                if ds.attrs.get(key) != value:
                    raise ValueError(f"Existing case differs: {output}: {key}")
    return output


def compare(candidate, reference):
    result = {}
    with xr.open_dataset(candidate) as a, xr.open_dataset(reference) as b:
        for field, axis in (("u", "x_u"), ("zeta", "x_zeta")):
            # Compare on candidate coordinates within the finer grid's support.
            points = a[axis].values
            mask = (points >= float(b[axis].min())) & (
                points <= float(b[axis].max())
            )
            values = a[field + "_ref"].values[mask]
            target = (
                b[field + "_ref"]
                .interp({axis: points[mask], "time": a.time})
                .values
            )
            error = values - target
            result[field + "_relative_l2"] = float(
                np.linalg.norm(error) / np.linalg.norm(target)
            )
            result[field + "_max_error"] = float(np.abs(error).max())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case", choices=("viscous", "inviscid"), default="viscous"
    )
    p.add_argument("--study", choices=("dx", "dt"), default="dx")
    p.add_argument("--duration", type=float, default=270000.0)
    p.add_argument("--save-interval", type=float, default=60.0)
    p.add_argument(
        "--workers", type=int, default=1, help="independent solver processes"
    )
    p.add_argument("--compiler", default="gfortran")
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--plan",
        action="store_true",
        help="list cases without compiling or running",
    )
    a = p.parse_args()
    if a.workers < 1:
        p.error("workers must be positive")
    viscosity = 50000.0 if a.case == "viscous" else 0.0
    grid = cases(a.study, viscosity, a.duration, a.save_interval)
    if a.plan:
        for dx, dt in grid:
            print(f"dx={dx:g} m, dt={dt:g} s")
        return
    directory = (
        a.output or HERE / "convergence_results" / a.case / a.study
    ).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        executable = compile_solver(temp, a.compiler)
        settings = [
            (
                directory / f"dx_{dx:g}_dt_{dt:g}.nc",
                dx,
                dt,
                viscosity,
                a.duration,
                a.save_interval,
                executable,
            )
            for dx, dt in grid
        ]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            outputs = list(pool.map(run_case, settings))
    rows = []
    for output, (dx, dt) in zip(outputs, grid):
        group = [
            (path, h, k)
            for path, (h, k) in zip(outputs, grid)
            if a.study == "dx" or h == dx
        ]
        reference = min(
            group, key=lambda item: item[1] if a.study == "dx" else item[2]
        )[0]
        row = {
            "dx_m": dx,
            "dt_s": dt,
            "reference": reference.name,
            "file": output.name,
        }
        row.update(compare(output, reference))
        rows.append(row)
    with (directory / "errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib.pyplot as plt

    for field in ("u", "zeta"):
        fig, axis = plt.subplots()
        groups = (
            [None] if a.study == "dx" else sorted({r["dx_m"] for r in rows})
        )
        for group in groups:
            selected = [
                r
                for r in rows
                if (group is None or r["dx_m"] == group)
                and r[field + "_relative_l2"] > 0
            ]
            key = "dx_m" if a.study == "dx" else "dt_s"
            selected.sort(key=lambda r: r[key])
            axis.loglog(
                [r[key] for r in selected],
                [r[field + "_relative_l2"] for r in selected],
                "o-",
                label=(
                    "grid refinement" if group is None else f"dx={group:g} m"
                ),
            )
        axis.set(
            xlabel="dx [m]" if a.study == "dx" else "dt [s]",
            ylabel=f"{field} relative L2 error",
        )
        axis.legend()
        fig.tight_layout()
        fig.savefig(directory / f"{field}_convergence.png")
        plt.close(fig)
    print(directory)


if __name__ == "__main__":
    main()
