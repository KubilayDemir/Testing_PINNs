"""Portable run metadata and NetCDF serialization."""

from datetime import datetime, timezone
from importlib.metadata import version, distributions
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import zipfile
import time

import numpy as np
import xarray as xr
import yaml


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_file(temp, path):
    """Retry brief Windows file locks without falling back to a partial write."""
    for attempt in range(8):
        try:
            Path(temp).replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * 2**attempt)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        temp.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        replace_file(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_config(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    write_json(directory / "config.json", config)


def provenance(root, reference, config):
    import torch

    root = reproduction_root(root)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            )
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", True
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: version(name)
            for name in (
                "numpy",
                "torch",
                "smt",
                "xarray",
                "netCDF4",
                "PyYAML",
                "matplotlib",
            )
        },
        "reference_path": str(reference),
        "reference_sha256": (
            sha256(reference) if reference is not None else None
        ),
        "seed": config["seed"],
        "device": config["device"],
        "dtype": "float32",
        "hardware": {
            "processor": platform.processor(),
            "cpu_threads": torch.get_num_threads(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": (
                torch.cuda.get_device_name(config["device"])
                if str(config["device"]).startswith("cuda")
                and torch.cuda.is_available()
                else None
            ),
        },
        ("checkpoint_selection"): (
            "minimum sum of dimensional reference MSE(u) and MSE(zeta),"
            " checked every epoch"
        ),
        "config_json": json.dumps(config, sort_keys=True),
    }


def reproduction_root(root):
    root = Path(root).resolve()
    checkout = Path(__file__).resolve().parents[2]
    return root if (root / "src/swe_pinn").is_dir() else checkout


def source_files(root):
    """Explicit source allowlist includes untracked code, excludes data and secrets."""
    root = reproduction_root(root)
    paths = []
    for directory in (
        "src/swe_pinn",
        "training_cases",
        "publication_cases",
        "software_tests",
        "scripts",
        "docs",
        "cluster_jobs",
        "reference_solutions/shallow_water",
    ):
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and path.suffix
            in (".py", ".yaml", ".json", ".md", ".sh", ".ps1", ".f90")
            and "__pycache__" not in path.parts
        )
    paths.extend(
        root / name
        for name in (
            "README.md",
            "LICENSE",
            "src/swe_pinn/optimizers/PCGrad_LICENSE",
            "pyproject.toml",
            "environment.yaml",
            "requirements-tested.txt",
        )
        if (root / name).is_file()
    )
    if not (root / "src/swe_pinn").is_dir():
        package = Path(__file__).resolve().parent
        paths.extend(
            path
            for path in package.rglob("*")
            if path.is_file()
            and path.suffix in (".py", ".yaml", ".json")
            and "__pycache__" not in path.parts
        )
    return sorted(paths)


def source_manifest(root):
    root = reproduction_root(root)
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in source_files(root)
    }


def save_reproduction_bundle(directory, root):
    directory, root = Path(directory), reproduction_root(root)
    manifest = source_manifest(root)
    with zipfile.ZipFile(
        directory / "source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in manifest:
            archive.write(root / path, path)
    write_json(directory / "source_manifest.json", manifest)
    packages = sorted(
        {
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in distributions()
            if dist.metadata.get("Name")
            and dist.metadata["Name"] != "swe-pinn-reflections"
        }
    )
    (directory / "environment-freeze.txt").write_text(
        "\n".join(packages) + "\n", encoding="utf-8"
    )
    return sha256(directory / "source.zip")


def write_netcdf(dataset, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temporary file, then replace only this generated dataset.
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        encoding = {
            name: {"zlib": True, "complevel": 4} for name in dataset.data_vars
        }
        dataset.to_netcdf(temp, engine="netcdf4", encoding=encoding)
        replace_file(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def history_dataset(model, metadata):
    coords = {
        "epoch": (
            "epoch",
            model.history_epochs,
            {"long_name": "completed training epochs"},
        )
    }
    ds = xr.Dataset(
        coords=coords,
        attrs={
            "title": "PINN training history",
            "schema_version": "2",
            "metadata_json": json.dumps(metadata),
            "segment_id": model.segment_id,
        },
    )
    for name in (
        "loss_total",
        "loss_pde",
        "loss_bc",
        "loss_ic",
        "loss_momentum",
        "loss_continuity",
        "loss_ic_u",
        "loss_ic_zeta",
        "loss_bc_left",
        "loss_bc_right",
        "mse_ref_u",
        "mse_ref_zeta",
    ):
        physical_units = {
            "mse_ref_u": "m2 s-2",
            "mse_ref_zeta": "m2",
            "loss_momentum": "m2 s-4",
            "loss_continuity": "m2 s-2",
            "loss_ic_u": "m2 s-2",
            "loss_ic_zeta": "m2",
            "loss_bc": "m2 s-2",
            "loss_bc_left": "m2 s-2",
            "loss_bc_right": "m2 s-2",
        }
        units = (
            physical_units[name]
            if name.startswith("mse_ref")
            else (
                "1"
                if model.nondimensionalize
                else physical_units.get(name, "mixed; see README.md")
            )
        )
        ds[name] = (
            "epoch",
            np.asarray(getattr(model, name + "_history")),
            {"units": units, "long_name": name.replace("_", " ")},
        )
    ds["epoch_seconds"] = (
        "training_epoch",
        model.time_per_epoch,
        {"units": "s"},
    )
    ds["iteration_seconds"] = (
        "iteration",
        model.time_per_iteration,
        {"units": "s"},
    )
    ds["improvement_epoch"] = ("improvement", model.improvement_steps)
    if model.adaptive_loss_weights:
        ds = ds.assign_coords(weight_step=("weight_step", model.weight_steps))
        for key, attr in (
            ("weight_pde", "pde_weight_history"),
            ("weight_ic", "ic_weight_history"),
            ("weight_bc", "bc_weight_history"),
        ):
            ds[key] = (
                "weight_step",
                np.asarray(getattr(model, attr)),
                {"units": "1"},
            )
    if model.snapshot_epochs:
        length = (
            model.horizontal_length_scale if model.nondimensionalize else 1
        )
        time = model.time_scale if model.nondimensionalize else 1
        height = model.vertical_length_scale if model.nondimensionalize else 1
        speed = length / time
        ds = ds.assign_coords(
            snapshot_epoch=("snapshot", model.snapshot_epochs),
            diagnostic_time=(
                "diagnostic_time",
                model.time_grid * time,
                {"units": "s"},
            ),
            diagnostic_x=(
                "diagnostic_x",
                model.x_grid * length,
                {"units": "m"},
            ),
        )
        for key, attr, factor, units in (
            ("u_pred", "velocity_history", speed, "m s-1"),
            ("zeta_pred", "elevation_history", height, "m"),
            (
                "momentum_residual",
                "momentum_residual_history",
                1,
                "1" if model.nondimensionalize else "m s-2",
            ),
            (
                "continuity_residual",
                "continuity_residual_history",
                1,
                "1" if model.nondimensionalize else "m s-1",
            ),
        ):
            values = getattr(model, attr)
            if values:
                ds[key] = (
                    ("snapshot", "diagnostic_x", "diagnostic_time"),
                    np.asarray(values) * factor,
                    {"units": units},
                )
    return ds
