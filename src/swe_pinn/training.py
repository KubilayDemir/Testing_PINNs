"""Run orchestration with explicit paths, checkpoints and configuration backups."""

from datetime import datetime, timezone
from pathlib import Path
import json
import random
import time
import uuid

import numpy as np
import torch
import xarray as xr

from .output_storage import (
    save_reproduction_bundle, source_manifest,
    history_dataset,
    provenance,
    save_config,
    sha256,
    write_json,
    write_netcdf,
)
from .config import model_kwargs, validate
from .evaluation import evaluate_model
from .model import PINN
from .reference import load_reference


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_path(root, path):
    path = Path(path)
    return (
        path.resolve() if path.is_absolute() else (Path(root) / path).resolve()
    )


def check_reference(config, path):
    reference = load_reference(path)
    t0, t1 = config["domain"]["time"]
    x0, x1 = config["domain"]["space"]
    if not np.isclose(reference.x_u[0], x0) or not np.isclose(
            reference.x_u[-1], x1
    ):
        raise ValueError(
            "Reference velocity grid must span the configured spatial domain"
        )
    if reference.x_zeta[0] <= x0 or reference.x_zeta[-1] >= x1:
        raise ValueError(
            "Elevation grid must lie inside velocity boundary nodes"
        )
    if (
            reference.time[0] > t0
            or reference.time[-1] + np.max(np.diff(reference.time)) < t1
    ):
        raise ValueError(
            (
                "Reference times do not cover the configured interval (e"
                "xclusive upper bound)"
            )
        )
    return reference


def create_model(
        config, reference_path, segment_id=0, previous=None, checkpoint_dir=None
):
    kwargs = model_kwargs(
        config, reference_path, segment_id, previous, checkpoint_dir
    )
    if previous is not None:
        kwargs["new_initial_conditions"] = torch.tensor(
            previous[0], dtype=torch.float32, device=config["device"]
        )
        kwargs["new_initial_condition_sampling_points"] = torch.tensor(
            previous[1],
            dtype=torch.float32,
            device=config["device"],
            requires_grad=True,
        )
    return PINN(**kwargs).to(config["device"]), kwargs


def train(config, root, run_id=None, resume_run=None):
    validate(config)
    root = Path(root).resolve()
    reference_path = resolve_path(root, config["reference"])
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Numerical reference missing: {reference_path}. "
            "Run python reference_solutions/shallow_water/generate.py --case viscous "
            "(or --case inviscid), or set reference in your YAML."
        )
    check_reference(config, reference_path)
    if (
            str(config["device"]).startswith("cuda")
            and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable; set device: cpu in YAML"
        )
    seed_everything(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    run_id = (
            run_id
            or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
    )
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run-id must be a single directory name")
    run = (
            resolve_path(root, config["output_root"])
            / config["experiment"]
            / run_id
    )
    if resume_run is not None:
        run = Path(resume_run).resolve()
        if json.loads((run / "config.json").read_text()) != config:
            raise ValueError("Resume requires the original unchanged configuration")
        metadata = json.loads((run / "metadata.json").read_text())
        if metadata["reference_sha256"] != sha256(reference_path):
            raise ValueError("Reference changed since the original run")
        if json.loads((run / "source_manifest.json").read_text()) != source_manifest(root):
            raise ValueError("Source changed since the original run; use the saved source.zip")
        if not config["training"]["resume_period"]:
            raise ValueError("This run did not enable training.resume_period")
    else:
        run.mkdir(parents=True, exist_ok=False)
        save_config(run, config)
        metadata = provenance(root, reference_path, config)
        metadata["source_sha256"] = save_reproduction_bundle(run, root)
        with xr.open_dataset(reference_path) as ds:
            metadata["reference_role"] = ds.attrs.get("data_role", "scientific_reference")
        write_json(run / "metadata.json", metadata)
    previous = None
    from .training_checkpoints import HistoryWriter, save_torch, trim_streams
    try:
        for segment_id in range(config["domain"]["n_time_segments"]):
            segment = run / "segments" / f"{segment_id:03d}"
            checkpoints = segment / "checkpoints"
            checkpoints.mkdir(parents=True, exist_ok=resume_run is not None)
            save_config(segment, config)
            started = time.perf_counter()
            model, kwargs = create_model(
                config, reference_path, segment_id, previous, checkpoints
            )
            recovery = checkpoints / "resume.pt"
            recovery = recovery if resume_run is not None and recovery.exists() else None
            if config["output"]["stream_history"]:
                if resume_run is not None:
                    trim_streams(segment, recovery)
                model.history_writer = HistoryWriter(segment, metadata)
            write_json(run / "status.json", {"status": "running", "segment": segment_id})
            model.fit(resume=recovery)
            elapsed = time.perf_counter() - started
            # Save the final iterate before loading the selected best checkpoint.
            save_torch(model.state_dict(), checkpoints / "final.pt")
            optimizer = (
                model.optimizer.optimizer
                if model.use_pcgrad
                else model.optimizer
            )
            save_torch(
                optimizer.state_dict(), checkpoints / "optimizer_final.pt"
            )
            model.load_state_dict(
                torch.load(
                    checkpoints / "best.pt",
                    map_location=config["device"],
                    weights_only=True,
                )
            )
            write_netcdf(
                history_dataset(model, metadata), segment / "history.nc"
            )
            evaluation_metadata = dict(
                metadata,
                checkpoint=str(checkpoints / "best.pt"),
                checkpoint_sha256=sha256(checkpoints / "best.pt"),
            )
            ds, metrics = evaluate_model(
                model,
                evaluation_metadata,
                config["output"]["evaluation_batch_size"],
            )
            write_netcdf(ds, segment / "evaluation.nc")
            metrics.update(
                best_epoch=model.best_step,
                best_reference_mse_sum=model.best_score,
                training_seconds=elapsed,
            )
            write_json(segment / "metrics.json", metrics)
            previous = model.final_state()
            # Retain the sampled handoff to reconstruct later time segments
            # independently.
            write_netcdf(
                xr.Dataset(
                    {
                        "state": (("sample", "field"), previous[0]),
                        "points": (("sample", "coordinate"), previous[1]),
                    },
                    coords={"field": ["u", "zeta"], "coordinate": ["t", "x"]},
                    attrs={
                        "coordinate_system": (
                            "model nondimensional"
                            if model.nondimensionalize
                            else "dimensional SI"
                        ),
                        "segment_id": segment_id,
                    },
                ),
                segment / "handoff.nc",
            )
            if config["output"]["plots"]:
                from .plotting import plot_segment

                plot_segment(segment)
            del model, optimizer, ds
        write_json(run / "status.json", {"status": "complete"})
    except BaseException as error:
        write_json(
            run / "status.json", {"status": "failed", "error": str(error)}
        )
        raise
    return run


def evaluate_checkpoint(
        config, root, checkpoint, output, segment_id=0, handoff=None
):
    """Evaluate any compatible state dict, including archived extensionless backups."""
    reference_path = resolve_path(root, config["reference"])
    check_reference(config, reference_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous = None
    if segment_id:
        if handoff is None:
            raise ValueError(
                "Later segments require the previous segment's --handoff file"
            )
        with xr.open_dataset(handoff) as ds:
            previous = (ds.state.values, ds.points.values)
    seed_everything(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    model, _ = create_model(config, reference_path, segment_id, previous)
    model.load_state_dict(
        torch.load(
            checkpoint, map_location=config["device"], weights_only=True
        ),
        strict=True,
    )
    metadata = provenance(root, reference_path, config)
    metadata["checkpoint"] = str(Path(checkpoint).resolve())
    metadata["checkpoint_sha256"] = sha256(checkpoint)
    with xr.open_dataset(reference_path) as ds:
        metadata["reference_role"] = ds.attrs.get(
            "data_role", "scientific_reference"
        )
    save_config(output, config)
    write_json(output / "metadata.json", metadata)
    ds, metrics = evaluate_model(
        model, metadata, config["output"]["evaluation_batch_size"]
    )
    write_netcdf(ds, output / "evaluation.nc")
    write_json(output / "metrics.json", metrics)
    return output
