"""Optional runtime persistence, independent of the optimization objective."""

from pathlib import Path
import json
import random

import numpy as np
import torch

from .artifacts import history_dataset, write_netcdf, sha256, write_json, replace_file

SNAPSHOT_LISTS = (
    "velocity_history", "elevation_history",
    "momentum_residual_history", "continuity_residual_history",
)


def save_torch(value, path):
    """Replace a generated checkpoint only after its temporary write succeeds."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temp)
        replace_file(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class HistoryWriter:
    """Stream large snapshots to individual compressed NetCDF files.

    Scalar history remains small and is also journaled for interrupted runs.
    No background thread touches HDF5/NetCDF handles.
    """

    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.metadata = metadata
        (self.directory / "snapshots").mkdir(exist_ok=True)

    def record(self, model, epoch):
        values = {
            name[:-8]: float(getattr(model, name)[-1])
            for name in model.history_names
            if name.endswith("_history") and name.startswith(("loss_", "mse_ref_"))
        }
        self._append("history.jsonl", dict(epoch=epoch, **values))

    def snapshot(self, model, epoch):
        ds = history_dataset(model, self.metadata)
        variables = [name for name in ds.data_vars if "snapshot" in ds[name].dims]
        ds = ds[variables]
        write_netcdf(ds, self.directory / "snapshots" / f"epoch_{epoch:06d}.nc")
        for name in SNAPSHOT_LISTS:
            getattr(model, name).clear()
        model.snapshot_epochs.clear()

    def timing(self, model, epoch):
        values = {
            "epoch": epoch, "epoch_seconds": model.time_per_epoch[-1],
            "iteration_seconds": model.time_per_iteration,
        }
        if model.adaptive_loss_weights:
            values["adaptive_weights"] = {
                "step": model.weight_steps,
                "weight_pde": model.pde_weight_history,
                "weight_ic": model.ic_weight_history,
                "weight_bc": model.bc_weight_history,
            }
        self._append("timing.jsonl", values)
        model.time_per_epoch.clear()
        model.time_per_iteration.clear()
        if model.adaptive_loss_weights:
            model.weight_steps.clear()
            model.pde_weight_history.clear()
            model.ic_weight_history.clear()
            model.bc_weight_history.clear()

    def _append(self, name, values):
        with (self.directory / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(values, allow_nan=False) + "\n")
            stream.flush()


def save_resume(model, epoch):
    """Epoch-boundary recovery includes pools, sampler states and all RNG states."""
    if model.checkpoint_dir is None:
        return
    optimizer = model.optimizer.optimizer if model.use_pcgrad else model.optimizer
    state = {
        "schema_version": 1, "epoch": epoch,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "weight_ic": model.weight_ic, "weight_bc": model.weight_bc,
        "best_step": model.best_step, "best_score": model.best_score,
        "best_model": torch.load(model.checkpoint_dir / "best.pt", map_location="cpu", weights_only=True),
        "improvement_steps": model.improvement_steps,
        "histories": {name: getattr(model, name) for name in model.history_names},
        "history_epochs": model.history_epochs,
        "snapshot_epochs": model.snapshot_epochs, "weight_steps": model.weight_steps,
        "time_per_epoch": model.time_per_epoch,
        "time_per_iteration": model.time_per_iteration,
        "datasets": {name: getattr(model, name) for name in (
            "pde_dataset", "lbc_dataset", "ubc_dataset", "ic_dataset", "final_state_dataset",
        )},
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    path = model.checkpoint_dir / "resume.pt"
    save_torch(state, path)
    write_json(path.with_suffix(".json"), {"sha256": sha256(path), "epoch": epoch})


def restore_resume(model, path):
    """Load only this run's locally generated recovery file (contains Python objects)."""
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != manifest["sha256"]:
        raise ValueError("Resume checkpoint checksum mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["schema_version"] != 1:
        raise ValueError("Unsupported resume checkpoint schema")
    model.load_state_dict(state["model"], strict=True)
    save_torch(state["best_model"], model.checkpoint_dir / "best.pt")
    optimizer = model.optimizer.optimizer if model.use_pcgrad else model.optimizer
    optimizer.load_state_dict(state["optimizer"])
    for name in ("weight_ic", "weight_bc", "best_step", "best_score", "improvement_steps",
                 "history_epochs", "snapshot_epochs", "weight_steps",
                 "time_per_epoch", "time_per_iteration"):
        value = state[name]
        if name in ("weight_ic", "weight_bc") and isinstance(value, torch.Tensor):
            value = value.to(model.device)
        setattr(model, name, value)
    for name, values in state["histories"].items():
        setattr(model, name, values)
    for name, dataset in state["datasets"].items():
        # Sampling pools always remain on CPU, including for CUDA recovery.
        dataset.tx = dataset.tx.detach().cpu().requires_grad_(True)
        dataset.t, dataset.x = dataset.tx[:, :1], dataset.tx[:, 1:2]
        setattr(model, name, dataset)
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if state["rng"]["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])
    return state["epoch"]


def trim_streams(directory, recovery):
    """Discard only generated journal records newer than the recovery boundary."""
    directory = Path(directory)
    epoch = -1 if recovery is None else json.loads(Path(recovery).with_suffix(".json").read_text())["epoch"]
    for path in (directory / "snapshots").glob("epoch_*.nc"):
        if int(path.stem.removeprefix("epoch_")) > epoch:
            path.unlink()
    for name in ("history.jsonl", "timing.jsonl"):
        path = directory / name
        if path.exists():
            records = []
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record["epoch"] <= epoch:
                    records.append(json.dumps(record))
            path.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")
