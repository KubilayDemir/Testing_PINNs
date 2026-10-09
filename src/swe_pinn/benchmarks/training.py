"""Heat diffusion and viscous Burgers benchmark training.
"""

from importlib.resources import files
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import xarray as xr
import yaml

from ..artifacts import provenance, save_config, save_reproduction_bundle, source_manifest, sha256, write_json, \
    write_netcdf
from ..config import _merge
from ..optim.pcgrad import PCGrad
from ..runtime import save_torch, trim_streams
from ..sampling import PDEDataset, LeftBoundaryDataset, RightBoundaryDataset, InitialConditionDataset
from ..training import seed_everything, resolve_path
from ..model import PINN


def load_config(path, overrides=None):
    base = yaml.safe_load(files("swe_pinn.benchmarks").joinpath("defaults.yaml").read_text())
    loaded = yaml.safe_load(Path(path).read_text())
    if isinstance(loaded, dict) and loaded.get("equation") == "burgers":
        base["physics"]["diffusivity"] = 0.01 / (4 * torch.atan(torch.ones(1)).item())
    config = _merge(base, {} if loaded is None else loaded)
    for extra in overrides or []:
        loaded = yaml.safe_load(Path(extra).read_text())
        _merge(config, {} if loaded is None else loaded)
    validate(config)
    return config


def validate(c):
    if c["equation"] not in ("heat", "burgers"):
        raise ValueError("equation must be heat or burgers")
    if c["experiment"] in ("", ".", "..") or Path(c["experiment"]).name != c["experiment"]:
        raise ValueError("experiment must be a single directory name")
    if type(c["seed"]) is not int or not 0 <= c["seed"] < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(c["cpu_threads"]) is not int or c["cpu_threads"] < 1:
        raise ValueError("cpu_threads must be positive")
    for section, keys in (("sampling", ("n_pde", "n_bc", "n_ic", "resample_period")),
                          ("training", ("epochs", "pde_batch_size", "bc_batch_size", "ic_batch_size", "report_period", "lr_step_size")),
                          ("output", ("snapshot_period", "evaluation_batch_size"))):
        for key in keys:
            if type(c[section][key]) is not int or c[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for key in ("dataloader_workers", "resume_period"):
        if type(c["training"][key]) is not int or c["training"][key] < 0:
            raise ValueError(f"training.{key} must be nonnegative")
    for value in (c["physics"]["diffusivity"], c["training"]["learning_rate"], *c["loss"].values()):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError("diffusivity, learning rate, and loss weights must be positive")
    if not c["network"]["hidden_layers"] or any(type(n) is not int or n < 1 for n in c["network"]["hidden_layers"]):
        raise ValueError("hidden_layers must contain positive integers")
    if c["network"]["activation"] not in ("tanh", "sin", "sigmoid"):
        raise ValueError("activation must be tanh, sin, or sigmoid")
    training = c["training"]
    if training["optimizer"] not in ("Adam", "LBFGS"):
        raise ValueError("benchmark optimizer must be Adam or LBFGS")
    if training["optimizer"] == "LBFGS" and (training["use_pcgrad"] or training["adaptive_loss_weights"]):
        raise ValueError("PCGrad and adaptive weights require benchmark optimizer Adam")
    if training["lr_schedule"] not in ("none", "step"):
        raise ValueError("lr_schedule must be none or step")
    if training["lr_schedule"] != "none" and training["optimizer"] != "Adam":
        raise ValueError("Learning-rate schedules require Adam")
    if not isinstance(training["lr_decay"], (int, float)) or not 0 < training["lr_decay"] <= 1:
        raise ValueError("lr_decay must be in (0, 1]")
    if training["initial_checkpoint"] is not None and not isinstance(training["initial_checkpoint"], str):
        raise ValueError("initial_checkpoint must be a path or null")
    iterations = training["iterations_per_epoch"]
    if iterations is not None and (type(iterations) is not int or iterations < 1):
        raise ValueError("iterations_per_epoch must be null or a positive integer")


def sample_pools(c):
    s = c["sampling"]
    return [
        PDEDataset(0, 1, -1, 1, s["n_pde"]).tx.detach(),
        LeftBoundaryDataset(0, 1, -1, s["n_bc"]).tx.detach(),
        RightBoundaryDataset(0, 1, 1, s["n_bc"]).tx.detach(),
        InitialConditionDataset(0, -1, 1, s["n_ic"]).tx.detach(),
    ]


def reference_data(c, root):
    if c["equation"] == "burgers":
        from scipy.io import loadmat
        ds = loadmat(resolve_path(root, c["reference"]))
        t, x, u = ds["t"].ravel(), ds["x"].ravel(), ds["usol"]
    else:
        # Heat diagnostic grid and closed-form solution.
        t, x = np.arange(0, 1 + 0.1, 0.1), np.arange(-1, 1 + 0.1, 0.1)
        tt, xx = np.meshgrid(t, x)
        u = np.sin(np.pi * xx) * np.exp(-c["physics"]["diffusivity"] * np.pi ** 2 * tt)
    if u.shape != (len(x), len(t)) or not np.isfinite(u).all() or not np.all(np.diff(t) > 0) or not np.all(
            np.diff(x) > 0):
        raise ValueError("Scalar reference must be finite on an increasing (x,time) grid")
    return t, x, u


def evaluation(model, c, root, reference_arrays=None):
    t, x, reference = reference_arrays if reference_arrays is not None else reference_data(c, root)
    tt, xx = np.meshgrid(t, x)
    points = torch.tensor(np.column_stack((tt.ravel(), xx.ravel())), dtype=torch.float32)
    predictions, residuals = [], []
    for start in range(0, len(points), c["output"]["evaluation_batch_size"]):
        batch = points[start:start + c["output"]["evaluation_batch_size"]].to(c["device"])
        with torch.no_grad():
            predictions.append(model(batch[:, :1], batch[:, 1:2]).cpu().numpy())
        if c["output"]["save_residuals"]:
            residuals.append(model.pde_residual(batch.requires_grad_(True)).detach().cpu().numpy())
    predicted = np.concatenate(predictions).reshape(tt.shape)
    data = {"u_pred": (("x", "time"), predicted), "u_ref": (("x", "time"), reference),
            "u_error": (("x", "time"), predicted - reference)}
    if residuals:
        data["pde_residual"] = (("x", "time"), np.concatenate(residuals).reshape(tt.shape))
    ds = xr.Dataset(data, coords={"x": x, "time": t}, attrs={"equation": c["equation"], "schema_version": "2"})
    for name in ds.variables:
        ds[name].attrs["units"] = "1"
    error = predicted.astype(np.float64) - reference
    relative_l2 = float(np.linalg.norm(error) / np.linalg.norm(reference))
    return ds, relative_l2


def report_losses(model, pools, c):
    size = c["output"]["evaluation_batch_size"]
    values = {"loss_pde": 0.0, "loss_bc": 0.0, "loss_ic": 0.0}
    for index, pool in enumerate(pools):
        key = "loss_pde" if index == 0 else "loss_ic" if index == 3 else "loss_bc"
        for start in range(0, len(pool), size):
            batch = pool[start:start + size].to(c["device"])
            if index == 0:
                error = model.pde_residual(batch.requires_grad_(True)).detach()
            else:
                with torch.no_grad():
                    prediction = model(batch[:, :1], batch[:, 1:2])
                    target = model.initial_condition(batch[:, 1:2]) if index == 3 else 0 * batch[:, 1:2]
                    error = prediction - target
            values[key] += float((error.square() / len(pool)).sum())
    return values


def train(c, root, run_id=None, resume_run=None):
    from datetime import datetime, timezone
    import uuid
    validate(c)
    root = Path(root).resolve()
    reference_arrays = reference_data(c, root)
    initial_checkpoint = c["training"]["initial_checkpoint"]
    initial_checkpoint = resolve_path(root, initial_checkpoint) if initial_checkpoint else None
    if initial_checkpoint is not None and not initial_checkpoint.is_file():
        raise FileNotFoundError(initial_checkpoint)
    if c["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose device: cpu")
    seed_everything(c["seed"])
    torch.set_num_threads(c["cpu_threads"])
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run-id must be a single directory name")
    run = Path(resume_run).resolve() if resume_run else resolve_path(root, c["output_root"]) / c["experiment"] / run_id
    if resume_run:
        if json.loads((run / "config.json").read_text()) != c or json.loads(
                (run / "source_manifest.json").read_text()) != source_manifest(root):
            raise ValueError("Resume requires unchanged configuration and source")
    else:
        run.mkdir(parents=True, exist_ok=False)
        save_config(run, c)
        metadata = provenance(root, resolve_path(root, c["reference"]) if c["equation"] == "burgers" else None, c)
        metadata["reference_role"] = "burgers_numerical_reference" if c[
                                                                          "equation"] == "burgers" else "analytic_heat_solution"
        metadata["checkpoint_selection"] = "minimum relative L2 reference error; final.pt always preserved"
        metadata["source_sha256"] = save_reproduction_bundle(run, root)
        if initial_checkpoint:
            metadata["initial_checkpoint"] = str(initial_checkpoint)
            metadata["initial_checkpoint_sha256"] = sha256(initial_checkpoint)
        write_json(run / "metadata.json", metadata)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(exist_ok=bool(resume_run))
    model = PINN(equation=c["equation"], **c["network"], diffusivity=c["physics"]["diffusivity"]).to(c["device"])
    if initial_checkpoint is not None and not resume_run:
        model.load_state_dict(torch.load(initial_checkpoint, map_location=c["device"], weights_only=True))
    training, output = c["training"], c["output"]
    optimizer = getattr(torch.optim, training["optimizer"])(model.parameters(), lr=training["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training["lr_step_size"], gamma=training["lr_decay"]) if training["lr_schedule"] == "step" else None
    projected = PCGrad(optimizer) if training["use_pcgrad"] else None
    weights = dict(c["loss"])
    pools = sample_pools(c)
    history, snapshots, best, completed, best_epoch = [], [], float("inf"), 0, 0
    recovery = checkpoints / "resume.pt"
    if resume_run:
        metadata = json.loads((run / "metadata.json").read_text())
        if c["equation"] == "burgers" and metadata["reference_sha256"] != sha256(resolve_path(root, c["reference"])):
            raise ValueError("Scalar reference changed since the original run")
        manifest = json.loads(recovery.with_suffix(".json").read_text())
        if sha256(recovery) != manifest["sha256"]:
            raise ValueError("Resume checkpoint checksum mismatch")
        state = torch.load(recovery, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        pools, history, snapshots, weights, best, completed = [p.cpu() for p in state["pools"]], state["history"], \
            state["snapshots"], state["weights"], state["best"], state["epoch"]
        save_torch(state["best_model"], checkpoints / "best.pt")
        best_epoch = state.get("best_epoch", 0)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if output["stream_history"]:
            trim_streams(run, recovery)
    sizes = (training["pde_batch_size"], training["bc_batch_size"], training["bc_batch_size"],
             training["ic_batch_size"])

    def select_best(epoch, score):
        nonlocal best, best_epoch
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite benchmark reference error")
        if score < best:
            best, best_epoch = score, epoch
            save_torch(model.state_dict(), checkpoints / "best.pt")
            if output["keep_improving_checkpoints"]:
                save_torch(model.state_dict(), checkpoints / f"best_epoch_{epoch:06d}.pt")

    def record(epoch, result):
        if output["batch_diagnostics"]:
            values = report_losses(model, pools, c)
        else:
            batches = [p.to(c["device"]).requires_grad_(True) for p in pools]
            terms = model.losses(batches)
            values = {key: float(value.detach()) for key, value in terms.items()}
        ds, score = result
        values.update(epoch=epoch, relative_l2=score,
                      loss_total=sum(weights[key.replace("loss_", "weight_")] * value for key, value in values.items()
                                     if key == "loss_pde" or not c["network"]["enforce_" + key.removeprefix("loss_")]))
        values["learning_rate"] = optimizer.param_groups[0]["lr"]
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError("Nonfinite benchmark training/validation loss")
        history.append(values)
        if output["stream_history"]:
            with (run / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(values) + "\n")
        if training["report_period"] > 1:
            print(f"Epoch {epoch}/{training['epochs']}: loss={values['loss_total']:.6g}, relative_l2={score:.6g}, lr={values['learning_rate']:.3g}", flush=True)

    def snapshot(epoch, ds):
        if epoch % output["snapshot_period"] == 0:
            keys = (["u_pred"] if output["save_predictions"] else []) + (
                ["pde_residual"] if output["save_residuals"] else [])
            if keys:
                snapshot = ds[keys].expand_dims(snapshot=[epoch]).assign_coords(snapshot_epoch=("snapshot", [epoch]))
                if output["stream_history"]:
                    (run / "snapshots").mkdir(exist_ok=True)
                    write_netcdf(snapshot, run / "snapshots" / f"epoch_{epoch:06d}.nc")
                else:
                    snapshots.append(snapshot)

    try:
        if not resume_run:
            initial = evaluation(model, c, root, reference_arrays)
            select_best(0, initial[1])
            record(0, initial)
            snapshot(0, initial[0])
        for epoch in range(completed + 1, training["epochs"] + 1):
            write_json(run / "status.json", {"status": "running", "epoch": epoch})
            # Resample after a completed sampling period.
            if epoch > 1 and (epoch - 1) % c["sampling"]["resample_period"] == 0:
                pools = sample_pools(c)
            loaders = [DataLoader(pool, batch_size=size, shuffle=True, num_workers=training["dataloader_workers"],
                                  pin_memory=training["pin_memory"]) for pool, size in zip(pools, sizes)] if training[
                "minibatch"] else []
            iterators = [iter(loader) for loader in loaders]
            iterations = training["iterations_per_epoch"] or (max(1, len(pools[0]) // sizes[0]) if loaders else 1)
            for iteration in range(iterations):
                batches = []
                if loaders:
                    for index, loader in enumerate(loaders):
                        try:
                            batch = next(iterators[index])
                        except StopIteration:
                            iterators[index] = iter(loader)
                            batch = next(iterators[index])
                        batches.append(
                            batch.to(c["device"], non_blocking=training["pin_memory"]).detach().requires_grad_(True))
                else:
                    batches = [p.to(c["device"]).detach().requires_grad_(True) for p in pools]

                def closure():
                    optimizer.zero_grad()
                    terms = model.losses(batches)
                    if training["adaptive_loss_weights"]:
                        params = list(model.parameters())
                        gradients = {key: torch.cat(
                            [g.reshape(-1) for g in torch.autograd.grad(value, params, retain_graph=True)]) for
                            key, value in terms.items()}
                        maximum = gradients["loss_pde"].abs().max()
                        for field in ("bc", "ic"):
                            if not c["network"][f"enforce_{field}"]:
                                target = maximum / gradients[f"loss_{field}"].abs().mean().clamp_min(
                                    torch.finfo(torch.float32).eps)
                                weights[f"weight_{field}"] = 0.1 * weights[f"weight_{field}"] + 0.9 * float(
                                    target.detach())
                    losses = [weights[key.replace("loss_", "weight_")] * value for key, value in terms.items()
                              if key == "loss_pde" or not c["network"]["enforce_" + key.removeprefix("loss_")]]
                    total = sum(losses)
                    if not torch.isfinite(total):
                        raise FloatingPointError("Nonfinite benchmark training loss")
                    if projected:
                        projected.pc_backward(losses)
                    else:
                        total.backward()
                    return total

                if training["optimizer"] == "LBFGS":
                    optimizer.step(closure)
                else:
                    closure()
                    optimizer.step()
            result = evaluation(model, c, root, reference_arrays)
            select_best(epoch, result[1])
            if epoch % training["report_period"] == 0 or epoch == training["epochs"]:
                record(epoch, result)
            snapshot(epoch, result[0])
            if scheduler is not None:
                scheduler.step()
            if training["resume_period"] and (epoch % training["resume_period"] == 0 or epoch == training["epochs"]):
                save_torch({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "pools": pools, "history": history, "snapshots": snapshots, "weights": weights,
                            "best": best, "best_epoch": best_epoch,
                            "scheduler": scheduler.state_dict() if scheduler is not None else None,
                            "best_model": torch.load(checkpoints / "best.pt", map_location="cpu", weights_only=True),
                            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, recovery)
                write_json(recovery.with_suffix(".json"), {"sha256": sha256(recovery), "epoch": epoch})
        save_torch(model.state_dict(), checkpoints / "final.pt")
        save_torch(optimizer.state_dict(), checkpoints / "optimizer_final.pt")
        ds, score = evaluation(model, c, root, reference_arrays)
        ds.attrs["checkpoint"] = "final.pt"
        write_netcdf(ds, run / "evaluation.nc")
        history_ds = xr.Dataset({key: ("epoch", [row[key] for row in history]) for key in history[0] if key != "epoch"},
                                coords={"epoch": [row["epoch"] for row in history]})
        if snapshots:
            history_ds = xr.merge([history_ds, xr.concat(snapshots, dim="snapshot")])
        write_netcdf(history_ds, run / "history.nc")
        model.load_state_dict(torch.load(checkpoints / "best.pt", map_location=c["device"], weights_only=True))
        best_ds, _ = evaluation(model, c, root, reference_arrays)
        best_ds.attrs["checkpoint"] = "best.pt"
        write_netcdf(best_ds, run / "best_evaluation.nc")
        write_json(run / "metrics.json", {"final_relative_l2": score, "best_relative_l2": best, "best_epoch": best_epoch})
        if output["plots"]:
            plot(run)
        write_json(run / "status.json", {"status": "complete"})
    except BaseException as error:
        write_json(run / "status.json", {"status": "failed", "error": str(error)})
        raise
    return run


def plot(run):
    from ..plotting import _save, plt, solution_panels
    run = Path(run)
    directory = run / "figures"
    directory.mkdir(exist_ok=True)
    datasets = {}
    for name, filename in (("solution", "evaluation.nc"), ("best_solution", "best_evaluation.nc")):
        if (run / filename).exists():
            with xr.open_dataset(run / filename) as ds:
                datasets[name] = ds.load()
    # These benchmark cases have amplitude one; share limits across cases and checkpoints.
    limit = max(1.0, *(float(np.abs(ds.u_pred).max()) for ds in datasets.values()))
    for name, ds in datasets.items():
        error = float(np.linalg.norm(ds.u_error) / np.linalg.norm(ds.u_ref))
        caption = "Best checkpoint" if name == "best_solution" else "Final checkpoint"
        fig = solution_panels(ds.time, ds.x, ds.u_ref, ds.u_pred, limit=limit,
                              xlabel="t", ylabel="x", title=f"{ds.attrs['equation'].capitalize()}: {caption}, relative L2 = {100 * error:.2f}%")
        _save(fig, directory, name)
    with xr.open_dataset(run / "history.nc") as ds:
        fig, axes = plt.subplots(1, 3 if "learning_rate" in ds else 2, figsize=(13, 4))
        for index, key in enumerate(("loss_total", "loss_pde", "loss_bc", "loss_ic")):
            axes[0].semilogy(ds.epoch, np.maximum(ds[key], 1e-30), label=key.removeprefix("loss_"),
                             color=plt.get_cmap("Dark2")(index))
        axes[0].set(ylabel="Training loss")
        axes[0].legend()
        axes[1].semilogy(ds.epoch, np.maximum(100 * ds.relative_l2, 1e-30), color=plt.get_cmap("Dark2")(0))
        axes[1].set(ylabel="Reference relative L2 [%]")
        metrics_path = run / "metrics.json"
        if metrics_path.exists():
            best_epoch = json.loads(metrics_path.read_text()).get("best_epoch")
            if best_epoch is not None:
                axes[1].axvline(best_epoch, color="0.4", linestyle="--", label="Best checkpoint")
                axes[1].legend()
        if len(axes) == 3:
            axes[2].semilogy(ds.epoch, ds.learning_rate, drawstyle="steps-post")
            axes[2].set(ylabel="Adam learning rate")
        for axis in axes:
            axis.set(xlabel="Completed epoch", xlim=(float(ds.epoch.min()), float(ds.epoch.max())))
            axis.grid(True, which="major", alpha=0.2)
        _save(fig, directory, "learning_curves")
    return directory
