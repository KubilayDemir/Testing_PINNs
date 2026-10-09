"""Internal diagnostics generated entirely from portable saved NetCDF arrays."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import xarray as xr
from .output_formats import canonical_dataset


def _save(fig, directory, name):
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=300)
    fig.savefig(directory / f"{name}.pdf", dpi=300)
    plt.close(fig)


def solution_panels(time, x, reference, prediction, *, units="1", limit=None,
                    xlabel="Time", ylabel="Position", title=None):
    """Shared field limits and six decades of absolute error in physical units.

    Zero errors are clipped only for display; saved signed arrays are untouched.
    Pass the same limit to compare several runs or checkpoints.
    """
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    limit = limit or max(float(np.abs(reference).max()), float(np.abs(prediction).max()), 1e-12)
    field_norm = Normalize(-limit, limit)
    error_norm = LogNorm(limit * 1e-6, 2 * limit)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True, layout="constrained")
    for axis, array, caption in zip(axes[:2], (reference, prediction), ("Reference", "PINN")):
        mesh = axis.pcolormesh(time, x, array, shading="auto", cmap="RdBu_r",
                              norm=field_norm, rasterized=True)
        axis.set(title=caption)
    fig.colorbar(mesh, ax=list(axes[:2]), label=units)
    error = np.maximum(np.abs(prediction - reference), error_norm.vmin)
    mesh = axes[2].pcolormesh(time, x, error, shading="auto", cmap="YlOrRd",
                              norm=error_norm, rasterized=True)
    fig.colorbar(mesh, ax=axes[2], label=f"Absolute error [{units}]")
    axes[2].set(title="Absolute error (log scale)")
    for axis in axes:
        axis.set(xlabel=xlabel, xlim=(np.min(time), np.max(time)), ylim=(np.min(x), np.max(x)))
    axes[0].set(ylabel=ylabel)
    if title:
        fig.suptitle(title)
    return fig


def plot_segment(segment):
    segment = Path(segment)
    directory = segment / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(segment / "evaluation.nc") as source:
        ds = canonical_dataset(source).load()
    for key, grid in (("u", "x_u"), ("zeta", "x_zeta")):
        fig = solution_panels(ds.time / 3600, ds[grid] / 1000, ds[f"{key}_ref"], ds[f"{key}_pred"],
                              units=ds[f"{key}_pred"].attrs["units"], xlabel="Time [h]",
                              ylabel="Position [km]", title=key)
        _save(fig, directory, f"solution_{key}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, key in zip(axes, ("momentum_residual", "continuity_residual")):
        image = axis.pcolormesh(
            ds.diagnostic_time / 3600,
            ds.diagnostic_x / 1000,
            np.log10(np.maximum(np.abs(ds[key]), 1e-15)),
            shading="auto",
            cmap="YlOrRd",
        )
        fig.colorbar(image, ax=axis, label="log10 absolute residual")
        axis.set(xlabel="Time [h]", ylabel="Position [km]", title=key)
    _save(fig, directory, "pde_residuals")
    fig, axis = plt.subplots(figsize=(7, 4))
    for key in ("u_bc_left", "u_bc_right"):
        axis.plot(ds.diagnostic_time / 3600, ds[key], label=key)
    axis.set(xlabel="Time [h]", ylabel="Boundary velocity [m/s]")
    axis.legend()
    _save(fig, directory, "boundary_conditions")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    order = np.argsort(ds.ic_x.values)
    for axis, key in zip(axes, ("u", "zeta")):
        for suffix in ("target", "pred"):
            axis.plot(
                ds.ic_x.values[order] / 1000,
                ds[f"ic_{key}_{suffix}"].values[order],
                label=suffix,
            )
        axis.set(
            xlabel="Position [km]",
            ylabel=ds[f"ic_{key}_pred"].attrs["units"],
            title=f"Initial {key}",
        )
        axis.legend()
    _save(fig, directory, "initial_conditions")
    if (segment / "history.nc").exists():
        with xr.open_dataset(segment / "history.nc") as source:
            history = canonical_dataset(source).load()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for key in ("loss_total", "loss_pde", "loss_bc", "loss_ic"):
            axes[0].semilogy(
                history.epoch, np.maximum(history[key], 1e-30), label=key
            )
        for key in ("mse_ref_u", "mse_ref_zeta"):
            axes[1].semilogy(
                history.epoch, np.maximum(history[key], 1e-30), label=key
            )
        for axis in axes:
            axis.set(
                xlabel="Completed epoch",
                ylabel="Mean squared error (see stored units)",
            )
            axis.legend()
        _save(fig, directory, "learning_curves")
    return directory


def animate_segment(segment, field="zeta", fps=8):
    """Regenerate a GIF of the final solution without loading PyTorch checkpoints."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    segment = Path(segment)
    field = "zeta" if field == "h" else field
    directory = segment / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(segment / "evaluation.nc") as source:
        ds = canonical_dataset(source).load()
    grid = ds["x_zeta" if field == "zeta" else "x_u"].values / 1000
    fig, axis = plt.subplots()
    (reference,) = axis.plot(
        grid, ds[f"{field}_ref"].values[:, 0], label="reference"
    )
    (predicted,) = axis.plot(
        grid, ds[f"{field}_pred"].values[:, 0], label="PINN"
    )
    values = np.concatenate(
        (ds[f"{field}_ref"].values.ravel(), ds[f"{field}_pred"].values.ravel())
    )
    margin = max(float(np.ptp(values)) * 0.05, 1e-6)
    axis.set(
        xlabel="Position [km]",
        ylabel=ds[f"{field}_pred"].attrs["units"],
        ylim=(values.min() - margin, values.max() + margin),
    )
    axis.legend()

    def update(index):
        reference.set_ydata(ds[f"{field}_ref"].values[:, index])
        predicted.set_ydata(ds[f"{field}_pred"].values[:, index])
        axis.set_title(f"Time: {ds.time.values[index] / 3600:.3f} h")
        return reference, predicted

    path = directory / f"solution_{field}.gif"
    try:
        FuncAnimation(
            fig, update, frames=ds.sizes["time"], interval=1000 / fps
        ).save(path, writer=PillowWriter(fps=fps))
    finally:
        plt.close(fig)
    return path


def animate_history(segment, field="zeta", kind="training", fps=8):
    """Animate the recorded training solution or signed PDE residual snapshots."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    segment = Path(segment)
    field = "zeta" if field == "h" else field
    epochs = snapshot_epochs(segment)
    if not epochs:
        raise ValueError("No snapshots saved; enable save_predictions or save_residuals")
    ds = load_snapshot(segment, epochs[0])
    key = f"{field}_pred" if kind == "training" else ("momentum_residual" if field == "u" else "continuity_residual")
    if key not in ds:
        raise ValueError(
            f"No {key} snapshots saved; enable the corresponding output option in YAML"
        )

    def frame(epoch):
        data = load_snapshot(segment, epoch)[key].values[0]
        return np.log10(np.maximum(np.abs(data), 1e-15)) if kind == "residuals" else data

    limits = [(float(frame(epoch).min()), float(frame(epoch).max())) for epoch in epochs]
    fig, axis = plt.subplots()
    mesh = axis.pcolormesh(
        ds.diagnostic_time / 3600,
        ds.diagnostic_x / 1000,
        frame(epochs[0]),
        shading="auto",
        cmap="YlOrRd" if kind == "residuals" else "RdBu_r",
        vmin=min(a for a, b in limits),
        vmax=max(b for a, b in limits),
    )
    fig.colorbar(
        mesh,
        ax=axis,
        label=(
            "log10 absolute residual"
            if kind == "residuals"
            else ds[key].attrs["units"]
        ),
    )
    axis.set(xlabel="Time [h]", ylabel="Position [km]")

    def update(index):
        mesh.set_array(frame(epochs[index]))
        axis.set_title(
            f"{key}, completed epoch {epochs[index]}"
        )
        return (mesh,)

    directory = segment / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}_{field}.gif"
    try:
        FuncAnimation(fig, update, frames=len(epochs), interval=1000 / fps).save(
            path, writer=PillowWriter(fps=fps)
        )
    finally:
        plt.close(fig)
    return path


def snapshot_epochs(segment):
    paths = sorted((Path(segment) / "snapshots").glob("epoch_*.nc"))
    if paths:
        return [int(path.stem.removeprefix("epoch_")) for path in paths]
    with xr.open_dataset(Path(segment) / "history.nc") as ds:
        return ds.snapshot_epoch.values.tolist() if "snapshot_epoch" in ds else []


def load_snapshot(segment, epoch):
    path = Path(segment) / "snapshots" / f"epoch_{epoch:06d}.nc"
    if path.exists():
        with xr.open_dataset(path) as ds:
            return canonical_dataset(ds).load()
    with xr.open_dataset(Path(segment) / "history.nc") as ds:
        indices = np.flatnonzero(ds.snapshot_epoch.values == epoch)
        if not len(indices):
            raise ValueError(f"No snapshot at epoch {epoch}")
        variables = [name for name in ds.data_vars if "snapshot" in ds[name].dims]
        return canonical_dataset(ds[variables].isel(snapshot=indices[:1])).load()


def plot_study(segments, labels=None, output=None, epochs=None):
    """Paper-style solution, loss and snapshot panels for NEW run outputs.

    Actual recorded epochs are shown. Publication data/notebook stays on Zenodo.
    """
    segments = [Path(path) for path in segments]
    labels = labels or [path.parent.parent.name for path in segments]
    if len(labels) != len(segments):
        raise ValueError("Provide one label per segment directory")
    directory = Path(output) if output else segments[0] / "study_figures"
    directory.mkdir(parents=True, exist_ok=True)
    field_limits = {"u": 0.0, "zeta": 0.0}
    for segment in segments:
        with xr.open_dataset(segment / "evaluation.nc") as source:
            ds = canonical_dataset(source)
            for field in field_limits:
                field_limits[field] = max(field_limits[field], *(float(np.abs(ds[f"{field}_{suffix}"]).max())
                                                                for suffix in ("ref", "pred")))
    for segment, label in zip(segments, labels):
        with xr.open_dataset(segment / "evaluation.nc") as source:
            ds = canonical_dataset(source).load()
        for field, grid in (("u", "x_u"), ("zeta", "x_zeta")):
            fig = solution_panels(ds.time / 3600, ds[grid] / 1000, ds[f"{field}_ref"], ds[f"{field}_pred"],
                                  units=ds[f"{field}_pred"].attrs["units"], limit=field_limits[field],
                                  xlabel="Time [h]", ylabel="Position [km]", title=f"{label}: {field}")
            _save(fig, directory, f"{segments.index(segment):02d}_solution_{field}")
        available = snapshot_epochs(segment)
        selected = [epoch for epoch in (epochs or available[:1] + available[-1:]) if epoch in available]
        selected = list(dict.fromkeys(selected))
        for keys, name in ((("u_pred", "zeta_pred"), "solution_over_training"),
                           (("momentum_residual", "continuity_residual"), "residuals_over_training")):
            if not selected:
                continue
            snapshots = [load_snapshot(segment, epoch) for epoch in selected]
            if not all(key in snapshots[0] for key in keys):
                continue
            fig, axes = plt.subplots(2, len(selected), figsize=(4 * len(selected), 7), squeeze=False)
            for row, key in enumerate(keys):
                data = [ds[key].values[0] for ds in snapshots]
                if name.startswith("residuals"):
                    data = [np.log10(np.maximum(np.abs(array), 1e-15)) for array in data]
                lo, hi = min(float(a.min()) for a in data), max(float(a.max()) for a in data)
                for col, (epoch, snapshot, array) in enumerate(zip(selected, snapshots, data)):
                    mesh = axes[row, col].pcolormesh(snapshot.diagnostic_time / 3600, snapshot.diagnostic_x / 1000,
                                                     array, shading="auto", vmin=lo, vmax=hi,
                                                     cmap="YlOrRd" if name.startswith("residuals") else "RdBu_r",
                                                     rasterized=True)
                    axes[row, col].set(xlabel="Time [h]", ylabel="Position [km]", title=f"{key}, epoch {epoch}")
                    fig.colorbar(mesh, ax=axes[row, col],
                                 label="log10 absolute residual" if name.startswith("residuals") else
                                 snapshot[key].attrs["units"])
            _save(fig, directory, f"{segments.index(segment):02d}_{name}")
    for keys, name in (
            (("loss_total",), "total_loss"),
            (("loss_pde", "loss_bc", "loss_ic"), "individual_losses"),
            (("loss_momentum", "loss_continuity"), "pde_losses"),
            (("mse_ref_u", "mse_ref_zeta"), "reference_errors"),
    ):
        fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4), squeeze=False)
        for segment, label in zip(segments, labels):
            with xr.open_dataset(segment / "history.nc") as source:
                history = canonical_dataset(source).load()
            for axis, key in zip(axes[0], keys):
                axis.semilogy(history.epoch, np.maximum(history[key], 1e-30), label=label)
                axis.set(xlabel="Completed epoch", ylabel=history[key].attrs["units"], title=key)
                axis.legend()
        _save(fig, directory, name)
    return directory
