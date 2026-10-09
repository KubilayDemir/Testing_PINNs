"""Evaluate on the staggered reference grid and save dimensional diagnostic arrays."""

import numpy as np
import json
import torch
import xarray as xr


def _predict(model, t, x, column, batch_size):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(t), batch_size):
            chunks.append(
                model(
                    t[start : start + batch_size].to(model.device),
                    x[start : start + batch_size].to(model.device),
                )[:, column]
                .cpu()
                .numpy()
            )
    return np.concatenate(chunks)


def evaluate_model(model, metadata, batch_size=8192):
    u = _predict(
        model,
        model.u_solution_time_input_grid,
        model.u_solution_x_input_grid,
        0,
        batch_size,
    ).reshape(model.u_solution_mesh_grid_shape)
    zeta = _predict(
        model,
        model.zeta_solution_time_input_grid,
        model.zeta_solution_x_input_grid,
        1,
        batch_size,
    ).reshape(model.zeta_solution_mesh_grid_shape)
    length = model.horizontal_length_scale if model.nondimensionalize else 1.0
    time = model.time_scale if model.nondimensionalize else 1.0
    height = model.vertical_length_scale if model.nondimensionalize else 1.0
    u, zeta = u * length / time, zeta * height
    u_ref, zeta_ref = model.u_ref.cpu().numpy(), model.zeta_ref.cpu().numpy()
    ds = xr.Dataset(
        coords={
            "time": (
                "time",
                model.solution_time_grid * time,
                {"units": "s", "long_name": "elapsed physical time"},
            ),
            "x_u": ("x_u", model.u_solution_x_grid * length, {"units": "m"}),
            "x_zeta": (
                "x_zeta",
                model.zeta_solution_x_grid * length,
                {"units": "m"},
            ),
            "diagnostic_time": (
                "diagnostic_time",
                model.time_grid * time,
                {"units": "s"},
            ),
            "diagnostic_x": (
                "diagnostic_x",
                model.x_grid * length,
                {"units": "m"},
            ),
        },
        attrs={
            "title": "Shallow-water PINN evaluation and internal diagnostics",
            "schema_version": "2",
            "segment_id": model.segment_id,
            "nondimensional_model": int(model.nondimensionalize),
            "reference_sha256": metadata["reference_sha256"],
            "config_json": metadata["config_json"],
            "git_revision": metadata["git_revision"],
            "created_utc": metadata["created_utc"],
            "checkpoint": metadata.get("checkpoint", "best.pt"),
            "metadata_json": json.dumps(metadata),
            "data_role": metadata.get(
                "reference_role", "scientific_reference"
            ),
            ("zeta_definition"): (
                "sea-level elevation relative to the undisturbed surface; "
                "total depth is d + zeta"
            ),
            ("residual_definition"): (
                "signed PDE residual, not squared loss; model coordinate"
                "s and scaling"
            ),
        },
    )
    for key, predicted, reference, grid, units in (
        ("u", u, u_ref, "x_u", "m s-1"),
        ("zeta", zeta, zeta_ref, "x_zeta", "m"),
    ):
        for suffix, values, title in (
            ("pred", predicted, "PINN prediction"),
            ("ref", reference, "numerical reference"),
            (
                "error",
                predicted - reference,
                "signed prediction minus reference",
            ),
        ):
            ds[f"{key}_{suffix}"] = (
                (grid, "time"),
                values,
                {"units": units, "long_name": f"{title}: {key}"},
            )
    old_points = model.pde_points if hasattr(model, "pde_points") else None
    momentum_chunks, continuity_chunks = [], []
    try:
        for start in range(0, len(model.time_input_grid), batch_size):
            t = (
                model.time_input_grid[start : start + batch_size]
                .detach()
                .clone()
                .requires_grad_(True)
            )
            x = (
                model.x_input_grid[start : start + batch_size]
                .detach()
                .clone()
                .requires_grad_(True)
            )
            model.pde_points = torch.hstack((t, x))
            model.compute_pde_residuals()
            momentum_chunks.append(
                model.momentum_residual.detach().cpu().numpy()
            )
            continuity_chunks.append(
                model.continuity_residual.detach().cpu().numpy()
            )
    finally:
        model.pde_points = old_points
    for key, values, units in (
        ("momentum_residual", momentum_chunks, "m s-2"),
        ("continuity_residual", continuity_chunks, "m s-1"),
    ):
        ds[key] = (
            ("diagnostic_x", "diagnostic_time"),
            np.concatenate(values).reshape(model.mesh_grid_shape),
            {
                "units": "1" if model.nondimensionalize else units,
                "long_name": "signed " + key,
            },
        )
    t = torch.tensor(
        model.time_grid, dtype=torch.float32, device=model.device
    ).reshape(-1, 1)
    for key, bound in (
        ("u_bc_left", model.x_min),
        ("u_bc_right", model.x_max),
    ):
        ds[key] = (
            "diagnostic_time",
            _predict(model, t, torch.full_like(t, bound), 0, batch_size)
            * length
            / time,
            {"units": "m s-1", "target": "zero for closed boundaries"},
        )
    # Save initial-condition targets at their actual sample positions for every segment.
    if model.segment_id == 0:
        points = model.ic_dataset.tx.to(model.device)
        with torch.no_grad():
            targets = torch.cat(
                (
                    model.initial_u(points[:, 1:2]),
                    model.initial_zeta(points[:, 1:2]),
                ),
                dim=1,
            )
    else:
        points, targets = (
            model.new_initial_condition_sampling_points,
            model.new_initial_conditions,
        )
    ds["ic_x"] = (
        "ic_sample",
        points[:, 1].detach().cpu().numpy() * length,
        {"units": "m"},
    )
    for key, column, factor, units in (
        ("u", 0, length / time, "m s-1"),
        ("zeta", 1, height, "m"),
    ):
        ds[f"ic_{key}_pred"] = (
            "ic_sample",
            _predict(model, points[:, :1], points[:, 1:2], column, batch_size)
            * factor,
            {"units": units},
        )
        ds[f"ic_{key}_target"] = (
            "ic_sample",
            targets[:, column].detach().cpu().numpy() * factor,
            {"units": units},
        )
    metrics = {}
    for name, prediction, reference in (
        ("u", u, u_ref),
        ("zeta", zeta, zeta_ref),
    ):
        error = prediction.astype(np.float64) - reference
        denominator = np.linalg.norm(reference.astype(np.float64))
        max_ref = np.max(np.abs(reference))
        metrics[f"mse_{name}"] = float(np.mean(error**2))
        metrics[f"relative_l2_{name}"] = (
            float(np.linalg.norm(error) / denominator) if denominator else None
        )
        metrics[f"relative_max_{name}"] = (
            float(np.max(np.abs(error)) / max_ref) if max_ref else None
        )
    return ds, metrics
