"""Optional batched diagnostics; training losses and sampling are unchanged."""

import numpy as np
import torch

from .physics import swe_residuals


def prediction_snapshot(model):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(model.time_input_grid), model.evaluation_batch_size):
            stop = start + model.evaluation_batch_size
            chunks.append(model(model.time_input_grid[start:stop], model.x_input_grid[start:stop]).cpu().numpy())
    values = np.concatenate(chunks)
    model.velocity_history.append(values[:, 0].reshape(model.mesh_grid_shape))
    model.elevation_history.append(values[:, 1].reshape(model.mesh_grid_shape))


def residual_snapshot(model):
    momentum, continuity = [], []
    for start in range(0, len(model.time_input_grid), model.evaluation_batch_size):
        stop = start + model.evaluation_batch_size
        t = model.time_input_grid[start:stop].detach().requires_grad_(True)
        x = model.x_input_grid[start:stop].detach().requires_grad_(True)
        rm, rc = swe_residuals(model, t, x)
        momentum.append(rm.detach().cpu().numpy())
        continuity.append(rc.detach().cpu().numpy())
    model.momentum_residual_history.append(np.concatenate(momentum).reshape(model.mesh_grid_shape))
    model.continuity_residual_history.append(np.concatenate(continuity).reshape(model.mesh_grid_shape))


def report_losses(model):
    """Accumulate point-weighted means, never means of unequal batch means."""
    size = model.evaluation_batch_size
    momentum = continuity = 0.0
    points = model.pde_dataset.tx
    for start in range(0, len(points), size):
        batch = points[start:start + size].detach().to(model.device).requires_grad_(True)
        rm, rc = swe_residuals(model, batch[:, :1], batch[:, 1:2])
        momentum += rm.detach().square().sum() / len(points)
        continuity += rc.detach().square().sum() / len(points)
    model.loss_momentum, model.loss_continuity = momentum, continuity
    model.loss_pde = momentum + continuity

    def mean_error(points, target, columns):
        total = torch.zeros(len(columns), device=model.device)
        with torch.no_grad():
            for start in range(0, len(points), size):
                batch = points[start:start + size].to(model.device)
                expected = target(start, batch)
                error = model(batch[:, :1], batch[:, 1:2])[:, columns] - expected
                total += error.square().sum(dim=0) / len(points)
        return total

    zero = lambda start, batch: torch.zeros_like(batch[:, :1])
    model.loss_bc_left = mean_error(model.lbc_dataset.tx, zero, [0])[0]
    model.loss_bc_right = mean_error(model.ubc_dataset.tx, zero, [0])[0]
    model.loss_bc = model.loss_bc_left + model.loss_bc_right
    if model.segment_id == 0:
        points = model.ic_dataset.tx
        target = lambda start, batch: torch.cat((model.initial_u(batch[:, 1:2]), model.initial_zeta(batch[:, 1:2])),
                                                dim=1)
    else:
        points = model.new_initial_condition_sampling_points
        target = lambda start, batch: model.new_initial_conditions[start:start + len(batch)]
    model.loss_ic_u, model.loss_ic_zeta = mean_error(points, target, [0, 1])
    model.loss_ic = model.loss_ic_u + model.loss_ic_zeta
    with torch.no_grad():
        model.compute_reference_error()
        model.compute_total_loss()
