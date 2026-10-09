"""Explicit training loop; model formulation and losses live in model/physics."""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .optimizers.lbfgs import FullBatchLBFGS


def fit(model, resume=None):
    """Train without changing working directory or requiring a batch scheduler."""
    names = (
        "loss_pde_history",
        "loss_bc_history",
        "loss_ic_history",
        "mse_ref_zeta_history",
        "mse_ref_u_history",
        "loss_total_history",
        "loss_ic_u_history",
        "loss_ic_zeta_history",
        "loss_momentum_history",
        "loss_continuity_history",
        "loss_bc_left_history",
        "loss_bc_right_history",
        "velocity_history",
        "elevation_history",
        "momentum_residual_history",
        "continuity_residual_history",
        "pde_weight_history",
        "ic_weight_history",
        "bc_weight_history",
    )
    for name in names:
        setattr(model, name, [])
    model.history_names = names
    model.history_epochs, model.snapshot_epochs = [], []
    model.weight_steps = []
    model.time_per_epoch, model.time_per_iteration = [], []
    from .training_checkpoints import restore_resume, save_resume, save_torch

    completed = restore_resume(model, resume) if resume else 0
    if resume is None:
        model._record_epoch(0)
        model._save_best(0, float(model.mse_ref_u + model.mse_ref_zeta))
    for epoch in range(completed + 1, model.epochs + 1):
        started = time.perf_counter()
        if epoch > 1 and epoch % model.batch_resampling_period == 0:
            model.resample_sampling_points()
        if not (model.batch_diagnostics and model.minibatch_training):
            model._use_full_pool()
        datasets = (
            model.pde_dataset,
            model.lbc_dataset,
            model.ubc_dataset,
            model.ic_dataset,
        )
        strategy = model.training_strategy
        if strategy.active:
            from .training_methods.controller import prepare_swe

            prepare_swe(model)
            if (epoch - 1) % strategy.settings["update_period"] == 0:
                dataset = model.pde_dataset
                dataset.tx = strategy.adapt(model, dataset.tx).requires_grad_(
                    True
                )
                dataset.t, dataset.x = dataset.tx[:, :1], dataset.tx[:, 1:2]
            if not model.minibatch_training:
                model._use_full_pool()
        sizes = (
            model.pde_batch_size,
            model.bc_batch_size,
            model.bc_batch_size,
            model.ic_batch_size,
        )
        loaders = [
            DataLoader(
                ds.tx.detach() if model.dataloader_workers else ds,
                batch_size=size,
                shuffle=True,
                num_workers=model.dataloader_workers,
                pin_memory=model.pin_memory,
            )
            for ds, size in zip(datasets, sizes)
        ]
        iterators = [iter(loader) for loader in loaders]
        for iteration in range(model.iterations_per_epoch):
            step_start = time.perf_counter()
            if model.minibatch_training:
                batches = []
                for index, loader in enumerate(loaders):
                    try:
                        batch = next(iterators[index])
                    except StopIteration:
                        iterators[index] = iter(loader)
                        batch = next(iterators[index])
                    batches.append(
                        batch.detach()
                        .to(model.device, non_blocking=model.pin_memory)
                        .requires_grad_(True)
                    )
                (
                    model.pde_points,
                    model.left_boundary_points,
                    model.right_boundary_points,
                    model.initial_condition_sampling_points,
                ) = batches
            if strategy.active:
                prepare_swe(model)
            if model.use_pcgrad and not model.train_on_solution:
                model.closure()
                losses = []
                if model.use_pde_loss:
                    losses.append(model.weight_pde * model.loss_pde)
                if (
                    model.use_ic_loss
                    and not model.initial_condition_transition_function
                ):
                    losses.append(model.weight_ic * model.loss_ic)
                if (
                    model.use_bc_loss
                    and not model.boundary_condition_transition_function
                ):
                    losses.append(model.weight_bc * model.loss_bc)
                model.optimizer.pc_backward(losses)
                model.optimizer.step()
            elif isinstance(model.optimizer, FullBatchLBFGS):
                # This optimizer gathers existing gradients before line search;
                # Wolfe then computes its own backward pass on candidate losses.
                model.closure()

                def objective():
                    adaptive = model.adaptive_loss_weights
                    model.adaptive_loss_weights = False
                    try:
                        return model.closure(backward=False)
                    finally:
                        model.adaptive_loss_weights = adaptive

                model.optimizer.step(
                    options={
                        "closure": objective,
                        "current_loss": model.loss_total.detach(),
                    }
                )
            elif isinstance(model.optimizer, torch.optim.LBFGS):
                model.optimizer.step(model.closure)
            else:
                model.closure()
                model.optimizer.step()
            if not torch.isfinite(model.loss_total):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}"
                )
            strategy.finish_step(model.loss_pde)
            if model.adaptive_loss_weights:
                model.save_weight_history()
                model.weight_steps.append(
                    (epoch - 1) * model.iterations_per_epoch + iteration + 1
                )
            model.time_per_iteration.append(time.perf_counter() - step_start)
        # Evaluate every epoch so output frequency cannot hide a better checkpoint.
        with torch.no_grad():
            model.compute_reference_error()
        score = float(model.mse_ref_u + model.mse_ref_zeta)
        if not np.isfinite(score):
            raise FloatingPointError(
                f"Non-finite reference error at epoch {epoch}"
            )
        if score < model.best_score:
            model._save_best(epoch, score)
        if epoch % model.output_period == 0 or epoch == model.epochs:
            model._record_epoch(epoch)
            print(
                f"Epoch {epoch}/{model.epochs}: "
                f"loss={float(model.loss_total.detach()):.6g}, "
                f"reference_mse_sum={score:.6g}"
            )
        elif epoch % model.snapshot_period == 0:
            model._record_snapshot(epoch)
        model.time_per_epoch.append(time.perf_counter() - started)
        if strategy.active and model.checkpoint_dir is not None:
            from .output_storage import write_json

            write_json(
                model.checkpoint_dir.parent / "training_strategy.json",
                dict(epoch=epoch, **strategy.summary()),
            )
        if model.history_writer is not None:
            model.history_writer.timing(model, epoch)
        if (
            model.checkpoint_dir is not None
            and model.checkpoint_period
            and epoch % model.checkpoint_period == 0
        ):
            save_torch(
                model.state_dict(),
                model.checkpoint_dir / f"epoch_{epoch:06d}.pt",
            )
        if model.resume_period and (
            epoch % model.resume_period == 0 or epoch == model.epochs
        ):
            save_resume(model, epoch)
    if not model.batch_diagnostics:
        model._use_full_pool()
