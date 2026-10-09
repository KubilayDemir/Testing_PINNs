"""Measure one SWE Adam update and batched diagnostics on the selected GPU."""

import argparse
import json
from pathlib import Path

import torch

from swe_pinn.config import load_config, model_kwargs
from swe_pinn.diagnostics import report_losses
from swe_pinn.model import PINN
from swe_pinn.training import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/swe.yaml"))
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpointing", action="store_true")
    args = parser.parse_args()
    if args.layers < 1 or args.width < 1:
        parser.error("layers and width must be positive")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("This check needs CUDA; run it on the allocated cluster GPU")
    config = load_config(args.config)
    if config["training"]["optimizer"] != "Adam":
        parser.error("This probe supports Adam; choose an Adam configuration")
    config["device"] = args.device
    config["network"].update(hidden_layers=[args.width] * args.layers,
                              activation_checkpointing=args.checkpointing)
    seed_everything(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    reference = Path(config["reference"]).resolve()
    model = PINN(**model_kwargs(config, reference)).to(args.device)
    pools = (model.pde_dataset, model.lbc_dataset, model.ubc_dataset, model.ic_dataset)
    sizes = (model.pde_batch_size, model.bc_batch_size, model.bc_batch_size, model.ic_batch_size)
    attributes = ("pde_points", "left_boundary_points", "right_boundary_points",
                  "initial_condition_sampling_points")
    for attribute, pool, size in zip(attributes, pools, sizes):
        setattr(model, attribute, pool.tx[:size].detach().to(args.device).requires_grad_(True))
    model.closure()
    if model.use_pcgrad and not model.train_on_solution:
        losses = []
        if model.use_pde_loss:
            losses.append(model.weight_pde * model.loss_pde)
        if model.use_ic_loss and not model.initial_condition_transition_function:
            losses.append(model.weight_ic * model.loss_ic)
        if model.use_bc_loss and not model.boundary_condition_transition_function:
            losses.append(model.weight_bc * model.loss_bc)
        model.optimizer.pc_backward(losses)
    model.optimizer.step()
    if not torch.isfinite(model.loss_total):
        raise FloatingPointError("Nonfinite probe loss")
    model.adaptive_loss_weights = False
    report_losses(model)
    torch.cuda.synchronize(args.device)
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(args.device),
        "hidden_layers": config["network"]["hidden_layers"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "pde_batch_size": model.pde_batch_size,
        "diagnostic_batch_size": model.evaluation_batch_size,
        "activation_checkpointing": model.activation_checkpointing,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(args.device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(args.device) / 2**20,
        "gpu_total_mib": torch.cuda.get_device_properties(args.device).total_memory / 2**20,
    }, indent=2))


if __name__ == "__main__":
    main()
