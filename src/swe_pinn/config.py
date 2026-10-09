"""Strict YAML settings with dimensional inputs and derived model arguments."""

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import math

import yaml

CASE_DIRECTORIES = {"swe": "shallow_water", "heat": "heat_diffusion", "burgers": "viscous_burgers"}


def default_case_path(equation):
    return Path("training_cases") / CASE_DIRECTORIES[equation] / "config.yaml"


def _merge(base, overrides, prefix=""):
    if not isinstance(overrides, dict):
        raise ValueError(f"{prefix or 'configuration'} must be a mapping")
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            expected = base[key]
            if isinstance(expected, bool) and not isinstance(value, bool):
                raise ValueError(f"{prefix}{key} must be a YAML boolean")
            if (
                    isinstance(expected, int)
                    and not isinstance(expected, bool)
                    and (not isinstance(value, int) or isinstance(value, bool))
            ):
                raise ValueError(f"{prefix}{key} must be an integer")
            if isinstance(expected, float) and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
            ):
                raise ValueError(f"{prefix}{key} must be a finite number")
            if isinstance(expected, str) and not isinstance(value, str):
                raise ValueError(f"{prefix}{key} must be a string")
            if isinstance(expected, list) and not isinstance(value, list):
                raise ValueError(f"{prefix}{key} must be a list")
            base[key] = value
    return base


def load_config(path, overrides=None):
    path = Path(path).resolve()
    base = yaml.safe_load(
        files("swe_pinn").joinpath("defaults.yaml").read_text()
    )
    loaded = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    config = _merge(base, {} if loaded is None else loaded)
    for extra in overrides or []:
        loaded = yaml.safe_load(Path(extra).read_text(encoding="utf-8-sig"))
        _merge(config, {} if loaded is None else loaded)
    validate(config)
    return config


def validate(c):
    if c["equation"] != "swe":
        raise ValueError("This configuration is for equation: swe")
    if c["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema_version")
    if (
            not isinstance(c["experiment"], str)
            or not c["experiment"]
            or c["experiment"] in (".", "..")
            or Path(c["experiment"]).name != c["experiment"]
    ):
        raise ValueError("experiment must be a single directory name")
    for section, keys in {
        "scales": ("length", "time", "vertical_factor"),
        "physics": ("gravity", "mean_depth"),
        "sampling": ("n_pde", "n_bc", "n_ic", "resample_period"),
        "training": (
                "epochs",
                "pde_batch_size",
                "bc_batch_size",
                "ic_batch_size",
                "report_period",
                "learning_rate",
        ),
        "output": ("snapshot_period", "evaluation_batch_size"),
        "domain": ("n_time_segments",),
    }.items():
        for key in keys:
            value = c[section][key]
            if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(value)
                    or value <= 0
            ):
                raise ValueError(
                    f"{section}.{key} must be positive and finite"
                )
    for key in ("time", "space"):
        bounds = c["domain"][key]
        if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or not all(
            isinstance(v, (int, float)) and math.isfinite(v)
            for v in bounds
        )
                or bounds[0] >= bounds[1]
        ):
            raise ValueError(
                f"domain.{key} must contain two increasing finite values"
            )
    # The inherited transition functions assume initial time zero and a symmetric
    # domain.
    if c["domain"]["time"][0] != 0:
        raise ValueError(
            "The Gaussian initial condition requires domain.time to begin at zero"
        )
    if c["network"]["enforce_bc"] and (
            c["domain"]["space"][0] != -c["domain"]["space"][1]
    ):
        raise ValueError(
            "Boundary transition functions require a symmetric spatial domain"
        )
    layers = c["network"]["hidden_layers"]
    if (
            not isinstance(layers, list)
            or not layers
            or not all(type(n) is int and n > 0 for n in layers)
    ):
        raise ValueError(
            "network.hidden_layers must be a nonempty list of positive integers"
        )
    if c["training"]["optimizer"] not in ("Adam", "LBFGS", "FullBatchLBFGS"):
        raise ValueError("optimizer must be Adam, LBFGS, or FullBatchLBFGS")
    if c["network"]["activation"] not in ("tanh", "sin", "sigmoid"):
        raise ValueError("activation must be tanh, sin, or sigmoid")
    if c["network"]["ic_transition"] not in ("TF2", "TF3"):
        raise ValueError("network.ic_transition must be TF2 or TF3")
    if c["domain"]["n_time_segments"] > 1 and not c["loss"]["use_ic"]:
        raise ValueError("Time decomposition requires the soft IC handoff loss")
    if c["training"]["use_pcgrad"] and c["training"]["optimizer"] != "Adam":
        raise ValueError("PCGrad requires optimizer: Adam")
    search = c["training"]["line_search"]
    allowed = (
        (None, "Armijo", "Wolfe")
        if c["training"]["optimizer"] == "FullBatchLBFGS"
        else (None, "strong_wolfe")
    )
    if search not in allowed:
        raise ValueError(f"line_search must be one of {allowed}")
    if c["physics"]["viscosity"] < 0 or c["physics"]["drag"] < 0:
        raise ValueError("viscosity and drag must be nonnegative")
    if type(c["seed"]) is not int or not 0 <= c["seed"] < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if (
            c["training"]["adaptive_loss_weights"]
            and c["training"]["optimizer"] == "LBFGS"
    ):
        raise ValueError(
            (
                "Adaptive loss weights cannot change the objective durin"
                "g PyTorch LBFGS line search"
            )
        )
    if type(c["cpu_threads"]) is not int or c["cpu_threads"] < 1:
        raise ValueError("cpu_threads must be a positive integer")
    for section, key in (
            ("training", "dataloader_workers"), ("training", "resume_period"),
            ("output", "checkpoint_period"),
    ):
        if type(c[section][key]) is not int or c[section][key] < 0:
            raise ValueError(f"{section}.{key} must be a nonnegative integer")
    active = (
            c["loss"]["use_reference"]
            or c["loss"]["use_pde"]
            or (c["loss"]["use_bc"] and not c["network"]["enforce_bc"])
            or (c["loss"]["use_ic"] and not c["network"]["enforce_ic"])
    )
    if not active:
        raise ValueError("At least one unconstrained loss must be enabled")
    for key in ("weight_pde", "weight_bc", "weight_ic"):
        if (
                not isinstance(c["loss"][key], (int, float))
                or not math.isfinite(c["loss"][key])
                or c["loss"][key] <= 0
        ):
            raise ValueError(f"loss.{key} must be positive and finite")
    for key in ("n_pde", "n_bc"):
        if c["sampling"][key] < c["domain"]["n_time_segments"]:
            raise ValueError(
                f"sampling.{key} must be at least n_time_segments"
            )
    if c["training"]["epochs"] < c["domain"]["n_time_segments"]:
        raise ValueError("epochs must be at least n_time_segments")
    iterations = c["training"]["iterations_per_epoch"]
    if iterations is not None and (
            type(iterations) is not int or iterations < 1
    ):
        raise ValueError(
            "iterations_per_epoch must be null or a positive integer"
        )


def model_kwargs(
        config, reference_path, segment_id=0, previous=None, checkpoint_dir=None
):
    """Translate user settings to the inherited numerical implementation."""
    import torch
    from .optimizers.lbfgs import FullBatchLBFGS

    c = deepcopy(config)
    p, s, d, n, loss, sample, train, out = (
        c[k]
        for k in (
        "physics",
        "scales",
        "domain",
        "network",
        "loss",
        "sampling",
        "training",
        "output",
    )
    )
    vertical = (
            s["vertical_factor"]
            * s["length"] ** 2
            / (s["time"] ** 2 * p["gravity"])
    )
    count = d["n_time_segments"]
    bounds = [
        d["time"][0] + index / count * (d["time"][1] - d["time"][0])
        for index in (segment_id, segment_id + 1)
    ]
    nondim = s["nondimensionalize"]
    x_scale, t_scale, h_scale = (
        (s["length"], s["time"], vertical) if nondim else (1.0, 1.0, 1.0)
    )
    n_pde = sample["n_pde"] // count
    epochs = train["epochs"] // count + int(
        segment_id < train["epochs"] % count
    )
    return dict(
        layer_sizes=[2, *n["hidden_layers"], 2],
        activation_function=getattr(torch, n["activation"]),
        activation_checkpointing=n["activation_checkpointing"],
        optimizer={
            "Adam": torch.optim.Adam,
            "LBFGS": torch.optim.LBFGS,
            "FullBatchLBFGS": FullBatchLBFGS,
        }[train["optimizer"]],
        learning_rate=train["learning_rate"],
        line_search=train["line_search"],
        weight_bc=loss["weight_bc"],
        weight_ic=loss["weight_ic"],
        weight_pde=loss["weight_pde"],
        n_bc_points=sample["n_bc"] // count,
        n_ic_points=sample["n_ic"],
        n_pde_points=n_pde,
        training_steps=epochs,
        batch_resampling_period=sample["resample_period"],
        console_output_period=train["report_period"],
        device=c["device"],
        gravitational_acceleration=p["gravity"] * h_scale * t_scale ** 2 / x_scale ** 2,
        average_sea_level=p["mean_depth"] / h_scale,
        momentum_dissipation=p["viscosity"] * t_scale / x_scale ** 2,
        nonlinear_drag_coefficient=p["drag"],
        initial_perturbation_amplitude=p["amplitude"] / h_scale,
        nondimensionalize=nondim,
        vertical_length_scale=vertical,
        vertical_scaling_factor=s["vertical_factor"],
        horizontal_length_scale=s["length"],
        time_scale=s["time"],
        t_min=bounds[0] / t_scale,
        t_max=bounds[1] / t_scale,
        x_min=d["space"][0] / x_scale,
        x_max=d["space"][1] / x_scale,
        use_pcgrad=train["use_pcgrad"],
        save_predictions=out["save_predictions"],
        save_residuals=out["save_residuals"],
        numerical_solution_time_interval=d["time"],
        numerical_solution_x_interval=d["space"],
        fraction_of_time_interval=[
            segment_id / count,
            (segment_id + 1) / count,
        ],
        segment_id=segment_id,
        n_time_segments=count,
        new_initial_conditions=previous[0] if previous else None,
        new_initial_condition_sampling_points=(
            previous[1] if previous else None
        ),
        train_on_solution=loss["use_reference"],
        use_pde_loss=loss["use_pde"],
        use_bc_loss=loss["use_bc"],
        use_ic_loss=loss["use_ic"],
        boundary_condition_transition_function=n["enforce_bc"],
        initial_condition_transition_function=n["enforce_ic"] and segment_id == 0,
        ic_transition=n["ic_transition"],
        split_networks=n["split_networks"],
        momentum_advection=p["momentum_advection"],
        mixed_activation_functions=n["mixed_activations"],
        sirens_initialization=n["siren_init"],
        adaptive_loss_weights=train["adaptive_loss_weights"],
        minibatch_training=train["minibatch"],
        pde_batch_size=train["pde_batch_size"],
        bc_batch_size=train["bc_batch_size"],
        ic_batch_size=train["ic_batch_size"],
        iterations_per_epoch=train["iterations_per_epoch"]
                             or (
                                 max(1, n_pde // train["pde_batch_size"])
                                 if train["minibatch"]
                                 else 1
                             ),
        reference_path=str(reference_path),
        checkpoint_dir=checkpoint_dir,
        snapshot_period=out["snapshot_period"],
        evaluation_batch_size=out["evaluation_batch_size"],
        dataloader_workers=train["dataloader_workers"],
        pin_memory=train["pin_memory"],
        resume_period=train["resume_period"],
        reference_on_cpu=train["reference_on_cpu"],
        stream_history=out["stream_history"],
        batch_diagnostics=out["batch_diagnostics"],
        keep_improving_checkpoints=out["keep_improving_checkpoints"],
        checkpoint_period=out["checkpoint_period"],
    )
