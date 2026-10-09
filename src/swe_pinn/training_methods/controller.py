"""Shared training strategies; equation residuals remain in physics.py."""

from copy import deepcopy
import math

import torch


def validate_methods(config):
    methods = config.get("training_methods", {})
    if not methods:
        return
    if methods["causal_weighting"] not in ("none", "residual", "attention"):
        raise ValueError(
            "causal_weighting must be none, residual or attention"
        )
    if methods["sampling"] not in (
        "lhs",
        "residual",
        "causal_as",
        "r3",
        "causal_r3",
    ):
        raise ValueError("Unknown training_methods.sampling")
    for key in (
        "time_bins",
        "probe_points",
        "candidate_points",
        "update_period",
    ):
        if type(methods[key]) is not int or methods[key] < 1:
            raise ValueError(
                f"training_methods.{key} must be a positive integer"
            )
    if methods["probe_points"] < methods["time_bins"]:
        raise ValueError("probe_points must be at least time_bins")
    if methods["power_schedule"] not in ("fixed", "temporal_alignment"):
        raise ValueError("power_schedule must be fixed or temporal_alignment")
    if not 0 < methods["alignment_rate"] < 1:
        raise ValueError("alignment_rate must be in (0, 1)")
    if not 0.5 <= methods["alignment_threshold"] < 1:
        raise ValueError("alignment_threshold must be in [0.5, 1)")
    for key in (
        "epsilon",
        "attention_scale",
        "gate_steepness",
        "gate_learning_rate",
        "gate_tolerance",
        "gate_max_step",
        "alignment_steepness",
    ):
        value = methods[key]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"training_methods.{key} must be positive and finite"
            )
    if not 0 < methods["uniform_fraction"] <= 1:
        raise ValueError("uniform_fraction must be in (0, 1]")
    if (
        not math.isfinite(methods["sampling_power"])
        or methods["sampling_power"] < 0
    ):
        raise ValueError("sampling_power must be nonnegative and finite")
    if not math.isfinite(methods["gate_initial"]):
        raise ValueError("gate_initial must be finite")
    if (
        methods["sampling"] == "causal_as"
        and methods["causal_weighting"] != "residual"
    ):
        raise ValueError("causal_as requires causal_weighting: residual")
    if (
        methods["sampling"] == "causal_r3"
        and methods["causal_weighting"] != "none"
    ):
        raise ValueError("causal_r3 supplies its own temporal weighting")
    if (
        methods["causal_weighting"] == "attention"
        and config["network"]["enforce_ic"]
    ):
        raise ValueError("Causal attention needs a soft IC learning error")
    active = (
        methods["sampling"] != "lhs" or methods["causal_weighting"] != "none"
    )
    if config["equation"] == "swe" and active:
        if not config["loss"]["use_pde"]:
            raise ValueError("Training strategies require the SWE PDE loss")
        if not config["output"]["batch_diagnostics"]:
            raise ValueError(
                "SWE training strategies require batch_diagnostics: true"
            )
        if (
            methods["causal_weighting"] == "attention"
            and not config["loss"]["use_ic"]
        ):
            raise ValueError("Causal attention requires a soft IC loss")


class TrainingStrategy:
    """Weights are frozen for each optimizer step, including all line searches.

    Temporal statistics use fixed, stratified diagnostic probes. Adaptive
    sampling uses a separate CPU generator and never expands the point budget.
    """

    def __init__(self, settings, bounds, seed, batch_size):
        self.settings = deepcopy(settings or {})
        self.active = bool(settings) and (
            settings["causal_weighting"] != "none"
            or settings["sampling"] != "lhs"
        )
        self.bounds = torch.tensor(bounds, dtype=torch.float32)
        self.batch_size = batch_size
        self.generator = None
        self.probes = None
        self.weights = None
        self.bin_losses = None
        self.ic_error = 0.0
        self.gate = settings["gate_initial"] if self.active else 0.0
        self.residual_evaluations = 0
        self.updates = 0
        self.sampling_power = (
            settings["sampling_power"] if self.active else 1.0
        )
        self.previous_adaptive_time = None
        self.condition_probes = None
        self.last_pde_objective = None
        if self.active:
            self.generator = torch.Generator().manual_seed(seed)
            self.probes = self.sample(settings["probe_points"])
            bins = settings["time_bins"]
            # One or more probes in every ordered time slice.
            strata = torch.arange(len(self.probes)) % bins
            jitter = torch.rand(len(self.probes), generator=self.generator)
            self.probes[:, 0] = self.bounds[0, 0] + (
                (strata + jitter) / bins
            ) * (self.bounds[0, 1] - self.bounds[0, 0])
            self.weights = torch.ones(bins)

    def sample(self, count):
        points = torch.rand(count, 2, generator=self.generator)
        return self.bounds[:, 0] + points * (
            self.bounds[:, 1] - self.bounds[:, 0]
        )

    def normalized_time(self, points):
        bounds = self.bounds[0].to(points.device)
        return ((points[:, 0] - bounds[0]) / (bounds[1] - bounds[0])).clamp(
            0, 1
        )

    def bin_indices(self, points):
        return (
            (self.normalized_time(points) * self.settings["time_bins"])
            .long()
            .clamp(max=self.settings["time_bins"] - 1)
        )

    def point_weights(self, points):
        if self.settings["sampling"] == "causal_r3":
            time = self.normalized_time(points)
            return torch.sigmoid(
                -2 * self.settings["gate_steepness"] * (time - self.gate)
            ).detach()
        if self.settings["causal_weighting"] == "attention":
            return torch.exp(
                -self.settings["attention_scale"]
                * self.ic_error
                * self.normalized_time(points)
            ).detach()
        if self.settings["causal_weighting"] == "none":
            return torch.ones(len(points), device=points.device)
        return self.weights.to(points.device)[
            self.bin_indices(points)
        ].detach()

    def pde_loss(self, points, squared_residual):
        return (
            squared_residual.reshape(-1) * self.point_weights(points)
        ).mean()

    def scores(self, model, points):
        values = []
        device = next(model.parameters()).device
        for batch in points.split(self.batch_size):
            batch = batch.detach().to(device).requires_grad_(True)
            with torch.enable_grad():
                residual = model.pde_residual(batch)
                if isinstance(residual, tuple):
                    score = sum(value.square() for value in residual)
                else:
                    score = residual.square()
            values.append(score.detach().cpu().reshape(-1))
        scores = torch.cat(values)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite adaptive-sampling residual")
        self.residual_evaluations += len(points)
        return scores

    def prepare(
        self,
        model,
        initial_points,
        initial_targets,
        boundary_points,
        boundary_targets,
        ic_weight=1.0,
        bc_weight=1.0,
    ):
        """Refresh temporal weights once, before the optimizer closure."""
        if not self.active:
            return
        if self.condition_probes is None:
            pairs = [(initial_points, initial_targets)] + list(
                zip(boundary_points, boundary_targets)
            )
            self.condition_probes = []
            for points, targets in pairs:
                count = min(len(points), self.settings["probe_points"])
                indices = torch.linspace(0, len(points) - 1, count).long()
                self.condition_probes.append(
                    (
                        points.detach().cpu()[indices],
                        targets.detach().cpu()[indices],
                    )
                )
        initial_points, initial_targets = self.condition_probes[0]
        boundary_points = [pair[0] for pair in self.condition_probes[1:]]
        boundary_targets = [pair[1] for pair in self.condition_probes[1:]]
        device = next(model.parameters()).device
        scores = self.scores(model, self.probes)
        indices = self.bin_indices(self.probes)
        bins = self.settings["time_bins"]
        counts = torch.bincount(indices, minlength=bins)
        losses = torch.zeros(bins).scatter_add_(0, indices, scores) / counts

        def errors(points, targets):
            chunks = []
            with torch.no_grad():
                for start in range(0, len(points), self.batch_size):
                    batch = points[start : start + self.batch_size].to(device)
                    target = targets[start : start + len(batch)].to(device)
                    predicted = model(batch[:, :1], batch[:, 1:2])
                    if target.shape[1] == 1 and predicted.shape[1] == 2:
                        predicted = predicted[:, :1]
                    chunks.append((predicted - target).square().sum(1).cpu())
            return torch.cat(chunks)

        ic = errors(initial_points, initial_targets)
        energy = initial_targets.square().sum(1).mean().item()
        self.ic_error = math.sqrt(float(ic.mean()) / max(energy, 1e-12))
        for points, targets in zip(boundary_points, boundary_targets):
            error = errors(points, targets)
            ids = self.bin_indices(points.cpu())
            counts_bc = torch.bincount(ids, minlength=bins).clamp_min(1)
            losses += (
                bc_weight
                * torch.zeros(bins).scatter_add_(0, ids, error)
                / counts_bc
            )
        self.bin_losses = losses
        cumulative = torch.cat((torch.zeros(1), losses.cumsum(0)[:-1]))
        self.weights = torch.exp(
            -self.settings["epsilon"] * (cumulative + ic_weight * ic.mean())
        )

    def finish_step(self, loss):
        if self.active:
            self.last_pde_objective = float(loss.detach())
        if self.active and self.settings["sampling"] == "causal_r3":
            speed = min(
                math.exp(
                    -self.settings["gate_tolerance"] * float(loss.detach())
                ),
                self.settings["gate_max_step"],
            )
            self.gate = min(
                1.5, self.gate + self.settings["gate_learning_rate"] * speed
            )

    def adapt(self, model, pool):
        """Replace adaptive points; retain a configurable coverage population."""
        method = self.settings["sampling"]
        if not self.active or method == "lhs":
            return pool
        pool = pool.detach().cpu()
        count = len(pool)
        coverage = max(1, math.ceil(count * self.settings["uniform_fraction"]))
        adaptive_count = count - coverage
        if method in ("r3", "causal_r3"):
            scores = self.scores(model, pool).sqrt()
            if method == "causal_r3":
                scores *= self.point_weights(pool)
            retained = torch.where(scores > scores.mean())[0]
            if len(retained) > adaptive_count:
                retained = retained[
                    scores[retained].topk(adaptive_count).indices
                ]
            selected = pool[retained]
            fresh = self.sample(count - len(selected))
        else:
            candidates = self.sample(
                max(adaptive_count, self.settings["candidate_points"])
            )
            scores = self.scores(model, candidates)
            if method == "causal_as":
                if (
                    self.settings["power_schedule"] == "temporal_alignment"
                    and self.previous_adaptive_time is not None
                ):
                    below = torch.where(
                        self.weights < self.settings["alignment_threshold"]
                    )[0]
                    frontier = (
                        (float(below[0]) + 0.5) / self.settings["time_bins"]
                        if len(below)
                        else 1.0
                    )
                    self.sampling_power *= 1 + self.settings[
                        "alignment_rate"
                    ] * math.tanh(
                        self.settings["alignment_steepness"]
                        * (self.previous_adaptive_time / frontier - 1)
                    )
                scores *= self.point_weights(candidates).pow(
                    self.sampling_power
                )
            selected = candidates[scores.topk(adaptive_count).indices]
            fresh = (
                pool[-coverage:]
                if method == "causal_as"
                else self.sample(coverage)
            )
            if len(selected):
                self.previous_adaptive_time = float(
                    self.normalized_time(selected).mean()
                )
        self.updates += 1
        return torch.cat((selected, fresh))

    def state_dict(self):
        if not self.active:
            return None
        return {
            "probes": self.probes,
            "weights": self.weights,
            "bin_losses": self.bin_losses,
            "ic_error": self.ic_error,
            "gate": self.gate,
            "updates": self.updates,
            "sampling_power": self.sampling_power,
            "previous_adaptive_time": self.previous_adaptive_time,
            "condition_probes": self.condition_probes,
            "last_pde_objective": self.last_pde_objective,
            "residual_evaluations": self.residual_evaluations,
            "rng": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        if state is None:
            return
        for key, value in state.items():
            if key == "rng":
                self.generator.set_state(value.cpu())
            else:
                setattr(self, key, value)

    def summary(self):
        bins = self.settings["time_bins"]
        centers = self.bounds[0, 0] + ((torch.arange(bins) + 0.5) / bins) * (
            self.bounds[0, 1] - self.bounds[0, 0]
        )
        points = torch.stack((centers, torch.zeros(bins)), dim=1)
        return {
            "gate": self.gate,
            "ic_relative_l2": self.ic_error,
            "weights": self.point_weights(points).tolist(),
            "sampling_power": self.sampling_power,
            "last_pde_objective": self.last_pde_objective,
            "sampling_updates": self.updates,
            "residual_evaluations": self.residual_evaluations,
        }


def prepare_swe(model):
    """Use actual segment IC targets, including the soft network handoff."""
    if model.segment_id == 0:
        initial = model.ic_dataset.tx.detach()
        targets = torch.cat(
            (
                model.initial_u(initial[:, 1:2]),
                model.initial_zeta(initial[:, 1:2]),
            ),
            dim=1,
        )
    else:
        initial = model.new_initial_condition_sampling_points.detach()
        targets = model.new_initial_conditions.detach()
    boundaries = [model.lbc_dataset.tx.detach(), model.ubc_dataset.tx.detach()]
    zero = [torch.zeros(len(points), 1) for points in boundaries]
    model.training_strategy.prepare(
        model,
        initial,
        targets,
        boundaries,
        zero,
        (
            float(model.weight_ic)
            if model.use_ic_loss
            and not model.initial_condition_transition_function
            else 0.0
        ),
        (
            float(model.weight_bc)
            if model.use_bc_loss
            and not model.boundary_condition_transition_function
            else 0.0
        ),
    )
