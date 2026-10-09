"""Opt-in, readable scientific checks; never part of paper training defaults."""

import numpy as np
import torch
from torch.autograd import grad

from .physics import swe_residuals


class PrescribedFields:
    """Supply known fields in place of a neural network for derivative checks."""
    nondimensionalize = False
    momentum_advection = True
    gravitational_acceleration = 9.81
    average_sea_level = 100.0
    momentum_dissipation = 0.5
    nonlinear_drag_coefficient = 0.0
    horizontal_length_scale = 1.0

    def __init__(self, function):
        self.function = function

    def __call__(self, t, x):
        return self.function(t, x)


def check(case):
    t = torch.tensor([[0.1], [0.4], [0.7]], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[-0.8], [0.1], [0.6]], dtype=torch.float64, requires_grad=True)
    if case == "still_water":
        # Zero fields retain derivative connectivity without introducing a value.
        zero = lambda t, x: torch.cat((0 * (t ** 2 + x ** 3), 0 * (t ** 2 + x ** 3)), dim=1)
        momentum, continuity = swe_residuals(PrescribedFields(zero), t, x)
        expected_momentum, expected_continuity = torch.zeros_like(t), torch.zeros_like(t)
        formulation = "nonlinear SWE, zero velocity and elevation"
    elif case == "manufactured_fields":
        model = PrescribedFields(lambda t, x: torch.cat((t + x ** 2, 2 * t + x ** 3), dim=1))
        momentum, continuity = swe_residuals(model, t, x)
        expected_momentum = 1 + 9.81 * 3 * x ** 2 + (t + x ** 2) * 2 * x - 0.5 * 2
        expected_continuity = 2 + (2 * t + x ** 3 + 100) * 2 * x + (t + x ** 2) * 3 * x ** 2
        formulation = "nonlinear SWE, independently derived polynomial residuals"
    elif case == "linear_standing_wave":
        # This checks explicitly LINEAR equations, not the nonlinear paper PDE.
        depth, gravity = 2.0, 3.0
        speed, wave_number = np.sqrt(depth * gravity), np.pi
        frequency = speed * wave_number
        zeta = torch.cos(wave_number * x) * torch.cos(frequency * t)
        u = speed / depth * torch.sin(wave_number * x) * torch.sin(frequency * t)
        momentum = grad(u.sum(), t, create_graph=True)[0] + gravity * grad(zeta.sum(), x, create_graph=True)[0]
        continuity = grad(zeta.sum(), t, create_graph=True)[0] + depth * grad(u.sum(), x, create_graph=True)[0]
        expected_momentum, expected_continuity = torch.zeros_like(t), torch.zeros_like(t)
        formulation = "linear inviscid SWE: u_t + g*zeta_x = 0, zeta_t + d*u_x = 0"
    else:
        raise ValueError("Unknown scientific example")
    errors = {"momentum_max_error": float((momentum - expected_momentum).abs().max().detach()),
              "continuity_max_error": float((continuity - expected_continuity).abs().max().detach())}
    if max(errors.values()) > 1e-10:
        raise AssertionError(errors)
    return {"case": case, "formulation": formulation, **errors}
