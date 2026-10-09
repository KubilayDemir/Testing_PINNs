"""Signed SWE residuals (paper Equations 9–10), in model coordinates.

The second network output is elevation zeta, not total depth h = d + zeta.
Coefficients are already scaled by config.model_kwargs for nondimensional runs.
"""

import torch
from torch.autograd import grad


def benchmark_residual(model, points):
    """Heat: u_t - diffusivity*u_xx; Burgers adds u*u_x."""
    t, x = points[:, :1], points[:, 1:2]
    u = model(t, x)
    u_t = grad(u.sum(), t, create_graph=True)[0]
    u_x = grad(u.sum(), x, create_graph=True)[0]
    u_xx = grad(u_x.sum(), x, create_graph=True)[0]
    if model.equation == "burgers":
        return u_t + u * u_x - model.diffusivity * u_xx
    return u_t - model.diffusivity * u_xx


def swe_residuals(model, t, x):
    output = model(t, x)
    u, zeta = output[:, :1], output[:, 1:2]
    u_t = grad(u.sum(), t, create_graph=True)[0]
    u_x = grad(u.sum(), x, create_graph=True)[0]
    u_xx = grad(u_x.sum(), x, create_graph=True)[0]
    zeta_t = grad(zeta.sum(), t, create_graph=True)[0]
    zeta_x = grad(zeta.sum(), x, create_graph=True)[0]
    pressure = (
        model.vertical_scaling_factor
        if model.nondimensionalize
        else model.gravitational_acceleration
    )
    drag_scale = model.horizontal_length_scale if model.nondimensionalize else 1
    momentum = (
            u_t
            + float(model.momentum_advection) * u * u_x
            + pressure * zeta_x
            + drag_scale * model.nonlinear_drag_coefficient * u * torch.abs(u)
            - model.momentum_dissipation * u_xx
    )
    continuity = zeta_t + u_x * (zeta + model.average_sea_level) + u * zeta_x
    return momentum, continuity
