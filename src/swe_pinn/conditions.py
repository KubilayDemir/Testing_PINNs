# import External Libraries
import numpy as np
import torch

# define true Initial and Boundary Conditions for the sea level elevation zeta and the
# zonal velocity u


def initial_zeta(
    initial_perturbation_amplitude,
    nondimensionalize,
    horizontal_length_scale,
):
    def f(x):
        mu = 0.0
        sigma = 100000.0
        pi = torch.tensor(np.pi)

        if nondimensionalize == True:
            x = x * horizontal_length_scale

        initial_elevation_target = (
            2.5
            * initial_perturbation_amplitude
            * (
                sigma
                * (1.0 / (sigma * torch.sqrt(2.0 * pi)))
                * torch.exp(-0.5 * ((x - mu) / sigma) ** 2.0)
            )
        )

        return initial_elevation_target

    return f


def initial_u(x):
    # the zonal velocity is zero everywhere at the initial state
    initial_velocity_target = torch.zeros_like(x)
    return initial_velocity_target


def boundary_u_right(t):
    # the zonal velocity is zero at the boundaries to ensure closed boundaries
    right_boundary_target = torch.zeros_like(t)
    return right_boundary_target


def boundary_u_left(t):
    # the zonal velocity is zero at the boundaries to ensure closed boundaries
    left_boundary_target = torch.zeros_like(t)
    return left_boundary_target
