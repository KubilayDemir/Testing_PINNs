"""Dimensional and nondimensional SWE residuals describe the same PDE."""

from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from swe_pinn.config import model_kwargs
from swe_pinn.physics import swe_residuals


@pytest.mark.parametrize("vertical_factor", [1.0, 2.5])
def test_residual_scaling(setup_run, vertical_factor):
    config, reference, _ = setup_run
    config["scales"]["vertical_factor"] = vertical_factor
    config["physics"]["drag"] = 0.002
    scaled = model_kwargs(config, reference)
    dimensional = deepcopy(config)
    dimensional["scales"]["nondimensionalize"] = False
    physical = model_kwargs(dimensional, reference)
    length, time = config["scales"]["length"], config["scales"]["time"]
    height = scaled["vertical_length_scale"]

    def field(t, x):
        return torch.cat(
            (
                0.1 + 0.02 * t / time + 0.03 * (x / length) ** 2,
                0.2 + 0.04 * (t / time) ** 2 + 0.05 * (x / length) ** 3,
            ),
            dim=1,
        )

    class Model:
        def __init__(self, settings, nondim):
            self.__dict__.update(settings)
            self.nondim = nondim

        def __call__(self, t, x):
            if self.nondim:
                output = field(t * time, x * length)
                return torch.cat(
                    (output[:, :1] * time / length, output[:, 1:] / height),
                    dim=1,
                )
            return field(t, x)

    t = torch.tensor([[0.2], [0.8]], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[-0.4], [0.6]], dtype=torch.float64, requires_grad=True)
    rm, rc = swe_residuals(Model(scaled, True), t, x)
    pm, pc = swe_residuals(Model(physical, False), t * time, x * length)
    torch.testing.assert_close(
        rm, pm * time**2 / length, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(rc, pc * time / height, rtol=1e-12, atol=1e-12)
    assert scaled["momentum_dissipation"] == pytest.approx(
        50000 * time / length**2
    )
    assert scaled["gravitational_acceleration"] == pytest.approx(
        vertical_factor
    )


@pytest.mark.parametrize("transition", ["TF2", "TF3"])
def test_initial_transition_and_soft_segment_handoff(setup_run, transition):
    from swe_pinn.training import create_model

    config, reference, _ = setup_run
    config["network"].update(
        enforce_bc=True, enforce_ic=True, ic_transition=transition
    )
    model, _ = create_model(config, reference)
    x = torch.linspace(-1, 1, 9).reshape(-1, 1)
    output = model(torch.zeros_like(x), x)
    torch.testing.assert_close(output[:, :1], torch.zeros_like(x))
    torch.testing.assert_close(output[:, 1:], model.initial_zeta(x))
    config["domain"]["n_time_segments"] = 2
    assert (
        model_kwargs(config, reference, 1)[
            "initial_condition_transition_function"
        ]
        is False
    )
