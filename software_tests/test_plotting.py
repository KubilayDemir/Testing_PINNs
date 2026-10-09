"""Figure scales must support direct comparisons, including exact solutions."""

import numpy as np
from matplotlib.colors import LogNorm
from swe_pinn.plotting import solution_panels, plt


def test_shared_limits_and_zero_error():
    field = np.zeros((3, 4))
    fig = solution_panels(np.arange(4), np.arange(3), field, field, limit=1)
    try:
        reference, prediction, error = [
            axis.collections[0] for axis in fig.axes[:3]
        ]
        assert reference.norm is prediction.norm
        assert (reference.norm.vmin, reference.norm.vmax) == (-1, 1)
        assert isinstance(error.norm, LogNorm)
        assert np.isfinite(error.get_array()).all()
        assert not field.any()
    finally:
        plt.close(fig)
