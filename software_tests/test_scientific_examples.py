"""Scientific demonstrations run only when explicitly selected with -m scientific."""

import pytest
from swe_pinn.equation_checks import check


@pytest.mark.scientific
@pytest.mark.parametrize(
    "case", ["still_water", "manufactured_fields", "linear_standing_wave"]
)
def test_prescribed_fields(case):
    result = check(case)
    assert result["momentum_max_error"] < 1e-10
    assert result["continuity_max_error"] < 1e-10
