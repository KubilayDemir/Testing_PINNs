# Software tests

Run `python -m pytest` from the repository root. These checks verify residuals,
scaling, training, checkpoint recovery, output files and cluster commands. They
use small networks and short runs; they do not reproduce convergence results.

`settings/shallow_water_quick_software_check.yaml` describes a two-epoch test.
Such a short execution check is sometimes called a *smoke test*. Its synthetic
input is artificial data used to test the software, not a numerical SWE
solution. Pytest's *fixtures* create reusable inputs in temporary directories.

To run the short configuration manually:

```bash
python -m swe_pinn make-test-data
check=software_tests/settings/shallow_water_quick_software_check.yaml
python -m swe_pinn train --config "$check"
```

Optional prescribed-field checks compare derivatives with known formulas:
`python -m swe_pinn check-physics --case still_water`. Use `python -m pytest -m
scientific` for those additional checks.
