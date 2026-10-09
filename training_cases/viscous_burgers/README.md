# Viscous Burgers test case

This optional case is outside the SWE publication. It uses the same `PINN`
class with the viscous Burgers residual and a supplied numerical reference for
viscosity `0.01/pi`, under `reference_solutions/viscous_burgers/`.

```bash
python -m swe_pinn train --equation burgers
```

`config.yaml` contains all new-run settings. `supplied_models/` contains
supplied weights and their compatible configuration. Training histories were
not provided. A different viscosity requires a matching reference solution.
