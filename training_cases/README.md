# Training cases

Each equation has one complete `config.yaml`. Copy it to define a new case,
or edit it and run from the repository root:

```bash
python -m swe_pinn train
python -m swe_pinn train --equation heat
python -m swe_pinn train --equation burgers
python -m swe_pinn train --config path/to/config.yaml --run-id my-run
```

SWE uses the included Fortran numerical reference. Heat diffusion has an
analytic reference; viscous Burgers uses `reference_solutions/viscous_burgers/viscous_burgers_reference_nu_0p01_over_pi.mat`.
Heat and Burgers are optional benchmarks, outside the SWE publication.
Publication cases and trained SWE models are under `publication_cases/`.

| Folder | Role |
|---|---|
| [shallow_water/](shallow_water/README.md) | Default SWE training; publication setups are in `publication_cases/` |
| [heat_diffusion/](heat_diffusion/README.md) | Optional heat-equation test case |
| [viscous_burgers/](viscous_burgers/README.md) | Optional Burgers-equation test case |

Heat/Burgers `supplied_models/` folders preserve supplied weights and their compatible
model configurations. Training histories were not supplied. The heat weights
use eight layers of 50; the new-run default uses nine. Burgers uses nine layers
of 20. New outputs go to `training_results/<experiment>/<run-id>/`.
