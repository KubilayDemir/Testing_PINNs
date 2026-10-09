# New runs

Each equation has one complete `config.yaml`. Copy it to define a new case,
or edit it and run from the repository root:

```bash
python -m swe_pinn train
python -m swe_pinn train --equation heat
python -m swe_pinn train --equation burgers
python -m swe_pinn train --config path/to/config.yaml --run-id my-run
```

SWE uses the included Fortran numerical reference. Heat diffusion has an
analytic reference; viscous Burgers uses `numerical_reference/burgers/shock.mat`.
Heat and Burgers are optional benchmarks, outside the SWE publication.
Publication cases and trained SWE models are under `paper/`.

Heat/Burgers `pretrained/` folders preserve supplied weights and their compatible
model configurations. Training histories were not supplied. The heat weights
use eight layers of 50; the new-run default uses nine. Burgers uses nine layers
of 20. New outputs go to `runs/<experiment>/<run-id>/`.
