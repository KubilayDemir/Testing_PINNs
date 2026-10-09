# Heat diffusion test case

This optional case is outside the SWE publication. It uses the same `PINN`
class with the heat-equation residual and an analytic reference solution.

```bash
python -m swe_pinn train --equation heat
```

`config.yaml` contains all new-run settings. `supplied_models/trained.pt`
contains the supplied weights; its adjacent configuration describes their
eight-layer architecture. The new-run default has nine layers. Training
histories for the supplied weights were not provided.
