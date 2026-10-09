# Heat and Burgers benchmarks

These optional cases are not part of the SWE publication.

```bash
python -m swe_pinn train --equation heat
python -m swe_pinn train --equation burgers
```

Both use the shared `PINN` class. Heat diffusion uses an analytic reference;
viscous Burgers uses `numerical_reference/burgers/shock.mat`.
Edit the complete settings in `configs/heat.yaml` or `configs/burgers.yaml`.

`heat/trained.pt` and `burgers/trained.pt` preserve supplied benchmark weights.
Their adjacent `config.yaml` describes the compatible model; training histories
were not supplied. The heat checkpoint has eight hidden layers of 50, while
the new-run default has nine. Burgers has nine hidden layers of 20.

Prescribed-field SWE checks are optional:
`python -m swe_pinn check-physics --case still_water`.
