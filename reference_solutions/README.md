# Reference solutions

Reference solutions are comparison fields for evaluating PINN accuracy. They
enter training losses only when reference supervision is explicitly enabled.

| Folder | Contents |
|---|---|
| [shallow_water/](shallow_water/README.md) | Included viscous/inviscid NetCDF solutions, Fortran generator and convergence studies |
| [viscous_burgers/](viscous_burgers/README.md) | Supplied MATLAB numerical reference for the Burgers test case |

Heat diffusion uses an analytic formula in the training code and needs no file.
Synthetic data under `software_tests/` serve software checks rather than accuracy studies.
