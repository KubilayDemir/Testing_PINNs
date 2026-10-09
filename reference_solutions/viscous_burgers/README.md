# Viscous Burgers numerical reference

`viscous_burgers_reference_nu_0p01_over_pi.mat` is the supplied numerical
solution used by the optional Burgers case, with viscosity `0.01/pi`. It stores
the time grid, spatial grid and reference field in MATLAB format. The loader
reads these arrays directly; the shared PINN predicts the field on the same
grid for error calculation. This reference is outside the SWE publication.
