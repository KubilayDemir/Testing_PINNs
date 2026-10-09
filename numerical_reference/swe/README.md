# SWE numerical reference

`solutions/viscous.nc` and `solutions/inviscid.nc` are included generated
references: dx=400 m, dt=1 s, 75 hours, saved every 60 s on a 10 km staggered grid.
Their NetCDF attributes record solver settings. They are distinct from the published Zenodo data.

To generate another solution, install/load gfortran and select a new output:

```bash
python numerical_reference/swe/generate.py --case viscous --output runs/reference/viscous.nc
python numerical_reference/swe/generate.py --case inviscid --output runs/reference/inviscid.nc
python numerical_reference/swe/convergence.py --study dx --case viscous
```

For Windows, run `powershell -ExecutionPolicy Bypass -File scripts/setup_fortran.ps1`.
The generator detects that local compiler automatically. On clusters, load the GCC module.
The semi-implicit solver uses upstream advection and explicit viscosity; flags
control grid, timestep, duration and output spacing. Existing outputs are not overwritten.
