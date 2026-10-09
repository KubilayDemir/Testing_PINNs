# Setup and memory utilities

Run these from the repository root:

| Script | Purpose |
|---|---|
| `setup_fortran.ps1` | Install a local Windows Fortran compiler under ignored `.tools/` |
| `check_gpu_memory.py` | Measure one proposed SWE update and batched diagnostics on an allocated GPU |
| `convert_fortran_output.py` | Convert solver text output into the training NetCDF format |

The reference-generation workflow is documented under `reference_solutions/shallow_water/`.
