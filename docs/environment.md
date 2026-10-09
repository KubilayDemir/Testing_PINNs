# Environment setup

Run installation commands from the repository root. `environment.yaml` creates
a Python 3.14 environment with the project, tests, JupyterLab and its Python kernel:

```bash
conda env create -f environment.yaml
conda activate testing-pinns
python scripts/check_environment.py --notebooks
```

After changing dependencies, synchronize the environment with
`conda env update -f environment.yaml`. Activation alone does not install packages.

For an existing Python environment (Python 3.10 or newer):

```bash
python -m pip install -e ".[test,notebooks]"
python -m pip check
python scripts/check_environment.py --notebooks
```

`pyproject.toml` declares the runtime dependencies: PyTorch, NumPy, SciPy, SMT
sampling, Matplotlib, PyYAML, xarray and netCDF4. Tests and notebook tools are
optional extras; omit them with `pip install -e .` for training only.
The live loss viewer uses the Python standard library.

`requirements-tested.txt` records the installed Windows CPU environment's
versions. It is an environment snapshot, rather than a CUDA or cross-platform lock.
Every training run records its own resolved packages and hardware.

## GPU and cluster environments

Select a CUDA-compatible PyTorch build using the
[official installer](https://pytorch.org/get-started/locally/), matching the cluster
driver and Python environment. Then install the project into that same environment.
An existing compatible PyTorch installation satisfies the project dependency.
Check GPU execution inside an allocated GPU job:

```bash
python scripts/check_environment.py --gpu
```

The check performs a small CUDA/autograd operation and reports the GPU. CPU
execution remains available with `--device cpu`. CUDA availability on a login
node does not determine availability inside a GPU allocation.

## Reference generation and notebooks

Included references allow training without a Fortran compiler. Regenerating SWE
references requires gfortran: load your cluster's compiler module or use
`scripts/setup_fortran.ps1` on Windows. Verify it with
`python scripts/check_environment.py --fortran`.

Launch `python -m jupyterlab` from the activated environment, or register it for
another Jupyter installation:

```bash
python -m ipykernel install --user --name testing-pinns --display-name "Testing PINNs"
```
