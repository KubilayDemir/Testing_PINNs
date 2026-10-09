"""Check the packages and numerical operations used by training and analysis."""

import argparse
import importlib
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", action="store_true", help="also check JupyterLab and ipykernel")
    parser.add_argument("--gpu", action="store_true", help="require a working CUDA GPU")
    parser.add_argument("--fortran", action="store_true", help="require a working gfortran compiler")
    args = parser.parse_args()
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    packages = [("torch", "torch"), ("numpy", "numpy"), ("scipy.io", "scipy"),
                ("smt.sampling_methods", "smt"), ("matplotlib", "matplotlib"),
                ("yaml", "PyYAML"), ("xarray", "xarray"), ("netCDF4", "netCDF4")]
    if args.notebooks:
        packages += [("jupyterlab", "jupyterlab"), ("ipykernel", "ipykernel")]
    failures = []
    for module, distribution in packages:
        try:
            importlib.import_module(module)
            print(f"{distribution}: {version(distribution)}")
        except Exception as error:
            failures.append(f"{distribution}: {error}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    import numpy as np
    import torch
    import xarray as xr
    try:
        value = torch.from_numpy(np.array([1.0, 2.0])).requires_grad_()
        gradient = torch.autograd.grad(value.square().sum(), value)[0]
        torch.testing.assert_close(gradient, 2 * value)
        with tempfile.TemporaryDirectory(prefix="pinn-environment-") as directory:
            path = Path(directory) / "check.nc"
            xr.Dataset({"value": ("point", [1.0, 2.0])}).to_netcdf(path, engine="netcdf4")
            with xr.open_dataset(path, engine="netcdf4") as data:
                np.testing.assert_array_equal(data.value.values, [1.0, 2.0])
        print("NumPy/PyTorch autograd and NetCDF round trip: passed")
        print(f"CUDA runtime: {torch.version.cuda}; available: {torch.cuda.is_available()}")
        if args.gpu:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in this environment/allocation")
            value = torch.tensor([1.0, 2.0], device="cuda:0", requires_grad=True)
            torch.testing.assert_close(torch.autograd.grad(value.square().sum(), value)[0], 2 * value)
            torch.cuda.synchronize()
            print(f"CUDA autograd: passed ({torch.cuda.get_device_name(0)})")
        if args.fortran:
            compiler = shutil.which("gfortran")
            local = Path(__file__).resolve().parents[1] / ".tools/compiler/mingw64/bin/gfortran.exe"
            if compiler is None and local.is_file():
                compiler = str(local)
            if compiler is None:
                raise RuntimeError("gfortran is unavailable; load/install the compiler")
            result = subprocess.run([compiler, "--version"], check=True, capture_output=True, text=True)
            print(result.stdout.splitlines()[0])
    except Exception as error:
        print(f"Environment check failed: {error}", file=sys.stderr)
        return 1
    print("Environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
