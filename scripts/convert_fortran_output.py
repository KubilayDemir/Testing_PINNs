"""Convert existing Fortran text snapshots without repeatedly copying history.

Default row selection preserves every raw row, including boundary/ghost rows.
Use explicit slices only after checking the grid. This does not run or alter
the solver, infer timestamps, or interpolate its output.
"""

import argparse
from pathlib import Path
import json

import numpy as np


def convert(source, output, count, u_slice=None, zeta_slice=None):
    source, output = Path(source), Path(output)
    if count < 1:
        raise ValueError("count must be positive")
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for field, subdir, prefix, indices in (
            ("u", "udata", "u", u_slice), ("zeta", "zdata", "z", zeta_slice),
    ):
        selection = slice(*indices) if indices else slice(None)
        first = np.loadtxt(source / subdir / f"{prefix}0000.dat", ndmin=1)[selection]
        if not len(first):
            raise ValueError("Selected spatial slice is empty")
        path = output / f"{field}.npy"
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(len(first), count))
        for index in range(count):
            values = np.loadtxt(source / subdir / f"{prefix}{index:04d}.dat", ndmin=1)[selection]
            if values.shape != first.shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid {field} snapshot {index}")
            array[:, index] = values
        array.flush()
        del array
        records[field] = {"file": path.name, "shape": [len(first), count], "row_slice": indices}
    (output / "conversion.json").write_text(json.dumps({
        "source": str(source.resolve()), "fields": records,
        "note": "Values preserved; verify coordinates, ghost rows and solver time labeling before import-reference.",
    }, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="impdat directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True, help="number of snapshots, including index 0000")
    parser.add_argument("--u-slice", nargs=3, type=int, metavar=("START", "STOP", "STEP"))
    parser.add_argument("--zeta-slice", nargs=3, type=int, metavar=("START", "STOP", "STEP"))
    args = parser.parse_args()
    print(convert(args.source, args.output, args.count, args.u_slice, args.zeta_slice))


if __name__ == "__main__":
    main()
