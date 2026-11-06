#!/usr/bin/env python3
"""
z_register.py -- is the GRF ice where the sample shape is, in z, in a render?

The ice is placed by the masker at the sample centre plus the lateral offset
of the simulated area (margin and padding). That offset must move the ice in
x and y only, as it moves the atoms. When it was added to z as well, the ice
slab rendered (margin + padding) * pixel_size above the shape, and the
validation scripts worked around it with margin 0. tests/test_ice_register.py
checks the arithmetic; this checks a render.

It renders the projected potential of an EMPTY cuboid of GRF ice with a
non-zero margin, and reads two things:

  * the z-range of the slices MULTEM computes (it slices only where there is
    ice or atoms; Parakeet's callback prints each slice's z0 -> z1);
  * which slices of the written potential volume, which spans exactly the
    shape's z-range, carry any ice at all.

The ice is in register when the computed slices span the shape's z-range to
within one slice and every slice whose midpoint is inside the shape carries
ice. The masker tests a slice at its midpoint against the half-open shape, so
a slice whose midpoint falls exactly on a face is decided by rounding; it is
reported as a boundary slice and not required.

    validation/grf/z_register.py --outdir /path/to/scratch --json z_register.json
    validation/grf/z_register.py --outdir ... --margin 0 100 --angpix 1.5

Needs a GPU build of the engine unless --device cpu.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

BOX_XY_A = 200.0
BOX_Z_A = 300.0
THICKNESS_A = 200.0
SLICE_A = 3.0

SLICE_LINE = re.compile(
    r"Calculating potential for slice: (-?[\d.]+) -> (-?[\d.]+) \(index: (-?\d+)\)"
)


def config(angpix, margin, device):
    nx = int(round(BOX_XY_A / angpix))
    centre = [BOX_XY_A / 2, BOX_XY_A / 2, BOX_Z_A / 2]
    return {
        "microscope": {
            "beam": {"energy": 300, "electrons_per_angstrom": 30.0},
            "lens": {"c_10": -20000, "c_30": 2.7, "c_c": 2.7},
            "detector": {"nx": nx, "ny": nx, "pixel_size": angpix},
        },
        "sample": {
            "box": [BOX_XY_A, BOX_XY_A, BOX_Z_A],
            "centre": centre,
            "shape": {
                "type": "cuboid",
                "cuboid": {
                    "length_x": BOX_XY_A,
                    "length_y": BOX_XY_A,
                    "length_z": THICKNESS_A,
                },
            },
        },
        "scan": {"mode": "still", "num_images": 1},
        "simulation": {"ice": True, "slice_thickness": SLICE_A, "margin": margin},
        "multiprocessing": {"device": device, "nproc": 1, "gpu_id": [0]},
    }


# Through the Python API rather than the command line: the command line also
# imports the analysis commands, whose dependencies a simulation does not need.
RUN = """
import sys
import parakeet.sample
import parakeet.simulate
config, sample, prefix = sys.argv[1:4]
parakeet.sample.new(config, sample)
parakeet.simulate.potential(config, sample, prefix)
"""


def render(cfg, sample, prefix):
    """Make the sample and render its potential in a fresh interpreter"""
    return subprocess.run(
        [sys.executable, "-c", RUN, str(cfg), str(sample), str(prefix)],
        capture_output=True,
        text=True,
    )


def measure(angpix, margin, device, outdir):
    tag = f"px{angpix:g}_margin{margin}"
    d = Path(outdir) / tag
    d.mkdir(parents=True, exist_ok=True)
    cfg, sample, pot = d / "c.yaml", d / "s.h5", d / "potential"
    cfg.write_text(yaml.safe_dump(config(angpix, margin, device), sort_keys=False))

    r = render(cfg, sample, pot)
    if r.returncode != 0:
        return {"tag": tag, "error": r.stderr[-800:]}

    computed = [
        (float(a), float(b), int(i)) for a, b, i in SLICE_LINE.findall(r.stdout)
    ]
    import mrcfile

    with mrcfile.open(d / "potential_0.mrc", permissive=True) as f:
        volume = np.asarray(f.data, dtype=np.float64)
    std = volume.reshape(volume.shape[0], -1).std(axis=1)

    shape_z0 = BOX_Z_A / 2 - THICKNESS_A / 2
    shape_z1 = BOX_Z_A / 2 + THICKNESS_A / 2
    z0 = min(c[0] for c in computed) if computed else None
    z1 = max(c[1] for c in computed) if computed else None
    with_ice = int((std > 0).sum())

    # The masker tests each slice at its midpoint against the half-open shape
    # [z0, z1). A slice whose midpoint falls on a face is decided by rounding,
    # so it is reported and not required. Every slice whose midpoint is inside
    # must carry ice.
    def midpoint(c):
        return (c[0] + c[1]) / 2

    boundary = [
        c[2]
        for c in computed
        if min(abs(midpoint(c) - shape_z0), abs(midpoint(c) - shape_z1)) < 1e-3
    ]
    required = [
        c[2]
        for c in computed
        if c[2] not in boundary
        and shape_z0 < midpoint(c) < shape_z1
        and 0 <= c[2] < len(std)
    ]
    missing = [i for i in required if std[i] == 0]
    in_register = (
        computed != []
        and abs(z0 - shape_z0) <= SLICE_A
        and abs(z1 - shape_z1) <= SLICE_A
        and not missing
    )
    return {
        "tag": tag,
        "angpix": angpix,
        "margin": margin,
        "offset_A": margin * angpix,
        "shape_z": [shape_z0, shape_z1],
        "computed_z": [z0, z1],
        "computed_slices": len(computed),
        "computed_out_of_volume": sum(
            1 for c in computed if c[2] < 0 or c[2] >= len(std)
        ),
        "volume_slices": int(len(std)),
        "volume_slices_with_ice": with_ice,
        "first_slice_with_ice": int(np.argmax(std > 0)) if with_ice else None,
        "volume_slices_without_ice": [int(i) for i in np.flatnonzero(std == 0)],
        "boundary_slices": boundary,
        "required_slices_without_ice": missing,
        "computed_indices_repeated": sorted(
            {c[2] for c in computed if sum(1 for d in computed if d[2] == c[2]) > 1}
        ),
        "computed": [list(c) for c in computed],
        "in_register": bool(in_register),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--angpix", type=float, nargs="+", default=[1.5])
    ap.add_argument("--margin", type=int, nargs="+", default=[0, 100])
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()

    results = []
    for angpix in args.angpix:
        for margin in args.margin:
            row = measure(angpix, margin, args.device, args.outdir)
            results.append(row)
            if "error" in row:
                print(f"{row['tag']}: FAILED {row['error']}")
                continue
            print(
                f"{row['tag']}: shape z {row['shape_z']}, computed z "
                f"{row['computed_z']} ({row['computed_slices']} slices, "
                f"{row['computed_out_of_volume']} outside the volume), "
                f"ice in {row['volume_slices_with_ice']}/{row['volume_slices']} "
                f"volume slices, boundary {row['boundary_slices']} "
                f"-> {'IN REGISTER' if row['in_register'] else 'OUT OF REGISTER'}"
            )
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    ok = all(r.get("in_register") for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
