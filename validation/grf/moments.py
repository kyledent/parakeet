#!/usr/bin/env python3
"""
moments.py -- decide whether the GRF ice's pixel-area corrections
are a bug or a compensation, before anyone tries to fix them.

The implicit ice scales its target mean by an empirical table, Cv_mean, which
falls from 1.0 to 0.488 at 1 A/px and 0.083 at 2 A/px. A stationary medium's
mean projected potential does not depend on sampling, so this looks wrong. But
the GRF is ADDED TO MULTEM's atomic potential, and if that path has the same
pixel-size dependence then Cv_mean is keeping the two consistent and removing
it would break them apart.

Parakeet contains its own ground truth for this: it can build ice out of real
water molecules. So the question is answerable without MD software, without
literature, and without a microscope.

    explicit water, GRF off   ->  how the ATOMIC path behaves with sampling
    empty box,     GRF on     ->  how the ICE MODEL behaves with sampling

Run both at several pixel sizes and compare the slopes.

    validation/grf/moments.py --outdir /path/to/scratch
    validation/grf/moments.py --outdir ... --angpix 0.5 1.0 2.0
    validation/grf/moments.py --outdir ... --pack --json moments_packed.json

--pack places the explicit water by hard-sphere packing (sample.ice.pack)
instead of uniformly at random, so no two molecules overlap: a closer stand-in
for real water at short range, which matters most for the variance. The
engine's band-limited potential (MULTEM_BANDLIMITED_POTENTIAL, on by default)
is recorded in the output; set it to 0 in the environment to reproduce the
numbers measured before it existed.

Outcomes:
  atomic flat, GRF falls   -> the tables are wrong; recalibrate against the
                              atomic path (this script's own output gives it)
  both fall together       -> compensation; the system is self-consistent, and
                              the question moves to whether it matches reality
  they fall differently    -> both need work, and the difference is the error
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

# Small box: this measures statistics, not structure, so it only has to be
# large enough that edge effects are a small fraction.
BOX_A = 200.0
THICKNESS_A = 200.0


def config(angpix, explicit_ice, outdir, pack=False):
    nx = int(round(BOX_A / angpix))
    return {
        "microscope": {
            "beam": {"energy": 300, "electrons_per_angstrom": 30.0},
            "lens": {"c_10": -20000, "c_30": 2.7, "c_c": 2.7},
            "detector": {"nx": nx, "ny": nx, "pixel_size": angpix},
        },
        "sample": {
            "box": [BOX_A, BOX_A, THICKNESS_A],
            "centre": [BOX_A / 2, BOX_A / 2, THICKNESS_A / 2],
            "shape": {
                "type": "cuboid",
                "cuboid": {
                    "length_x": BOX_A,
                    "length_y": BOX_A,
                    "length_z": THICKNESS_A,
                },
            },
            # explicit water molecules, the ground truth the GRF should match
            **(
                {"ice": {"generate": True, "density": 940.0, "pack": pack}}
                if explicit_ice
                else {}
            ),
        },
        "scan": {"mode": "still", "num_images": 1},
        "simulation": {
            "ice": not explicit_ice,  # implicit GRF only in the other arm
            "slice_thickness": 3.0,
            # The GRF used to sit margin * pixel_size above the atoms in z; it
            # no longer does (z_register.py), but a margin of 0 keeps both arms
            # to the same box and costs nothing here.
            "margin": 0,
            # This script measures the ice model where its tables run out, so
            # the guard against that warns instead of refusing
            "allow_uncalibrated_sampling": True,
        },
        "multiprocessing": {"device": "gpu", "nproc": 1, "gpu_id": [0]},
    }


def measure(angpix, explicit_ice, outdir, pack=False):
    """Mean and variance of the projected potential per unit thickness."""
    tag = f"{'atomic' if explicit_ice else 'grf'}_{angpix:g}"
    if explicit_ice and pack:
        tag += "_packed"
    d = Path(outdir) / tag
    d.mkdir(parents=True, exist_ok=True)
    cfg, sample, pot = d / "c.yaml", d / "s.h5", d / "potential"
    cfg.write_text(
        yaml.safe_dump(config(angpix, explicit_ice, outdir, pack), sort_keys=False)
    )

    for cmd in (
        ["parakeet", "sample", "new", "-c", str(cfg), "-s", str(sample)],
        [
            "parakeet",
            "simulate",
            "potential",
            "-c",
            str(cfg),
            "-s",
            str(sample),
            "-p",
            str(pot),
        ],
    ):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(
                f"  FAILED {tag}: {' '.join(cmd)}\n{r.stderr[-800:]}", file=sys.stderr
            )
            return None

    import mrcfile

    slices = sorted(d.glob("potential_*.mrc"))
    if not slices:
        print(f"  {tag}: no potential slices written", file=sys.stderr)
        return None
    vals = []
    for s in slices:
        with mrcfile.open(s, permissive=True) as f:
            vals.append(np.asarray(f.data, dtype=np.float64).ravel())
    v = np.concatenate(vals)
    return {
        "angpix": angpix,
        "n_slices": len(slices),
        "mean": float(v.mean()),
        "var": float(v.var()),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--angpix", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5]
    )
    ap.add_argument(
        "--pack", action="store_true", help="hard-sphere packed explicit water"
    )
    ap.add_argument("--json", help="where to write the results (default outdir)")
    args = ap.parse_args()

    import os

    results = {
        "atomic": [],
        "grf": [],
        "packed": args.pack,
        "band_limited": os.environ.get("MULTEM_BANDLIMITED_POTENTIAL", "1") != "0",
    }
    for angpix in args.angpix:
        for explicit in (True, False):
            r = measure(angpix, explicit, args.outdir, args.pack)
            if r:
                results["atomic" if explicit else "grf"].append(r)
                print(
                    f"  {'atomic' if explicit else 'grf   '} {angpix:4.2f} A/px  "
                    f"mean {r['mean']:12.5g}  var {r['var']:12.5g}"
                )

    out = Path(args.json) if args.json else Path(args.outdir) / "moments.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\n  -> {out}")

    ratios = [
        (a["angpix"], g["mean"] / a["mean"], g["var"] / a["var"])
        for a, g in zip(results["atomic"], results["grf"])
        if a["angpix"] == g["angpix"]
    ]
    results["grf_over_atomic"] = [
        {"angpix": p, "mean": m, "var": v} for p, m, v in ratios
    ]
    out.write_text(json.dumps(results, indent=1))
    print("\n  GRF / atomic:   angpix    mean     var")
    for p, m, v in ratios:
        print(f"                 {p:5.2f}  {m:7.4f} {v:7.4f}")

    for name in ("atomic", "grf"):
        rows = results[name]
        if len(rows) < 2:
            continue
        base = rows[0]
        print(f"\n  {name}: mean relative to {base['angpix']:g} A/px")
        for r in rows:
            print(f"    {r['angpix']:4.2f} A/px  {r['mean']/base['mean']:8.4f}")
    print("\n  If the atomic column is flat and the GRF column falls, the Cv tables")
    print("  are wrong and this run's atomic column is the calibration to use.")


if __name__ == "__main__":
    sys.exit(main())
