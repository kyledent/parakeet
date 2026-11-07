#!/usr/bin/env python3
"""
spectrum.py -- decide WHERE the GRF ice's variance shortfall lives, so the fix
targets the right thing.

moments.py established that the implicit ice reproduces explicit water's mean
projected potential to within 2% across sampling, but undershoots its variance
by 22% at 0.5 A/px, 37% at 1.0 and 67% at 2.0. Variance is the integral of the
power spectrum, so that deficit sits somewhere in frequency, and where it sits
decides the repair:

    short by a constant factor at all frequencies
        -> the amplitude is wrong; fix V0 or the Cv_var table

    short only at high frequency
        -> the two-Gaussian power spectrum is missing power, most likely near
           the 2.88 A water peak the second Gaussian is meant to represent;
           fix the spectrum shape (a1, a2, m1, m2, s1, s2)

    short only at low frequency
        -> the long-range component is wrong, which would show up as missing
           mottle rather than missing grain

This builds the same box twice -- once from explicit water molecules with the
GRF off, once from the GRF on an empty box -- projects both, and compares
radially averaged power spectra of the projected potential.

    validation/grf/spectrum.py --outdir /path/to/scratch
    validation/grf/spectrum.py --outdir ... --angpix 1.0 --box 400
    validation/grf/spectrum.py --outdir ... --pack --json spectrum_packed.json

--pack places the explicit water by hard-sphere packing (sample.ice.pack), so no
two molecules overlap: random placement overlaps them, which most affects the
high-frequency comparison near the 2.88 A water peak.

Note --margin 0 is forced: the GRF slab is z-offset from the atoms by
margin * pixel_size, so any other value compares two different places.
"""

import argparse
import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def config(angpix, box, thickness, explicit_ice, pack=False):
    nx = int(round(box / angpix))
    return {
        "microscope": {
            "beam": {"energy": 300, "electrons_per_angstrom": 30.0},
            "lens": {"c_10": -20000, "c_30": 2.7, "c_c": 2.7},
            "detector": {"nx": nx, "ny": nx, "pixel_size": angpix},
        },
        "sample": {
            "box": [box, box, thickness],
            "centre": [box / 2, box / 2, thickness / 2],
            "shape": {
                "type": "cuboid",
                "cuboid": {"length_x": box, "length_y": box, "length_z": thickness},
            },
            **(
                {"ice": {"generate": True, "density": 940.0, "pack": pack}}
                if explicit_ice
                else {}
            ),
        },
        "scan": {"mode": "still", "num_images": 1},
        "simulation": {
            "ice": not explicit_ice,
            "slice_thickness": 3.0,
            "margin": 0,
            "allow_uncalibrated_sampling": True,
        },
        "multiprocessing": {"device": "gpu", "nproc": 1, "gpu_id": [0]},
    }


def project(angpix, box, thickness, explicit_ice, outdir, pack=False):
    """Return the projected potential summed over slices, as a 2D array."""
    tag = f"{'atomic' if explicit_ice else 'grf'}_{angpix:g}"
    if explicit_ice and pack:
        tag += "_packed"
    d = Path(outdir) / tag
    d.mkdir(parents=True, exist_ok=True)
    cfg, sample, pot = d / "c.yaml", d / "s.h5", d / "potential"
    cfg.write_text(
        yaml.safe_dump(
            config(angpix, box, thickness, explicit_ice, pack), sort_keys=False
        )
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
            print(f"  FAILED {tag}\n{r.stderr[-800:]}", file=sys.stderr)
            return None

    import mrcfile

    total = None
    for s in sorted(d.glob("potential_*.mrc")):
        with mrcfile.open(s, permissive=True) as f:
            a = np.asarray(f.data, dtype=np.float64)
            a = a[0] if a.ndim == 3 else a
        total = a if total is None else total + a
    return total


def radial_power(image, angpix, n_bins=48):
    """Radially averaged power spectrum of the mean-subtracted image."""
    a = image - image.mean()
    f = np.fft.fftshift(np.abs(np.fft.fft2(a)) ** 2) / a.size
    ny, nx = a.shape
    qy = np.fft.fftshift(np.fft.fftfreq(ny, d=angpix))
    qx = np.fft.fftshift(np.fft.fftfreq(nx, d=angpix))
    q = np.hypot(*np.meshgrid(qy, qx, indexing="ij"))
    qmax = min(qy.max(), qx.max())
    edges = np.linspace(0, qmax, n_bins + 1)
    idx = np.digitize(q.ravel(), edges) - 1
    power, centres = [], []
    flat = f.ravel()
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() > 8:
            power.append(flat[sel].mean())
            centres.append(0.5 * (edges[b] + edges[b + 1]))
    return np.array(centres), np.array(power)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--angpix", type=float, default=1.0)
    ap.add_argument("--box", type=float, default=300.0)
    ap.add_argument("--thickness", type=float, default=200.0)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument(
        "--pack", action="store_true", help="hard-sphere packed explicit water"
    )
    ap.add_argument("--json", help="where to write the results (default outdir)")
    args = ap.parse_args()

    atomic = project(
        args.angpix, args.box, args.thickness, True, args.outdir, args.pack
    )
    grf = project(args.angpix, args.box, args.thickness, False, args.outdir)
    if atomic is None or grf is None:
        return 1

    qa, pa = radial_power(atomic, args.angpix)
    qg, pg = radial_power(grf, args.angpix)
    ratio = pg / pa

    print(
        f"\n  Projected potential, {args.box:g} A box, {args.thickness:g} A thick, "
        f"{args.angpix:g} A/px"
    )
    print(
        f"  variance   atomic {atomic.var():12.5g}   grf {grf.var():12.5g}   "
        f"ratio {grf.var()/atomic.var():.3f}\n"
    )

    hdr = ["q (1/A)", "d (A)", "atomic power", "grf power", "grf/atomic"]
    rows = []
    for q, a_, g_, r in zip(qa, pa, pg, ratio):
        if q <= 0:
            continue
        rows.append([f"{q:.3f}", f"{1/q:.1f}", f"{a_:.4g}", f"{g_:.4g}", f"{r:.3f}"])
    step = max(1, len(rows) // 16)
    rows = rows[::step]
    if args.markdown:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join("---" for _ in hdr) + "|")
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        print("  " + "".join(h.rjust(14) for h in hdr))
        for r in rows:
            print("  " + "".join(c.rjust(14) for c in r))

    lo = ratio[(qa > 0) & (qa < 0.1)]
    hi = ratio[qa > 0.3]
    print(f"\n  mean ratio below 0.1 1/A (>10 A):  {lo.mean():.3f}" if lo.size else "")
    print(f"  mean ratio above 0.3 1/A (<3.3 A): {hi.mean():.3f}" if hi.size else "")
    print("\n  A flat ratio means the amplitude is wrong -- fix V0 or Cv_var.")
    print("  A ratio that falls with q means the spectrum shape is short of power,")
    print("  and near 1/2.88 = 0.347 1/A that points at the second Gaussian.")

    out = Path(args.json) if args.json else Path(args.outdir) / "spectrum.json"
    out.write_text(
        json.dumps(
            {
                "angpix": args.angpix,
                "box": args.box,
                "thickness": args.thickness,
                "packed": args.pack,
                "band_limited": os.environ.get("MULTEM_BANDLIMITED_POTENTIAL", "1")
                != "0",
                "variance": {"atomic": float(atomic.var()), "grf": float(grf.var())},
                "q": qa.tolist(),
                "atomic_power": pa.tolist(),
                "grf_power": pg.tolist(),
            },
            indent=1,
        )
    )
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
