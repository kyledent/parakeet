#!/usr/bin/env python3
"""
edge_bands.py -- measure the edge bands of the GRF ice (upstream issue #87)
and test one explanation of them.

With simulation.ice on, images carry bands near the edges of the field (and a
vertical band) that vanish with ice off on the identical sample. One
explanation: the ice is a masked shape, and when the shape ends inside the
simulation grid -- the grid is the field plus (margin + padding) on every
side -- its edge is a step in projected potential. At the defocus used, the
step's ringing reaches into the field, and the grid is periodic, so it can
wrap around too. If that is the cause, a shape that covers the whole grid has
no edge in it and no bands.

Two renders with the optics of benchmark/configs/spa_70s_small.yaml (300 kV,
2 um underfocus, 1024 px at 1 A/px, 100 nm of ice, zero-loss filtered,
margin 100) and ice only, apart from two hydrogen atoms that make MULTEM slice
the slab (ANCHOR_Z_A):

  field  the ice shape is the field (as in the benchmark): its edge lies
         inside the grid
  grid   the ice shape is 300 A larger than the field on every side, which
         is more than (margin + padding) x pixel size: no edge in the grid

The two share the sample box and detector origin, and differ only in the
shape's lateral size.

For each it measures, on the exit-wave phase and on the image after the
optics (no detector noise):

  * the column- and row-mean profiles, and their RMS deviation from the
    interior mean in bands of distance from the nearest field edge, in units
    of (a) the pixel standard deviation of the interior texture and (b) the
    standard deviation of the interior's column means, which is what a
    profile of pure texture would show (a ratio near 1 = no band);
  * the worst single column or row, in the same units (a vertical band).

Bands present in the phase as well come from how the field is generated, not
from the optics.

    validation/grf/edge_bands.py --outdir /path/to/scratch --json edge_bands.json [--png edge_bands.png]

Needs a GPU build of the engine unless --device cpu.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

NX = 1024
ANGPIX = 1.0
FIELD_A = NX * ANGPIX
THICKNESS_A = 1000.0
MARGIN = 100
PADDING = 100  # parakeet's default
EXTRA_A = 300.0  # how far the "grid" shape extends past the field
BOX_A = FIELD_A + 2 * EXTRA_A
DISTANCE_BINS = [0, 8, 16, 32, 64, 128, 256]
ARMS = {"field": FIELD_A, "grid": BOX_A}


# MULTEM slices only where there are atoms: with none, the exit wave is a plane
# wave and the ice is never applied. Two hydrogen atoms at the centre of the
# field, just inside the two faces of the ice, make it slice the whole slab; one
# hydrogen atom adds nothing measurable beside 100 nm of ice texture, and the
# centre is as far from the edges being measured as anything can be.
ANCHOR_Z_A = (1.0, THICKNESS_A - 1.0)


def anchor_pdb(path):
    lines = []
    for serial, z in enumerate(ANCHOR_Z_A, start=1):
        x = y = BOX_A / 2
        lines.append(
            "HETATM%5d  H   HOH A%4d    %8.3f%8.3f%8.3f  1.00  0.00           H"
            % (serial, serial, x, y, z)
        )
    Path(path).write_text("\n".join(lines + ["END", ""]))


def config(shape_xy, device, anchors):
    return {
        "microscope": {
            "beam": {
                "energy": 300,
                "electrons_per_angstrom": 40.0,
                "energy_spread": 2.66e-6,
                "acceleration_voltage_spread": 0.8e-6,
                "illumination_semiangle": 0.02,
            },
            "lens": {"c_10": -20000, "c_30": 2.7, "c_c": 2.7},
            "detector": {
                "nx": NX,
                "ny": NX,
                "pixel_size": ANGPIX,
                # The field is sample x, y in [origin, origin + FIELD_A]
                "origin": [int(EXTRA_A), int(EXTRA_A)],
            },
        },
        "sample": {
            "coords": {"filename": str(anchors), "recentre": False},
            "box": [BOX_A, BOX_A, THICKNESS_A],
            "centre": [BOX_A / 2, BOX_A / 2, THICKNESS_A / 2],
            "shape": {
                "type": "cuboid",
                "cuboid": {
                    "length_x": shape_xy,
                    "length_y": shape_xy,
                    "length_z": THICKNESS_A,
                },
            },
        },
        "scan": {"mode": "still", "num_images": 1},
        "simulation": {
            "ice": True,
            "inelastic_model": "zero_loss",
            "slice_thickness": 3.0,
            "margin": MARGIN,
            "padding": PADDING,
        },
        "multiprocessing": {"device": device, "nproc": 1, "gpu_id": [0]},
    }


# Through the Python API: the command line also imports the analysis commands,
# whose dependencies a simulation does not need.
RUN = """
import sys
import parakeet.sample
import parakeet.simulate
config, sample, exit_wave, optics = sys.argv[1:5]
parakeet.sample.new(config, sample)
parakeet.simulate.exit_wave(config, sample, exit_wave)
parakeet.simulate.optics(config, exit_wave, optics)
"""


def render(directory, shape_xy, device):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    cfg = d / "c.yaml"
    anchors = d / "anchors.pdb"
    anchor_pdb(anchors)
    cfg.write_text(yaml.safe_dump(config(shape_xy, device, anchors), sort_keys=False))
    paths = [d / "s.h5", d / "exit_wave.mrc", d / "optics.mrc"]
    r = subprocess.run(
        [sys.executable, "-c", RUN, str(cfg), *map(str, paths)],
        capture_output=True,
        text=True,
    )
    (d / "render.out").write_text(r.stdout)
    (d / "render.err").write_text(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    import mrcfile

    with mrcfile.open(paths[1], permissive=True) as f:
        psi = np.asarray(f.data[0]).astype(np.complex128)
    # The exit wave is stored with the margin; the image after the optics is
    # cropped to the field. Measure both on the field alone.
    my, mx = (psi.shape[0] - NX) // 2, (psi.shape[1] - NX) // 2
    psi = psi[my : my + NX, mx : mx + NX]
    assert psi.shape == (NX, NX), psi.shape
    # 100 nm of ice shifts the phase by about 3 rad, so the raw angle would wrap.
    # Referencing it to the mean phasor leaves the fluctuations about the mean.
    reference = psi.mean()
    phase = np.angle(psi * np.conj(reference / abs(reference)))
    amplitude = np.abs(psi)

    with mrcfile.open(paths[2], permissive=True) as f:
        image = np.asarray(f.data[0], dtype=np.float64)
    return phase, image, amplitude


def basic_stats(img):
    return {
        "mean": float(img.mean()),
        "std": float(img.std()),
        "min": float(img.min()),
        "max": float(img.max()),
    }


def profile_stats(img):
    """Edge-band statistics of one image; None for a flat image"""
    ny, nx = img.shape
    q0, q1 = nx // 4, 3 * nx // 4
    interior = img[q0:q1, q0:q1]
    sigma_pixel = float(interior.std())
    out = {"basic": basic_stats(img), "sigma_pixel": sigma_pixel}
    # A flat interior (no texture at all) has no scale to measure bands in
    if not sigma_pixel > 1e-6 * max(abs(float(interior.mean())), 1e-30):
        out["flat"] = True
        return out
    out["flat"] = False
    for axis, name in ((0, "columns"), (1, "rows")):
        profile = img.mean(axis=axis)
        inner = profile[q0:q1]
        mean, sigma_profile = float(inner.mean()), float(inner.std())
        n = len(profile)
        distance = np.minimum(np.arange(n), n - 1 - np.arange(n))
        bands = []
        for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
            sel = (distance >= lo) & (distance < hi)
            rms = float(np.sqrt(np.mean((profile[sel] - mean) ** 2)))
            bands.append(
                {
                    "distance_px": [lo, hi],
                    "rms_over_sigma_pixel": rms / sigma_pixel,
                    "rms_over_sigma_profile": rms / sigma_profile,
                }
            )
        worst = int(np.argmax(np.abs(profile - np.median(profile))))
        out[name] = {
            "sigma_profile_interior": sigma_profile,
            "bands": bands,
            "worst_index": worst,
            "worst_over_sigma_profile": float(
                abs(profile[worst] - np.median(profile)) / sigma_profile
            ),
            "profile": profile.tolist(),
        }
    return out


def summary_line(arm, kind, stats):
    if stats["flat"]:
        b = stats["basic"]
        return (
            f"{arm:5s} {kind:9s} FLAT: mean {b['mean']:.6g}, s.d. {b['std']:.3g}, "
            f"min {b['min']:.6g}, max {b['max']:.6g}"
        )
    c, r = stats["columns"], stats["rows"]
    edge = max(
        c["bands"][0]["rms_over_sigma_profile"], r["bands"][0]["rms_over_sigma_profile"]
    )
    inner = max(
        c["bands"][-1]["rms_over_sigma_profile"],
        r["bands"][-1]["rms_over_sigma_profile"],
    )
    worst = max(c["worst_over_sigma_profile"], r["worst_over_sigma_profile"])
    return (
        f"{arm:5s} {kind:9s} edge band (0-8 px) {edge:7.2f}, 128-256 px {inner:6.2f}, "
        f"worst line {worst:7.2f}  (x interior profile s.d.)"
    )


def figure(results, png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    if any(results[a][k]["flat"] for a in ARMS for k in ("phase", "intensity")):
        return
    for row, kind in enumerate(("phase", "intensity")):
        for col, arm in enumerate(ARMS):
            img = results[arm][kind + "_image"]
            lo, hi = np.percentile(img, [1, 99])
            axes[row, col].imshow(img, cmap="gray", vmin=lo, vmax=hi)
            axes[row, col].set_title(f"{arm}: {kind}")
            axes[row, col].axis("off")
        ax = axes[row, 2]
        for arm in ARMS:
            s = results[arm][kind]["columns"]
            p = np.array(s["profile"])
            ax.plot((p - np.median(p)) / s["sigma_profile_interior"], label=arm, lw=0.8)
        ax.set_title(f"{kind}: column means (interior s.d. units)")
        ax.set_xlabel("x (px)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=110)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--json", help="write the statistics here")
    ap.add_argument("--png", help="write a figure here (needs matplotlib)")
    args = ap.parse_args()

    results = {}
    for arm, shape_xy in ARMS.items():
        phase, image, amplitude = render(Path(args.outdir) / arm, shape_xy, args.device)
        results[arm] = {
            "shape_xy_A": shape_xy,
            "exit_wave_amplitude": basic_stats(amplitude),
            "phase": profile_stats(phase),
            "intensity": profile_stats(image),
            "phase_image": phase,
            "intensity_image": image,
        }
        print(f"{arm:5s} exit-wave amplitude {results[arm]['exit_wave_amplitude']}")
        for kind in ("phase", "intensity"):
            print(summary_line(arm, kind, results[arm][kind]))

    if args.png:
        figure(results, args.png)
    if args.json:
        record = {
            "geometry": {
                "nx": NX,
                "angpix": ANGPIX,
                "field_A": FIELD_A,
                "grid_px": NX + 2 * (MARGIN + PADDING),
                "box_A": BOX_A,
                "thickness_A": THICKNESS_A,
                "defocus_A": 20000,
                "distance_bins_px": DISTANCE_BINS,
            },
            "arms": {
                arm: {k: v for k, v in r.items() if not k.endswith("_image")}
                for arm, r in results.items()
            },
        }
        Path(args.json).write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
