#!/usr/bin/env python3
"""
slice_thickness.py -- how thick can the multislice slices be at 1.5 A/px?

Thicker slices make every render proportionally faster, and multislice error
grows with them. This renders one specimen at a ladder of slice thicknesses and
measures what changes, at 0 and 60 degrees of tilt, in two arms:

  molecules  three 70S ribosomes (PDB 4v5d) and three apoferritins (4v1w),
             both shipped with Parakeet, at fixed positions and orientations
             in a 500 A deep box, with no ice. The specimen is identical at
             every thickness, so differences come from the slicing alone.
  ice        the same box filled with GRF ice and nothing else (apart from two
             hydrogen atoms that make MULTEM slice the whole slab). Each
             slicing draws a different field, so this arm compares statistics.

For the molecules it reports, against the finest slicing and against 3 A (and, by
spatial frequency, the power of the difference image as a fraction of the
image's own, with where that first reaches 1e-4, 1e-3 and 1e-2):

  * the exit wave's relative difference, |psi_s - psi_ref| / |psi_ref| (L2
    over the field);
  * the noiseless image's difference after the optics (300 kV, -4 um, no
    inelastic model), as an RMS over the image's own contrast (its standard
    deviation), and over the shot noise of a 3 e/A^2 exposure at 1.5 A/px and
    at 3 A/px (the image binned 2x2), which is what a tilt series of that dose
    can resolve;
  * the GPU's run-to-run floor: the finest slicing rendered twice.

For the ice: the exit-wave phase mean and standard deviation, the image
contrast, and the image power in bands of spatial frequency, each as a ratio
to the finest slicing. And for both arms, the wall time of each exit wave.

    validation/slicing/slice_thickness.py --outdir /scratch/... --json out.json [--png out.png]

Needs a GPU build of the engine unless --device cpu.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

NX = 512
ANGPIX = 1.5
FIELD_A = NX * ANGPIX
BOX_XY = (
    4000.0  # wider than the simulation grid at 60 deg too, so the ice has no edge in it
)
DEPTH_A = 500.0
CENTRE = (BOX_XY / 2, BOX_XY / 2, DEPTH_A / 2)
MARGIN = 100
PADDING = 100
DEFOCUS_A = -40000.0
DOSE = 3.0  # e/A^2, for the shot-noise scale
THICKNESSES = (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0)
TILTS = (0.0, 60.0)
SEED = 1234

# (dx, dy, rotation vector) about the centre, A and rad
RIBOSOMES = [
    (-200, -150, (0.3, 1.1, -0.4)),
    (180, -120, (-1.2, 0.2, 0.9)),
    (0, 200, (0.7, -0.5, 2.0)),
]
APOFERRITINS = [
    (-230, 220, (0.1, 0.2, 0.3)),
    (230, 200, (1.0, 0.0, 0.0)),
    (30, -260, (0.0, 1.4, 0.2)),
]


def config(slice_thickness, ice, molecules):
    origin = int(round(CENTRE[0] - FIELD_A / 2))
    sample = {
        "box": [BOX_XY, BOX_XY, DEPTH_A],
        "centre": list(CENTRE),
        "shape": {
            "type": "cuboid",
            "cuboid": {"length_x": BOX_XY, "length_y": BOX_XY, "length_z": DEPTH_A},
        },
    }
    if molecules:

        def poses(entries):
            return [
                {
                    "position": [CENTRE[0] + dx, CENTRE[1] + dy, CENTRE[2]],
                    "orientation": list(r),
                }
                for dx, dy, r in entries
            ]

        sample["molecules"] = {
            "pdb": [
                {"id": "4v5d", "instances": poses(RIBOSOMES)},
                {"id": "4v1w", "instances": poses(APOFERRITINS)},
            ]
        }
    return {
        "microscope": {
            "beam": {"energy": 300, "electrons_per_angstrom": DOSE},
            "detector": {
                "nx": NX,
                "ny": NX,
                "pixel_size": ANGPIX,
                "origin": [origin, origin],
            },
            "lens": {"c_10": DEFOCUS_A, "c_30": 2.7, "c_c": 2.7},
        },
        "sample": sample,
        "scan": {
            "mode": "tilt_series",
            "start_angle": TILTS[0],
            "step_angle": TILTS[1] - TILTS[0],
            "num_images": len(TILTS),
        },
        "simulation": {
            "ice": ice,
            "seed": SEED,
            "margin": MARGIN,
            "padding": PADDING,
            "slice_thickness": slice_thickness,
        },
    }


def anchors(path):
    """Two hydrogen atoms, one just inside each face of the box, at a corner of the field"""
    x = CENTRE[0] + FIELD_A / 2 - 6
    path.write_text(
        "".join(
            "HETATM%5d  H   HOH A%4d    %8.3f%8.3f%8.3f  1.00  0.00           H\n"
            % (i, i, x, x, z)
            for i, z in enumerate((1.0, DEPTH_A - 1.0), start=1)
        )
        + "END\n"
    )


def render(workdir, name, slice_thickness, ice, molecules, device, sample_file=None):
    """Exit wave and noiseless image; returns (psi, image, seconds for the exit wave)"""
    import parakeet.io
    import parakeet.sample
    import parakeet.simulate

    d = workdir / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = config(slice_thickness, ice, molecules)
    cfg["multiprocessing"] = {"device": device, "nproc": 1, "gpu_id": [0]}
    if not molecules:
        anchors(d / "anchors.pdb")
        cfg["sample"]["coords"] = {
            "filename": str(d / "anchors.pdb"),
            "recentre": False,
        }
    path = d / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    if sample_file is None:
        sample_file = d / "sample.h5"
        parakeet.sample.new(str(path), str(sample_file))
        if molecules:
            parakeet.sample.add_molecules(str(path), str(sample_file))
    t0 = time.time()
    parakeet.simulate.exit_wave(str(path), str(sample_file), str(d / "exit_wave.h5"))
    seconds = time.time() - t0
    parakeet.simulate.optics(str(path), str(d / "exit_wave.h5"), str(d / "optics.h5"))
    psi = np.asarray(
        parakeet.io.open(str(d / "exit_wave.h5")).data[:], dtype=np.complex128
    )
    image = np.asarray(parakeet.io.open(str(d / "optics.h5")).data[:], dtype=np.float64)
    # The exit wave keeps the margin; the image after the optics does not
    psi = psi[:, MARGIN : MARGIN + NX, MARGIN : MARGIN + NX]
    return psi, image, seconds, sample_file


def bin2(a):
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3))


# Thresholds on the relative difference spectrum: where the slicing error's power
# reaches this fraction of the image's own power at that frequency
RELATIVE_POWER_LEVELS = (1e-4, 1e-3, 1e-2)


def relative_difference_spectrum(image, image_ref, nbins=64):
    """
    |FT(image - ref)|^2 / |FT(ref - mean)|^2, radially averaged: how much of
    the signal at each spatial frequency the slicing changes

    """
    n = image.shape[0]
    d = np.abs(np.fft.fft2(image - image_ref)) ** 2
    r = np.abs(np.fft.fft2(image_ref - image_ref.mean())) ** 2
    f = np.fft.fftfreq(n, ANGPIX)
    q = np.hypot(*np.meshgrid(f, f, indexing="ij")).ravel()
    edges = np.linspace(0, 0.5 / ANGPIX, nbins + 1)
    idx = np.digitize(q, edges) - 1
    ok = (idx >= 0) & (idx < nbins) & (q > 0)
    num = np.bincount(idx[ok], d.ravel()[ok], minlength=nbins)
    den = np.bincount(idx[ok], r.ravel()[ok], minlength=nbins)
    return 0.5 * (edges[1:] + edges[:-1]), num / np.maximum(den, 1e-300)


def first_crossing(q, ratio, level):
    """The lowest spatial frequency (1/A) where ratio reaches level, or None"""
    above = np.flatnonzero(ratio >= level)
    return float(q[above[0]]) if above.size else None


def compare(psi, image, psi_ref, image_ref):
    """Molecule-arm differences, per tilt"""
    out = []
    for k, tilt in enumerate(TILTS):
        dpsi = np.linalg.norm(psi[k] - psi_ref[k]) / np.linalg.norm(psi_ref[k])
        mean = image_ref[k].mean()
        diff = image[k] - image_ref[k]
        rms = float(np.sqrt(np.mean(diff**2)))
        rms_binned = float(np.sqrt(np.mean(bin2(diff) ** 2)))
        out.append(
            {
                "tilt_deg": tilt,
                "exit_wave_rel_l2": float(dpsi),
                "image_rms_over_contrast": rms / float(image_ref[k].std()),
                # Shot noise of a flat image of mean intensity `mean`, in the
                # same units: sqrt(N) / N x mean, N = dose x pixel area
                "image_rms_over_noise_1p5": rms / (mean / np.sqrt(DOSE * ANGPIX**2)),
                "image_rms_over_noise_3p0": rms_binned
                / (mean / np.sqrt(DOSE * (2 * ANGPIX) ** 2)),
            }
        )
        q, ratio = relative_difference_spectrum(image[k], image_ref[k])
        out[-1]["relative_difference_spectrum"] = {
            "q": q.tolist(),
            "ratio": ratio.tolist(),
        }
        out[-1]["crossings_1_over_A"] = {
            "%g" % level: first_crossing(q, ratio, level)
            for level in RELATIVE_POWER_LEVELS
        }
    return out


def radial_bands(image, edges_per_A):
    """Power of an image in bands of spatial frequency (1/A)"""
    f = np.fft.fftshift(np.abs(np.fft.fft2(image - image.mean())) ** 2)
    q = np.fft.fftshift(np.fft.fftfreq(image.shape[0], d=ANGPIX))
    qq = np.hypot(*np.meshgrid(q, q, indexing="ij"))
    return [
        float(f[(qq >= lo) & (qq < hi)].mean())
        for lo, hi in zip(edges_per_A[:-1], edges_per_A[1:])
    ]


def ice_stats(psi, image):
    bands = [0.0, 0.01, 0.03, 0.1, 0.2, 1 / (2 * ANGPIX)]
    out = []
    for k, tilt in enumerate(TILTS):
        phase = np.angle(psi[k] / psi[k].mean())
        out.append(
            {
                "tilt_deg": tilt,
                "phase_mean_rad": float(np.angle(psi[k].mean())),
                "phase_sd_rad": float(phase.std()),
                "image_contrast": float(image[k].std() / image[k].mean()),
                "band_edges_per_A": bands,
                "band_power": radial_bands(image[k], bands),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--png")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--thicknesses", type=float, nargs="+", default=list(THICKNESSES))
    args = ap.parse_args()

    workdir = Path(args.outdir)
    thicknesses = sorted(args.thicknesses)
    finest = thicknesses[0]
    result = {
        "nx": NX,
        "angpix": ANGPIX,
        "depth_A": DEPTH_A,
        "defocus_A": DEFOCUS_A,
        "dose_e_per_A2": DOSE,
        "tilts_deg": list(TILTS),
        "thicknesses_A": thicknesses,
        "molecules": [],
        "ice": [],
    }

    # Molecules: one sample, rendered at every thickness, and the finest twice
    renders = {}
    sample_file = None
    for s in thicknesses + [finest]:
        name = "mol_%g" % s if s not in renders else "mol_%g_again" % s
        psi, image, seconds, sample_file = render(
            workdir,
            name,
            s,
            ice=False,
            molecules=True,
            device=args.device,
            sample_file=sample_file,
        )
        key = s if s not in renders else "again"
        renders[key] = (psi, image, seconds)
        print("molecules  %-6s %7.1f s" % (name, seconds), flush=True)
    psi_ref, image_ref, _ = renders[finest]
    psi_3, image_3, _ = renders.get(3.0, renders[finest])
    floor = compare(*renders["again"][:2], psi_ref, image_ref)
    result["floor"] = floor
    for s in thicknesses:
        psi, image, seconds = renders[s]
        result["molecules"].append(
            {
                "slice_A": s,
                "exit_wave_seconds": seconds,
                "vs_finest": compare(psi, image, psi_ref, image_ref),
                "vs_3A": compare(psi, image, psi_3, image_3),
            }
        )

    # Ice
    sample_file = None
    stats = {}
    for s in thicknesses:
        psi, image, seconds, sample_file = render(
            workdir,
            "ice_%g" % s,
            s,
            ice=True,
            molecules=False,
            device=args.device,
            sample_file=sample_file,
        )
        stats[s] = (ice_stats(psi, image), seconds)
        print("ice        %-6s %7.1f s" % ("ice_%g" % s, seconds), flush=True)
    ref = stats[finest][0]
    for s in thicknesses:
        st, seconds = stats[s]
        result["ice"].append(
            {
                "slice_A": s,
                "exit_wave_seconds": seconds,
                "stats": st,
                "ratio_to_finest": [
                    {
                        "tilt_deg": a["tilt_deg"],
                        "phase_mean": a["phase_mean_rad"] / b["phase_mean_rad"],
                        "phase_sd": a["phase_sd_rad"] / b["phase_sd_rad"],
                        "image_contrast": a["image_contrast"] / b["image_contrast"],
                        "band_power": [
                            x / y for x, y in zip(a["band_power"], b["band_power"])
                        ],
                    }
                    for a, b in zip(st, ref)
                ],
            }
        )

    Path(args.json).write_text(json.dumps(result, indent=1))
    print_summary(result)
    if args.png:
        plot(result, args.png)


def print_summary(r):
    print(
        "\nfloor (finest twice): "
        + ", ".join(
            "%g deg exit wave %.2g, image/noise(3A px) %.2g"
            % (f["tilt_deg"], f["exit_wave_rel_l2"], f["image_rms_over_noise_3p0"])
            for f in r["floor"]
        )
    )
    print(
        "\nmolecules: slice  t(s)   | per tilt: exit-wave rel L2, image RMS / contrast, / noise at 3 e/A^2 (1.5 A px, 3 A px)  [vs finest]"
    )
    for m in r["molecules"]:
        cells = "  ".join(
            "%g deg %.2g %.2g %.2g %.2g"
            % (
                c["tilt_deg"],
                c["exit_wave_rel_l2"],
                c["image_rms_over_contrast"],
                c["image_rms_over_noise_1p5"],
                c["image_rms_over_noise_3p0"],
            )
            for c in m["vs_finest"]
        )
        print("  %5g  %6.1f  | %s" % (m["slice_A"], m["exit_wave_seconds"], cells))
    print(
        "\nmolecules: where the slicing error's power first reaches a fraction of"
        " the image's, vs finest (as a resolution, A)"
    )
    for m in r["molecules"][1:]:
        cells = "  ".join(
            "%g deg %s"
            % (
                c["tilt_deg"],
                " ".join(
                    "%s:%s" % (lv, ("%.1f" % (1 / x)) if x else "none")
                    for lv, x in c["crossings_1_over_A"].items()
                ),
            )
            for c in m["vs_finest"]
        )
        print("  %5g  | %s" % (m["slice_A"], cells))
    print(
        "\nice: slice t(s) | per tilt: ratio to finest of phase mean, phase s.d., image contrast; band power ratios"
    )
    for i in r["ice"]:
        cells = "  ".join(
            "%g deg %.3f %.3f %.3f [%s]"
            % (
                c["tilt_deg"],
                c["phase_mean"],
                c["phase_sd"],
                c["image_contrast"],
                " ".join("%.2f" % b for b in c["band_power"]),
            )
            for c in i["ratio_to_finest"]
        )
        print("  %5g %6.1f | %s" % (i["slice_A"], i["exit_wave_seconds"], cells))


def plot(r, png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = [m["slice_A"] for m in r["molecules"]]
    # The finest slicing is the reference, so its difference is zero: leave it
    # off the log axis
    coarse = r["molecules"][1:]
    sc = s[1:]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    for k, tilt in enumerate(r["tilts_deg"]):
        ax[0].plot(
            sc,
            [m["vs_finest"][k]["image_rms_over_noise_3p0"] for m in coarse],
            "o-",
            label="image diff / shot noise (3 A px), %g deg" % tilt,
        )
        ax[0].plot(
            sc,
            [m["vs_finest"][k]["exit_wave_rel_l2"] for m in coarse],
            "s--",
            label="exit-wave rel. L2, %g deg" % tilt,
        )
        ax[0].axhline(
            r["floor"][k]["exit_wave_rel_l2"], color="grey", lw=0.8, ls=":"
        )  # the GPU's run-to-run floor
    ax[0].set_yscale("log")
    ax[0].set_xlabel("slice thickness (A)")
    ax[0].set_title("Molecules: difference from the finest slicing")
    ax[0].legend(fontsize=7)
    for k, tilt in enumerate(r["tilts_deg"]):
        ax[1].plot(
            s,
            [i["ratio_to_finest"][k]["phase_sd"] for i in r["ice"]],
            "o-",
            label="phase s.d., %g deg" % tilt,
        )
        ax[1].plot(
            s,
            [i["ratio_to_finest"][k]["phase_mean"] for i in r["ice"]],
            "s--",
            label="phase mean, %g deg" % tilt,
        )
        ax[1].plot(
            s,
            [i["ratio_to_finest"][k]["image_contrast"] for i in r["ice"]],
            "^:",
            label="image contrast, %g deg" % tilt,
        )
    ax[1].axhline(1, color="grey", lw=0.8)
    ax[1].set_xlabel("slice thickness (A)")
    ax[1].set_title("Ice: ratio to the finest slicing")
    ax[1].legend(fontsize=7)
    ax[2].plot(
        s, [m["exit_wave_seconds"] for m in r["molecules"]], "o-", label="molecules"
    )
    ax[2].plot(s, [i["exit_wave_seconds"] for i in r["ice"]], "s-", label="ice")
    ax[2].set_xlabel("slice thickness (A)")
    ax[2].set_ylabel("exit wave, s (%d tilts)" % len(r["tilts_deg"]))
    ax[2].set_title("Time")
    ax[2].legend(fontsize=7)
    for m in coarse:
        sp = m["vs_finest"][0]["relative_difference_spectrum"]
        ax[3].semilogy(sp["q"], sp["ratio"], label="%g A slices" % m["slice_A"])
    for level in RELATIVE_POWER_LEVELS:
        ax[3].axhline(level, color="grey", lw=0.6, ls=":")
    ax[3].set_xlabel("q (1/A)")
    ax[3].set_ylabel("difference power / image power")
    ax[3].set_title("Molecules, 0 deg: slicing error by frequency", fontsize=9)
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=130)


if __name__ == "__main__":
    main()
