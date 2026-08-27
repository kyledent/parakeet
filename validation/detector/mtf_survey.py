#!/usr/bin/env python3
"""
mtf_survey.py -- what detector MTF data is available, and what it says.

Parakeet models the detector as a single DQE curve measured at 300 keV, applied
as a Fourier-space multiplication, with no MTF, no per-detector selection and no
voltage dependence (parakeet.dqe). Improving that needs real curves. This
surveys what is already on disk in RELION-compatible form, reports the voltage
dependence, and names what is missing.

RELION ships MTF curves as two-column STAR files:

    _rlnResolutionInversePixel   _rlnMtfValue

with q running 0 to 0.5, where 0.5 is the physical Nyquist. Those files are the
natural interchange format -- Gatan and Thermo both publish in it -- so a
detector model for Parakeet should read them rather than embed its own tables.

    validation/detector/mtf_survey.py --relion-data ~/github_repos/relion/data
    validation/detector/mtf_survey.py --relion-data ... --markdown
"""

import argparse
import glob
import os
import sys

import numpy as np

# What each shipped curve is, and what it is not
PROVENANCE = {
    "mtf_k2_300kV": "Gatan K2, 300 kV",
    "mtf_falcon2_300kV": "Thermo Falcon 2, 300 kV",
    "mtf_falcon3EC_200kV": "Thermo Falcon 3EC, 200 kV",
    "mtf_falcon3EC_300kV": "Thermo Falcon 3EC, 300 kV",
    "mtf_falcon4EC_200kV": "Thermo Falcon 4EC, 200 kV",
    "mtf_falcon4EC_300kV": "Thermo Falcon 4EC, 300 kV",
    "mtf_de20_300kV": "Direct Electron DE-20, 300 kV",
}

MISSING = [
    ("Gatan K3, 200 kV, standard", "gatan.com K3 page, behind a request form"),
    ("Gatan K3, 200 kV, CDS", "gatan.com K3 page, behind a request form"),
    ("Gatan K3, 300 kV, standard", "gatan.com K3 page, behind a request form"),
    ("Gatan K3, 300 kV, CDS", "gatan.com K3 page, behind a request form"),
    (
        "Thermo Falcon 4i, any voltage",
        "not published in STAR form; 4EC is the closest proxy",
    ),
    (
        "Any detector at 100 kV",
        "not shipped by RELION; Gatan Alpine is the 100-200 kV part",
    ),
]


def read_mtf(path):
    """Read a RELION-format MTF star file -> (q, mtf)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    if not rows:
        raise ValueError(f"no two-column data in {path}")
    a = np.array(rows)
    return a[:, 0], a[:, 1]


def table(rows, header, md):
    if md:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            print("| " + " | ".join(str(c) for c in r) + " |")
    else:
        w = [
            max(len(str(h)), *(len(str(r[i])) for r in rows))
            for i, h in enumerate(header)
        ]
        print("  " + "  ".join(str(h).rjust(x) for h, x in zip(header, w)))
        for r in rows:
            print("  " + "  ".join(str(c).rjust(x) for c, x in zip(r, w)))
    print()


def survey(data_dir, md):
    print("1. Curves available locally, in RELION STAR form")
    rows = []
    curves = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "mtf_*.star"))):
        name = os.path.splitext(os.path.basename(path))[0]
        q, m = read_mtf(path)
        curves[name] = (q, m)
        g = lambda x: np.interp(x, q, m)
        rows.append(
            [
                PROVENANCE.get(name, name),
                len(q),
                f"{g(0.1):.3f}",
                f"{g(0.25):.3f}",
                f"{g(0.4):.3f}",
                f"{g(0.49):.3f}",
            ]
        )
    table(rows, ["detector", "points", "MTF@0.1", "@0.25", "@0.4", "@0.49"], md)
    return curves


def voltage_dependence(curves, md):
    print("2. Voltage dependence, where both voltages are available")
    pairs = [
        ("Falcon 4EC", "mtf_falcon4EC_200kV", "mtf_falcon4EC_300kV"),
        ("Falcon 3EC", "mtf_falcon3EC_200kV", "mtf_falcon3EC_300kV"),
    ]
    rows = []
    for label, lo, hi in pairs:
        if lo not in curves or hi not in curves:
            continue
        ql, ml = curves[lo]
        qh, mh = curves[hi]
        for qq in (0.1, 0.25, 0.4):
            a, b = np.interp(qq, ql, ml), np.interp(qq, qh, mh)
            rows.append([label, f"{qq:g}", f"{a:.3f}", f"{b:.3f}", f"{a/b:.3f}"])
    table(rows, ["detector", "q", "200 kV", "300 kV", "200/300"], md)
    print("  Both are worse at 200 kV, and the deficit grows with spatial frequency:")
    print("  lower-energy electrons spread further laterally in the sensor, widening")
    print("  the point spread. Any per-detector model must carry voltage with it.\n")


def missing(md):
    print("3. What is not available here")
    table([[what, where] for what, where in MISSING], ["curve", "where it is"], md)
    print("  The K3 curves matter most: Gatan publishes four (200 and 300 kV, each")
    print("  in standard and CDS mode) and CDS is a different detector response, not")
    print("  a processing option. They are behind a request form, so a person has to")
    print("  ask for them -- this script cannot.\n")


def caution(md):
    print("4. One caution before fitting anything to these")
    print("  MTF measured on a counting detector is not a simple linear response.")
    print("  The counting process is non-linear, so a measured MTF is only valid at")
    print("  the very low dose rate it was measured at, and coincidence loss changes")
    print("  the effective response at working dose rates. A model that applies one")
    print("  MTF at all dose rates is making the same assumption Parakeet's single")
    print("  DQE curve already makes -- worth stating rather than inheriting.\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--relion-data",
        default=os.path.expanduser("~/github_repos/relion/data"),
        help="directory holding RELION's mtf_*.star files",
    )
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.relion_data):
        print(f"no such directory: {args.relion_data}", file=sys.stderr)
        return 2

    print("Detector MTF: what is available and what it says")
    print("=" * 72 + "\n")
    curves = survey(args.relion_data, args.markdown)
    voltage_dependence(curves, args.markdown)
    missing(args.markdown)
    caution(args.markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
