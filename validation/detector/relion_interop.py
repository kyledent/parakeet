#!/usr/bin/env python3
"""
relion_interop.py -- read an MTF the way RELION does, and check it agrees.

RELION is where these curves come from and where they are used, so its
conventions decide what a curve means. Two of them matter here.

**The frequency axis is the physical detector pixel.** RELION never assumes the
curve's pixel is the data's pixel: it converts the tabulated axis to absolute
spatial frequency by dividing by a separately supplied pixel size, and only then
looks it up on the map's own grid.

    mtf_resol[i] = resol_inv_pixel / mtf_angpix        # 1/A
    res          = sqrt(r2) / (angpix * boxsize)       # 1/A

That pixel size is `--mtf_angpix` in relion_postprocess and
`rlnMicrographOriginalPixelSize` in the optics table for refinement, and the GUI
defines it as "the original pixel size (in Angstroms) in the raw
(non-super-resolution!) micrographs". So a curve is tabulated against the
**physical** pixel, and super-resolution data is handled by the pixel-size ratio
rather than by a different curve.

**Interpolation is linear, and clamps at both ends.** Past the last tabulated
point the last value is held, not extrapolated to zero -- which is what
parakeet.mtf.MTF.at does.

This script reproduces RELION's lookup exactly and compares it against direct
interpolation of the same table, because the two do not quite agree: RELION
derives its index spacing by dividing the tabulated span by the number of
points rather than the number of intervals.

    validation/detector/relion_interop.py <curve.star> [...]
    validation/detector/relion_interop.py <curve.star> --angpix 0.5 --mtf-angpix 1.0
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import parakeet.mtf  # noqa: E402


def relion_lookup(res, mtf_resol, mtf_value, per_elem_divisor=None):
    """
    RELION's own MTF interpolation, transcribed.

    src/postprocessing.cpp:divideByMtf and
    src/jaz/single_particle/obs_model.cpp:getMtfImage carry identical copies.
    Both compute the index spacing as

        res_per_elem = (mtf_resol[n-1] - mtf_resol[0]) / n

    where n is the number of points. The spacing between n points is the span
    divided by n-1, so pass per_elem_divisor=n-1 to see what the same code does
    with the interval count instead.

    Args:
        res (array): frequencies to evaluate, in the same units as mtf_resol
        mtf_resol (array): the tabulated axis
        mtf_value (array): the tabulated MTF
        per_elem_divisor (int): override the divisor; None for RELION's

    Returns:
        array: the MTF at res

    """
    n = len(mtf_value)
    divisor = n if per_elem_divisor is None else per_elem_divisor
    res_per_elem = (mtf_resol[n - 1] - mtf_resol[0]) / float(divisor)

    out = np.empty_like(np.asarray(res, dtype=float))
    for k, r in enumerate(np.atleast_1d(res)):
        i_0 = int(np.floor(r / res_per_elem))
        if i_0 >= n - 1:
            out[k] = mtf_value[n - 1]
        elif i_0 <= 0:
            out[k] = mtf_value[0]
        else:
            x_0, y_0 = mtf_resol[i_0], mtf_value[i_0]
            x_1, y_1 = mtf_resol[i_0 + 1], mtf_value[i_0 + 1]
            out[k] = y_0 + (y_1 - y_0) * (r - x_0) / (x_1 - x_0)
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("curves", nargs="+", help="RELION MTF star files")
    p.add_argument(
        "--mtf-angpix",
        type=float,
        help="pixel size the curve was measured at (A), for the conversion demo",
    )
    p.add_argument(
        "--angpix", type=float, help="pixel size of the data (A), for the demo"
    )
    args = p.parse_args()

    print("1. RELION's index spacing: span/n against span/(n-1)\n")
    print(
        f"  {'curve':<30} {'n':>4} {'worst diff':>11} {'at q':>7} {'as % of MTF':>12}"
    )
    worst_overall = 0.0
    for path in args.curves:
        m = parakeet.mtf.MTF.from_file(path)
        # The worst case is a supremum over a continuum, so a coarser probe
        # simply finds less of it. 2001 points matches
        # relion/mtf_index_bug/reproduce_mtf_index.py so the two agree; an
        # earlier 501-point probe under-reported Falcon 4EC as 0.0084 / 8.3%.
        q = np.linspace(0.0, m.q[-1], 2001)
        got = relion_lookup(q, m.q, m.mtf)
        ref = relion_lookup(q, m.q, m.mtf, per_elem_divisor=len(m.q) - 1)
        d = np.abs(got - ref)
        k = int(np.argmax(d))
        worst_overall = max(worst_overall, float(d[k]))
        print(
            f"  {os.path.basename(path):<30} {len(m.q):>4} {d[k]:>11.5f} "
            f"{q[k]:>7.3f} {100 * d[k] / max(ref[k], 1e-9):>11.2f}%"
        )

    print(
        f"\n  Worst difference across these curves: {worst_overall:.5f}\n"
        "  The divisor is the point count where the spacing between points is the\n"
        "  span over one fewer. The index therefore runs slightly ahead, and the\n"
        "  linear segment is extrapolated backwards to reach the requested\n"
        "  frequency. Both RELION copies do the same thing, so a curve read by\n"
        "  RELION and one read here will differ by about this much.\n"
        "  Sensitivity to the point count is the thing to note: a densely sampled\n"
        "  curve is barely affected, a sparse one more so.\n"
    )

    print("2. Interval count against the point count, per curve density\n")
    print(f"  {'points':>7} {'span/n vs span/(n-1)':>22}")
    for n in (11, 33, 65, 129, 1001):
        print(f"  {n:>7} {100.0 / (n - 1):>21.2f}%")
    print("  (the index is ahead by this fraction of an interval)\n")

    if args.mtf_angpix and args.angpix:
        print("3. What the pixel-size conversion does\n")
        m = parakeet.mtf.MTF.from_file(args.curves[0])
        print(
            f"  curve measured at {args.mtf_angpix:g} A/px, data at "
            f"{args.angpix:g} A/px\n"
        )
        print(f"  {'1/A':>9} {'q on the curve':>15} {'q in the data':>14} {'MTF':>8}")
        for f in np.linspace(0.05, 0.5 / args.angpix, 8):
            q_curve = f * args.mtf_angpix
            q_data = f * args.angpix
            print(f"  {f:>9.4f} {q_curve:>15.4f} {q_data:>14.4f} {m.at(q_curve):>8.4f}")
        print(
            f"\n  The data's Nyquist, {0.5 / args.angpix:.4f} 1/A, sits at "
            f"q = {0.5 * args.mtf_angpix / args.angpix:.4f} on the curve.\n"
            "  Where that exceeds 0.5 the curve has no data and the last value is\n"
            "  held -- which is what super-resolution data does to a curve measured\n"
            "  on the physical pixel.\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
