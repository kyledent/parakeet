#!/usr/bin/env python3
"""
fit_mtf.py -- fit a compact analytic form to a detector MTF curve.

Most published MTF data is not tabulated. RELION ships seven curves as STAR
files, Gatan puts four more behind a request form, and everything else --
McMullan 2014, Sun 2021, the Thermo application notes -- exists only as plots.
Points read off a plot are sparse and noisy, and cannot be interpolated
sensibly on their own. Fitting them to a physically motivated form turns them
into something evaluable at any frequency, and reduces a curve to a few numbers
that can sit in a config with its provenance attached.

The form is the pixel aperture times a charge-spread term:

    MTF(q) = sinc(q) * [ c * exp(-(q/a)^2) + (1 - c) / (1 + (q/b)^2) ]

with q in cycles per pixel, so q = 0.5 is Nyquist and sinc(q) = sin(pi q)/(pi q)
is the response of a square pixel. The bracket is a weighted sum of a Gaussian
and a Lorentzian charge spread; both are used in the detector literature, and
together they fit modern counting detectors to better than 1%.

A curve is only meaningful alongside the pixel it was tabulated against, so a
curve measured on a super-sampled grid has to be brought onto the pixel of the
data before it is fitted or used. --mtf-angpix and --angpix do that, and
--supersample N is the shorthand for a curve on an N-times finer grid:

    validation/detector/fit_mtf.py ~/github_repos/relion/data/mtf_k2_300kV.star
    validation/detector/fit_mtf.py <file.star> --json out.json
    validation/detector/fit_mtf.py <digitised.csv> --detector "K3 CDS" --kv 200
    validation/detector/fit_mtf.py <superres.star> --supersample 2 --rebin
    validation/detector/fit_mtf.py <file.star> --mtf-angpix 0.5 --angpix 1.06 \
        --write-star mtf_bin1.star

Use --rebin when the data really was binned, so the pixel aperture changed too;
leave it off when only the axis convention differs, which is what RELION's
--mtf_angpix does. See parakeet.mtf.resample.

Always reports the residual. A fit without one is a claim without evidence.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import parakeet.mtf  # noqa: E402

# Parameters are bounded because an unbounded fit is happy to return a negative
# component weight -- it did so for Falcon 2, giving c = -0.63, which is
# meaningless and was fitting noise rather than the curve.
BOUNDS = ([0.01, 0.005, 0.0], [2.0, 2.0, 1.0])
P0 = [0.3, 0.1, 0.9]

# Above this the form is not describing the data and the caller should know.
GOOD_FIT = 0.02


def mtf(q, a, b, c):
    """Pixel aperture times a Gaussian/Lorentzian charge spread. MTF(0) = 1."""
    return np.sinc(q) * (c * np.exp(-((q / a) ** 2)) + (1 - c) / (1 + (q / b) ** 2))


def read_curve(path):
    """Read a RELION MTF star file, or any two-column text/CSV of q, MTF."""
    q, m = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.replace(",", " ").split()
            if len(parts) == 2:
                try:
                    a, b = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                q.append(a)
                m.append(b)
    if len(q) < 4:
        raise ValueError(f"need at least 4 points, found {len(q)} in {path}")
    return np.array(q), np.array(m)


def fit(q, m):
    sel = (q > 0) & (q <= 0.5)
    q, m = q[sel], m[sel]
    popt, _ = curve_fit(mtf, q, m, p0=P0, bounds=BOUNDS, maxfev=40000)
    resid = mtf(q, *popt) - m
    return popt, float(np.max(np.abs(resid))), float(np.sqrt(np.mean(resid**2)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("curve", help="RELION mtf STAR file, or two-column q,MTF text")
    ap.add_argument("--detector", default=None, help="name to record with the fit")
    ap.add_argument("--kv", type=float, default=None, help="accelerating voltage")
    ap.add_argument(
        "--source",
        default=None,
        help="where the curve came from -- a DOI, a URL, 'vendor STAR', "
        "'digitised from figure N of ...'",
    )
    ap.add_argument("--json", default=None, help="write the fit here")
    ap.add_argument(
        "--mtf-angpix",
        type=float,
        default=None,
        help="pixel size (A) the curve is tabulated against",
    )
    ap.add_argument(
        "--angpix",
        type=float,
        default=None,
        help="pixel size (A) to express the curve against before fitting",
    )
    ap.add_argument(
        "--supersample",
        type=float,
        default=None,
        help="shorthand: the curve is on a grid this many times finer than the "
        "target pixel (2 for a super-resolution curve going to bin 1)",
    )
    ap.add_argument(
        "--rebin",
        action="store_true",
        help="the data was genuinely binned, so swap the pixel aperture too; "
        "leave off when only the axis convention differs",
    )
    ap.add_argument(
        "--write-star",
        default=None,
        help="write the (resampled) curve as a RELION MTF star file",
    )
    args = ap.parse_args()

    q, m = read_curve(args.curve)

    resampled_from = None
    if args.supersample is not None:
        if args.mtf_angpix or args.angpix:
            print(
                "  --supersample is a shorthand for --mtf-angpix/--angpix; "
                "give one or the other",
                file=sys.stderr,
            )
            return 2
        from_angpix, to_angpix = 1.0, float(args.supersample)
    elif args.mtf_angpix is not None or args.angpix is not None:
        if args.mtf_angpix is None or args.angpix is None:
            print("  --mtf-angpix and --angpix go together", file=sys.stderr)
            return 2
        from_angpix, to_angpix = args.mtf_angpix, args.angpix
    else:
        from_angpix = to_angpix = None

    if from_angpix is not None:
        curve = parakeet.mtf.MTF(q, m, source=args.curve)
        curve, covered = parakeet.mtf.resample(
            curve, from_angpix, to_angpix, rebin=args.rebin
        )
        q, m = curve.q, curve.mtf
        resampled_from = (from_angpix, to_angpix, args.rebin, covered)

    (a, b, c), max_resid, rms = fit(q, m)

    print(f"  curve      {args.curve}")
    if resampled_from is not None:
        f, tt, rb, covered = resampled_from
        how = "rebinned (aperture swapped)" if rb else "axis re-expressed"
        print(f"  resampled  {f:g} -> {tt:g} A/px, {how}")
        print(f"  coverage   {covered:.2f} of the new axis came from the table")
        if covered < 0.999:
            print(
                "             the rest holds the last tabulated value, because "
                "the curve\n             does not reach that far -- treat those "
                "frequencies as unmeasured"
            )
    print(f"  points     {len(q)} over q = {q.min():.3f} to {q.max():.3f}")
    print(f"  form       sinc(q) * [ c*exp(-(q/a)^2) + (1-c)/(1+(q/b)^2) ]")
    print(f"  a          {a:.5f}     Gaussian width, cycles/px")
    print(f"  b          {b:.5f}     Lorentzian width, cycles/px")
    print(f"  c          {c:.5f}     Gaussian weight")
    print(f"  max resid  {max_resid:.5f}")
    print(f"  rms resid  {rms:.5f}")
    if max_resid > GOOD_FIT:
        print(
            f"\n  WARNING: max residual exceeds {GOOD_FIT}. This form is built around"
        )
        print("  counting-detector physics and fits scintillator-based and older")
        print("  detectors poorly. Use the tabulated curve rather than this fit.")

    print(f"\n  {'q':>6} {'d (px)':>7} {'data':>8} {'fit':>8} {'diff':>8}")
    for qq in (0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.49):
        if qq > q.max():
            continue
        d = float(np.interp(qq, q, m))
        f = float(mtf(qq, a, b, c))
        print(f"  {qq:6.2f} {1/qq:7.1f} {d:8.4f} {f:8.4f} {f-d:+8.4f}")

    if args.write_star:
        parakeet.mtf.write_mtf_star(
            parakeet.mtf.MTF(q, m, source=args.curve), args.write_star
        )
        print(f"\n  -> {args.write_star}")

    if args.json:
        # Provenance travels with the fit. A fitted curve is derived from
        # someone's measurement, and fitting does not change whose it is.
        payload = {
            "detector": args.detector,
            "kv": args.kv,
            "source": args.source or args.curve,
            "form": "sinc(q) * (c*exp(-(q/a)**2) + (1-c)/(1+(q/b)**2))",
            "q_units": "cycles per pixel, Nyquist = 0.5",
            "a": a,
            "b": b,
            "c": c,
            "max_residual": max_resid,
            "rms_residual": rms,
            "n_points": int(len(q)),
            "usable": bool(max_resid <= GOOD_FIT),
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\n  -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
