#!/usr/bin/env python3
"""
delivered_dqe.py -- measure the DQE that Parakeet's detector model delivers.

A detector model is only as good as the DQE it actually produces, and that is
measurable: take a flat field, simulate it with and without the detector, and
form

    DQE(q) = MTF(q)^2 . NPS_ideal(q) / NPS_detected(q)

which is the definition, evaluated on noise power spectra. This is how the
number would be measured on a real detector, so it is the honest way to check a
model of one. It is also the only measurement that separates the two curves:
the MTF sets how much signal survives, the DQE how much signal-to-noise, and a
model that confuses them looks perfectly reasonable in an image.

Four configurations are compared:

    ideal      no MTF, no DQE -- Poisson only, DQE = 1 by construction
    mtf only   an ideal counting detector that blurs; DQE should still be 1
    mtf + dqe  the full model; DQE should reproduce the tabulated curve
    legacy     the DQE multiplied onto the signal before Poisson noise is
               added, which is what Parakeet did before this model existed

The middle two are the interesting ones. Blurring attenuates signal and noise
together, so it cannot change signal-to-noise; a model that applies the DQE as
a signal filter fails here, because it attenuates the signal without touching
the noise. The legacy case shows what that costs: multiplying by DQE and then
adding noise at the reduced mean delivers DQE^2/DQE(0), so a detector specified
at 0.35 at Nyquist behaves like one at 0.13.

    validation/detector/delivered_dqe.py
    validation/detector/delivered_dqe.py --size 512 --repeats 16 --plot out.png
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import parakeet.dqe  # noqa: E402
import parakeet.mtf  # noqa: E402


def radial_bands(n, edges):
    """Annulus masks on a centred grid, in fractions of Nyquist."""
    Y, X = np.mgrid[0:n, 0:n]
    q = 2 * np.sqrt(((Y - n // 2) / n) ** 2 + ((X - n // 2) / n) ** 2)
    return [(q > lo) & (q <= hi) for lo, hi in zip(edges[:-1], edges[1:])]


def nps(a):
    """Noise power spectrum of a flat field, centred."""
    return np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean()))) ** 2 / a.size


def legacy_detect(image, dose, mtf, dqe, rng):
    """
    What Parakeet did before: multiply the image by the DQE in Fourier space,
    clip, then draw Poisson noise. Kept here so the comparison is a measurement
    rather than an assertion about code that no longer exists.

    """
    fft = np.fft.fft2(image) * np.fft.ifftshift(dqe)
    image = np.clip(np.real(np.fft.ifft2(fft)), 0, None)
    return rng.poisson(image * dose).astype("float64")


def measure(n, dose, repeats, detect_fn, rng, contrast=0.15):
    """
    Measure the delivered DQE the way a real one is measured, in two parts.

    The noise power comes from a flat field. The signal transfer needs
    something with signal in it: a flat field is uniform, so above DC it
    carries nothing, and a filter applied to the signal alone is invisible in
    it. So a white random pattern is pushed through, many realisations are
    averaged to recover the mean, and the transfer is the ratio of output to
    input power spectra with the residual noise -- which falls as 1/repeats --
    subtracted off.

        DQE(q) = T(q)^2 . N / NPS(q)

    with N the electrons per pixel, since an ideal detector has NPS = N.

    """
    flat = np.ones((n, n))
    nps_meas = np.mean(
        [nps(detect_fn(flat, dose, rng)) for _ in range(repeats)], axis=0
    )

    # The same pattern for every case, so what is compared is detector models
    # and not random fields.
    pattern = np.clip(
        1.0 + contrast * np.random.default_rng(12345).normal(size=(n, n)), 0.05, None
    )
    mean_out = np.mean([detect_fn(pattern, dose, rng) for _ in range(repeats)], axis=0)
    out_power = nps(mean_out)
    in_power = nps(pattern * dose)

    # Averaging a ratio mode by mode is not the ratio of the averages, and is
    # far worse behaved: a mode whose input power happens to be small gives a
    # huge ratio that survives the average. Numerator and denominator are
    # averaged over the annulus separately.
    edges = np.linspace(0.05, 1.0, 20)
    centres, values = [], []
    for lo, hi, sel in zip(edges[:-1], edges[1:], radial_bands(n, edges)):
        if sel.sum() < 20:
            continue
        noise = float(np.mean(nps_meas[sel]))
        signal = float(np.mean(out_power[sel])) - noise / repeats
        transfer_sq = max(signal, 0.0) / float(np.mean(in_power[sel]))
        centres.append(0.5 * (lo + hi))
        values.append(transfer_sq * dose / noise)
    return np.array(centres), np.array(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512, help="Flat field size")
    parser.add_argument("--dose", type=float, default=40.0, help="Electrons/pixel")
    parser.add_argument("--repeats", type=int, default=16, help="Fields averaged")
    parser.add_argument("--energy", type=float, default=300, help="keV")
    parser.add_argument("--detector", default="falcon4EC_300kV", help="MTF preset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="Write results here")
    parser.add_argument("--plot", help="Write a figure here")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    table = parakeet.dqe.DQETable()
    dqe = table.dqe_fs(args.energy, args.dose, (args.size, args.size))
    mtf = parakeet.mtf.MTF.from_preset(args.detector)

    def modelled(m, d):
        return lambda im, dose, r: parakeet.mtf.detect(im, dose, mtf=m, dqe=d, rng=r)

    cases = {
        "ideal": modelled(None, None),
        "mtf only": modelled(mtf, None),
        "mtf + dqe": modelled(mtf, dqe),
        "legacy": lambda im, dose, r: legacy_detect(im, dose, mtf, dqe, r),
    }
    results = {}
    for name, fn in cases.items():
        q, v = measure(args.size, args.dose, args.repeats, fn, rng)
        results[name] = (q, v)

    target = np.interp(
        results["mtf + dqe"][0],
        table.spatial_freq,
        table.dqe_table(args.energy, args.dose),
    )

    print(
        f"\n  {args.detector}, {args.energy:.0f} keV, {args.dose:g} e/pixel, "
        f"{args.size}^2, {args.repeats} fields\n"
    )
    print(
        f"  {'frac Nyq':>9} {'ideal':>8} {'mtf only':>9} {'mtf+dqe':>9} "
        f"{'table':>8} {'error':>8} {'legacy':>8}"
    )
    worst = 0.0
    for i, q in enumerate(results["mtf + dqe"][0]):
        e = results["mtf + dqe"][1][i] - target[i]
        worst = max(worst, abs(e))
        print(
            f"  {q:9.3f} {results['ideal'][1][i]:8.3f} "
            f"{results['mtf only'][1][i]:9.3f} {results['mtf + dqe'][1][i]:9.3f} "
            f"{target[i]:8.3f} {e:+8.3f} {results['legacy'][1][i]:8.3f}"
        )
    print(f"\n  Worst error against the tabulated DQE: {worst:.3f}")
    legacy_worst = float(np.max(np.abs(results["legacy"][1] - target)))
    print(f"  Worst error for the legacy model:      {legacy_worst:.3f}")
    for name in ("ideal", "mtf only"):
        d = float(np.max(np.abs(results[name][1] - 1.0)))
        print(f"  {name:<10} departs from DQE = 1 by at most {d:.3f}")

    if args.json:
        payload = {
            "detector": args.detector,
            "energy_keV": args.energy,
            "dose_e_per_pixel": args.dose,
            "size": args.size,
            "repeats": args.repeats,
            "seed": args.seed,
            "produced_by": "validation/detector/delivered_dqe.py",
            "fraction_of_nyquist": results["mtf + dqe"][0].tolist(),
            "tabulated_dqe": target.tolist(),
            "delivered": {k: v[1].tolist() for k, v in results.items()},
            "worst_error": worst,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  -> {args.json}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        q = results["mtf + dqe"][0]
        ax.plot(q, target, "k-", lw=2.5, label="tabulated DQE", zorder=1)
        for name, style in (
            ("ideal", "o"),
            ("mtf only", "s"),
            ("mtf + dqe", "^"),
            ("legacy", "v"),
        ):
            ax.plot(q, results[name][1], style, ms=5, label=f"delivered, {name}")
        ax.plot(q, target**2 / target[0], "r--", lw=1.2, label="DQE$^2$/DQE(0)")
        ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
        ax.set_xlabel("spatial frequency (fraction of Nyquist)")
        ax.set_ylabel("DQE, measured from output/input SNR$^2$")
        ax.set_title(
            f"{args.detector}, {args.dose:g} e/pixel, worst error {worst:.3f}",
            fontsize=10,
        )
        ax.set_ylim(0, 1.25)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"  -> {args.plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
