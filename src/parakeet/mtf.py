#
# parakeet.mtf
#
# Copyright (C) 2026 Diamond Light Source and Rosalind Franklin Institute
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
"""
Detector modulation transfer functions.

The MTF says how much *signal contrast* survives to each spatial frequency. It
is a different quantity from the DQE, which says how much *signal to noise*
survives, and a detector needs both: the MTF blurs, the DQE sets how much noise
the detector adds on top of the arrival statistics.

Curves are read rather than embedded. RELION distributes MTF as a two-column
STAR file, and both Gatan and Thermo publish in that format, so a curve
obtained from a vendor can be used directly:

    data_
    loop_
    _rlnResolutionInversePixel
    _rlnMtfValue
    0            0.9995
    0.0078125    0.9966
    ...

with q in cycles per pixel, so q = 0.5 is Nyquist.

Where only a plot is available -- which is the case for most published curves --
a fit to a compact analytic form can be supplied instead of a table. The form
is the square-pixel aperture times a two-component charge spread,

    MTF(q) = sinc(q) * [ c*exp(-(q/a)^2) + (1 - c)/(1 + (q/b)^2) ]

which fits modern counting detectors to better than one percent. See
validation/detector/ for the fitting tool and the measured residuals.
"""

import os

import numpy as np


__all__ = ["MTF", "read_mtf_star", "analytic_mtf"]


def read_mtf_star(filename):
    """
    Read a RELION-format MTF star file.

    Args:
        filename (str): The file to read

    Returns:
        tuple: (q, mtf) with q in cycles per pixel, Nyquist at 0.5

    """
    q, m = [], []
    with open(filename) as infile:
        for line in infile:
            parts = line.replace(",", " ").split()
            if len(parts) == 2:
                try:
                    a, b = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                q.append(a)
                m.append(b)
    if len(q) < 2:
        raise RuntimeError(f"No MTF data found in {filename}")
    q = np.array(q)
    m = np.array(m)
    order = np.argsort(q)
    return q[order], m[order]


def analytic_mtf(q, a, b, c):
    """
    The fitted form: square pixel aperture times a two-component charge spread.

    Args:
        q (array): Spatial frequency in cycles per pixel
        a (float): Gaussian width (cycles per pixel)
        b (float): Lorentzian width (cycles per pixel)
        c (float): Weight of the Gaussian component, 0 to 1

    Returns:
        array: The MTF, which is 1 at q = 0 by construction

    """
    return np.sinc(q) * (c * np.exp(-((q / a) ** 2)) + (1 - c) / (1 + (q / b) ** 2))


# Analytic fits to the MTF curves distributed with RELION, which are the
# vendors' own measurements in the community's interchange format. Each entry is
# (a, b, c) for analytic_mtf, with the RMS and worst-case residual of the fit so
# a caller can see what the three numbers cost against the tabulated curve. The
# numbers are a convenience: if the curve itself is to hand, read it with
# MTF.from_file, which uses the measurement rather than a description of it.
#
# Reproduce with validation/detector/fit_mtf.py; see validation/detector/README.md.
PRESETS = {
    "k2_300kV": (0.7470, 0.0192, 0.9195, 0.0065, 0.0141, "Gatan K2, 300 kV"),
    "falcon2_300kV": (
        0.3473,
        0.0938,
        0.2736,
        0.0144,
        0.0470,
        "Thermo Falcon 2, 300 kV",
    ),
    "falcon3EC_200kV": (
        0.3864,
        0.0834,
        0.9160,
        0.0042,
        0.0163,
        "Thermo Falcon 3EC, 200 kV",
    ),
    "falcon3EC_300kV": (
        0.3977,
        0.1392,
        0.9161,
        0.0046,
        0.0197,
        "Thermo Falcon 3EC, 300 kV",
    ),
    "falcon4EC_200kV": (
        0.3357,
        0.0666,
        0.9038,
        0.0029,
        0.0070,
        "Thermo Falcon 4EC, 200 kV",
    ),
    "falcon4EC_300kV": (
        0.3582,
        0.0699,
        0.9624,
        0.0038,
        0.0084,
        "Thermo Falcon 4EC, 300 kV",
    ),
    "de20_300kV": (
        0.5189,
        0.0735,
        0.7824,
        0.0071,
        0.0268,
        "Direct Electron DE-20, 300 kV",
    ),
}


def presets_available():
    """The names accepted by MTF.from_preset, with what each one is."""
    return {k: v[5] for k, v in PRESETS.items()}


class MTF(object):
    """
    A detector MTF, from a table or a fitted form.

    Carries its provenance because a curve without one cannot be checked, and
    because these are somebody else's measurements.

    """

    def __init__(self, q=None, mtf=None, params=None, source=None):
        """
        Args:
            q (array): Spatial frequency, cycles per pixel (tabulated form)
            mtf (array): The MTF values (tabulated form)
            params (dict): {a, b, c} for the analytic form instead
            source (str): Where this curve came from

        """
        if params is not None:
            self.params = {k: float(params[k]) for k in ("a", "b", "c")}
            self.q = None
            self.mtf = None
        else:
            if q is None or mtf is None:
                raise RuntimeError("Give either a table or analytic parameters")
            self.params = None
            self.q = np.asarray(q, dtype=float)
            self.mtf = np.asarray(mtf, dtype=float)
        self.source = source

    @classmethod
    def from_file(cls, filename):
        """Read a RELION-format MTF star file."""
        q, m = read_mtf_star(filename)
        return cls(q=q, mtf=m, source=os.path.abspath(filename))

    @classmethod
    def from_params(cls, a, b, c, source=None):
        """Use the fitted analytic form."""
        return cls(params={"a": a, "b": b, "c": c}, source=source)

    @classmethod
    def from_preset(cls, name):
        """
        Build from one of the built-in fits.

        Args:
            name (str): A key of PRESETS

        Returns:
            MTF: The MTF

        """
        if name not in PRESETS:
            raise ValueError(
                "Unknown MTF preset %s. Known: %s" % (name, ", ".join(sorted(PRESETS)))
            )
        a, b, c, rms, worst, what = PRESETS[name]
        self = cls.from_params(a, b, c, source=f"preset {name} ({what})")
        self.residual = (rms, worst)
        return self

    @classmethod
    def resolve(cls, spec):
        """
        Build from whatever the config gave us: a preset name, or a file path.

        A name that is neither a known preset nor an existing file is an error
        rather than a silent fallback, because a silently ignored MTF is
        indistinguishable in the output from a detector that has none.

        Args:
            spec (str): A preset name or a path to a RELION MTF star file

        Returns:
            MTF: The MTF, or None if spec is None or empty

        """
        if not spec:
            return None
        if spec in PRESETS:
            return cls.from_preset(spec)
        if os.path.exists(spec):
            return cls.from_file(spec)
        raise ValueError(
            "MTF %s is neither a known preset (%s) nor an existing file"
            % (spec, ", ".join(sorted(PRESETS)))
        )

    def at(self, q):
        """
        Evaluate at spatial frequencies q, in cycles per pixel.

        Beyond Nyquist the response is zero: a sampled detector has none there,
        and a table would otherwise be clamped to its endpoint.

        """
        q = np.asarray(q, dtype=float)
        if self.params is not None:
            value = analytic_mtf(q, **self.params)
        else:
            # Tables usually stop a little short of Nyquist -- the RELION files
            # end at 0.492 -- so hold the last value out to Nyquist rather than
            # dropping to zero there, and zero only beyond it.
            value = np.interp(q, self.q, self.mtf, left=self.mtf[0], right=self.mtf[-1])
        return np.where(q > 0.5, 0.0, np.clip(value, 0.0, 1.0))

    def fs(self, shape):
        """
        Evaluate on a Fourier-space grid of the given shape.

        The grid is unshifted-centred, matching parakeet.dqe.DQETable.dqe_fs,
        so callers apply fftshift in the same way for both.

        """
        Y, X = np.mgrid[0 : shape[0], 0 : shape[1]]
        q = np.sqrt(
            ((Y - shape[0] // 2) / shape[0]) ** 2
            + ((X - shape[1] // 2) / shape[1]) ** 2
        )
        return self.at(q)

    def __repr__(self):
        kind = "analytic" if self.params is not None else f"{len(self.q)}-point table"
        return f"<MTF {kind} from {self.source}>"


def detect(image, electrons_per_pixel, mtf=None, dqe=None, rng=None, backend=None):
    """
    Take a noiseless intensity image to detected counts.

    The three steps are the ones the detector performs, in order:

    1. **Arrival.** Electrons arrive as a Poisson process at the delivered
       dose. This is shot noise and belongs to the beam, not the detector.
    2. **Charge spreading.** The detector blurs by its MTF. This blurs the
       noise as well as the signal, which is why a blur alone does not change
       signal to noise -- and why applying an MTF is not the same operation as
       applying a DQE.
    3. **The detector's own noise.** Whatever remains between an ideal detector
       and the measured DQE is added here, with the spectrum that makes the
       output signal-to-noise come out right.

    For an input signal S at frequency q with N electrons per pixel, after
    steps 1 and 2 the signal is MTF(q).S.N and the noise power is MTF(q)^2.N,
    so the signal to noise is unchanged and the effective DQE is 1. Requiring

        DQE(q) = MTF(q)^2 . N / NPS_total(q)

    gives NPS_total = MTF^2.N/DQE, so the noise added in step 3 has power
    MTF(q)^2.N.(1/DQE(q) - 1), which is what this adds.

    Passing neither mtf nor dqe gives plain Poisson noise, which is what
    Parakeet did before either existed.

    Args:
        image (array): The noiseless image, in units of intensity
        electrons_per_pixel (float): The dose
        mtf (MTF): The detector MTF, or None for an ideal one
        dqe (array): The DQE on the same Fourier grid, or None for DQE = 1
        rng: A numpy random generator, or None
        backend: A module supplying fft2, ifft2 and to_numpy, or None for numpy

    Returns:
        array: The detected counts

    """
    if rng is None:
        rng = np.random
    if backend is None:
        fft2, ifft2, to_numpy = np.fft.fft2, np.fft.ifft2, (lambda x: x)
    else:
        fft2, ifft2, to_numpy = backend.fft2, backend.ifft2, backend.to_numpy

    # 1. Arrival
    counts = rng.poisson(np.clip(image, 0, None) * electrons_per_pixel).astype(
        "float64"
    )
    if mtf is None and dqe is None:
        return counts

    shape = counts.shape
    m = np.fft.ifftshift(mtf.fs(shape)) if mtf is not None else np.ones(shape)

    # 2. Charge spreading, which blurs signal and noise alike
    detected = np.real(to_numpy(ifft2(fft2(counts) * m)))

    if dqe is None:
        return detected

    # 3. The detector's own noise. DQE is clamped away from zero because the
    # required noise power goes as 1/DQE; where the DQE has fallen to nothing
    # the MTF has too, so the product stays finite, but the guard keeps the
    # arithmetic well behaved.
    d = np.clip(np.fft.ifftshift(dqe), 1e-3, 1.0)
    excess_power = m**2 * (1.0 / d - 1.0)
    excess_power = np.where(m > 0, excess_power, 0.0)

    white = rng.normal(size=shape)
    excess = np.real(to_numpy(ifft2(fft2(white) * np.sqrt(excess_power))))
    return detected + excess * np.sqrt(electrons_per_pixel)
