#
# parakeet.elastic.py
#
# Copyright (C) 2026 Diamond Light Source and Rosalind Franklin Institute
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
"""
Elastic scattering that a multislice grid cannot hold.

A grid of spacing dx holds spatial frequencies only to its Nyquist frequency
q_N = 1/(2 dx), that is scattering angles to theta_g = lambda q_N (6.56 mrad at
300 kV and 1.5 A/px). A material scatters beyond that as well, and with the
band-limited potential (MULTEM_BANDLIMITED_POTENTIAL=1) the engine also
attenuates what it does draw, by its window W(q): flat to q_N/2, a raised
cosine to zero at q_N. Kinematically the simulation therefore scatters
W(q)^2 dsigma/dOmega below q_N and nothing above, and every electron the real
material would scatter beyond that is kept in the exit wave instead.

This module gives, per material, the cross-section the simulation leaves out,
split by the objective aperture's acceptance q_acc = beta / lambda:

    lost:  int_{q > q_acc} (1 - w(q)) dsigma    (outside the aperture)
    band:  int_{q < q_acc} (1 - w(q)) dsigma    (inside it; displaced, not lost)

with w(q) = W(q)^2 below q_N and 0 above. With no aperture, q_acc is infinite
and everything is band.

The cross-section is that of independent atoms. The default model, "iam",
uses the five-Gaussian electron scattering factors of International Tables
for Crystallography Vol. C, Table 4.3.2.2 (Peng et al. 1996, Acta Cryst A52,
257), as gemmi ships them, with the relativistic factor gamma^2:
dsigma/dOmega = gamma^2 |f(s)|^2, s = theta / (2 lambda) = q / 2. The fit is
valid to s = 2 1/A; beyond it the Gaussians fall faster than the real
theta^-4 tail, so the cross-section there is underestimated
(`tail_fraction` reports how much lies there). "lenz" is the screened
Rutherford form (Egerton, Electron Energy-Loss Spectroscopy in the Electron
Microscope, 3rd edn, sec. 3.1), the analytic check.

Stated approximations: atoms scatter independently (S(q) = 1; the
intermolecular structure factor and bonding are left out); the part inside
the grid is taken to be what the material's random field already draws, which
holds for a texture calibrated to the material (the water presets) and not
for a placeholder; scattering is single and kinematic, and losses multiply.

"""

import math

import numpy as np

__all__ = [
    "wavelength",
    "gamma",
    "grid_q",
    "window",
    "cross_section",
    "missing_cross_sections",
    "water_number_density",
    "tail_fraction",
    "band_rings",
    "halo",
    "MODELS",
]

MODELS = ("iam", "lenz")

# Physical constants (CODATA 2018)
ELECTRON_REST_ENERGY_KEV = 510.99895
BOHR_RADIUS_A = 0.529177210903
AVOGADRO = 6.02214076e23
WATER_MOLAR_MASS = 18.01528  # g/mol

# The upper limit of validity of the Table 4.3.2.2 fit, s = sin(theta)/lambda
PENG_S_MAX = 2.0  # 1/A

# With no aperture the halo's rings are sampled to this q; the weight beyond is
# far below band_rings' default tail cut
HALO_Q_MAX = 16.0  # 1/A

# The halo's coarse canvas, padded for the widest ring that can land, is at
# most this many coarse pixels across
MAX_HALO_CANVAS = 4096

ATOMIC_NUMBER = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "Na": 11,
    "Mg": 12,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "K": 19,
    "Ca": 20,
    "Fe": 26,
    "Zn": 30,
    "Au": 79,
}


def wavelength(energy_kev):
    """The relativistic electron wavelength (A) at a beam energy (keV)"""
    v = energy_kev * 1e3
    return 12.2643247 / math.sqrt(v * (1 + 0.978476e-6 * v))


def gamma(energy_kev):
    """The relativistic mass factor m/m0 at a beam energy (keV)"""
    return 1 + energy_kev / ELECTRON_REST_ENERGY_KEV


def grid_q(pixel_size):
    """The grid's Nyquist frequency q_N = 1/(2 dx) (1/A)"""
    return 0.5 / pixel_size


def window(q, pixel_size):
    """
    The band-limited potential's window W(q) (python-multem README): 1 to
    q_N/2, a raised cosine to 0 at q_N, 0 beyond

    """
    q = np.asarray(q, dtype="float64")
    qn = grid_q(pixel_size)
    q1 = 0.5 * qn
    w = np.where(q <= q1, 1.0, 0.5 * (1 + np.cos(np.pi * (q - q1) / (qn - q1))))
    return np.where(q >= qn, 0.0, w)


def _peng(element):
    import gemmi

    c = gemmi.Element(element).c4322
    a, b = np.asarray(c.a, dtype="float64"), np.asarray(c.b, dtype="float64")
    if not np.any(a):
        raise ValueError(
            "no electron scattering factors (Table 4.3.2.2) for element %r" % element
        )
    return a, b


def _iam_closed(element, lam, g, q_lo, q_hi):
    """int_{q_lo}^{q_hi} gamma^2 f(s)^2 dOmega, closed form (A^2)"""
    a, b = _peng(element)
    s_lo = 0.5 * q_lo
    B = b[:, None] + b[None, :]
    aa = a[:, None] * a[None, :]
    term_lo = np.exp(-B * s_lo**2) / (2 * B)
    term_hi = 0.0 if not np.isfinite(q_hi) else np.exp(-B * (0.5 * q_hi) ** 2) / (2 * B)
    # dOmega = 2 pi theta dtheta = 2 pi lam^2 q dq = 8 pi lam^2 s ds
    return float(8 * np.pi * lam**2 * g**2 * np.sum(aa * (term_lo - term_hi)))


def _lenz_closed(element, lam, g, q_lo, q_hi):
    """The Lenz (screened Rutherford) cross-section between q_lo and q_hi (A^2)"""
    Z = ATOMIC_NUMBER[element]
    k0 = 2 * np.pi / lam
    theta0 = Z ** (1.0 / 3.0) / (k0 * BOHR_RADIUS_A)
    pre = 4 * np.pi * g**2 * Z**2 / (BOHR_RADIUS_A**2 * k0**4)
    t_lo = lam * q_lo
    hi = 0.0 if not np.isfinite(q_hi) else 1.0 / ((lam * q_hi) ** 2 + theta0**2)
    return float(pre * (1.0 / (t_lo**2 + theta0**2) - hi))


def _dsigma_dq(element, lam, g, q, model):
    """dsigma/dq (A^2 A), for the numerical part"""
    q = np.asarray(q, dtype="float64")
    if model == "iam":
        a, b = _peng(element)
        s = 0.5 * q
        f = np.sum(a[:, None] * np.exp(-b[:, None] * s[None, :] ** 2), axis=0)
        return g**2 * f**2 * 2 * np.pi * lam**2 * q
    Z = ATOMIC_NUMBER[element]
    k0 = 2 * np.pi / lam
    theta0 = Z ** (1.0 / 3.0) / (k0 * BOHR_RADIUS_A)
    pre = 4 * g**2 * Z**2 / (BOHR_RADIUS_A**2 * k0**4)
    theta = lam * q
    return pre / (theta**2 + theta0**2) ** 2 * 2 * np.pi * lam**2 * q


def cross_section(element, energy_kev, q_lo, q_hi=np.inf, model="iam"):
    """
    The elastic cross-section of one atom between spatial frequencies q_lo and
    q_hi (1/A; scattering angle theta = lambda q), A^2

    Args:
        element: The element symbol
        energy_kev: The beam energy (keV)
        q_lo, q_hi: The limits (q_hi may be inf)
        model: "iam" (Table 4.3.2.2 scattering factors) or "lenz"

    """
    if model not in MODELS:
        raise ValueError("elastic model must be one of %s" % (MODELS,))
    if not (0 <= q_lo <= q_hi):
        raise ValueError("cross_section needs 0 <= q_lo <= q_hi")
    lam = wavelength(energy_kev)
    g = gamma(energy_kev)
    closed = _iam_closed if model == "iam" else _lenz_closed
    return closed(element, lam, g, q_lo, q_hi)


def _windowed_part(element, lam, g, q_lo, q_hi, pixel_size, model, n=256):
    """int over [q_lo, q_hi] within the window's slope of W(q)^2 dsigma (A^2)"""
    qn = grid_q(pixel_size)
    q1 = 0.5 * qn
    lo, hi = max(q_lo, q1), min(q_hi, qn)
    flat_hi = min(q_hi, q1)
    total = 0.0
    if flat_hi > q_lo:  # W = 1 below q_N/2
        closed = _iam_closed if model == "iam" else _lenz_closed
        total += closed(element, lam, g, q_lo, flat_hi)
    if hi > lo:
        x, w = np.polynomial.legendre.leggauss(n)
        q = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        total += float(
            0.5
            * (hi - lo)
            * np.sum(w * window(q, pixel_size) ** 2 * _dsigma_dq(element, lam, g, q, model))
        )
    return total


def missing_cross_sections(
    number_density, energy_kev, pixel_size, q_acc=np.inf, model="iam"
):
    """
    The inverse mean free paths (1/A) of the elastic scattering a band-limited
    multislice simulation of a material leaves out, outside and inside an
    aperture

    Args:
        number_density: Element -> atoms per A^3
        energy_kev: The beam energy (keV)
        pixel_size: The simulation's grid spacing (A)
        q_acc: The aperture's acceptance beta / lambda (1/A); inf with none
        model: "iam" or "lenz"

    Returns:
        (mu_lost, mu_band): 1/lambda for scattering beyond q_acc and below it

    """
    if model not in MODELS:
        raise ValueError("elastic model must be one of %s" % (MODELS,))
    if not q_acc > 0:
        raise ValueError("the aperture's acceptance must be positive")
    lam = wavelength(energy_kev)
    g = gamma(energy_kev)
    closed = _iam_closed if model == "iam" else _lenz_closed
    mu_lost = 0.0
    mu_band = 0.0
    for element, n in number_density.items():
        if n < 0:
            raise ValueError("negative number density for %s" % element)
        if n == 0:
            continue
        # What the simulation draws itself (W^2 dsigma below q_N) it scatters
        # itself, and below q_acc Parakeet's aperture mask removes its share
        if np.isfinite(q_acc):
            mu_lost += n * (
                closed(element, lam, g, q_acc, np.inf)
                - _windowed_part(element, lam, g, q_acc, np.inf, pixel_size, model)
            )
        mu_band += n * (
            closed(element, lam, g, 0.0, q_acc)
            - _windowed_part(element, lam, g, 0.0, q_acc, pixel_size, model)
        )
    return mu_lost, mu_band


def tail_fraction(number_density, energy_kev, q_lo, model="iam"):
    """
    The share of the cross-section above q_lo that lies beyond the Table
    4.3.2.2 fit's validity (s > 2 1/A, q > 4 1/A), where the fit
    underestimates it. Weighted by the number density.

    """
    q_max = 2 * PENG_S_MAX
    above = 0.0
    beyond = 0.0
    for element, n in number_density.items():
        above += n * cross_section(element, energy_kev, q_lo, np.inf, model)
        beyond += n * cross_section(element, energy_kev, max(q_lo, q_max), np.inf, model)
    return beyond / above if above > 0 else 0.0


def water_number_density(density):
    """H2O at a mass density (g/cm^3): element -> atoms per A^3"""
    n = density * AVOGADRO / WATER_MOLAR_MASS * 1e-24
    return {"H": 2 * n, "O": n}


def band_rings(
    number_density,
    energy_kev,
    pixel_size,
    q_acc,
    c_10,
    c_30_A,
    model="iam",
    n=4000,
    keep=1 - 1e-5,
):
    """
    Where the band's electrons land: the scattering the aperture passes but
    the grid cannot hold, as rings about the pixel it left. An electron
    scattered to q is displaced, in geometric optics, by the gradient of the
    aberration function, r(q) = lambda q |c_10 + C_s lambda^2 q^2| (Parakeet's
    chi = -(pi lambda c_10 q^2 + pi/2 lambda^3 C_s q^4)); its weight is the
    material's (1 - W(q)^2) dsigma/dq.

    Args:
        number_density: Element -> atoms per A^3
        energy_kev: The beam energy (keV)
        pixel_size: The grid spacing (A)
        q_acc: The aperture's acceptance (1/A); inf with none
        c_10: The frame's defocus (A, Parakeet's sign)
        c_30_A: The spherical aberration (A)
        model: "iam" or "lenz"
        n: The number of q samples
        keep: The share of the band's weight to resolve; the rest, the far
            tail, is reported and treated as lost

    Returns:
        (r, w, record): ring radii (A) and weights (1/A, summing to mu_band
        less the tail), and the q range and tail share

    """
    lam = wavelength(energy_kev)
    g = gamma(energy_kev)
    qn = grid_q(pixel_size)
    q_lo = 0.5 * qn  # W = 1 below: nothing missing there
    q_hi = q_acc if np.isfinite(q_acc) else HALO_Q_MAX
    if q_hi <= q_lo:
        return np.zeros(0), np.zeros(0), {"q": [q_lo, q_hi], "tail_share": 0.0}
    q = np.geomspace(q_lo, q_hi, n + 1)
    dq = np.diff(q)
    qc = 0.5 * (q[1:] + q[:-1])
    dens = np.zeros_like(qc)
    for element, nd in number_density.items():
        dens += nd * _dsigma_dq(element, lam, g, qc, model)
    w = dens * (1 - window(qc, pixel_size) ** 2) * dq
    _, total = missing_cross_sections(number_density, energy_kev, pixel_size, q_acc, model)
    if total <= 0 or w.sum() <= 0:
        return np.zeros(0), np.zeros(0), {"q": [q_lo, q_hi], "tail_share": 0.0}
    # Rescale the quadrature to the closed form, then cut the far tail
    w *= total / w.sum()
    c = np.cumsum(w) / total
    m = int(np.searchsorted(c, keep)) + 1
    r = lam * qc[:m] * np.abs(c_10 + c_30_A * lam**2 * qc[:m] ** 2)
    record = {
        "q": [float(q_lo), float(qc[m - 1])],
        "r_A": [float(r.min()), float(r.max())],
        "tail_share": float(1 - w[:m].sum() / total),
    }
    return r, w[:m], record


def halo(source, pixel_size, r, w):
    """
    The band's electrons put back: each pixel's removed intensity spread over
    its rings (`band_rings`), summed incoherently. The convolution is linear
    on a zero-padded canvas, so electrons whose rings leave the field are
    lost there (the specimen stops at its edge). It is computed on a coarse
    grid, a quarter of the smallest ring radius, and interpolated back.

    Args:
        source: The intensity each pixel's band removes, (ny, nx), before the
            ring weights (the share of each pixel's band, summed to 1)
        pixel_size: The grid spacing (A)
        r, w: The ring radii (A) and their weights, normalised here

    Returns:
        (halo, record): the (ny, nx) intensity that lands, and the coarse
        pixel and the share of the source that leaves the field

    """
    from scipy.ndimage import map_coordinates

    ny, nx = source.shape
    total = float(source.sum())
    if len(r) == 0 or total == 0:
        return np.zeros_like(source), {"coarse_binning": None, "left_field": 0.0}
    r = np.asarray(r, dtype="float64")
    w = np.asarray(w, dtype="float64") / np.sum(w)

    # A ring wider than the field's diagonal lands outside it from anywhere
    diagonal = np.hypot(ny, nx) * pixel_size
    beyond = float(w[r > diagonal].sum())
    keep = r <= diagonal
    r, w = r[keep], w[keep]
    if len(r) == 0:
        return np.zeros_like(source), {"coarse_binning": None, "left_field": 1.0}

    # The coarse pixel: a quarter of the ring radius below which a tenth of
    # the weight lies (smaller rings fall in their own coarse pixel), and no
    # finer than keeps the padded canvas within MAX_HALO_CANVAS
    order = np.argsort(r)
    r10 = r[order][np.searchsorted(np.cumsum(w[order]), 0.1 * w.sum())]
    a = max(pixel_size, 0.25 * r10, (max(ny, nx) * pixel_size + 2 * r.max()) / MAX_HALO_CANVAS)
    b = max(1, int(np.floor(a / pixel_size)))
    a = b * pixel_size  # the coarse pixel (A)
    cy, cx = -(-ny // b), -(-nx // b)
    s = np.zeros((cy * b, cx * b))
    s[:ny, :nx] = source
    coarse = s.reshape(cy, b, cx, b).sum(axis=(1, 3))

    # The kernel: each ring's weight spread over its circumference
    pad = int(np.ceil(r.max() / a)) + 1
    ky, kx = cy + 2 * pad, cx + 2 * pad
    fy, fx = 2 * ky, 2 * kx  # linear convolution
    yy = np.fft.fftfreq(fy, 1.0 / fy)[:, None] * a
    xx = np.fft.fftfreq(fx, 1.0 / fx)[None, :] * a
    rho = np.sqrt(yy**2 + xx**2)
    step = 0.25 * a
    edges = np.arange(0.0, r.max() + 2 * step, step)
    profile, _ = np.histogram(r, bins=edges, weights=w)
    centres = 0.5 * (edges[1:] + edges[:-1])
    density = profile / (2 * np.pi * np.maximum(centres, 1e-12) * step)  # per A^2
    kernel = np.interp(rho, centres, density, right=0.0) * a**2
    inner = w[r < 0.5 * a].sum()  # rings inside the centre pixel
    kernel[0, 0] = inner
    if kernel.sum() > 0:
        kernel *= 1.0 / kernel.sum()

    canvas = np.zeros((fy, fx))
    canvas[pad : pad + cy, pad : pad + cx] = coarse
    out = np.fft.irfft2(np.fft.rfft2(canvas) * np.fft.rfft2(kernel), s=(fy, fx))
    field = out[pad : pad + cy, pad : pad + cx]
    landed = float(field.sum())

    # Back onto the fine grid: the coarse density at each fine pixel's centre
    dens = field / b**2
    jy = (np.arange(ny) + 0.5) / b - 0.5
    jx = (np.arange(nx) + 0.5) / b - 0.5
    gy, gx = np.meshgrid(jy, jx, indexing="ij")
    fine = map_coordinates(dens, [gy, gx], order=1, mode="nearest")
    landed *= 1 - beyond
    if fine.sum() > 0:
        fine *= landed / fine.sum()
    return fine, {
        "coarse_binning": b,
        "coarse_pixel_A": a,
        "beyond_diagonal": beyond,
        "left_field": 1 - landed / total,
    }
