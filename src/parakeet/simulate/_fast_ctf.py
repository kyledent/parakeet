"""
Pure-Python CTF computation matching MULTEM's output.

Replaces the expensive SimulationEngine + multem.compute_ctf() round-trip
with direct NumPy/CuPy computation.

The CTF includes:
- Aberration phase function chi(q) with C1 (defocus) and C3 (Cs)
- Temporal coherence envelope E_t (defocus spread)
- Spatial coherence envelope E_s (source size)

Sign convention: chi uses negative sign (MULTEM convention).

Relationship to MULTEM
----------------------
This is an approximation, not a transcription. MULTEM's apply_PCTF
(cgpu_fcns.cuh) uses the Ishizuka *coupled* partial-coherence envelope,

    u      = 1 + (pi * theta_c * delta_t)^2 * g^2
    st_inc = exp(-(spa_inc + temp_inc) / u) / sqrt(u)

whereas the envelopes below are applied independently, with no 1/sqrt(u)
prefactor and no coupling term. Since theta_c and delta_t enter u as a
product, the discrepancy grows with both. Measured max |fast - multem| over
a 256^2 grid at 300 kV, C1 = -2 um:

    energy_spread / semiangle    0.02 (default)     0.1        0.5
    2.66e-6 (default)                3.8e-05     6.7e-06    6.0e-06
    1e-5                             1.5e-05     7.3e-05    7.6e-05
    1e-4                             9.9e-05     2.1e-03    7.4e-03
    1e-3                             9.9e-04     2.3e-02    1.8e-01

So the approximation holds to ~1e-4 across realistic cryo-EM coherence and
degrades by orders of magnitude outside it. tests/test_fast_ctf.py pins both
the agreement and that boundary. If you need exactness for a low-coherence
source, call MULTEM instead.

Reference: Kirkland, "Advanced Computing in Electron Microscopy" (2010).
"""

import logging
import numpy as np
from math import pi, sqrt

from parakeet.simulate._fft_backend import HAS_CUPY

if HAS_CUPY:
    import cupy as cp

logger = logging.getLogger(__name__)

# Physical constants
_M0C2 = 510.99895  # electron rest mass energy in keV
_HC = 12.3984198  # h*c in keV*A


def electron_wavelength(energy_kev):
    """Relativistic electron wavelength in Angstroms."""
    return _HC / sqrt(energy_kev * (2 * _M0C2 + energy_kev))


def compute_ctf(
    shape,
    pixel_size,
    energy_kev,
    c_10,
    c_30_mm,
    sigma_df,
    sigma_source,
    q2=None,
    q=None,
):
    """
    Compute the CTF with temporal and spatial coherence envelopes.

    Uses CuPy (GPU) if available, otherwise NumPy.

    Args:
        shape: (ny, nx) grid dimensions
        pixel_size: Real-space pixel size in Angstroms
        energy_kev: Beam energy in keV
        c_10: Defocus in Angstroms (negative = underfocus)
        c_30_mm: Spherical aberration Cs in mm
        sigma_df: Defocus spread sigma in Angstroms
            (from multem.iehwgd_to_sigma(defocus_spread(...)))
        sigma_source: Source size sigma in 1/A
            (from multem.mrad_to_sigma(energy, illumination_semiangle))
        q2: Pre-computed q^2 grid (optional, for caching). NumPy or CuPy array.
        q: Pre-computed |q| grid (optional, for caching). NumPy or CuPy array.

    Returns:
        Complex CTF array of shape (ny, nx). NumPy array (transferred from GPU
        if CuPy was used).
    """
    xp = cp if HAS_CUPY else np
    lam = electron_wavelength(energy_kev)
    c_30 = c_30_mm * 1e7  # mm -> A

    # Frequency grids
    if q2 is None:
        qy = xp.fft.fftfreq(shape[0], d=pixel_size)
        qx = xp.fft.fftfreq(shape[1], d=pixel_size)
        q2 = qy[:, None] ** 2 + qx[None, :] ** 2
    elif HAS_CUPY and not isinstance(q2, cp.ndarray):
        q2 = cp.asarray(q2)
    if q is None:
        q = xp.sqrt(q2)
    elif HAS_CUPY and not isinstance(q, cp.ndarray):
        q = cp.asarray(q)

    # Aberration phase (negative sign = MULTEM convention)
    chi = -(pi * lam * c_10 * q2 + 0.5 * pi * lam**3 * c_30 * q2**2)

    # Temporal coherence envelope
    E_t = xp.exp(-0.5 * (pi * lam * sigma_df) ** 2 * q2**2)

    # Spatial coherence envelope
    dchi_dq = 2 * pi * lam * c_10 * q + 2 * pi * lam**3 * c_30 * q**3
    E_s = xp.exp(-0.5 * sigma_source**2 * dchi_dq**2)

    # Bandwidth limit at Nyquist (match MULTEM behaviour)
    q_nyquist = 1.0 / (2.0 * pixel_size)
    bwl = q < q_nyquist

    ctf = xp.exp(1j * chi) * E_t * E_s * bwl

    if HAS_CUPY:
        return cp.asnumpy(ctf)
    return ctf
