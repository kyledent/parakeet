#
# tests/test_fast_ctf.py
#
# Parity tests for the pure-Python CTF against MULTEM's own implementation.
#
# The optics stage no longer builds a SimulationEngine just to call
# multem.compute_ctf(); it evaluates the CTF directly (see
# parakeet.simulate._fast_ctf). That replacement sits underneath every
# simulated image, so it is pinned here rather than trusted to a one-off
# comparison.
#
# Note the agreement is REGIME-BOUNDED, not exact. MULTEM's apply_PCTF uses
# the Ishizuka coupled envelope (1/sqrt(u) prefactor, spatial/temporal terms
# divided by u); _fast_ctf applies the two envelopes independently. They
# match closely for realistic cryo-EM coherence and drift apart as the
# defocus spread grows. Both facts are pinned below, so neither a regression
# in the agreement nor a silent change to the approximation goes unnoticed.
#

import numpy as np
import pytest

import multem

import parakeet.config
import parakeet.microscope
from parakeet.simulate._fast_ctf import compute_ctf, electron_wavelength
from parakeet.simulate.engine import SimulationEngine, defocus_spread

# MULTEM and the direct evaluation agree to ~1e-5; allow a little headroom so
# the test pins behaviour without tripping on last-bit float differences.
CTF_ATOL = 1e-4


def _microscope(energy=300, c_10=-20000, c_30=2.7, nx=256, ny=256, pixel_size=1.0):
    """A microscope with an explicit, fully-specified lens and detector."""
    config = parakeet.config.Microscope()
    config.beam.energy = energy
    config.lens.c_10 = c_10
    config.lens.c_30 = c_30
    config.detector.nx = nx
    config.detector.ny = ny
    config.detector.pixel_size = pixel_size
    return parakeet.microscope.new(config)


def _reference_ctf(microscope, margin=0, slice_thickness=5.0):
    """The CTF as MULTEM computes it, via the engine the optics stage used to build."""
    nx = microscope.detector.nx
    ny = microscope.detector.ny
    pixel_size = microscope.detector.pixel_size
    x_fov = nx * pixel_size
    y_fov = ny * pixel_size
    offset = margin * pixel_size

    engine = SimulationEngine("gpu", 0, microscope, slice_thickness, margin, "HRTEM")
    engine.input.spec_lx = x_fov + offset * 2
    engine.input.spec_ly = y_fov + offset * 2
    engine.input.spec_lz = x_fov
    return engine.ctf()


def _fast_ctf(microscope, margin=0):
    """The CTF as the optics stage now computes it."""
    nx = microscope.detector.nx
    ny = microscope.detector.ny
    pixel_size = microscope.detector.pixel_size

    Cc_A = microscope.lens.c_c * 1e7  # mm -> A
    ds = defocus_spread(
        Cc_A,
        microscope.beam.energy_spread,
        microscope.lens.current_spread,
        microscope.beam.acceleration_voltage_spread,
    )
    return compute_ctf(
        (ny + 2 * margin, nx + 2 * margin),
        pixel_size,
        microscope.beam.energy,
        microscope.lens.c_10,
        microscope.lens.c_30,
        multem.iehwgd_to_sigma(ds),
        multem.mrad_to_sigma(
            microscope.beam.energy, microscope.beam.illumination_semiangle
        ),
    )


def test_electron_wavelength():
    # Standard relativistic values, Kirkland table 2.1
    assert electron_wavelength(300) == pytest.approx(0.019687, abs=1e-6)
    assert electron_wavelength(200) == pytest.approx(0.025079, abs=1e-6)
    assert electron_wavelength(100) == pytest.approx(0.037014, abs=1e-6)


@pytest.mark.parametrize("energy,c_10", [(300, -20000), (300, -5000), (200, -30000)])
def test_fast_ctf_matches_multem(energy, c_10):
    microscope = _microscope(energy=energy, c_10=c_10)
    reference = _reference_ctf(microscope)
    fast = _fast_ctf(microscope)

    assert fast.shape == reference.shape
    assert np.max(np.abs(fast - reference)) < CTF_ATOL


def test_fast_ctf_matches_multem_with_margin():
    # The optics stage evaluates on the detector grid plus its margin.
    microscope = _microscope()
    margin = 16
    reference = _reference_ctf(microscope, margin=margin)
    fast = _fast_ctf(microscope, margin=margin)

    assert fast.shape == (
        microscope.detector.ny + 2 * margin,
        microscope.detector.nx + 2 * margin,
    )
    assert np.max(np.abs(fast - reference)) < CTF_ATOL


def test_fast_ctf_is_bandwidth_limited_at_nyquist():
    # Everything beyond Nyquist must be zeroed, matching MULTEM.
    pixel_size = 1.0
    microscope = _microscope(pixel_size=pixel_size)
    ctf = _fast_ctf(microscope)

    n = microscope.detector.nx
    qy = np.fft.fftfreq(n, d=pixel_size)
    qx = np.fft.fftfreq(n, d=pixel_size)
    q = np.sqrt(qy[:, None] ** 2 + qx[None, :] ** 2)

    assert np.all(ctf[q >= 1.0 / (2.0 * pixel_size)] == 0)
    assert np.any(np.abs(ctf[q < 1.0 / (2.0 * pixel_size)]) > 0)


def test_precomputed_grids_match_internal_grids():
    # _optics.py passes cached q2/q grids; that path must not change the answer.
    microscope = _microscope()
    nx = microscope.detector.nx
    ny = microscope.detector.ny
    pixel_size = microscope.detector.pixel_size

    qy = np.fft.fftfreq(ny, d=pixel_size)
    qx = np.fft.fftfreq(nx, d=pixel_size)
    q2 = qy[:, None] ** 2 + qx[None, :] ** 2

    args = ((ny, nx), pixel_size, 300, -20000, 2.7, 30.0, 1e-5)
    assert np.allclose(
        compute_ctf(*args), compute_ctf(*args, q2=q2, q=np.sqrt(q2)), atol=1e-12
    )


# theta_c and delta_t enter MULTEM's coupling term u as a product, so the
# agreement has to be checked over both, not just one.
@pytest.mark.parametrize("semiangle", [0.02, 0.1, 0.5])
@pytest.mark.parametrize("energy_spread", [2.66e-6, 1.0e-5])
def test_fast_ctf_matches_multem_across_realistic_coherence(semiangle, energy_spread):
    # Across the coherence range real cryo-EM sources occupy, the decoupled
    # envelopes stay within tolerance of MULTEM's coupled form.
    microscope = _microscope()
    microscope.beam.illumination_semiangle = semiangle
    microscope.beam.energy_spread = energy_spread
    assert np.max(np.abs(_fast_ctf(microscope) - _reference_ctf(microscope))) < CTF_ATOL


def test_fast_ctf_diverges_from_multem_outside_realistic_coherence():
    # Documents the limit of the approximation rather than hiding it. Well
    # outside realistic cryo-EM coherence the missing coupling term produces a
    # difference three orders of magnitude above the working tolerance. If this
    # ever starts agreeing, the coupled envelope was implemented and both this
    # test and the _fast_ctf docstring need updating.
    microscope = _microscope()
    microscope.beam.illumination_semiangle = 0.5
    microscope.beam.energy_spread = 1.0e-3
    difference = np.max(np.abs(_fast_ctf(microscope) - _reference_ctf(microscope)))
    assert difference > 1e-2
