#
# tests/test_energy_filter.py
#
# Tests for the zero-loss energy filter slit.
#
# The slit width used to be inert in the zero_loss model -- a 10 eV and a 40 eV
# slit gave identical images -- because the model returned exp(-t/lambda)
# regardless. It now admits plasmon replicas that fall inside the slit, which
# is the Poisson plural-scattering picture, and these pin that behaviour and
# its backwards compatibility.
#

import math

import pytest

import parakeet.inelastic
from parakeet.inelastic import ICE_PLASMON_ENERGY, MEAN_FREE_PATH


def cuboid(thickness_A):
    return {
        "type": "cuboid",
        "cuboid": {"length_x": 1e4, "length_y": 1e4, "length_z": thickness_A},
    }


def poisson(t_over_lambda, n):
    return t_over_lambda**n * math.exp(-t_over_lambda) / math.factorial(n)


@pytest.mark.parametrize("thickness", [500.0, 1000.0, 2000.0])
def test_no_slit_is_the_zero_loss_peak(thickness):
    # With no slit the answer is P(0) = exp(-t/lambda), as it always was.
    got = parakeet.inelastic.zero_loss_fraction(cuboid(thickness), 0.0)
    assert got == pytest.approx(math.exp(-thickness / MEAN_FREE_PATH))


@pytest.mark.parametrize("slit", [None, 0, 10, 20, 40])
def test_narrow_slits_are_unchanged(slit):
    # Anything narrower than two plasmon energies admits no replica, so every
    # slit in normal use gives exactly the previous behaviour. This is what
    # makes the change safe for existing configs.
    shape = cuboid(1000.0)
    assert parakeet.inelastic.zero_loss_fraction(
        shape, 0.0, slit_width=slit
    ) == pytest.approx(math.exp(-1000.0 / MEAN_FREE_PATH))


def test_slit_admits_the_first_plasmon():
    # A slit reaching past the plasmon energy on each side passes P(0) + P(1).
    shape = cuboid(1000.0)
    tl = 1000.0 / MEAN_FREE_PATH
    wide = 2 * ICE_PLASMON_ENERGY + 2.0
    assert parakeet.inelastic.zero_loss_fraction(
        shape, 0.0, slit_width=wide
    ) == pytest.approx(poisson(tl, 0) + poisson(tl, 1))


def test_slit_admits_the_second_plasmon():
    shape = cuboid(1000.0)
    tl = 1000.0 / MEAN_FREE_PATH
    wide = 4 * ICE_PLASMON_ENERGY + 2.0
    assert parakeet.inelastic.zero_loss_fraction(
        shape, 0.0, slit_width=wide
    ) == pytest.approx(poisson(tl, 0) + poisson(tl, 1) + poisson(tl, 2))


def test_transmission_increases_with_slit_width():
    shape = cuboid(1000.0)
    widths = [10, 50, 100, 200, 400]
    got = [
        parakeet.inelastic.zero_loss_fraction(shape, 0.0, slit_width=w) for w in widths
    ]
    assert got == sorted(got)
    # a very wide slit approaches passing everything
    assert got[-1] > 0.99


def test_thicker_ice_transmits_less():
    for slit in (None, 60.0):
        thin = parakeet.inelastic.zero_loss_fraction(
            cuboid(500.0), 0.0, slit_width=slit
        )
        thick = parakeet.inelastic.zero_loss_fraction(
            cuboid(2000.0), 0.0, slit_width=slit
        )
        assert thin > thick


def test_tilting_transmits_less():
    # The path through the ice goes as 1/cos(theta).
    shape = cuboid(1000.0)
    flat = parakeet.inelastic.zero_loss_fraction(shape, 0.0, slit_width=60.0)
    tilted = parakeet.inelastic.zero_loss_fraction(shape, 60.0, slit_width=60.0)
    assert tilted < flat


def test_plasmon_energy_is_overridable():
    # A material with a lower plasmon energy admits replicas through a narrower
    # slit than ice would.
    shape = cuboid(1000.0)
    ice = parakeet.inelastic.zero_loss_fraction(shape, 0.0, slit_width=30.0)
    other = parakeet.inelastic.zero_loss_fraction(
        shape, 0.0, slit_width=30.0, plasmon_energy=10.0
    )
    assert other > ice
