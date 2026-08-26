#
# tests/test_energy_filter.py
#
# Tests for the energy filter slit.
#
# The slit width used to be inert in the zero_loss model -- a 10 eV and a 40 eV
# slit gave identical images -- because the model returned exp(-t/lambda)
# regardless. filter_transmission models the low-loss spectrum as a zero-loss
# peak plus Poisson-weighted plasmon replicas, each of finite width, and
# integrates the slit over it.
#
# Note the distinction from zero_loss_fraction, which is unchanged: that is the
# fraction of electrons IN the zero-loss peak, a property of the specimen. This
# is the fraction the filter TRANSMITS, which depends on the slit and is 1.0
# when the filter is retracted.
#

import math

import pytest

import parakeet.inelastic
from parakeet.inelastic import (
    DEFAULT_ZLP_FWHM,
    ICE_PLASMON_ENERGY,
    MEAN_FREE_PATH,
)


def cuboid(thickness_A):
    return {
        "type": "cuboid",
        "cuboid": {"length_x": 1e4, "length_y": 1e4, "length_z": thickness_A},
    }


def poisson(t_over_lambda, n):
    return t_over_lambda**n * math.exp(-t_over_lambda) / math.factorial(n)


def fraction(thickness=1000.0, angle=0.0, **kw):
    return parakeet.inelastic.filter_transmission(cuboid(thickness), angle, **kw)


@pytest.mark.parametrize("slit", [None, 0, -1])
def test_no_slit_means_retracted_and_transmits_everything(slit):
    # A retracted filter removes nothing. This is deliberately NOT the
    # zero-loss-peak-only case, which needs a narrow slit.
    assert fraction(slit_width=slit) == 1.0


def test_a_wide_slit_approaches_transmitting_everything():
    assert fraction(slit_width=500.0) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("thickness", [500.0, 1000.0, 2000.0])
def test_a_narrow_slit_gives_the_zero_loss_peak(thickness):
    # Narrow compared with the plasmon but wide compared with the source: the
    # classic exp(-t/lambda) zero-loss fraction.
    got = fraction(thickness=thickness, slit_width=5.0)
    assert got == pytest.approx(math.exp(-thickness / MEAN_FREE_PATH), rel=0.02)


def test_a_slit_narrower_than_the_source_cuts_the_zero_loss_peak():
    # The zero-loss peak has the width of the source, so a slit of comparable
    # width loses part of it. A delta-function model would miss this.
    zlp = math.exp(-1000.0 / MEAN_FREE_PATH)
    wide_enough = fraction(slit_width=5.0, zlp_fwhm=DEFAULT_ZLP_FWHM)
    at_the_fwhm = fraction(slit_width=DEFAULT_ZLP_FWHM, zlp_fwhm=DEFAULT_ZLP_FWHM)
    much_narrower = fraction(slit_width=0.2, zlp_fwhm=DEFAULT_ZLP_FWHM)

    assert wide_enough == pytest.approx(zlp, rel=0.02)
    # a slit of exactly one FWHM passes erf(sqrt(ln2)) of a Gaussian
    assert at_the_fwhm == pytest.approx(
        zlp * math.erf(math.sqrt(math.log(2))), rel=0.02
    )
    assert much_narrower < at_the_fwhm < wide_enough


def test_a_narrower_source_survives_a_narrow_slit_better():
    # A cold FEG through a 1 eV slit keeps more than a Schottky does.
    cold = fraction(slit_width=1.0, zlp_fwhm=0.3)
    schottky = fraction(slit_width=1.0, zlp_fwhm=0.7)
    assert cold > schottky
    # but with a normal slit the source width is irrelevant
    assert fraction(slit_width=20.0, zlp_fwhm=0.3) == pytest.approx(
        fraction(slit_width=20.0, zlp_fwhm=0.7), rel=1e-6
    )


def test_a_typical_slit_clips_the_plasmon_edge():
    # The plasmon is broad, so a 20 eV slit on the zero-loss peak is not a
    # clean cut: it admits the plasmon's leading edge and so passes slightly
    # more than the zero-loss peak alone.
    zlp = math.exp(-1000.0 / MEAN_FREE_PATH)
    assert fraction(slit_width=20.0) > zlp
    assert fraction(slit_width=20.0) < zlp * 1.1


def test_offsetting_the_slit_gives_plasmon_imaging():
    # Centred on the plasmon, the zero-loss peak falls outside the slit, so
    # the image is formed from inelastically scattered electrons only.
    tl = 1000.0 / MEAN_FREE_PATH
    got = fraction(slit_width=20.0, slit_centre=ICE_PLASMON_ENERGY)
    assert got < poisson(tl, 0)  # the zero-loss peak is excluded
    assert got == pytest.approx(poisson(tl, 1), rel=0.3)  # mostly the first plasmon


def test_transmission_increases_with_slit_width():
    widths = [1, 5, 20, 50, 100, 300]
    got = [fraction(slit_width=w) for w in widths]
    assert got == sorted(got)


def test_thicker_ice_transmits_less_through_a_zero_loss_slit():
    assert fraction(thickness=500.0, slit_width=20.0) > fraction(
        thickness=2000.0, slit_width=20.0
    )


def test_tilting_transmits_less():
    # The path through the ice goes as 1/cos(theta).
    assert fraction(angle=60.0, slit_width=20.0) < fraction(angle=0.0, slit_width=20.0)


def test_plasmon_energy_is_overridable():
    # A material whose plasmon sits lower admits it through a narrower slit.
    ice = fraction(slit_width=20.0)
    lower = fraction(slit_width=20.0, plasmon_energy=8.0)
    assert lower > ice


def test_zero_loss_fraction_is_a_specimen_property_not_a_filter_one():
    # zero_loss_fraction keeps its original meaning: the electrons in the
    # zero-loss peak, independent of any filter. A retracted filter transmits
    # everything but does not change how many electrons were unscattered.
    shape = cuboid(1000.0)
    assert parakeet.inelastic.zero_loss_fraction(shape, 0.0) == pytest.approx(
        math.exp(-1000.0 / MEAN_FREE_PATH)
    )
    assert parakeet.inelastic.filter_transmission(shape, 0.0, slit_width=None) == 1.0
