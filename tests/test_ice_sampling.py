#
# test_ice_sampling.py
#
# Without the band-limited potential, the GRF ice model's corrections are
# tabulated to a pixel area of 4 A^2 and clamp beyond it, so a coarser pixel
# would silently get a finer one's mean and variance. That case is refused.
#
import logging
import types

import pytest

import parakeet.config
import parakeet.simulate.engine as engine
from parakeet.simulate.engine import (
    MAX_CALIBRATED_ICE_PIXEL_AREA,
    check_ice_sampling,
)


def test_the_table_limit_is_two_angstrom_pixels():
    assert MAX_CALIBRATED_ICE_PIXEL_AREA == 4.0


@pytest.mark.parametrize("pixel_size", [0.5, 1.0, 1.5, 2.0])
def test_calibrated_pixels_pass_either_way(pixel_size):
    check_ice_sampling(pixel_size, band_limited=False)
    check_ice_sampling(pixel_size, band_limited=True)


@pytest.mark.parametrize("pixel_size", [2.01, 2.5, 3.0, 20.0])
def test_coarse_pixels_are_refused_without_band_limit(pixel_size):
    with pytest.raises(ValueError, match="allow_uncalibrated_sampling"):
        check_ice_sampling(pixel_size, band_limited=False)


@pytest.mark.parametrize("pixel_size", [2.5, 3.0, 20.0])
def test_coarse_pixels_pass_with_band_limit(pixel_size):
    check_ice_sampling(pixel_size, band_limited=True)


def test_the_override_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=engine.logger.name):
        check_ice_sampling(3.0, band_limited=False, allow_uncalibrated=True)
    assert any("clamp" in r.getMessage() for r in caplog.records)


def test_the_override_is_in_the_config_and_off_by_default():
    assert parakeet.config.Simulation().allow_uncalibrated_sampling is False
    assert (
        parakeet.config.Simulation(
            allow_uncalibrated_sampling=True
        ).allow_uncalibrated_sampling
        is True
    )


@pytest.mark.parametrize(
    "fake, expected",
    [
        (types.SimpleNamespace(), False),  # an engine with no switch predates it
        (types.SimpleNamespace(band_limited_potential=lambda: False), False),
        (types.SimpleNamespace(band_limited_potential=lambda: True), True),
    ],
)
def test_engine_band_limited(monkeypatch, fake, expected):
    monkeypatch.setattr(engine, "multem", fake, raising=False)
    assert engine.engine_band_limited() is expected


def test_masker_checks_before_building(monkeypatch):
    # An engine with the switch off and no Masker: the guard must raise before
    # the masker is built (which would be an AttributeError here)
    fake = types.SimpleNamespace(band_limited_potential=lambda: False)
    monkeypatch.setattr(engine, "multem", fake, raising=False)
    simulate = object.__new__(engine.SimulationEngine)
    simulation = parakeet.config.Simulation(ice=True).model_dump()
    with pytest.raises(ValueError, match="band-limited"):
        simulate.masker(0, 3.0, (0, 0), 0, (0, 0, 0), (0, 0, 0), None, None, simulation)
