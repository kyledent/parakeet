#
# test_ice_presets.py
#
# The shipped ice spectrum presets: opt-in only, each a valid table whose V0 is
# its own integral by the engine's rule, and a config that names one survives
# being written out and read back.
#
import json

import numpy as np
import pytest

from parakeet.config import IceParameters, Material, Simulation
from parakeet.data import ice_spectrum_path

PRESETS = ("vitreous_ice_md", "liquid_water_md")


def table_integral(q, P):
    """int P 2 pi q dq, held at P(q0) below the first knot (the engine's rule)"""
    q = np.concatenate([[0.0], q])
    P = np.concatenate([[P[0]], P])
    return float(np.sum(0.5 * (P[1:] * q[1:] + P[:-1] * q[:-1]) * np.diff(q)) * 2 * np.pi)


def test_the_default_ice_has_no_table():
    ice = IceParameters()
    assert ice.preset is None
    assert ice.spectrum_table is None
    assert ice.variance_constant is None


@pytest.mark.parametrize("name", PRESETS)
def test_a_preset_sets_its_table_and_its_variance(name):
    with open(ice_spectrum_path(name)) as f:
        shipped = json.load(f)
    ice = IceParameters(preset=name)
    assert ice.spectrum_table.q == shipped["q"]
    assert ice.spectrum_table.P == shipped["P"]
    assert ice.variance_constant == shipped["V0"]
    assert shipped["V0"] == pytest.approx(
        table_integral(np.array(shipped["q"]), np.array(shipped["P"])), rel=1e-9
    )


def test_a_given_variance_constant_is_kept():
    ice = IceParameters(preset="vitreous_ice_md", variance_constant=123.0)
    assert ice.variance_constant == 123.0


def test_a_preset_and_another_table_are_refused():
    with pytest.raises(ValueError, match="preset or a spectrum_table"):
        IceParameters(
            preset="vitreous_ice_md", spectrum_table={"q": [0.1, 0.2], "P": [1.0, 1.0]}
        )


def test_a_material_takes_a_preset():
    m = Material(preset="liquid_water_md", density=1.0)
    assert m.spectrum_table is not None


def test_a_preset_config_survives_a_round_trip():
    s = Simulation(ice=True, ice_parameters={"preset": "vitreous_ice_md"})
    again = Simulation(**s.model_dump())
    assert again.ice_parameters.spectrum_table.q == s.ice_parameters.spectrum_table.q
    assert again.ice_parameters.variance_constant == s.ice_parameters.variance_constant


def test_an_unknown_preset_is_refused():
    with pytest.raises(ValueError):
        IceParameters(preset="hexagonal_ice")
