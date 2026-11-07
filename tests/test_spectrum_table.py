#
# test_spectrum_table.py
#
# A tabulated GRF spectrum and the ice's own mean and variance per molecule:
# the config checks a table, reads one from a file, and the engine passes both
# to python-multem, refusing a table the loaded engine cannot take. CPU only.
#
import json
import types

import pydantic
import pytest

import parakeet.config
import parakeet.simulate.engine as engine
from parakeet.engine_check import EngineFeatureError


def test_a_table_is_checked():
    parakeet.config.SpectrumTable(q=[0.01, 0.5], P=[1.0, 2.0])
    for q, P, message in (
        ([0.01], [1.0], "at least 2 knots"),
        ([0.01, 0.5], [1.0], "at least 2 knots"),
        ([0.0, 0.5], [1.0, 2.0], "start above 0"),
        ([0.5, 0.5], [1.0, 2.0], "increase"),
        ([0.01, 0.5], [1.0, -2.0], "negative"),
        ([0.01, float("nan")], [1.0, 2.0], "finite"),
    ):
        with pytest.raises(pydantic.ValidationError, match=message):
            parakeet.config.SpectrumTable(q=q, P=P)


def test_a_table_is_read_from_a_file(tmp_path):
    path = tmp_path / "table.json"
    path.write_text(json.dumps({"q": [0.01, 0.2, 0.5], "P": [3.0, 5.0, 1.0], "V0": 9}))
    table = parakeet.config.SpectrumTable(file=str(path))
    assert table.q == [0.01, 0.2, 0.5] and table.P == [3.0, 5.0, 1.0]
    with pytest.raises(pydantic.ValidationError, match="not both"):
        parakeet.config.SpectrumTable(file=str(path), q=[0.1, 0.2], P=[1, 1])


def test_the_ice_and_every_material_can_carry_a_table_and_constants():
    for model in (parakeet.config.IceParameters, parakeet.config.Material):
        m = model(
            spectrum_table={"q": [0.01, 0.5], "P": [1, 2]}, variance_constant=11195.5
        )
        assert m.spectrum_table.P == [1, 2] and m.variance_constant == 11195.5
        assert model().spectrum_table is None and model().mean_constant is None


class Recorder:
    """A stand-in for multem.IceParameters that records what it is given"""

    def __init__(self, table=True):
        self.m0 = self.v0 = -1.0
        self.calls = []
        if table:
            self.spectrum_q = []

    def set_spectrum_table(self, q, P):
        self.calls.append((q, P))


def fake_multem(table=True):
    module = types.ModuleType("multem")
    module.IceParameters = lambda: Recorder(table)
    module.Masker = type("Masker", (), {"set_seed": lambda self, s: None})
    module.MultiRegionMasker = object
    return module


def test_the_table_and_constants_reach_the_engine(monkeypatch):
    monkeypatch.setattr(engine, "multem", fake_multem(), raising=False)
    material = parakeet.config.IceParameters(
        spectrum_table={"q": [0.01, 0.5], "P": [1, 2]},
        mean_constant=150.0,
        variance_constant=11195.5,
    ).model_dump()
    p = engine.grf_parameters(material)
    assert p.calls == [([0.01, 0.5], [1, 2])]
    assert p.m0 == 150.0 and p.v0 == 11195.5


def test_without_a_table_nothing_is_set(monkeypatch):
    monkeypatch.setattr(engine, "multem", fake_multem(), raising=False)
    p = engine.grf_parameters(parakeet.config.IceParameters().model_dump())
    assert p.calls == [] and p.m0 == -1.0 and p.v0 == -1.0


def test_a_table_on_an_engine_without_one_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(engine, "multem", fake_multem(table=False), raising=False)
    material = parakeet.config.IceParameters(
        spectrum_table={"q": [0.01, 0.5], "P": [1, 2]}
    ).model_dump()
    with pytest.raises(EngineFeatureError, match="spectrum_table"):
        engine.grf_parameters(material)


def test_plain_ice_with_its_own_constants_goes_through_the_regions_path():
    assert not engine.own_constants(parakeet.config.IceParameters().model_dump())
    assert engine.own_constants(
        parakeet.config.IceParameters(variance_constant=11195.5).model_dump()
    )
    import parakeet.sample._regions as R

    medium, regions = R.parse(engine.PLAIN_ICE)
    assert medium == "water" and regions == []
