#
# test_seeding.py
#
# Seeds for the GRF ice and the detector noise, derived from one base seed. The
# derivation must give the same values in every process and on every run, so a
# table of values pins it; the engine's side is checked with a stand-in.
#
import types

import numpy as np
import pytest

import parakeet.config
import parakeet.io
import parakeet.seeding as seeding
import parakeet.simulate._exit_wave as exit_wave_module
import parakeet.simulate.engine as engine
from parakeet.engine_check import EngineFeatureError

# (base, image_number) -> grf_seed. If these change, every seeded field changes
# and every cached render keyed on a seed is stale.
GRF_SEEDS = {
    (0, 0): 1817441351,
    (1234, 0): 752531711,
    (1234, 1): 776020064,
    (1234, 40): 1941713578,
    (2**62, 7): 993647843,
}


@pytest.mark.parametrize("key, value", sorted(GRF_SEEDS.items()))
def test_grf_seed_table(key, value):
    assert seeding.grf_seed(*key) == value


def test_grf_seeds_stay_inside_the_engines_range():
    # minstd takes seeds mod 2^31 - 1 and maps 0 to 1: inside 1 .. 2^31 - 2
    # neither happens, so two derived seeds cannot collide through it
    values = [
        seeding.grf_seed(base, i) for base in (0, 1, 99, 2**62) for i in range(500)
    ]
    assert min(values) >= seeding.GRF_SEED_MIN == 1
    assert max(values) <= seeding.GRF_SEED_MAX == 2**31 - 2


def test_each_image_gets_its_own_ice():
    values = [seeding.grf_seed(1234, i) for i in range(1000)]
    assert len(set(values)) == len(values)
    assert seeding.grf_seed(1234, 0) != seeding.grf_seed(1235, 0)


def test_detector_noise_is_per_frame_and_reproducible():
    def draw(*args):
        return seeding.detector_rng(*args).integers(0, 2**31, 3).tolist()

    assert draw(1234, 3, 5) == [374650386, 755691912, 440915922]
    assert draw(1234, 3, 5) == draw(1234, 3, 5)
    assert draw(1234, 3, 5) != draw(1234, 3, 6)
    assert draw(1234, 3, 5) != draw(1234, 4, 5)


def test_resolve():
    assert seeding.resolve(42) == 42
    drawn = [seeding.resolve(None) for _ in range(4)]
    assert all(0 <= d <= seeding.BASE_SEED_MAX for d in drawn)
    assert len(set(drawn)) == 4
    with pytest.raises(ValueError):
        seeding.resolve(-1)


def test_config():
    assert parakeet.config.Simulation().seed is None
    assert parakeet.config.Simulation(seed=7).seed == 7
    with pytest.raises(ValueError):
        parakeet.config.Simulation(seed=-1)


class FakeMasker:
    def __init__(self, *args):
        self.calls = {}

    def __getattr__(self, name):
        # Every setter but set_seed, whose presence is the feature under test
        if name.startswith("set_") and name != "set_seed":
            return lambda *args: self.calls.setdefault(name, []).append(args)
        raise AttributeError(name)


def fake_engine(seedable=True):
    class Masker(FakeMasker):
        pass

    if seedable:
        Masker.set_seed = lambda self, seed: self.calls.setdefault(
            "set_seed", []
        ).append((seed,))
    return types.SimpleNamespace(
        Masker=Masker,
        IceParameters=lambda: types.SimpleNamespace(),
        band_limited_potential=lambda: True,
    )


def build_masker(monkeypatch, fake, seed, image_number=3):
    monkeypatch.setattr(engine, "multem", fake, raising=False)
    simulate = object.__new__(engine.SimulationEngine)
    simulate.input = types.SimpleNamespace(nx=64, ny=64)
    simulation = parakeet.config.Simulation(ice=True, seed=seed).model_dump()
    sample = types.SimpleNamespace(
        shape={
            "type": "cuboid",
            "cuboid": {"length_x": 100.0, "length_y": 100.0, "length_z": 50.0},
        },
        centre=(50.0, 50.0, 25.0),
    )
    scan = types.SimpleNamespace(
        is_uniform_angular_scan=True,
        image_number=np.array([0, 0, 0, image_number]),
    )
    return simulate.masker(
        3, 1.5, (0, 0), 0, (0, 0, 0), (0, 0, 0), sample, scan, simulation
    )


def test_masker_is_seeded_per_image(monkeypatch):
    masker = build_masker(monkeypatch, fake_engine(), seed=1234, image_number=40)
    assert masker.calls["set_seed"] == [(GRF_SEEDS[(1234, 40)],)]


def test_unseeded_masker_is_left_alone(monkeypatch):
    masker = build_masker(monkeypatch, fake_engine(), seed=None)
    assert "set_seed" not in masker.calls


def test_a_seed_needs_an_engine_that_takes_one(monkeypatch):
    with pytest.raises(EngineFeatureError, match="grf_seed"):
        build_masker(monkeypatch, fake_engine(seedable=False), seed=1234)


@pytest.mark.parametrize(
    "simulation, seedable, expect",
    [
        ({"ice": True, "seed": 9}, True, 9),
        ({"ice": True, "seed": 9}, False, 9),  # kept; the masker refuses it later
        ({"ice": True, "seed": None}, False, None),  # unseedable engine: as before
        ({"ice": False, "seed": None}, True, None),  # no ice, nothing to seed
    ],
)
def test_resolve_ice_seed(monkeypatch, simulation, seedable, expect):
    monkeypatch.setattr(
        exit_wave_module, "engine_features", lambda: {"grf_seed": seedable}
    )
    assert exit_wave_module.resolve_ice_seed(simulation).get("seed") == expect


def test_resolve_ice_seed_draws_one_when_the_engine_takes_it(monkeypatch):
    monkeypatch.setattr(exit_wave_module, "engine_features", lambda: {"grf_seed": True})
    seed = exit_wave_module.resolve_ice_seed({"ice": True, "seed": None})["seed"]
    assert isinstance(seed, int) and 0 <= seed <= seeding.BASE_SEED_MAX


def test_seed_fields_are_in_the_header():
    for key in ("seed", "grf_seed", "noise_seed"):
        assert parakeet.io.METADATA_DTYPE.fields[key][0] == np.dtype("i8")


def test_an_older_nexus_file_reads_a_missing_field_as_zero():
    handle = {"data": np.zeros((3, 4, 4)), "dose": np.array([1.0, 2.0, 3.0])}
    header = parakeet.io.NexusHeader(handle)
    assert header.get(1, "dose") == 2.0
    assert header.get(1, "seed") == 0
