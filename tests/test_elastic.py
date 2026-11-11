#
# Copyright (C) 2026 Diamond Light Source and Rosalind Franklin Institute
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
"""The elastic scattering a band-limited grid cannot hold (parakeet.elastic)"""

from types import SimpleNamespace

import numpy as np
import pytest

import parakeet.config
import parakeet.elastic as E
import parakeet.inelastic

KV = 300.0
DX = 1.5
QN = E.grid_q(DX)
WATER = E.water_number_density(0.94)


def quad(element, q_lo, q_hi, model, weight=None, n=200001):
    """dsigma integrated by the trapezoid rule on a fine grid (A^2)"""
    q = np.linspace(q_lo, q_hi, n)
    lam, g = E.wavelength(KV), E.gamma(KV)
    f = E._dsigma_dq(element, lam, g, q, model)
    if weight is not None:
        f = f * weight(q)
    return float(np.trapezoid(f, q) if hasattr(np, "trapezoid") else np.trapz(f, q))


@pytest.mark.parametrize("model", E.MODELS)
@pytest.mark.parametrize("element", ["H", "C", "O", "Au"])
def test_the_closed_form_is_the_integral(element, model):
    for lo, hi in [(0.0, 0.2), (QN, 0.6), (0.1, 3.0)]:
        assert E.cross_section(element, KV, lo, hi, model) == pytest.approx(
            quad(element, lo, hi, model), rel=1e-6
        )


@pytest.mark.parametrize("model", E.MODELS)
def test_limits_and_monotonicity(model):
    assert E.cross_section("O", KV, 0.3, 0.3, model) == 0.0
    a = E.cross_section("O", KV, QN, 0.5, model)
    b = E.cross_section("O", KV, QN, 0.8, model)
    c = E.cross_section("O", KV, 0.4, 0.8, model)
    assert 0 < c < b and a < b
    total = E.cross_section("O", KV, 0.0, np.inf, model)
    assert total == pytest.approx(
        E.cross_section("O", KV, 0.0, QN, model) + E.cross_section("O", KV, QN, np.inf, model)
    )
    with pytest.raises(ValueError):
        E.cross_section("O", KV, 0.5, 0.4, model)


def test_the_two_models_agree_in_order_of_magnitude():
    # Lenz is a screened Rutherford atom, iam the tabulated one: the same
    # physics, not the same numbers
    for element in ("C", "O", "Au"):
        r = E.cross_section(element, KV, QN, np.inf, "lenz") / E.cross_section(
            element, KV, QN, np.inf, "iam"
        )
        assert 0.3 < r < 3, (element, r)


def test_the_missing_part_is_one_minus_the_window_squared():
    # No aperture: everything the grid leaves out is band, nothing lost
    lost, band = E.missing_cross_sections(WATER, KV, DX)
    assert lost == 0.0
    expect = 0.0
    for el, n in WATER.items():
        below = quad(el, 0.0, QN, "iam", weight=lambda q: 1 - E.window(q, DX) ** 2)
        expect += n * (below + E.cross_section(el, KV, QN, np.inf))
    assert band == pytest.approx(expect, rel=1e-5)


def test_an_aperture_splits_lost_from_band():
    lam = E.wavelength(KV)
    for beta in (4e-3, 11.7e-3):  # inside the grid's angles, and beyond them
        q_acc = beta / lam
        lost, band = E.missing_cross_sections(WATER, KV, DX, q_acc)
        _, everything = E.missing_cross_sections(WATER, KV, DX)
        assert lost + band == pytest.approx(everything, rel=1e-6)
        assert lost > 0 and band > 0


def test_density_scales_and_mixtures_add():
    a = E.missing_cross_sections(WATER, KV, DX)[1]
    b = E.missing_cross_sections({k: 2 * v for k, v in WATER.items()}, KV, DX)[1]
    assert b == pytest.approx(2 * a)
    carbon = {"C": 0.05}
    mix = {"H": WATER["H"], "O": WATER["O"], "C": 0.05}
    assert E.missing_cross_sections(mix, KV, DX)[1] == pytest.approx(
        a + E.missing_cross_sections(carbon, KV, DX)[1]
    )


def test_the_tail_beyond_the_fit_is_small_for_water():
    assert 0 < E.tail_fraction(WATER, KV, QN) < 0.02


def test_the_config_checks_a_composition():
    m = parakeet.config.Material(density=0.94, number_density_A3={"H": 0.06, "O": 0.03})
    assert m.number_density_A3 == {"H": 0.06, "O": 0.03}
    with pytest.raises(ValueError):
        parakeet.config.Material(number_density_A3={"Xx": 0.1})
    with pytest.raises(ValueError):
        parakeet.config.Material(number_density_A3={"O": -1.0})
    s = parakeet.config.Simulation(elastic_out={"model": "lenz"})
    assert s.elastic_out.model == "lenz" and s.elastic_out.halo is False
    assert parakeet.config.Simulation(elastic_out={"halo": True}).elastic_out.halo
    with pytest.raises(ValueError):
        parakeet.config.Simulation(elastic_out={"model": "mott"})


# -- the halo -----------------------------------------------------------------


def test_the_rings_follow_the_aberration_function():
    lam = E.wavelength(KV)
    c10, cs = -30000.0, 2.7e7
    q_acc = 11.7e-3 / lam
    r, w, rec = E.band_rings(WATER, KV, DX, q_acc, c10, cs)
    # Every ring is the displacement of some q in the band
    q = np.linspace(0.5 * QN, q_acc, 100001)
    radius = lam * q * np.abs(c10 + cs * lam**2 * q**2)
    assert r.min() >= radius.min() * (1 - 1e-3) and r.max() <= radius.max() * (1 + 1e-3)
    # and the weights are the band's cross-section, less the reported tail
    band = E.missing_cross_sections(WATER, KV, DX, q_acc)[1]
    assert w.sum() == pytest.approx(band * (1 - rec["tail_share"]), rel=1e-9)
    assert rec["tail_share"] < 1e-4


def test_a_uniform_source_gets_its_band_back_where_the_rings_fit():
    lam = E.wavelength(KV)
    r, w, _ = E.band_rings(WATER, KV, DX, 11.7e-3 / lam, -30000.0, 2.7e7)
    src = np.ones((600, 600))
    h, info = E.halo(src, DX, r, w)
    # Rings of 100-310 A, a field of 900 A: the centre receives what it gave
    assert h[300, 300] == pytest.approx(1.0, abs=1e-3)
    assert info["beyond_diagonal"] == 0.0
    # and the field loses only what lands past its edge
    assert 0 < info["left_field"] < 0.5
    assert h.sum() == pytest.approx(src.sum() * (1 - info["left_field"]), rel=1e-9)


def test_a_point_source_lands_on_its_rings():
    r = np.array([60.0])
    w = np.array([1.0])
    src = np.zeros((301, 301))
    src[150, 150] = 1.0
    h, info = E.halo(src, DX, r, w)
    assert h.sum() == pytest.approx(1.0, rel=1e-9) and info["left_field"] == pytest.approx(0, abs=1e-12)
    y, x = np.indices(h.shape)
    rho = np.hypot(y - 150, x - 150) * DX
    assert np.sum(h * rho) / h.sum() == pytest.approx(60.0, rel=0.05)


def test_rings_wider_than_the_field_leave_it():
    src = np.ones((64, 64))
    h, info = E.halo(src, DX, np.array([1e6]), np.array([1.0]))
    assert not h.any() and info["left_field"] == 1.0


# -- the optics' use of it, with no engine ------------------------------------

NX = 64
MARGIN = 4
R = 30.0


def simulator(document, materials=None, aperture=None, monkeypatch=None, band=True):
    from parakeet.simulate._optics import OpticsImageSimulator

    if monkeypatch is not None:
        import parakeet.simulate.engine

        monkeypatch.setattr(parakeet.simulate.engine, "engine_band_limited", lambda: band)
    field = NX * DX
    return OpticsImageSimulator(
        microscope=SimpleNamespace(
            detector=SimpleNamespace(origin=(0.0, 0.0), nx=NX, ny=NX, pixel_size=DX),
            beam=SimpleNamespace(energy=KV),
            objective_aperture_cutoff_freq=aperture,
        ),
        scan=SimpleNamespace(orientation=[(0.0, 0.0, 0.0)], position=[(0.0, 0.0, 0.0)]),
        simulation={
            "elastic_out": {"model": "iam", "halo": False},
            "materials": materials or {},
            "ice_parameters": {"density": 0.94},
            "inelastic_map_stride": 8,
        },
        sample={
            "regions": {"document": document},
            "shape": {
                "type": "cuboid",
                "cuboid": {"length_x": field, "length_y": field, "length_z": 2 * R + 20},
            },
            "centre": [field / 2, field / 2, R + 10],
        },
    )


def sphere_doc(material="water"):
    c = NX * DX / 2
    return {
        "schema": "parakeet-regions/1",
        "medium": "vacuum",
        "regions": [
            {
                "id": "s",
                "kind": "sphere",
                "material": material,
                "centre_A": [c, c, R + 10],
                "radius_A": R,
            }
        ],
    }


def test_the_amplitude_loss_is_the_exact_chord(monkeypatch):
    sim = simulator(sphere_doc(), monkeypatch=monkeypatch)
    tau, record = sim.elastic_out(0, MARGIN)
    mu = sum(E.missing_cross_sections(WATER, KV, DX))
    k = np.arange(NX + 2 * MARGIN)
    x = (k - MARGIN) * DX - NX * DX / 2
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    chord = 2 * np.sqrt(np.clip(R**2 - r2, 0, None))
    assert np.max(np.abs(tau - mu * chord)) < 1e-9
    assert record["regime"].startswith("band-limited")
    assert record["materials"]["water"]["lambda_lost_A"] is None  # no aperture
    assert record["materials_without_composition"] == []


def test_without_the_band_limited_potential_it_refuses(monkeypatch):
    sim = simulator(sphere_doc(), monkeypatch=monkeypatch, band=False)
    with pytest.raises(ValueError, match="band-limited"):
        sim.elastic_out(0, MARGIN)


def test_a_material_without_a_composition_is_named(monkeypatch, caplog):
    sim = simulator(
        sphere_doc("glass"), materials={"glass": {"density": 1.0}}, monkeypatch=monkeypatch
    )
    tau, record = sim.elastic_out(0, MARGIN)
    assert not tau.any()
    assert record["materials_without_composition"] == ["glass"]
    assert "glass" in caplog.text


def test_a_material_with_a_composition_uses_it(monkeypatch):
    dense = {k: 2 * v for k, v in WATER.items()}
    sim = simulator(
        sphere_doc("dense"),
        materials={"dense": {"density": 1.88, "number_density_A3": dense}},
        monkeypatch=monkeypatch,
    )
    tau, _ = sim.elastic_out(0, MARGIN)
    water, _ = simulator(sphere_doc(), monkeypatch=monkeypatch).elastic_out(0, MARGIN)
    assert np.allclose(tau, 2 * water, rtol=1e-12, atol=0)


def test_the_zero_loss_depth_is_unchanged_by_sharing_the_paths():
    # Without elastic_out the shared maps are the sampled ones the zero-loss
    # filter always used (parakeet.inelastic.optical_depth_map)
    import parakeet.config as C
    import parakeet.sample._new
    import parakeet.sample._regions

    sim = simulator(sphere_doc())
    sim.simulation = dict(sim.simulation, elastic_out=None)
    regions = parakeet.sample._regions.parse(
        parakeet.sample._new.regions_document(C.SampleRegions(document=sphere_doc()))
    )
    expect = parakeet.inelastic.optical_depth_map(
        regions,
        {"water": parakeet.inelastic.MEAN_FREE_PATH},
        sim.sample["shape"],
        sim.sample["centre"],
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0),
        NX,
        NX,
        MARGIN,
        DX,
        stride=8,
    )
    assert np.array_equal(sim.optical_depth(0, MARGIN), expect)
