#
# test_detector_profiles.py
#
# Named detector profiles: the MTF follows the detector, mode and exact voltage
# with no substitution between voltages, apply() fills a detector config, and
# benchmark/make_config.py uses the same table rather than a copy.
#
import importlib.util
import os

import pytest

import parakeet.config
import parakeet.detector_profiles as dp


def test_names():
    assert dp.names() == ["falcon4i", "k3"]
    assert dp.READOUT_MODES == ["cds", "counting", "standard"]


def test_mtf_follows_detector_mode_and_voltage():
    assert dp.mtf_for("k3", 300) == "k3 standard 300kV"
    assert dp.mtf_for("k3", 200, mode="cds") == "k3 CDS 200kV"
    assert dp.mtf_for("falcon4i", 300) == "falcon4EC_300kV"
    assert dp.mtf_for("falcon4i", 200.0) == "falcon4EC_200kV"


def test_no_substitution_between_voltages():
    assert dp.mtf_for("k3", 250) is None
    assert dp.mtf_for("falcon4i", 120) is None


def test_unknown_names_are_errors():
    with pytest.raises(ValueError, match="falcon4i, k3"):
        dp.mtf_for("de64", 300)
    with pytest.raises(ValueError, match="cds, standard"):
        dp.mtf_for("k3", 300, mode="counting")


def test_apply_to_a_config_dict():
    detector = {"nx": 1000, "ny": 1000, "pixel_size": 1, "dqe": True, "mtf": None}
    warnings = dp.apply(detector, "falcon4i", kv=300)
    assert detector == {
        "nx": 4096,
        "ny": 4096,
        "pixel_size": 1.20,
        "dqe": True,
        "mtf": "falcon4EC_300kV",
    }
    assert warnings == []  # a built-in preset: found


def test_apply_to_a_config_object_with_a_pixel_size():
    detector = parakeet.config.Detector()
    dp.apply(detector, "k3", kv=300, mode="cds", pixel_size=0.83)
    assert (detector.nx, detector.ny, detector.pixel_size) == (5760, 4092, 0.83)
    assert detector.mtf == "k3 CDS 300kV"


def test_apply_warns_without_a_curve():
    detector = {}
    warnings = dp.apply(detector, "k3", kv=120)
    assert detector["mtf"] is None
    assert len(warnings) == 1 and "120 kV has no MTF curve" in warnings[0]


def test_apply_warns_naming_a_curve_it_cannot_find(monkeypatch, tmp_path):
    # the K3 curves are not distributed: with an empty search path they are not found
    import parakeet.mtf

    monkeypatch.setenv("PARAKEET_MTF_PATH", str(tmp_path))
    monkeypatch.setattr(parakeet.mtf, "DEFAULT_MTF_DIR", str(tmp_path))
    warnings = dp.apply({}, "k3", kv=300)
    assert len(warnings) == 1
    assert "NOT FOUND" in warnings[0] and "'k3 standard 300kV'" in warnings[0]


def test_make_config_uses_the_package_table():
    path = os.path.join(os.path.dirname(__file__), "..", "benchmark", "make_config.py")
    if not os.path.exists(path):
        pytest.skip("benchmark/ is not in this checkout")
    spec = importlib.util.spec_from_file_location("make_config", path)
    make_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_config)
    assert make_config.DETECTORS is dp.DETECTORS
    assert make_config.READOUT_MODES is dp.READOUT_MODES
