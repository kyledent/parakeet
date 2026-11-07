#
# test_ice_pack_config.py
#
# sample.ice.pack reaches add_ice, so packed explicit water can be asked for
# from a config. CPU only; add_ice itself is replaced by a recorder.
#
import parakeet.config
import parakeet.sample._new as new


def run(monkeypatch, tmp_path, ice):
    calls = []
    monkeypatch.setattr(
        new, "add_ice", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    config = parakeet.config.Sample(ice=ice)
    parakeet.sample.new(config, str(tmp_path / "sample.h5"))
    return calls


def test_pack_is_passed_to_add_ice(monkeypatch, tmp_path):
    calls = run(monkeypatch, tmp_path, {"generate": True, "pack": True})
    assert len(calls) == 1 and calls[0][1]["pack"] is True


def test_random_placement_stays_the_default(monkeypatch, tmp_path):
    assert parakeet.config.Ice().pack is False
    calls = run(monkeypatch, tmp_path, {"generate": True})
    assert calls[0][1]["pack"] is False
