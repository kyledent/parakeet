#
# test_pdb_cache.py
#
# The PDB cache directory is created with its parents, so a home directory
# without ~/.cache (a fresh account, a batch node) does not break every lookup.
#
import parakeet.data


def test_the_cache_is_created_with_its_parents(tmp_path, monkeypatch):
    cache = tmp_path / "home" / ".cache" / "parakeet"
    monkeypatch.setenv("PARAKEET_CACHE", str(cache))
    parakeet.data.get_pdb_cache()
    assert cache.is_dir()


def test_an_existing_cache_is_read(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "1abc.pdb").write_text("END\n")
    monkeypatch.setenv("PARAKEET_CACHE", str(cache))
    assert parakeet.data.get_pdb_cache()["1abc"] == str(cache / "1abc.pdb")
