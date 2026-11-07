#
# test_engine_check.py
#
# The engine checks against stand-in engines and a hand-built fat binary, so
# they run with no GPU and whatever python-multem is installed.
#
import struct
import types

import pytest

from parakeet import engine_check
from parakeet.engine_check import (
    EngineFeatureError,
    commit_from_version,
    cuda_architectures,
    engine_features,
    require_engine_features,
)


def fake_engine(
    m0=True,
    regions=True,
    toggle=True,
    spacing=True,
    seed=True,
    vacuum=True,
    table=True,
):
    class Masker:
        pass

    if seed:
        Masker.set_seed = lambda self, value: None

    class IceParameters:
        def __init__(self):
            self.m1 = 0.0
            if m0:
                self.m0 = -1.0
            if table:
                self.spectrum_q = []

    module = types.ModuleType("multem")
    module.__version__ = "0.4.2.dev7+g50a3a34"
    module.IceParameters = IceParameters
    if regions:

        class MultiRegionMasker:
            pass

        if vacuum:
            MultiRegionMasker.supports_vacuum_regions = True
        module.MultiRegionMasker = MultiRegionMasker
    if toggle:
        module.band_limited_potential = lambda: True
    if spacing:
        module.band_limit_spacing = lambda x, y: 1.0
    module.Masker = Masker
    return module


def test_every_feature_present():
    assert all(engine_features(fake_engine()).values())


def test_an_old_engine_is_refused_by_name():
    old = fake_engine(
        m0=False,
        regions=False,
        toggle=False,
        spacing=False,
        seed=False,
        vacuum=False,
        table=False,
    )
    assert not any(engine_features(old).values())
    with pytest.raises(EngineFeatureError) as error:
        require_engine_features("multi_region_masker", "ice_m0", multem=old)
    message = str(error.value)
    assert "multi_region_masker" in message and "ice_m0" in message
    assert "PYTHON_MULTEM_COMMIT" in message


def test_only_the_named_features_are_required():
    engine = fake_engine(m0=False)
    require_engine_features("multi_region_masker", multem=engine)
    with pytest.raises(EngineFeatureError):
        require_engine_features("ice_m0", multem=engine)


def test_unknown_feature_is_an_error():
    with pytest.raises(KeyError):
        require_engine_features("no_such_feature", multem=fake_engine())


def test_no_engine(monkeypatch):
    monkeypatch.setattr(engine_check, "import_multem", lambda: None)
    assert not any(engine_features().values())
    with pytest.raises(EngineFeatureError):
        require_engine_features("ice_m0")


@pytest.mark.parametrize(
    "version, commit",
    [
        ("0.4.2.dev7+g50a3a34", "50a3a34"),
        ("0.4.2.dev7+g50a3a34.d20261106", "50a3a34"),
        (
            "0.4.2.dev7+g50a3a34a8913ce6072529b539e10e5cbda1532fe",
            "50a3a34a8913ce6072529b539e10e5cbda1532fe",
        ),
        ("0.4.1", None),
        ("unknown", None),
        (None, None),
    ],
)
def test_commit_from_version(version, commit):
    assert commit_from_version(version) == commit


def fatbin(entries):
    """
    A fat binary as nvcc lays it out: a 16-byte wrapper header (magic, version,
    header size, size of the entries) and entries with a 64-byte header (kind,
    version, header size, payload size, ..., architecture at byte 28)

    """
    body = b""
    for kind, arch, payload in entries:
        header = bytearray(64)
        struct.pack_into("<HHIQ", header, 0, kind, 0x0101, 64, len(payload))
        struct.pack_into("<I", header, 28, arch)
        body += bytes(header) + payload
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(body)) + body


def test_cuda_architectures(tmp_path):
    library = tmp_path / "ext.so"
    library.write_bytes(
        b"\x7fELF padding"
        + fatbin([(2, 89, b"\x00" * 40), (1, 89, b"ptx"), (2, 120, b"\x01" * 8)])
        + b"more"
        + fatbin([(2, 89, b"x")])
    )
    assert cuda_architectures(library) == [
        ("cubin", "sm_120"),
        ("cubin", "sm_89"),
        ("ptx", "sm_89"),
    ]


def test_no_cuda_code(tmp_path):
    library = tmp_path / "ext.so"
    library.write_bytes(b"\x7fELF nothing here")
    assert cuda_architectures(library) == []
    assert cuda_architectures(tmp_path / "missing.so") == []
