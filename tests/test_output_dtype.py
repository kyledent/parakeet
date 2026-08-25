#
# tests/test_output_dtype.py
#
# Tests for the fork's configurable output dtype, and in particular for the
# integer rescaling, which had no coverage at all and was silently wrong for
# multi-frame output: the mapping was recomputed per frame, so every tilt was
# normalised independently and intensities could not be compared between them.
#

import numpy as np
import pytest

import parakeet.io


def _write(tmp_path, name, shape, dtype, data, intensity_range=None):
    path = str(tmp_path / name)
    writer = parakeet.io.new(
        path,
        shape=shape,
        pixel_size=1.0,
        dtype=dtype,
        intensity_range=intensity_range,
    )
    for i in range(shape[0]):
        writer.data[i, :, :] = data[i]
    del writer
    return parakeet.io.open(path).data


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_float_output_is_written_unscaled(tmp_path, dtype):
    data = np.array([np.full((4, 4), 1.0), np.full((4, 4), 2.0)], dtype=np.float32)
    out = _write(tmp_path, f"f_{dtype}.mrc", (2, 4, 4), dtype, data)
    assert np.allclose(out[0], 1.0)
    assert np.allclose(out[1], 2.0)


@pytest.mark.parametrize("dtype", ["int16", "uint16"])
def test_single_frame_integer_uses_its_own_range(tmp_path, dtype):
    data = np.linspace(0, 1, 16).reshape(1, 4, 4).astype(np.float32)
    out = _write(tmp_path, f"one_{dtype}.mrc", (1, 4, 4), dtype, data)
    lo, hi = (-32767, 32767) if dtype == "int16" else (0, 65535)
    assert out.min() == lo and out.max() == hi


@pytest.mark.parametrize("dtype", ["int16", "uint16"])
def test_multi_frame_integer_without_a_range_is_refused(tmp_path, dtype):
    # The bug: this used to succeed and silently rescale each frame separately.
    data = np.array([np.full((4, 4), 1.0), np.full((4, 4), 2.0)], dtype=np.float32)
    with pytest.raises(RuntimeError, match="explicit intensity range"):
        _write(tmp_path, f"multi_{dtype}.mrc", (2, 4, 4), dtype, data)


@pytest.mark.parametrize("dtype", ["int16", "uint16"])
def test_multi_frame_integer_preserves_relative_intensity(tmp_path, dtype):
    # Two frames differing only in brightness, as tilts of differing dose do.
    # With one shared mapping the ratio between them must survive the round trip.
    data = np.array([np.full((4, 4), 1.0), np.full((4, 4), 2.0)], dtype=np.float32)
    out = _write(
        tmp_path,
        f"range_{dtype}.mrc",
        (2, 4, 4),
        dtype,
        data,
        intensity_range=(0.0, 4.0),
    )
    lo = -32767.0 if dtype == "int16" else 0.0
    hi = 32767.0 if dtype == "int16" else 65535.0

    def expected(v):
        return lo + (v - 0.0) * (hi - lo) / 4.0

    assert np.allclose(out[0], expected(1.0), atol=1)
    assert np.allclose(out[1], expected(2.0), atol=1)
    # the frames stay distinguishable, which is the whole point
    assert out[1].mean() > out[0].mean()


def test_explicit_range_is_invertible(tmp_path):
    # Because the range lives in the config rather than in the data, the
    # mapping can be undone exactly.
    vmin, vmax = -1.0, 3.0
    data = np.array([np.full((4, 4), 0.0), np.full((4, 4), 2.0)], dtype=np.float32)
    out = _write(
        tmp_path, "inv.mrc", (2, 4, 4), "uint16", data, intensity_range=(vmin, vmax)
    )
    recovered = vmin + out.astype(np.float64) * (vmax - vmin) / 65535.0
    assert np.allclose(recovered[0], 0.0, atol=1e-3)
    assert np.allclose(recovered[1], 2.0, atol=1e-3)


def test_out_of_range_values_clip_rather_than_wrap(tmp_path):
    data = np.array([np.full((4, 4), 10.0), np.full((4, 4), -10.0)], dtype=np.float32)
    out = _write(
        tmp_path, "clip.mrc", (2, 4, 4), "uint16", data, intensity_range=(0.0, 1.0)
    )
    assert out[0].max() == 65535
    assert out[1].min() == 0
