#
# tests/test_mtf.py
#
# Tests for the detector transfer model: the DQE frequency axis, MTF loading,
# and the detection step that puts the two together.
#

import numpy as np
import pytest

import parakeet.dqe
import parakeet.mtf


# A short RELION-format MTF star file, written to a temporary directory rather
# than depending on a RELION installation being present.
STAR = """
data_

loop_
_rlnResolutionInversePixel #1
_rlnMtfValue #2
0.000000 1.000000
0.100000 0.800000
0.200000 0.600000
0.300000 0.420000
0.400000 0.280000
0.492000 0.200000
"""


@pytest.fixture
def star_file(tmp_path):
    path = tmp_path / "mtf_test_300kV.star"
    path.write_text(STAR)
    return str(path)


class TestDQEFrequencies:
    """
    The DQE table is tabulated against fraction of Nyquist, running 0 to 1.
    The Fourier grid is in cycles per pixel, where Nyquist is 0.5. Getting the
    factor of two wrong reads the table at half the frequency it should, which
    inflates the DQE everywhere and cannot be seen by eye in an image.

    """

    def test_dc_is_the_first_table_value(self):
        table = parakeet.dqe.DQETable()
        dqe = table.dqe_fs(300, 1.5, (64, 64))
        expected = table.dqe_table(300, 1.5)[0]
        assert dqe[32, 32] == pytest.approx(expected, abs=1e-6)

    def test_nyquist_is_the_last_table_value(self):
        table = parakeet.dqe.DQETable()
        dqe = table.dqe_fs(300, 1.5, (64, 64))
        expected = table.dqe_table(300, 1.5)[-1]
        # The grid point one Nyquist along an axis from the centre
        assert dqe[32, 0] == pytest.approx(expected, abs=1e-6)

    def test_beyond_nyquist_is_zero(self):
        # The corners of a square grid are at sqrt(2) Nyquist, where the
        # detector cannot respond and the table has no data.
        dqe = parakeet.dqe.DQETable().dqe_fs(300, 1.5, (64, 64))
        assert dqe[0, 0] == 0.0

    def test_decreases_with_frequency(self):
        dqe = parakeet.dqe.DQETable().dqe_fs(300, 1.5, (64, 64))
        radial = dqe[32, 32:]
        assert np.all(np.diff(radial) <= 1e-9)


class TestMTFLoading:
    def test_read_star_file(self, star_file):
        mtf = parakeet.mtf.MTF.from_file(star_file)
        assert mtf.at(0.0) == pytest.approx(1.0)
        assert mtf.at(0.2) == pytest.approx(0.6)

    def test_table_holds_to_nyquist(self, star_file):
        # RELION's files stop at 0.492, a little short of Nyquist. Interpolating
        # to zero there would put a discontinuity in the response that the
        # detector does not have.
        mtf = parakeet.mtf.MTF.from_file(star_file)
        assert mtf.at(0.5) == pytest.approx(0.2)
        assert mtf.at(0.6) == 0.0

    def test_analytic_is_unity_at_dc(self):
        for name in parakeet.mtf.PRESETS:
            mtf = parakeet.mtf.MTF.from_preset(name)
            assert mtf.at(0.0) == pytest.approx(1.0, abs=1e-6), name

    def test_presets_decrease_with_frequency(self):
        q = np.linspace(0, 0.5, 50)
        for name in parakeet.mtf.PRESETS:
            m = parakeet.mtf.MTF.from_preset(name).at(q)
            assert np.all(np.diff(m) <= 1e-9), name

    def test_resolve_accepts_preset_and_path(self, star_file):
        assert parakeet.mtf.MTF.resolve(None) is None
        assert parakeet.mtf.MTF.resolve("k2_300kV").params is not None
        assert parakeet.mtf.MTF.resolve(star_file).params is None

    def test_resolve_rejects_unknown(self):
        # Silently ignoring an unknown MTF would be indistinguishable in the
        # output from a detector that has none.
        with pytest.raises(ValueError):
            parakeet.mtf.MTF.resolve("no_such_detector")

    def test_grid_matches_the_curve(self):
        mtf = parakeet.mtf.MTF.from_preset("falcon4EC_300kV")
        grid = mtf.fs((64, 64))
        assert grid[32, 32] == pytest.approx(1.0, abs=1e-6)
        assert grid[32, 0] == pytest.approx(mtf.at(0.5), abs=1e-6)
        assert grid[0, 0] == 0.0


class TestDetect:
    """
    The delivered DQE is measured the way a real one is: from the ratio of
    output to input signal-to-noise. These are the tests that would fail if the
    MTF were applied where the DQE belongs, or the other way round.

    """

    def test_no_model_is_plain_poisson(self):
        rng = np.random.default_rng(0)
        image = np.full((256, 256), 0.8)
        out = parakeet.mtf.detect(image, 30.0, rng=rng)
        assert out.mean() == pytest.approx(24.0, rel=0.02)
        assert out.var() == pytest.approx(24.0, rel=0.05)

    def test_mtf_alone_does_not_change_signal_to_noise(self):
        # Blurring attenuates the signal and the noise together, so an ideal
        # detector with a poor MTF still has DQE = 1.
        rng = np.random.default_rng(1)
        n, dose = 256, 40.0
        mtf = parakeet.mtf.MTF.from_preset("falcon4EC_300kV")
        flat = np.ones((n, n))
        ideal = rng.poisson(flat * dose).astype(float)
        got = parakeet.mtf.detect(flat, dose, mtf=mtf, rng=rng)
        m = mtf.fs((n, n))
        for band, tol in ((0.3, 0.1), (0.6, 0.1)):
            delivered = _delivered_dqe(ideal, got, m, band)
            assert delivered == pytest.approx(1.0, abs=tol)

    def test_delivered_dqe_matches_the_table(self):
        rng = np.random.default_rng(2)
        n, dose = 256, 40.0
        table = parakeet.dqe.DQETable()
        dqe = table.dqe_fs(300, dose, (n, n))
        mtf = parakeet.mtf.MTF.from_preset("falcon4EC_300kV")
        flat = np.ones((n, n))

        ideal = np.mean(
            [rng.poisson(flat * dose).astype(float) for _ in range(8)], axis=0
        )
        got = np.mean(
            [
                parakeet.mtf.detect(flat, dose, mtf=mtf, dqe=dqe, rng=rng)
                for _ in range(8)
            ],
            axis=0,
        )
        m = mtf.fs((n, n))
        for band in (0.3, 0.5, 0.7, 0.9):
            delivered = _delivered_dqe(ideal, got, m, band)
            expected = np.interp(band, table.spatial_freq, table.dqe_table(300, dose))
            assert delivered == pytest.approx(expected, rel=0.1), band

    def test_negative_input_does_not_break_poisson(self):
        rng = np.random.default_rng(3)
        image = np.full((32, 32), -0.5)
        out = parakeet.mtf.detect(image, 10.0, rng=rng)
        assert np.all(out == 0)


def _delivered_dqe(ideal, got, mtf_grid, band, width=0.04):
    """
    DQE = MTF^2 . NPS_in / NPS_out, averaged over an annulus at a given
    fraction of Nyquist. This is the definition, evaluated on noise power
    spectra, and is how the number would be measured on a real detector.

    """
    n = ideal.shape[0]
    Y, X = np.mgrid[0:n, 0:n]
    q = 2 * np.sqrt(((Y - n // 2) / n) ** 2 + ((X - n // 2) / n) ** 2)
    sel = (q > band - width) & (q < band + width)

    def nps(a):
        return np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean()))) ** 2 / a.size

    return float(
        np.mean(mtf_grid[sel] ** 2) * np.mean(nps(ideal)[sel]) / np.mean(nps(got)[sel])
    )


class _DeviceArray(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Stands in for a GPU array: it wraps a numpy array but refuses to combine
    with a bare host array, which is what CuPy does. A backend that leaves a
    kernel on the host works fine under numpy and fails only on a machine with
    a GPU, so the contract is tested here rather than left to the hardware.

    """

    def __init__(self, a):
        self.a = np.asarray(a)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        converted = []
        for x in inputs:
            if isinstance(x, _DeviceArray):
                converted.append(x.a)
            elif isinstance(x, np.ndarray) and x.ndim > 0:
                raise TypeError("host array combined with a device array")
            else:
                converted.append(x)
        return _DeviceArray(getattr(ufunc, method)(*converted, **kwargs))


class _FakeBackend:
    fft2 = staticmethod(lambda x: _DeviceArray(np.fft.fft2(getattr(x, "a", x))))
    ifft2 = staticmethod(lambda x: _DeviceArray(np.fft.ifft2(getattr(x, "a", x))))
    to_numpy = staticmethod(lambda x: getattr(x, "a", x))
    from_numpy = staticmethod(_DeviceArray)


class TestBackend:
    def test_kernels_reach_the_device(self):
        rng = np.random.default_rng(4)
        image = np.full((32, 32), 0.5)
        dqe = parakeet.dqe.DQETable().dqe_fs(300, 10.0, (32, 32))
        mtf = parakeet.mtf.MTF.from_preset("k2_300kV")
        out = parakeet.mtf.detect(
            image, 10.0, mtf=mtf, dqe=dqe, rng=rng, backend=_FakeBackend
        )
        assert isinstance(out, np.ndarray)
        assert out.shape == (32, 32)

    def test_numpy_and_backend_agree(self):
        image = np.full((32, 32), 0.5)
        mtf = parakeet.mtf.MTF.from_preset("k2_300kV")
        a = parakeet.mtf.detect(image, 10.0, mtf=mtf, rng=np.random.default_rng(5))
        b = parakeet.mtf.detect(
            image, 10.0, mtf=mtf, rng=np.random.default_rng(5), backend=_FakeBackend
        )
        assert np.allclose(a, b)


class TestResample:
    """
    An MTF is in cycles per pixel, so the curve means nothing without the pixel
    it was measured on. RELION keeps the two apart as --mtf_angpix and converts
    to absolute frequency before use; these check the same conversion.

    """

    def test_same_pixel_is_a_no_op(self, star_file):
        mtf = parakeet.mtf.MTF.from_file(star_file)
        out, covered = parakeet.mtf.resample(mtf, 1.0, 1.0)
        assert covered == pytest.approx(1.0)
        for q in (0.0, 0.1, 0.3, 0.45):
            assert out.at(q) == pytest.approx(mtf.at(q), abs=1e-6)

    def test_axis_only_preserves_absolute_frequency(self, star_file):
        # The same physical frequency must give the same MTF, whichever pixel
        # the curve is expressed against.
        mtf = parakeet.mtf.MTF.from_file(star_file)
        out, _ = parakeet.mtf.resample(mtf, 0.5, 1.0)
        # f is capped so that f * 1.0 stays within the new curve's Nyquist
        for f in (0.1, 0.2, 0.4, 0.5):  # 1/A
            assert out.at(f * 1.0) == pytest.approx(mtf.at(f * 0.5), abs=2e-3)

    def test_rebin_applies_the_aperture_swap(self, star_file):
        # Binning replaces the pixel aperture: the sinc of the old pixel comes
        # out and the sinc of the new one goes in.
        mtf = parakeet.mtf.MTF.from_file(star_file)
        out, _ = parakeet.mtf.resample(mtf, 0.5, 1.0, rebin=True)
        for q in (0.2, 0.35, 0.5):
            expected = mtf.at(q / 2.0) * np.sinc(q) / np.sinc(q / 2.0)
            assert out.at(q) == pytest.approx(expected, abs=2e-3)

    def test_rebin_blurs_rather_than_sharpens(self, star_file):
        mtf = parakeet.mtf.MTF.from_file(star_file)
        axis, _ = parakeet.mtf.resample(mtf, 0.5, 1.0)
        binned, _ = parakeet.mtf.resample(mtf, 0.5, 1.0, rebin=True)
        assert binned.at(0.5) < axis.at(0.5)

    def test_going_finer_reports_missing_coverage(self, star_file):
        # Asking for a super-resolution curve from a physical-pixel one needs
        # the response above the original Nyquist, which the curve does not
        # contain. That has to be reported, not silently invented.
        mtf = parakeet.mtf.MTF.from_file(star_file)
        _, covered = parakeet.mtf.resample(mtf, 1.0, 0.5)
        assert covered == pytest.approx(0.5, abs=0.02)

    def test_rejects_nonsense_pixel_sizes(self, star_file):
        mtf = parakeet.mtf.MTF.from_file(star_file)
        with pytest.raises(ValueError):
            parakeet.mtf.resample(mtf, 0.0, 1.0)
        with pytest.raises(ValueError):
            parakeet.mtf.resample(mtf, 1.0, -1.0)


class TestStarRoundTrip:
    def test_written_curve_reads_back(self, star_file, tmp_path):
        mtf = parakeet.mtf.MTF.from_file(star_file)
        out = str(tmp_path / "written.star")
        parakeet.mtf.write_mtf_star(mtf, out)
        back = parakeet.mtf.MTF.from_file(out)
        for q in (0.0, 0.1, 0.25, 0.4):
            assert back.at(q) == pytest.approx(mtf.at(q), abs=1e-6)

    def test_never_writes_a_value_relion_rejects(self, tmp_path):
        # RELION divides by the MTF and errors on anything <= 1e-10.
        mtf = parakeet.mtf.MTF.from_params(0.2, 0.05, 1.0)
        out = str(tmp_path / "small.star")
        parakeet.mtf.write_mtf_star(mtf, out)
        back = parakeet.mtf.MTF.from_file(out)
        assert np.all(back.mtf > 1e-10)


class TestCurveLibrary:
    """
    Vendor curves are not distributed with this code, so a detector profile
    names the curve it needs and the file is found on the machine that has it.

    """

    @pytest.fixture
    def library(self, tmp_path, monkeypatch):
        for name in (
            "mtf_k3_CDS_300kV_FL1.star",
            "mtf_k3_standard_300kV_FL2.star",
            "notes.txt",
        ):
            (tmp_path / name).write_text(STAR)
        monkeypatch.setenv(parakeet.mtf.MTF_PATH_VAR, str(tmp_path))
        return tmp_path

    def test_finds_by_tokens_despite_vendor_suffixes(self, library):
        # The K3 curves arrive with an _FL1 or _FL2 suffix that is not
        # explained and differs between them, so an exact filename cannot be
        # named in advance.
        found = parakeet.mtf.find_curve("k3 CDS 300kV")
        assert found is not None
        assert found.endswith("mtf_k3_CDS_300kV_FL1.star")

    def test_matching_is_case_insensitive(self, library):
        assert parakeet.mtf.find_curve("K3 cds 300KV") is not None

    def test_ambiguity_is_an_error(self, library):
        # Silently picking one of two curves for a detector is exactly the kind
        # of thing that is never noticed.
        with pytest.raises(ValueError):
            parakeet.mtf.find_curve("k3 300kV")

    def test_ignores_non_star_files(self, library):
        assert parakeet.mtf.find_curve("notes") is None

    def test_resolve_falls_through_to_the_library(self, library):
        mtf = parakeet.mtf.MTF.resolve("k3 standard 300kV")
        assert mtf.params is None
        assert mtf.at(0.0) == pytest.approx(1.0)

    def test_preset_still_wins_over_the_library(self, library):
        mtf = parakeet.mtf.MTF.resolve("k2_300kV")
        assert mtf.params is not None

    def test_missing_curve_names_where_it_looked(self, library):
        with pytest.raises(ValueError, match=parakeet.mtf.MTF_PATH_VAR):
            parakeet.mtf.MTF.resolve("no such detector 999kV")
