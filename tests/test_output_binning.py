#
# test_output_binning.py
#
# simulation.output_binning: the noiseless intensity is averaged over b x b
# blocks before the detector, so the shot noise is that of the bigger pixel.
# CPU only; no multislice engine is needed.
#
import numpy as np
import pydantic
import pytest

import parakeet.config
import parakeet.io
import parakeet.simulate
from parakeet.simulate._image import bin_mean, output_binning

NX = 64
ANGPIX = 1.5
DOSE = 10.0  # electrons per A^2


def test_bin_mean_is_the_block_mean():
    image = np.random.default_rng(0).random((6, 8))
    binned = bin_mean(image, 2)
    assert binned.shape == (3, 4)
    for j in range(3):
        for i in range(4):
            block = image[2 * j : 2 * j + 2, 2 * i : 2 * i + 2]
            assert binned[j, i] == pytest.approx(block.mean(), rel=1e-14)


def test_bin_mean_by_one_leaves_the_image_alone():
    image = np.ones((4, 4), dtype="float32")
    assert bin_mean(image, 1) is image


def test_a_binning_that_does_not_divide_the_detector_is_refused():
    with pytest.raises(ValueError, match="multiples"):
        output_binning({"output_binning": 3}, 64, 64)
    assert output_binning({"output_binning": 4}, 64, 32) == 4
    assert output_binning(None, 7, 5) == 1


def test_the_config_refuses_a_binning_below_one():
    with pytest.raises(pydantic.ValidationError):
        parakeet.config.Simulation(output_binning=0)


def write_flat_optics(path):
    """A one-frame optics file of unit intensity, as the optics stage writes it"""
    writer = parakeet.io.new(
        str(path), shape=(1, NX, NX), pixel_size=ANGPIX, dtype="float32"
    )
    writer.data[0, :, :] = np.ones((NX, NX), dtype="float32")
    writer.header[0]["image_number"] = 0
    writer.header[0]["fraction_number"] = 0
    writer.header[0]["tilt_alpha"] = 0
    writer.header[0]["tilt_axis_y"] = 1
    writer.header[0]["image_size_x"] = NX
    writer.header[0]["image_size_y"] = NX


def simulate(tmp_path, binning):
    optics = tmp_path / ("optics_%d.h5" % binning)
    write_flat_optics(optics)
    config = parakeet.config.default()
    config.microscope.detector.nx = NX
    config.microscope.detector.ny = NX
    config.microscope.detector.pixel_size = ANGPIX
    config.microscope.beam.electrons_per_angstrom = DOSE
    config.simulation.output_binning = binning
    config.simulation.seed = 1234
    config.multiprocessing.device = "cpu"
    config.multiprocessing.nproc = 1
    config.multiprocessing.gpu_id = [0]
    out = tmp_path / ("image_%d.h5" % binning)
    parakeet.simulate.image(config, str(optics), str(out))
    reader = parakeet.io.open(str(out))
    return np.asarray(reader.data[:], dtype="float64"), reader


@pytest.mark.parametrize("binning", [1, 2, 4])
def test_binned_counts_are_the_shot_noise_of_the_bigger_pixel(tmp_path, binning):
    data, _ = simulate(tmp_path, binning)
    n = NX // binning
    assert data.shape == (1, n, n)

    # The dose of the output pixel, and Poisson: variance equal to the mean.
    # Detecting at the fine pixel and averaging afterwards would give the fine
    # pixel's mean and a variance of mean / binning^2 instead.
    expected = DOSE * (ANGPIX * binning) ** 2
    counts = data[0].ravel()
    m = counts.size
    assert abs(counts.mean() - expected) < 5 * np.sqrt(expected / m)
    assert abs(counts.var(ddof=1) - expected) < 5 * expected * np.sqrt(2 / (m - 1))


def test_the_image_file_carries_the_output_pixel(tmp_path):
    _, reader = simulate(tmp_path, 2)
    assert reader.pixel_size == pytest.approx(2 * ANGPIX)
    header = reader.header
    assert header[0]["pixel_size_x"] == pytest.approx(2 * ANGPIX)
    assert header[0]["pixel_size_y"] == pytest.approx(2 * ANGPIX)
    assert header[0]["image_size_x"] == NX // 2
    assert header[0]["image_size_y"] == NX // 2
