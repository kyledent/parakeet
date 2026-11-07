#
# test_dose_per_image.py
#
# scan.dose_per_image: each image of a scan gets its own dose, split evenly over
# its fractions, and the damage model sees the cumulative dose. CPU only.
#
import numpy as np
import pytest

import parakeet.config
import parakeet.io
import parakeet.scan
import parakeet.simulate


def test_each_image_gets_its_dose_split_over_its_fractions():
    scan = parakeet.scan.new(
        mode="tilt_series",
        start_angle=-3,
        step_angle=3,
        num_images=3,
        num_fractions=2,
        dose_per_image=[30, 3, 6],
    )
    assert list(scan.image_number) == [0, 0, 1, 1, 2, 2]
    assert np.allclose(scan.electrons_per_angstrom, [15, 15, 1.5, 1.5, 3, 3])


def test_dose_follows_the_order_the_images_are_taken():
    # A dose-symmetric scan takes the images in order of |angle|, so the first
    # dose goes to the untilted image
    scan = parakeet.scan.new(
        mode="dose_symmetric",
        start_angle=-6,
        step_angle=3,
        num_images=5,
        dose_per_image=[30, 3, 3, 3, 3],
    )
    angles = np.asarray(scan.angles)
    dose = np.asarray(scan.electrons_per_angstrom)
    assert angles[0] == pytest.approx(0)
    assert dose[0] == 30
    assert np.all(dose[np.abs(angles) > 0] == 3)


def test_without_a_list_the_beam_dose_is_spread_evenly():
    scan = parakeet.scan.new(
        mode="tilt_series", num_images=4, num_fractions=2, electrons_per_angstrom=8
    )
    assert np.allclose(scan.electrons_per_angstrom, 4)


def test_the_cumulative_dose_is_the_running_sum():
    scan = parakeet.scan.new(
        mode="tilt_series", num_images=3, num_fractions=2, dose_per_image=[30, 3, 6]
    )
    assert np.allclose(scan.cumulative_dose, [15, 30, 31.5, 33, 36, 39])
    uniform = parakeet.scan.new(
        mode="tilt_series", num_images=5, electrons_per_angstrom=3
    )
    assert np.allclose(uniform.cumulative_dose, 3 * np.arange(1, 6))


@pytest.mark.parametrize(
    "dose, message",
    [([30, 3], "3 images"), ([30, -3, 3], "non-negative"), ([30, np.nan, 3], "finite")],
)
def test_a_bad_list_is_refused(dose, message):
    with pytest.raises(ValueError, match=message):
        parakeet.scan.new(mode="tilt_series", num_images=3, dose_per_image=dose)


def test_the_config_carries_the_list_to_the_scan():
    config = parakeet.config.Scan(
        mode="tilt_series", num_images=3, dose_per_image=[30, 3, 3]
    )
    scan = parakeet.scan.new(electrons_per_angstrom=100, **config.model_dump())
    assert np.allclose(scan.electrons_per_angstrom, [30, 3, 3])


def test_the_image_stage_detects_each_image_at_its_own_dose(tmp_path):
    # Two images of unit intensity; the detected mean is the image's dose
    nx, angpix, dose = 32, 1.0, [30.0, 3.0]
    optics = tmp_path / "optics.h5"
    writer = parakeet.io.new(
        str(optics), shape=(2, nx, nx), pixel_size=angpix, dtype="float32"
    )
    for i in range(2):
        writer.data[i, :, :] = np.ones((nx, nx), dtype="float32")
        writer.header[i]["image_number"] = i
        writer.header[i]["fraction_number"] = 0
        writer.header[i]["tilt_alpha"] = 3 * i
        writer.header[i]["tilt_axis_y"] = 1
    writer = None

    config = parakeet.config.default()
    config.microscope.detector.nx = nx
    config.microscope.detector.ny = nx
    config.microscope.detector.pixel_size = angpix
    config.microscope.beam.electrons_per_angstrom = 100
    config.scan.mode = "tilt_series"
    config.scan.start_angle = 0
    config.scan.step_angle = 3
    config.scan.num_images = 2
    config.scan.dose_per_image = dose
    config.simulation.seed = 1
    config.multiprocessing.device = "cpu"
    config.multiprocessing.nproc = 1
    config.multiprocessing.gpu_id = [0]
    out = tmp_path / "image.h5"
    parakeet.simulate.image(config, str(optics), str(out))

    reader = parakeet.io.open(str(out))
    data = np.asarray(reader.data[:], dtype="float64")
    for i, d in enumerate(dose):
        assert abs(data[i].mean() - d) < 5 * np.sqrt(d / nx**2)
        assert reader.header[i]["dose"] == pytest.approx(d)
