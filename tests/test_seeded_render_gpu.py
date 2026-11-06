#
# test_seeded_render_gpu.py
#
# A seeded render through parakeet: the same base seed gives the same ice, the
# fractions of one image share it, and two images get their own. Needs a CUDA
# GPU and an engine that takes a seed; skipped otherwise (as in CI).
#
import numpy as np
import pytest
import yaml

import parakeet.io
import parakeet.sample
import parakeet.seeding
import parakeet.simulate
from parakeet.engine_check import engine_features, import_multem

multem = import_multem()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        multem is None
        or not getattr(multem, "is_gpu_available", lambda: False)()
        or not engine_features(multem)["grf_seed"],
        reason="needs a CUDA GPU and an engine with grf_seed",
    ),
]

NX = 128
ANGPIX = 1.5
FIELD = NX * ANGPIX
THICKNESS = 120.0

# The atomic part of an exit wave varies by about 6e-6 relative from run to run
# on the GPU; the seeded ice adds nothing to that. Fields from different seeds,
# or different images, differ by order one.
SAME = 1e-4
DIFFERENT = 0.1


def render(directory, seed):
    directory.mkdir()
    # MULTEM slices only where there are atoms: two hydrogen atoms, one just
    # inside each face of the ice, make it slice the whole slab
    anchors = directory / "anchors.pdb"
    anchors.write_text(
        "".join(
            "HETATM%5d  H   HOH A%4d    %8.3f%8.3f%8.3f  1.00  0.00           H\n"
            % (i, i, FIELD / 2, FIELD / 2, z)
            for i, z in enumerate((1.0, THICKNESS - 1.0), start=1)
        )
        + "END\n"
    )
    config = {
        "microscope": {
            "beam": {"energy": 300, "electrons_per_angstrom": 30.0},
            "detector": {"nx": NX, "ny": NX, "pixel_size": ANGPIX},
        },
        "sample": {
            "coords": {"filename": str(anchors), "recentre": False},
            "box": [FIELD, FIELD, THICKNESS],
            "centre": [FIELD / 2, FIELD / 2, THICKNESS / 2],
            "shape": {
                "type": "cuboid",
                "cuboid": {"length_x": FIELD, "length_y": FIELD, "length_z": THICKNESS},
            },
        },
        # Two images at the same angle, two fractions each: four frames
        "scan": {
            "mode": "tilt_series",
            "start_angle": 0,
            "step_angle": 0,
            "num_images": 2,
            "num_fractions": 2,
        },
        "simulation": {"ice": True, "seed": seed, "margin": 20, "padding": 20},
        "multiprocessing": {"device": "gpu", "nproc": 1, "gpu_id": [0]},
    }
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    parakeet.sample.new(str(path), str(directory / "sample.h5"))
    parakeet.simulate.exit_wave(
        str(path), str(directory / "sample.h5"), str(directory / "exit_wave.h5")
    )
    reader = parakeet.io.open(str(directory / "exit_wave.h5"))
    data = np.asarray(reader.data[:], dtype=np.complex128)
    header = reader.header
    meta = {
        key: [header[i][key] for i in range(data.shape[0])]
        for key in ("image_number", "fraction_number", "seed", "grf_seed")
    }
    return data, meta


def phase(psi):
    reference = psi.mean()
    return np.angle(psi * np.conj(reference / abs(reference)))


def relative(a, b):
    return float(np.abs(a - b).max() / np.abs(a).max())


@pytest.fixture(scope="module")
def renders(tmp_path_factory):
    base = tmp_path_factory.mktemp("seeded")
    return {
        "a": render(base / "a", 1234),
        "a_again": render(base / "a_again", 1234),
        "b": render(base / "b", 1235),
    }


def test_the_frames_are_two_images_of_two_fractions(renders):
    data, meta = renders["a"]
    assert data.shape[0] == 4
    assert list(meta["image_number"]) == [0, 0, 1, 1]
    assert list(meta["fraction_number"]) == [0, 1, 0, 1]


def test_the_ice_is_there(renders):
    data, _ = renders["a"]
    assert phase(data[0]).std() > 0.05


def test_the_same_seed_gives_the_same_ice(renders):
    (a, _), (again, _) = renders["a"], renders["a_again"]
    worst = max(relative(a[i], again[i]) for i in range(4))
    print("same seed, worst frame: %.3g relative" % worst)
    assert worst < SAME


def test_a_different_seed_gives_different_ice(renders):
    (a, _), (b, _) = renders["a"], renders["b"]
    assert relative(a[0], b[0]) > DIFFERENT


def test_the_fractions_of_an_image_share_its_ice(renders):
    data, _ = renders["a"]
    assert relative(data[0], data[1]) < SAME
    assert relative(data[2], data[3]) < SAME


def test_each_image_has_its_own_ice(renders):
    data, _ = renders["a"]
    assert relative(data[0], data[2]) > DIFFERENT


def test_the_seeds_are_recorded(renders):
    _, meta = renders["a"]
    assert list(meta["seed"]) == [1234] * 4
    assert list(meta["grf_seed"]) == [
        parakeet.seeding.grf_seed(1234, i) for i in meta["image_number"]
    ]
