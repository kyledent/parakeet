#
# test_sample_cylinder.py
#
# A cylindrical sample whose centre has different x, y and z, through the code
# that fills it with ice and the code that mills it. The cylinder lies along y.
# The reference here is the shape's definition, written out in the test, so
# these tests depend on nothing but the public entry points.
#
import numpy as np
import pandas
import pytest

import parakeet.config
import parakeet.sample
from parakeet.sample._mill import _mill_Sample
from parakeet.sample._new import add_ice

CENTRE = np.array([300.0, 700.0, 500.0])


def inside(points, centre, length, radius):
    x, y, z = np.asarray(points, dtype=float).T
    along = (y >= centre[1] - length / 2) & (y < centre[1] + length / 2)
    across = (x - centre[0]) ** 2 + (z - centre[2]) ** 2 <= radius**2
    return along & across


def cylinder(length, radius):
    return {"type": "cylinder", "cylinder": {"length": length, "radius": radius}}


def oxygen(sample):
    atoms = sample.get_atoms().data
    return atoms[atoms["atomic_number"] == 8][["x", "y", "z"]].to_numpy()


@pytest.mark.parametrize("pack", [False, True])
def test_cylinder_ice_fills_the_cylinder(tmp_path, pack):
    # Longer than it is wide, so its length and diameter differ
    length, radius = 120.0, 15.0
    sample = parakeet.sample.Sample(str(tmp_path / "ice.h5"), mode="w")
    np.random.seed(1)
    add_ice(sample, CENTRE, cylinder(length, radius), density=940.0, pack=pack)
    o = oxygen(sample)

    expected = parakeet.sample.number_of_water_molecules(np.pi * radius**2 * length)
    assert len(o) == pytest.approx(expected, rel=0.2 if pack else 0.01)

    # An oxygen is within about 1 A of the point drawn for its molecule
    assert inside(o, CENTRE, length + 4.0, radius + 2.0).all()

    # The molecules fill the whole length, evenly
    y = o[:, 1] - (CENTRE[1] - length / 2)
    counts, _ = np.histogram(y, bins=4, range=(0.0, length))
    assert counts.min() > 0.8 * len(o) / 4


def test_milling_keeps_the_atoms_inside_the_cylinder(tmp_path):
    rng = np.random.default_rng(2)
    n = 50_000
    points = rng.uniform(0.0, 1000.0, size=(n, 3)).astype("float32")
    sample = parakeet.sample.Sample(str(tmp_path / "mill.h5"), mode="w")
    sample.add_atoms(
        parakeet.sample.AtomData(
            data=pandas.DataFrame(
                {
                    "atomic_number": np.full(n, 6, dtype="uint8"),
                    "x": points[:, 0],
                    "y": points[:, 1],
                    "z": points[:, 2],
                    "sigma": np.full(n, 0.085, dtype="float32"),
                    "occupancy": np.ones(n, dtype="float32"),
                    "charge": np.zeros(n, dtype="int8"),
                    "group": np.full(n, -1, dtype="int32"),
                }
            )
        )
    )
    length, radius = 400.0, 150.0
    config = parakeet.config.Sample(
        box=(1000.0, 1000.0, 1000.0),
        centre=tuple(CENTRE),
        shape=cylinder(length, radius),
    )
    _mill_Sample(config, sample)
    kept = sample.get_atoms().data[["x", "y", "z"]].to_numpy()
    expected = inside(points, CENTRE, length, radius)
    assert expected.sum() > 1000
    assert len(kept) == expected.sum()
    assert inside(kept, CENTRE, length, radius).all()
