#
# test_shape_contains.py
#
# Which points the sample shape contains, checked against the shape's definition
# written out independently, at a centre whose x, y and z differ. The cylinder
# lies along y. With x = y at the centre (the default), swapping x for y in the
# filter gives the same answer, so only an off-diagonal centre can tell.
#
import numpy as np
import pandas
import pytest

from parakeet.sample import shape_contains

CENTRE = np.array([300.0, 700.0, 500.0])
CYLINDER = {"type": "cylinder", "cylinder": {"length": 200.0, "radius": 80.0}}


def reference_cylinder(points, centre, length, radius):
    x, y, z = points.T
    along = (y >= centre[1] - length / 2) & (y < centre[1] + length / 2)
    across = (x - centre[0]) ** 2 + (z - centre[2]) ** 2 <= radius**2
    return along & across


def random_points(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    return CENTRE + rng.uniform(-450.0, 450.0, size=(n, 3))


def test_cylinder_matches_its_definition_off_the_diagonal():
    points = random_points()
    mask = shape_contains(points, CENTRE, CYLINDER)
    expected = reference_cylinder(points, CENTRE, 200.0, 80.0)
    assert mask.sum() > 1000
    assert np.array_equal(mask, expected)


def test_cylinder_is_around_its_own_axis():
    inside = np.array(
        [
            CENTRE,
            CENTRE + [79.0, 0.0, 0.0],
            CENTRE + [0.0, 0.0, -79.0],
            CENTRE + [0.0, -100.0, 0.0],  # the closed end
        ]
    )
    outside = np.array(
        [
            CENTRE + [81.0, 0.0, 0.0],
            CENTRE + [0.0, 100.0, 0.0],  # the open end
            # The centre with x and y swapped: inside the filter that confused them
            [CENTRE[1], CENTRE[0], CENTRE[2]],
        ]
    )
    assert shape_contains(inside, CENTRE, CYLINDER).all()
    assert not shape_contains(outside, CENTRE, CYLINDER).any()


@pytest.mark.parametrize(
    "shape, length",
    [
        ({"type": "cube", "cube": {"length": 300.0}}, (300.0, 300.0, 300.0)),
        (
            {
                "type": "cuboid",
                "cuboid": {"length_x": 500.0, "length_y": 200.0, "length_z": 100.0},
            },
            (500.0, 200.0, 100.0),
        ),
    ],
)
def test_boxes_match_their_definition(shape, length):
    points = random_points()
    half = np.array(length) / 2
    expected = ((points >= CENTRE - half) & (points < CENTRE + half)).all(axis=1)
    assert np.array_equal(shape_contains(points, CENTRE, shape), expected)


def test_a_data_frame_gives_the_same_mask():
    points = random_points(10_000)
    frame = pandas.DataFrame(points, columns=["x", "y", "z"])
    assert np.array_equal(
        shape_contains(frame, CENTRE, CYLINDER),
        shape_contains(points, CENTRE, CYLINDER),
    )


def test_unknown_shape_raises():
    with pytest.raises(RuntimeError):
        shape_contains(np.zeros((1, 3)), CENTRE, {"type": "sphere", "sphere": {}})
