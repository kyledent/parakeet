#
# test_optical_depth.py
#
# The zero-loss filter over a sample with regions: each pixel's optical depth
# t/lambda is traced through the regions' materials, and the filter passes the
# fraction that depth gives. CPU only.
#
import numpy as np
import pytest

import parakeet.inelastic as I
import parakeet.sample._regions as R

SHAPE = {
    "type": "cuboid",
    "cuboid": {"length_x": 400.0, "length_y": 400.0, "length_z": 200.0},
}
CENTRE = (200.0, 200.0, 100.0)
PX = 4.0
N = 100  # the field is 400 A


def doc(*regions, medium="vacuum"):
    return {"schema": R.SCHEMA, "medium": medium, "regions": list(regions)}


def tau(document, mfp, orientation=(0, 0, 0), stride=1, step=0.5):
    return I.optical_depth_map(
        R.parse(document),
        mfp,
        SHAPE,
        CENTRE,
        orientation,
        (0, 0, 0),
        (0, 0),
        N,
        N,
        0,
        PX,
        stride=stride,
        step=step,
    )


def test_the_transmission_of_a_depth_is_the_shapes_transmission():
    for t in (0.0, 0.1, 0.5, 1.0, 2.0):
        shape = {
            "type": "cuboid",
            "cuboid": {"length_x": 1, "length_y": 1, "length_z": t * I.MEAN_FREE_PATH},
        }
        assert I.filter_transmission_tau(t, slit_width=20) == pytest.approx(
            I.filter_transmission(shape, 0, slit_width=20), rel=1e-12
        )
    depths = np.array([[0.0, 0.5], [1.0, 2.0]])
    each = [
        [I.filter_transmission_tau(float(t), slit_width=20) for t in row]
        for row in depths
    ]
    assert np.allclose(I.filter_transmission_tau(depths, slit_width=20), each)
    assert np.all(I.filter_transmission_tau(depths, slit_width=None) == 1)


def slab(id, x0, x1, material, z0=50.0, z1=150.0, **kw):
    return dict(
        id=id,
        kind="box",
        material=material,
        centre_A=[(x0 + x1) / 2, 200, (z0 + z1) / 2],
        half_lengths_A=[(x1 - x0) / 2, 2000, (z1 - z0) / 2],
        **kw
    )


def test_a_slab_gives_its_thickness_over_its_mean_free_path():
    t = tau(
        doc(slab("w", -2000, 200, "water"), slab("d", 200, 2400, "dense")),
        {"water": 1000.0, "dense": 500.0},
    )
    x = np.arange(N) * PX
    water, dense = t[:, (x > 20) & (x < 180)], t[:, (x > 220) & (x < 380)]
    assert water == pytest.approx(100 / 1000.0, abs=0.002)
    assert dense == pytest.approx(100 / 500.0, abs=0.002)


def test_vacuum_and_the_medium():
    hole = dict(
        id="h",
        kind="cylinder",
        material="vacuum",
        parent="w",
        centre_A=[200, 200, 100],
        axis=[0, 0, 1],
        radius_A=60,
        length_A=100,
    )
    t = tau(doc(slab("w", -2000, 2400, "water"), hole), {"water": 1000.0})
    assert t[50, 50] == pytest.approx(0.0, abs=1e-12)
    assert t[5, 5] == pytest.approx(0.1, abs=0.002)
    # A water medium fills the sample shape (200 A deep) where no region is
    medium = tau(doc(medium="water"), {"water": 1000.0})
    assert medium[50, 50] == pytest.approx(0.2, abs=0.002)


def test_a_tilt_lengthens_the_path():
    # 60 degrees about y: a slab wide in x and y gives t / cos(60)
    t = tau(
        doc(slab("w", -5000, 5000, "water")),
        {"water": 1000.0},
        orientation=(0, np.radians(60), 0),
    )
    assert t[50, 50] == pytest.approx(0.1 / np.cos(np.radians(60)), rel=0.01)


def test_a_sphere_and_the_interpolated_map():
    ball = dict(
        id="b", kind="sphere", material="water", centre_A=[200, 200, 100], radius_A=80
    )
    fine = tau(doc(ball), {"water": 1000.0})
    assert fine[50, 50] == pytest.approx(0.16, abs=0.002)
    # On a coarse grid the traced pixels are exact, and the ones between them
    # are interpolated linearly: across a sphere's curved chord that smooths
    # by a few per cent at this stride (32 A)
    coarse = tau(doc(ball), {"water": 1000.0}, stride=8)
    assert coarse.shape == fine.shape
    assert coarse[48, 48] == pytest.approx(fine[48, 48], abs=1e-12)
    assert coarse[50, 50] == pytest.approx(fine[50, 50], rel=0.05)
