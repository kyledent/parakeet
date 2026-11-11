#
# test_optical_depth.py
#
# The zero-loss filter over a sample with regions: each pixel's optical depth
# t/lambda is traced through the regions' materials, and the filter passes the
# fraction that depth gives. CPU only.
#
import math

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


def test_path_lengths_per_material_add_up_to_the_depth():
    ball = dict(
        id="b",
        kind="sphere",
        material="dense",
        parent="w",
        centre_A=[200, 200, 100],
        radius_A=40,
    )
    document = doc(slab("w", -2000, 2400, "water"), ball)
    paths = I.path_length_maps(
        R.parse(document),
        SHAPE,
        CENTRE,
        (0, 0, 0),
        (0, 0, 0),
        (0, 0),
        N,
        N,
        0,
        PX,
        stride=1,
        step=0.5,
    )
    assert sorted(paths) == ["dense", "water"]
    # Through the sphere's centre: 80 A of it and the rest of the 100 A slab water
    assert paths["dense"][50, 50] == pytest.approx(80, abs=1)
    assert paths["water"][50, 50] == pytest.approx(20, abs=1)
    assert paths["dense"][5, 5] == 0 and paths["water"][5, 5] == pytest.approx(
        100, abs=1
    )
    t = tau(document, {"water": 1000.0, "dense": 400.0})
    assert np.allclose(t, paths["water"] / 1000.0 + paths["dense"] / 400.0)


def test_a_coarse_stride_keeps_edges_sharp():
    # A film with an empty hole: traced every pixel and every 8th, the maps are
    # the same, edge included, because cells crossing a boundary are traced fully
    hole = dict(
        id="h",
        kind="cylinder",
        material="vacuum",
        parent="w",
        centre_A=[203, 197, 100],
        axis=[0, 0, 1],
        radius_A=90,
        length_A=100,
    )
    document = doc(slab("w", -2000, 2400, "water"), hole)
    every = tau(document, {"water": 1000.0}, stride=1)
    eighth = tau(document, {"water": 1000.0}, stride=8)
    print(
        "max difference, stride 8 vs 1: %.3g (film 0.1)" % np.abs(eighth - every).max()
    )
    assert np.abs(eighth - every).max() < 1e-12


@pytest.mark.parametrize(
    "feature",
    [
        dict(
            id="vesicle",
            kind="sphere",
            material="dense",
            parent="w",
            centre_A=[214.0, 214.0, 100.0],
            radius_A=4.0,
        ),
        dict(
            id="fibre",
            kind="cylinder",
            material="dense",
            parent="w",
            centre_A=[214.0, 200.0, 100.0],
            axis=[0, 1, 0],
            radius_A=3.0,
            length_A=300.0,
        ),
        dict(
            id="pore",
            kind="sphere",
            material="vacuum",
            parent="w",
            centre_A=[214.0, 214.0, 100.0],
            radius_A=4.0,
        ),
    ],
    ids=["vesicle", "fibre", "pore"],
)
def test_a_feature_smaller_than_a_cell_is_not_missed(feature):
    # Centred between the rays of a stride-8 grid (rays every 32 A at 4 A/px, at
    # x = 192 and 224): the corner rays all miss it, but it is still traced
    document = doc(slab("w", -2000, 2400, "water"), feature)
    mfp = {"water": 1000.0, "dense": 300.0}
    every = tau(document, mfp, stride=1)
    eighth = tau(document, mfp, stride=8)
    assert np.abs(every - 0.1).max() > 0.005  # the feature is there at stride 1
    assert np.abs(eighth - every).max() < 1e-12


def test_a_thin_fibre_across_the_field_is_not_missed():
    # A 3 A fibre running diagonally: its projected bounding box is large, but
    # it is thin, and between rays it would be lost
    fibre = dict(
        id="fibre",
        kind="cylinder",
        material="dense",
        parent="w",
        centre_A=[214.0, 200.0, 100.0],
        axis=[1, 1, 0],
        radius_A=3.0,
        length_A=300.0,
    )
    document = doc(slab("w", -2000, 2400, "water"), fibre)
    mfp = {"water": 1000.0, "dense": 300.0}
    every = tau(document, mfp, stride=1)
    eighth = tau(document, mfp, stride=8)
    assert np.abs(every - 0.1).max() > 0.005
    print(
        "diagonal fibre, stride 8 vs 1: max difference %.3g"
        % np.abs(eighth - every).max()
    )
    assert np.abs(eighth - every).max() < 1e-12


def paths(document, orientation=(0, 0, 0), stride=1, exact=True, step=0.5):
    return I.path_length_maps(
        R.parse(document),
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
        exact=exact,
    )


def test_exact_paths_through_a_sphere_are_its_chords():
    ball = dict(
        id="b",
        kind="sphere",
        material="dense",
        centre_A=[201.3, 198.7, 100.0],
        radius_A=77.0,
    )
    L = paths(doc(ball))["dense"]
    x = np.arange(N) * PX
    rho2 = (x[None, :] - 201.3) ** 2 + (x[:, None] - 198.7) ** 2
    chord = 2 * np.sqrt(np.clip(77.0**2 - rho2, 0, None))
    assert np.abs(L - chord).max() < 1e-9


def test_exact_paths_through_a_tilted_slab_and_a_nested_sphere():
    document = doc(
        slab("w", -5000, 5000, "water"),
        dict(
            id="b",
            kind="sphere",
            material="dense",
            parent="w",
            centre_A=[200, 200, 100],
            radius_A=40,
        ),
    )
    L = paths(document)
    x = np.arange(N) * PX
    rho2 = (x[None, :] - 200) ** 2 + (x[:, None] - 200) ** 2
    chord = 2 * np.sqrt(np.clip(40.0**2 - rho2, 0, None))
    assert np.abs(L["dense"] - chord).max() < 1e-9
    assert np.abs(L["water"] + L["dense"] - 100).max() < 1e-9
    tilted = paths(
        doc(slab("w", -5000, 5000, "water")), orientation=(0, np.radians(60), 0)
    )
    assert np.abs(tilted["water"][10:-10, 10:-10] - 200).max() < 1e-9


@pytest.mark.parametrize(
    "entry",
    [
        dict(
            id="e",
            kind="ellipsoid",
            material="dense",
            parent="w",
            centre_A=[200, 200, 100],
            semi_axes_A=[90, 50, 40],
            axes=[
                [np.cos(0.4), -np.sin(0.4), 0],
                [np.sin(0.4), np.cos(0.4), 0],
                [0, 0, 1],
            ],
            cuts=[{"normal": [1, 0.3, 0.2], "offset_A": 30}],
        ),
        dict(
            id="s",
            kind="ellipsoidal_shell",
            material="dense",
            parent="w",
            centre_A=[200, 200, 100],
            semi_axes_A=[90, 60, 45],
            inner={"mode": "offset", "thickness_A": 8},
        ),
        dict(
            id="t",
            kind="slab",
            material="dense",
            parent="w",
            origin_A=[200, 200, 100],
            normal=[0.2, 0.1, 1],
            u_axis=[1, 0, 0],
            thickness_A=30,
            footprint={
                "kind": "disc",
                "centre_uv_A": [10, -5],
                "radius_A": 90,
                "clips": [{"normal_uv": [1, 1], "offset_A": 40}],
            },
        ),
    ],
    ids=["ellipsoid_cut", "shell", "slab_disc_clip"],
)
def test_exact_paths_agree_with_fine_sampling(entry):
    # The film reaches past the field at 35 degrees (+-600 A about its centre)
    # without making the rays long, which the sampled reference pays for
    document = doc(slab("w", -400, 800, "water", z0=0, z1=200), entry)
    exact = paths(document, orientation=(0, np.radians(35), 0))
    fine = paths(document, orientation=(0, np.radians(35), 0), exact=False, step=0.1)
    for m in exact:
        # Sampling is good to about one step per boundary crossed
        assert np.abs(exact[m] - fine[m]).max() < 0.1 * 6
    assert (
        np.abs(exact["water"] + exact["dense"] - 200 / np.cos(np.radians(35))).max()
        < 1e-9
    )


def test_an_exact_coarse_map_equals_the_exact_full_one():
    document = doc(
        slab("w", -5000, 5000, "water"),
        dict(
            id="b",
            kind="sphere",
            material="dense",
            parent="w",
            centre_A=[203, 197, 100],
            radius_A=60,
        ),
    )
    full = paths(document, stride=1)
    coarse = paths(document, stride=8)
    for m in full:
        assert np.abs(full[m] - coarse[m]).max() < 1e-9


def crowded(tilt_deg=0.0):
    """A slab with nested spheres and an ellipsoid inside the field, and regions
    whose bounds no ray of it meets (far along x and y, beyond the field)"""
    inside = [
        dict(id="s%d" % i, kind="sphere", material=("dense", "water")[i % 2],
             parent="w", centre_A=[80.0 + 60 * i, 120.0 + 25 * i, 100.0], radius_A=30.0)
        for i in range(5)
    ]
    far = [
        dict(id="f%d" % i, kind="sphere", material="dense", centre_A=[5000.0 + 900 * i, -4000.0, 100.0],
             radius_A=200.0)
        for i in range(4)
    ]
    ell = dict(id="e", kind="ellipsoid", material="dense", parent="w", centre_A=[300.0, 300.0, 100.0],
               semi_axes_A=[60.0, 30.0, 20.0], axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return doc(slab("w", -5000, 5000, "water"), *inside, ell, *far), (0.0, math.radians(tilt_deg), 0.0)


@pytest.mark.parametrize("tilt", [0.0, 50.0])
def test_the_per_chunk_region_cull_does_not_change_the_exact_maps(tilt, monkeypatch):
    # The same scene traced twice: as is (each chunk asks only the regions whose
    # bounds its lines meet; the far ones here are met by none) and with the
    # cull off, every region asked. Bit for bit the same
    document, orientation = crowded(tilt)
    culled = paths(document, orientation=orientation)
    monkeypatch.setattr(I, "CULL_REGIONS", False)
    kept = paths(document, orientation=orientation)
    assert culled.keys() == kept.keys()
    for m in culled:
        assert np.array_equal(culled[m], kept[m])


@pytest.mark.parametrize("workers", [2, 3])
def test_worker_processes_give_the_same_exact_maps(workers):
    document, orientation = crowded(50.0)
    kw = dict(stride=1, step=0.5, exact=True)
    args = (R.parse(document), SHAPE, CENTRE, orientation, (0, 0, 0), (0, 0), 200, 200, 0, 2.0)
    one = I.path_length_maps(*args, **kw)
    many = I.path_length_maps(*args, workers=workers, **kw)
    assert one.keys() == many.keys()
    for m in one:
        assert np.array_equal(one[m], many[m])
