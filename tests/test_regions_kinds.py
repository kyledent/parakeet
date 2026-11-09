#
# test_regions_kinds.py
#
# The region kinds the multislice engine cannot render yet (rotated boxes,
# ellipsoids with cuts, ellipsoidal shells, slabs): their inside-tests, bounds,
# nesting and ownership on the CPU, which are the reference for the engine's
# label raster; and the renderer's refusal of them by name. CPU only.
#
import numpy as np
import pytest

from parakeet.sample import _regions as R


def doc(*regions, medium="vacuum"):
    return {"schema": R.SCHEMA, "medium": medium, "regions": list(regions)}


def one(entry):
    return R.parse(doc(entry))[1][0]


def rot_z(deg):
    t = np.radians(deg)
    return [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]


def inside(region, pts):
    return list(R.contains(region, np.asarray(pts, float)))


def test_a_rotated_box_follows_its_axes():
    r = one(
        dict(
            id="b",
            kind="box",
            material="water",
            centre_A=[0, 0, 0],
            half_lengths_A=[10, 2, 2],
            axes=rot_z(45),
        )
    )
    assert inside(r, [[5, 5, 0], [8, 8, 0], [-1, 1, 0], [-2, 2, 0]]) == [
        True,
        False,
        True,
        False,
    ]


def test_an_ellipsoid_follows_its_semi_axes_axes_and_cuts():
    e = dict(
        id="e",
        kind="ellipsoid",
        material="water",
        centre_A=[0, 0, 0],
        semi_axes_A=[10, 5, 2],
    )
    assert inside(one(e), [[9, 0, 0], [0, 4.9, 0], [0, 0, 2.1]]) == [True, True, False]
    assert inside(one(dict(e, axes=rot_z(90))), [[0, 9, 0], [9, 0, 0]]) == [True, False]
    cut = one(dict(e, cuts=[{"normal": [2, 0, 0], "offset_A": 0}]))
    assert inside(cut, [[-5, 0, 0], [5, 0, 0]]) == [True, False]
    # A cut is relative to the centre: moving the ellipsoid moves it
    moved = one(
        dict(e, centre_A=[100, 0, 0], cuts=[{"normal": [1, 0, 0], "offset_A": 0}])
    )
    assert inside(moved, [[95, 0, 0], [105, 0, 0]]) == [True, False]


def test_a_shell_is_its_outer_ellipsoid_less_its_inner():
    s = dict(
        id="s",
        kind="ellipsoidal_shell",
        material="water",
        centre_A=[0, 0, 0],
        semi_axes_A=[10, 5, 2],
    )
    scaled = one(dict(s, inner={"mode": "scale", "factor": 0.5}))
    assert inside(scaled, [[7, 0, 0], [2, 0, 0], [11, 0, 0]]) == [True, False, False]
    offset = one(dict(s, inner={"mode": "offset", "thickness_A": 1}))
    assert inside(offset, [[0, 0, 1.5], [0, 0, 0.5]]) == [True, False]
    # cut_offset_A moves the inner surface's cut inward, leaving a wall at the cut
    capped = one(
        dict(
            s,
            inner={"mode": "scale", "factor": 0.5},
            cuts=[{"normal": [1, 0, 0], "offset_A": 0}],
            cut_offset_A=1,
        )
    )
    assert inside(capped, [[-0.5, 0, 0], [-2, 0, 0], [0.5, 0, 0]]) == [
        True,
        False,
        False,
    ]


def test_a_slab_is_its_thickness_over_its_footprint():
    s = dict(
        id="s",
        kind="slab",
        material="water",
        origin_A=[0, 0, 0],
        normal=[0, 0, 1],
        u_axis=[1, 0, 0],
        thickness_A=10,
    )
    rect = one(
        dict(s, footprint={"kind": "rectangle", "half_u_A": 100, "half_v_A": 50})
    )
    assert inside(rect, [[99, 0, 4.9], [0, 51, 0], [0, 0, 5], [0, 0, -5]]) == [
        True,
        False,
        False,
        True,
    ]
    disc = one(
        dict(s, footprint={"kind": "disc", "centre_uv_A": [20, 0], "radius_A": 10})
    )
    assert inside(disc, [[25, 0, 0], [5, 0, 0]]) == [True, False]
    clipped = one(
        dict(
            s,
            footprint={
                "kind": "rectangle",
                "half_u_A": 100,
                "half_v_A": 50,
                "clips": [{"normal_uv": [1, 0], "offset_A": 0}],
            },
        )
    )
    assert inside(clipped, [[-10, 0, 0], [10, 0, 0]]) == [True, False]
    # A tilted slab: u_axis is made perpendicular to the normal
    tilted = one(
        dict(
            s,
            normal=[0, 1, 1],
            u_axis=[1, 0, 1],
            footprint={"kind": "rectangle", "half_u_A": 100, "half_v_A": 100},
        )
    )
    n = np.array([0, 1, 1]) / np.sqrt(2)
    assert inside(
        tilted, [4.9 * n, 5.1 * n, 60 * np.array([0, 1, -1]) / np.sqrt(2)]
    ) == [True, False, True]


@pytest.mark.parametrize(
    "entry, message",
    [
        (
            dict(kind="ellipsoid", centre_A=[0, 0, 0], semi_axes_A=[1, 0, 1]),
            "semi_axes_A must be positive",
        ),
        (
            dict(
                kind="ellipsoid",
                centre_A=[0, 0, 0],
                semi_axes_A=[1, 1, 1],
                axes=[[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            ),
            "must be a rotation",
        ),
        (
            dict(
                kind="ellipsoid",
                centre_A=[0, 0, 0],
                semi_axes_A=[1, 1, 1],
                cuts=[{"normal": [0, 0, 0], "offset_A": 1}],
            ),
            "must not be zero",
        ),
        (
            dict(
                kind="ellipsoidal_shell",
                centre_A=[0, 0, 0],
                semi_axes_A=[5, 5, 5],
                inner={"mode": "scale", "factor": 1.5},
            ),
            "factor must be in",
        ),
        (
            dict(
                kind="ellipsoidal_shell",
                centre_A=[0, 0, 0],
                semi_axes_A=[5, 5, 2],
                inner={"mode": "offset", "thickness_A": 2},
            ),
            "less than the smallest",
        ),
        (
            dict(
                kind="ellipsoidal_shell",
                centre_A=[0, 0, 0],
                semi_axes_A=[5, 5, 5],
                inner={"mode": "grow"},
            ),
            "inner must be",
        ),
        (
            dict(
                kind="slab",
                origin_A=[0, 0, 0],
                normal=[0, 0, 1],
                u_axis=[0, 0, 2],
                thickness_A=1,
                footprint={"kind": "disc", "centre_uv_A": [0, 0], "radius_A": 1},
            ),
            "parallel",
        ),
        (
            dict(
                kind="slab",
                origin_A=[0, 0, 0],
                normal=[0, 0, 1],
                u_axis=[1, 0, 0],
                thickness_A=1,
                footprint={"kind": "triangle"},
            ),
            "rectangle or disc",
        ),
        (
            dict(
                kind="slab",
                origin_A=[0, 0, 0],
                normal=[0, 0, 1],
                u_axis=[1, 0, 0],
                thickness_A=1,
                footprint={"kind": "disc", "centre_uv_A": [0, 0], "radius_A": 1},
                thickness_map={"array": "x"},
            ),
            "thickness_map",
        ),
    ],
)
def test_bad_geometry_is_refused_by_name(entry, message):
    with pytest.raises(R.RegionsError, match=message):
        one(dict(entry, id="r", material="water"))


KINDS_SAMPLE = [
    dict(
        id="box",
        kind="box",
        material="water",
        centre_A=[3, -2, 1],
        half_lengths_A=[10, 4, 2],
        axes=rot_z(30),
    ),
    dict(id="sph", kind="sphere", material="water", centre_A=[1, 2, 3], radius_A=7),
    dict(
        id="cyl",
        kind="cylinder",
        material="water",
        centre_A=[0, 0, 0],
        axis=[1, 2, 3],
        radius_A=3,
        length_A=20,
    ),
    dict(
        id="ell",
        kind="ellipsoid",
        material="water",
        centre_A=[2, 0, 0],
        semi_axes_A=[12, 6, 3],
        axes=rot_z(60),
        cuts=[{"normal": [1, 1, 0], "offset_A": 4}],
    ),
    dict(
        id="shl",
        kind="ellipsoidal_shell",
        material="water",
        centre_A=[0, 1, 0],
        semi_axes_A=[12, 6, 3],
        inner={"mode": "offset", "thickness_A": 1},
    ),
    dict(
        id="slb",
        kind="slab",
        material="water",
        origin_A=[0, 0, 0],
        normal=[1, 0, 2],
        u_axis=[0, 1, 0],
        thickness_A=6,
        footprint={"kind": "disc", "centre_uv_A": [3, 1], "radius_A": 9},
    ),
]


@pytest.mark.parametrize("entry", KINDS_SAMPLE, ids=[e["id"] for e in KINDS_SAMPLE])
def test_bounds_contain_the_shape(entry):
    r = one(entry)
    lo, hi = R.bounds(r)
    p = np.random.default_rng(3).uniform(-40, 40, size=(300000, 3))
    p = p[R.contains(r, p)]
    assert len(p) > 100
    assert np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9)


def test_the_json_round_trip_keeps_every_kind():
    medium, regions = R.parse(doc(*KINDS_SAMPLE))
    again = R.parse(R.to_json(medium, regions))[1]
    p = np.random.default_rng(4).uniform(-25, 25, size=(50000, 3))
    for a, b in zip(regions, again):
        assert np.array_equal(R.contains(a, p), R.contains(b, p))


def test_sampled_nesting():
    slab = dict(
        id="section",
        kind="slab",
        material="water",
        origin_A=[0, 0, 0],
        normal=[0, 0, 1],
        u_axis=[1, 0, 0],
        thickness_A=200,
        footprint={"kind": "rectangle", "half_u_A": 500, "half_v_A": 500},
    )
    cell = dict(
        id="cell",
        kind="ellipsoid",
        material="water",
        parent="section",
        centre_A=[0, 0, 0],
        semi_axes_A=[300, 200, 90],
    )
    poking = dict(cell, id="poking", semi_axes_A=[300, 200, 110])
    wall = dict(
        id="wall",
        kind="ellipsoidal_shell",
        material="water",
        parent="cell",
        centre_A=[0, 0, 0],
        semi_axes_A=[300, 200, 90],
        inner={"mode": "offset", "thickness_A": 10},
    )
    _, regions = R.parse(doc(slab, cell, wall))
    R.check_nesting(regions)
    _, regions = R.parse(doc(slab, poking))
    with pytest.raises(R.RegionsError, match="'poking' is not inside its parent"):
        R.check_nesting(regions)


def test_the_paint_plan_draws_the_owner_with_every_kind():
    section = dict(
        id="section",
        kind="slab",
        material="water",
        origin_A=[0, 0, 0],
        normal=[0, 0, 1],
        u_axis=[1, 0, 0],
        thickness_A=60,
        footprint={"kind": "rectangle", "half_u_A": 100, "half_v_A": 100},
    )
    cell = dict(
        id="cell",
        kind="ellipsoid",
        material="water",
        parent="section",
        centre_A=[-20, 0, 0],
        semi_axes_A=[60, 40, 25],
        axes=rot_z(20),
    )
    wall = dict(
        id="wall",
        kind="ellipsoidal_shell",
        material="water",
        parent="cell",
        centre_A=[-20, 0, 0],
        semi_axes_A=[60, 40, 25],
        axes=rot_z(20),
        inner={"mode": "offset", "thickness_A": 4},
    )
    nucleus = dict(
        id="nucleus",
        kind="ellipsoid",
        material="water",
        parent="cell",
        centre_A=[-30, 0, 0],
        semi_axes_A=[15, 12, 10],
        cuts=[{"normal": [0, 1, 0], "offset_A": 8}],
    )
    crack = dict(
        id="crack",
        kind="box",
        material="water",
        priority=1,
        centre_A=[40, 0, 0],
        half_lengths_A=[40, 3, 25],
        axes=rot_z(-35),
    )
    medium, regions = R.parse(doc(section, cell, wall, nucleus, crack))
    R.check_nesting(regions)
    p = np.random.default_rng(5).uniform(-110, 110, size=(200000, 3))
    p[:, 2] /= 3
    owner = R.owner(regions, p)
    plan = R.paint_plan(regions)
    shapes = {r.id: R.contains(r, p) for r in regions}
    drawn = np.zeros((len(regions), len(p)), bool)
    for i, r in enumerate(regions):
        m = shapes[r.id].copy()
        for other in plan[r.id]:
            m &= ~shapes[other]
        drawn[i] = m
    count = drawn.sum(axis=0)
    assert np.all(count == (owner >= 0))
    assert np.all(np.argmax(drawn, axis=0)[owner >= 0] == owner[owner >= 0])
    assert set(owner) == set(range(-1, len(regions)))


def test_the_engine_renders_what_it_can_and_names_the_rest():
    sphere_like = one(
        dict(
            id="e",
            kind="ellipsoid",
            material="water",
            centre_A=[1, 2, 3],
            semi_axes_A=[5, 5, 5],
        )
    )
    kind, g = R.engine_shape(sphere_like)
    assert kind == "sphere" and g["radius_A"] == 5 and list(g["centre_A"]) == [1, 2, 3]
    assert (
        R.engine_shape(
            one(
                dict(
                    id="b",
                    kind="box",
                    material="water",
                    centre_A=[0, 0, 0],
                    half_lengths_A=[1, 1, 1],
                )
            )
        )[0]
        == "box"
    )
    for entry, what in (
        (KINDS_SAMPLE[0], "rotated box"),
        (KINDS_SAMPLE[3], "ellipsoid"),
        (KINDS_SAMPLE[4], "ellipsoidal_shell"),
        (KINDS_SAMPLE[5], "slab"),
    ):
        with pytest.raises(
            R.RegionsError,
            match=r"\(%s\) cannot be rendered by this multislice engine" % what,
        ):
            R.engine_shape(one(entry))


def test_clip_containment_allows_a_child_outside_its_parent():
    film = dict(
        id="film",
        kind="box",
        material="water",
        centre_A=[0, 0, 0],
        half_lengths_A=[100, 100, 10],
    )
    wall = dict(
        id="wall",
        kind="ellipsoidal_shell",
        material="water",
        parent="film",
        centre_A=[0, 0, 0],
        semi_axes_A=[60, 60, 60],
        inner={"mode": "offset", "thickness_A": 5},
    )
    refuse = doc(film, wall)
    clip = dict(doc(film, wall), containment="clip")
    assert R.containment(refuse) == "refuse" and R.containment(clip) == "clip"
    _, regions = R.parse(clip)
    with pytest.raises(R.RegionsError, match="not inside its parent"):
        R.check_nesting(regions)
    clipped = R.check_nesting(regions, "clip")
    assert [c[:2] for c in clipped] == [("wall", "film")]
    assert 0.5 < clipped[0][2] < 1.0  # most of a 60 A shell lies outside a 20 A film
    # The reference clips it: the wall's points above the film belong to nobody
    assert list(R.owner(regions, [[0, 0, 58], [0, 58, 0], [0, 0, 0]])) == [-1, 1, 0]
    # The setting survives the round trip, and a bad one is refused
    assert R.containment(R.to_json(*R.parse(clip), "clip")) == "clip"
    with pytest.raises(R.RegionsError, match="containment must be"):
        R.parse(dict(doc(film), containment="grow"))


CONVEX_EXTRA = [
    dict(
        id="cut_shell",
        kind="ellipsoidal_shell",
        material="water",
        centre_A=[2, -1, 4],
        semi_axes_A=[12, 9, 7],
        axes=rot_z(25),
        cuts=[{"normal": [0, 0, 1], "offset_A": 3}, {"normal": [1, -1, 0.5], "offset_A": 5}],
        inner={"mode": "scale", "factor": 0.7},
        cut_offset_A=1.5,
    ),
    dict(
        id="clipped_slab",
        kind="slab",
        material="water",
        origin_A=[0, 0, 0],
        normal=[0.2, 0.1, 1],
        u_axis=[1, 0, 0],
        thickness_A=6,
        footprint={
            "kind": "rectangle",
            "half_u_A": 15,
            "half_v_A": 10,
            "clips": [{"normal_uv": [1, 1], "offset_A": 8}],
        },
    ),
    dict(
        id="tilted_disc",
        kind="slab",
        material="water",
        origin_A=[3, 2, -4],
        normal=[0.5, 0, 1],
        u_axis=[0, 1, 0],
        thickness_A=5,
        footprint={"kind": "disc", "centre_uv_A": [-2, 3], "radius_A": 8},
    ),
]


@pytest.mark.parametrize(
    "entry", KINDS_SAMPLE + CONVEX_EXTRA, ids=lambda e: e["id"]
)
def test_the_convex_form_is_the_shape(entry):
    # The engine's convex set (a quadric and half-spaces, less an inner set)
    # holds exactly the points contains() does, off the surfaces
    region = one(entry)
    form = R.convex_form(region)
    lo, hi = R.bounds(region)
    rng = np.random.default_rng(7)
    pts = lo - 2 + (hi - lo + 4) * rng.random((200000, 3))
    expected = R.contains(region, pts)
    assert expected.sum() > 500 and (~expected).sum() > 500
    assert np.array_equal(R.convex_contains(form, pts), expected)


def test_the_engine_renders_every_kind_with_convex_regions():
    for entry in KINDS_SAMPLE + CONVEX_EXTRA:
        kind, _ = R.engine_shape(one(entry), convex=True)
        assert kind in ("box", "sphere", "cylinder", "convex")


def test_a_membrane_thin_shell_is_sampled_for_nesting():
    # A 40 A bilayer on a ~2 um cell: its bounding box holds almost none of it,
    # so nesting is sampled in the ellipsoid's own coordinates; nested in its
    # cell when it is the cell's outer surface, and caught when it pokes out
    cell = dict(
        id="cell",
        kind="ellipsoid",
        material="water",
        centre_A=[0, 0, 0],
        semi_axes_A=[21000, 17000, 17000],
        axes=rot_z(20),
        cuts=[{"normal": [0, 0, 1], "offset_A": 900}, {"normal": [0, 0, -1], "offset_A": 900}],
    )
    bilayer = dict(
        cell,
        id="bilayer",
        kind="ellipsoidal_shell",
        parent="cell",
        inner={"mode": "offset", "thickness_A": 40},
    )
    medium, regions = R.parse(doc(cell, bilayer))
    assert R.nested(regions[1], regions[0])
    bigger = dict(bilayer, semi_axes_A=[21050, 17000, 17000])
    medium, regions = R.parse(doc(cell, bigger))
    assert not R.nested(regions[1], regions[0])


TILTED_SLAB = dict(
    id="slab",
    kind="slab",
    material="water",
    origin_A=[0, 0, 0],
    normal=[0.2, 0.1, 1],
    u_axis=[1, 0, 0],
    thickness_A=20,
    footprint={"kind": "rectangle", "half_u_A": 200, "half_v_A": 200},
)
CLIP_CHILDREN = [
    dict(
        id="wall",
        kind="ellipsoidal_shell",
        material="water",
        parent="slab",
        centre_A=[3, -2, 1],
        semi_axes_A=[60, 45, 35],
        axes=rot_z(30),
        cuts=[{"normal": [1, 0, 0], "offset_A": 40}],
        inner={"mode": "offset", "thickness_A": 6},
        cut_offset_A=3,
    ),
    dict(
        id="cell",
        kind="ellipsoid",
        material="water",
        parent="slab",
        centre_A=[-5, 4, 0],
        semi_axes_A=[50, 40, 30],
    ),
    dict(
        id="block",
        kind="box",
        material="water",
        parent="slab",
        centre_A=[2, 3, 0],
        half_lengths_A=[30, 15, 25],
        axes=rot_z(25),
    ),
]
VACUOLE = dict(
    id="vacuole",
    kind="ellipsoid",
    material="water",
    parent="cell",
    centre_A=[-5, 4, 0],
    semi_axes_A=[20, 15, 25],
)


def clip_doc(*regions):
    d = doc(TILTED_SLAB, *regions)
    d["containment"] = "clip"
    return d


def test_clipping_by_planes_only_ancestors_is_owners_rule():
    # Children poking through a tilted slab are clipped by its planes, a shell's
    # cut_offset_A staying with its own cuts, and a grandchild (a vacuole in an
    # ellipsoid cell, inside the cell but through the slab) by the slab too:
    # the clipped convex sets hold exactly the points in the child and in
    # every ancestor
    medium, regions = R.parse(clip_doc(*CLIP_CHILDREN, VACUOLE))
    plan = R.clip_plan(regions)
    by_id = {r.id: r for r in regions}
    rng = np.random.default_rng(3)
    pts = -80 + 160 * rng.random((300000, 3))
    for r in regions[1:]:
        assert r.id in plan, r.id
        form = R.clipped_form(r, plan[r.id])
        expected = R.contains(r, pts)
        a = r.parent
        while a is not None:
            expected &= R.contains(by_id[a], pts)
            a = by_id[a].parent
        assert expected.sum() > 500
        assert np.array_equal(R.convex_contains(form, pts), expected), r.id


def test_a_child_outside_a_curved_ancestor_is_refused_when_clipping():
    big = dict(VACUOLE, semi_axes_A=[70, 15, 25])  # pokes out of the cell in x
    medium, regions = R.parse(clip_doc(CLIP_CHILDREN[1], big))
    with pytest.raises(R.RegionsError, match="pokes out of its ancestor 'cell'"):
        R.clip_plan(regions)


def test_the_engine_plane_limit_is_named_when_clipping():
    cuts = [
        {"normal": [np.cos(t), np.sin(t), 0.1], "offset_A": 45}
        for t in np.linspace(0, 2 * np.pi, R.MAX_CONVEX_PLANES, endpoint=False)
    ]
    many = dict(CLIP_CHILDREN[1], cuts=cuts)
    medium, regions = R.parse(clip_doc(many))
    with pytest.raises(R.RegionsError, match="more than the engine's %d" % R.MAX_CONVEX_PLANES):
        R.clip_plan(regions)


@pytest.mark.parametrize("entry", KINDS_SAMPLE + CONVEX_EXTRA, ids=lambda e: e["id"])
def test_support_bounds_the_shape(entry):
    # support(region, n) is the largest n . p over the shape (cuts ignored):
    # never below a point inside it, and, without cuts or clips, reached
    region = one(entry)
    lo, hi = R.bounds(region)
    rng = np.random.default_rng(5)
    pts = lo + (hi - lo) * rng.random((400000, 3))
    pts = pts[R.contains(region, pts)]
    size = float(np.max(hi - lo))
    uncut = not region.geometry.get("cuts") and not region.geometry.get("footprint", {}).get("clips")
    for n in rng.normal(size=(12, 3)):
        n /= np.linalg.norm(n)
        h, reached = R.support(region, n), float(np.max(pts @ n))
        assert h >= reached - 1e-9
        if uncut:
            assert h - reached < 0.03 * size


def test_only_the_planes_a_region_crosses_clip_it():
    # A cell poking through the slab's faces, far inside its footprint, gets the
    # two face planes and none of the four edges
    medium, regions = R.parse(clip_doc(CLIP_CHILDREN[1]))
    assert len(R.clip_plan(regions)["cell"]) == 2
    inside = dict(CLIP_CHILDREN[1], semi_axes_A=[5, 5, 5])  # inside the slab
    medium, regions = R.parse(clip_doc(inside))
    assert "cell" not in R.clip_plan(regions)


def test_the_plane_limit_comes_from_the_engine():
    cuts = [
        {"normal": [np.cos(t), np.sin(t), 0.1], "offset_A": 45}
        for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ]
    medium, regions = R.parse(clip_doc(dict(CLIP_CHILDREN[1], cuts=cuts)))
    with pytest.raises(R.RegionsError, match="18 half-spaces .* more than the engine's 16"):
        R.clip_plan(regions, max_planes=16)
    assert len(R.clip_plan(regions, max_planes=32)["cell"]) == 2


def test_check_regions_reports_every_finding(tmp_path):
    # Two bad regions in one clip document: a vacuole poking out of its curved
    # cell, and a region needing more half-spaces than the engine has. The
    # check reports both, with the renderer's own wording, and the renderer
    # refuses with the first and a count of the rest
    import parakeet.sample

    big = dict(VACUOLE, semi_axes_A=[70, 15, 25])
    cuts = [
        {"normal": [np.cos(t), np.sin(t), 0.1], "offset_A": 45}
        for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ]
    crowded = dict(CLIP_CHILDREN[1], id="crowded", cuts=cuts, centre_A=[40, 0, 0])
    document = clip_doc(CLIP_CHILDREN[1], big, crowded)
    findings = parakeet.sample.check_regions(document, max_planes=16)
    assert {(f["region"], f["reason"]) for f in findings} == {
        ("vacuole", "outside_curved_ancestor"),
        ("crowded", "plane_budget"),
    }
    medium, regions = R.parse(document)
    with pytest.raises(R.RegionsError) as error:
        R.clip_plan(regions, max_planes=16)
    assert str(error.value).startswith(findings[0]["message"])
    assert "(and 1 more" in str(error.value)
    # the same document with room for the planes leaves only the poke-out
    assert [f["reason"] for f in parakeet.sample.check_regions(document, max_planes=32)] == [
        "outside_curved_ancestor"
    ]


def test_check_regions_on_a_clean_document_and_without_clipping():
    import parakeet.sample

    assert parakeet.sample.check_regions(clip_doc(*CLIP_CHILDREN, VACUOLE)) == []
    # without containment: clip, a child outside its parent is a finding
    findings = parakeet.sample.check_regions(doc(TILTED_SLAB, CLIP_CHILDREN[1]))
    assert [(f["region"], f["reason"]) for f in findings] == [("cell", "outside_parent")]
    assert parakeet.sample.check_regions('{"schema": "nope"}')[0]["reason"] == "parse"


def test_the_check_regions_command(tmp_path, capsys):
    import json
    from parakeet.command_line.sample._check_regions import check_regions_impl, get_parser

    clean, bad = tmp_path / "clean.json", tmp_path / "bad.json"
    clean.write_text(json.dumps(clip_doc(*CLIP_CHILDREN)))
    bad.write_text(json.dumps(clip_doc(CLIP_CHILDREN[1], dict(VACUOLE, semi_axes_A=[70, 15, 25]))))
    assert check_regions_impl(get_parser().parse_args([str(clean), "--max-planes", "16"])) == 0
    assert check_regions_impl(get_parser().parse_args([str(bad), "--max-planes", "16", "--json"])) == 1
    first, _, rest = capsys.readouterr().out.partition("\n")
    assert first.startswith("0 findings (max_planes 16, --max-planes")
    out = json.loads(rest)
    assert out["max_planes"] == 16 and out["max_planes_source"] == "--max-planes"
    assert out["findings"][0]["reason"] == "outside_curved_ancestor"


def test_a_regions_own_planes_count_against_the_budget():
    # With or without clipping, a region the engine draws as a convex set may
    # not have more half-spaces of its own than the engine allows
    cuts = [
        {"normal": [np.cos(t), np.sin(t), 0.2], "offset_A": 45}
        for t in np.linspace(0, 2 * np.pi, 17, endpoint=False)
    ]
    cell = dict(CLIP_CHILDREN[1], parent=None, cuts=cuts)
    import parakeet.sample

    for document in (doc(cell), dict(doc(cell), containment="clip")):
        findings = parakeet.sample.check_regions(document, max_planes=16)
        assert [(f["region"], f["reason"]) for f in findings] == [("cell", "plane_budget")]
        assert "17 half-spaces (17 of its own, 0 from its ancestors)" in findings[0]["message"]
        assert parakeet.sample.check_regions(document, max_planes=32) == []


def test_an_engine_without_convex_regions_is_a_finding():
    import parakeet.sample

    document = clip_doc(*CLIP_CHILDREN)
    findings = parakeet.sample.check_regions(document, convex=False)
    assert {f["reason"] for f in findings} == {"engine_feature"}
    # the tilted slab is not a native kind either
    assert {f["region"] for f in findings} == {"slab", "wall", "cell", "block"}
    assert parakeet.sample.check_regions(document, convex=True) == []


def test_the_check_runs_without_the_engine(tmp_path):
    # With no importable engine (as on a node with no CUDA driver) the command
    # still runs: parakeet imports without multem, and the limit falls back to
    # the default, saying so
    import json
    import subprocess
    import sys

    path = tmp_path / "doc.json"
    path.write_text(json.dumps(clip_doc(*CLIP_CHILDREN)))
    # both CUDA-bound extensions made unimportable, and the command entered
    # the way the console script enters it (the whole command-line package)
    code = (
        "import sys; sys.modules['multem'] = None; sys.modules['guanaco'] = None\n"
        "import parakeet.command_line.sample as cli\n"
        "sys.exit(cli._check_regions.check_regions_impl(\n"
        "    cli._check_regions.get_parser().parse_args([sys.argv[1], '--json'])))\n"
    )
    run = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
    out = json.loads(run.stdout)
    assert out["max_planes"] == R.MAX_CONVEX_PLANES
    assert out["max_planes_source"].startswith("default: the engine could not be imported")
    assert out["engine_feature_checked"] is False
    assert out["findings"] == []
