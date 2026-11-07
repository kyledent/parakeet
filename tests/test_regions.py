#
# test_regions.py
#
# The parakeet-regions/1 document: parsing, the inside-tests, which region owns
# a point, the nesting check, and the subtraction plan the multislice path
# paints with. CPU only.
#
import json

import numpy as np
import pytest

import parakeet.config
import parakeet.sample
from parakeet.sample import _regions as R


def doc(*regions, medium="water"):
    return {"schema": R.SCHEMA, "medium": medium, "regions": list(regions)}


def box(id, centre, half, material="water", **kw):
    return dict(
        id=id, kind="box", material=material, centre_A=centre, half_lengths_A=half, **kw
    )


def sphere(id, centre, radius, material="water", **kw):
    return dict(
        id=id, kind="sphere", material=material, centre_A=centre, radius_A=radius, **kw
    )


def cylinder(id, centre, axis, radius, length, material="water", **kw):
    return dict(
        id=id,
        kind="cylinder",
        material=material,
        centre_A=centre,
        axis=axis,
        radius_A=radius,
        length_A=length,
        **kw,
    )


# --- parsing ----------------------------------------------------------------


def test_a_document_parses_into_regions_with_their_depths():
    medium, regions = R.parse(
        doc(
            box("film", [0, 0, 0], [100, 100, 10], material="carbon"),
            sphere("bead", [0, 0, 0], 5, material="gold", parent="film", priority=1),
        )
    )
    assert medium == "water"
    assert [r.id for r in regions] == ["film", "bead"]
    assert [r.depth for r in regions] == [0, 1]
    assert regions[1].ancestors == ["film"]
    assert regions[1].priority == 1


def test_the_document_round_trips_through_json():
    medium, regions = R.parse(doc(cylinder("c", [1, 2, 3], [0, 0, 2], 4, 5)))
    again = R.parse(R.to_json(medium, regions))
    assert again[0] == medium
    assert np.allclose(again[1][0].geometry["axis"], [0, 0, 1])


@pytest.mark.parametrize(
    "document, message",
    [
        ({"schema": "something-else/1", "regions": []}, "Not a parakeet-regions/1"),
        (
            doc(box("a", [0, 0, 0], [1, 1, 1]), box("a", [0, 0, 0], [1, 1, 1])),
            "used twice",
        ),
        (doc(dict(id="a", kind="torus", material="water")), "kind 'torus'"),
        (
            doc(dict(id="a", kind="sphere", material="water", centre_A=[0, 0, 0])),
            "missing radius_A",
        ),
        (doc(sphere("a", [0, 0, 0], -1)), "radius_A must be positive"),
        (doc(box("a", [0, 0, 0], [1, 0, 1])), "half_lengths_A must be positive"),
        (doc(cylinder("a", [0, 0, 0], [0, 0, 0], 1, 1)), "axis must not be zero"),
        (doc(sphere("a", [0, 0, 0], 1, parent="nobody")), "does not exist"),
        (
            doc(
                sphere("a", [0, 0, 0], 1, parent="b"),
                sphere("b", [0, 0, 0], 1, parent="a"),
            ),
            "its own ancestor",
        ),
        (doc(sphere("a", [0, 0, 0], 1, priority=1.5)), "priority must be an integer"),
        (
            doc(box("a", [0, 0, 0], [1, 1, 1], axes=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])),
            "rotated box",
        ),
    ],
)
def test_a_bad_document_is_refused_by_name(document, message):
    with pytest.raises(R.RegionsError, match=message):
        R.parse(document)


def test_materials_are_checked_when_they_are_known():
    document = doc(sphere("a", [0, 0, 0], 1, material="carbon"))
    R.parse(document)  # the sample stage does not know the materials
    R.parse(document, materials=["carbon"])
    with pytest.raises(R.RegionsError, match="'carbon' is not a known material"):
        R.parse(document, materials=[])
    # water and vacuum are always known
    R.parse(doc(sphere("a", [0, 0, 0], 1, material="vacuum")), materials=[])


# --- inside-tests -----------------------------------------------------------


def test_a_box_is_half_open():
    _, (r,) = R.parse(doc(box("b", [0, 0, 0], [1, 2, 3])))
    p = np.array([[-1, -2, -3], [0.999, 1.999, 2.999], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    assert list(R.contains(r, p)) == [True, True, False, False, False]


def test_a_sphere_contains_points_nearer_than_its_radius():
    _, (r,) = R.parse(doc(sphere("s", [10, 0, 0], 2)))
    p = np.array([[10, 0, 0], [11.9, 0, 0], [12, 0, 0], [10, 1.5, 1.5]])
    assert list(R.contains(r, p)) == [True, True, False, False]


def test_a_cylinder_follows_its_axis():
    # Along x, length 10 about x = 0, radius 2
    _, (r,) = R.parse(doc(cylinder("c", [0, 0, 0], [3, 0, 0], 2, 10)))
    p = np.array([[-5, 0, 0], [4.9, 1.9, 0], [5, 0, 0], [0, 0, 2], [0, 1.5, 1.5]])
    assert list(R.contains(r, p)) == [True, True, False, False, False]


# --- ownership ----------------------------------------------------------------


def test_the_deepest_region_owns_a_point_when_priorities_tie():
    _, regions = R.parse(
        doc(
            box("cell", [0, 0, 0], [50, 50, 50]),
            sphere("nucleus", [0, 0, 0], 20, parent="cell"),
            sphere("nucleolus", [0, 0, 0], 5, parent="nucleus"),
        )
    )
    p = np.array([[0, 0, 0], [10, 0, 0], [30, 0, 0], [60, 0, 0]])
    assert list(R.owner(regions, p)) == [2, 1, 0, -1]


def test_priority_beats_depth():
    _, regions = R.parse(
        doc(
            box("cell", [0, 0, 0], [50, 50, 50]),
            sphere("organelle", [0, 0, 0], 20, parent="cell"),
            sphere("crack", [15, 0, 0], 10, priority=1),
        )
    )
    # The crack (top level, priority 1) cuts through the organelle
    assert list(R.owner(regions, [[15, 0, 0], [0, 0, 0]])) == [2, 1]


def test_a_tie_on_both_goes_to_the_later_region():
    _, regions = R.parse(doc(sphere("a", [0, 0, 0], 10), sphere("b", [5, 0, 0], 10)))
    assert list(R.owner(regions, [[3, 0, 0], [-8, 0, 0], [13, 0, 0]])) == [1, 0, 1]


def test_a_child_is_clipped_to_its_parent_in_the_reference():
    # The reference applies containment even where the paint plan cannot
    _, regions = R.parse(
        doc(
            sphere("parent", [0, 0, 0], 10),
            sphere("child", [9, 0, 0], 5, parent="parent"),
        )
    )
    assert list(R.owner(regions, [[12, 0, 0], [8, 0, 0]])) == [-1, 1]


# --- nesting ------------------------------------------------------------------


@pytest.mark.parametrize(
    "child, inside",
    [
        (sphere("c", [0, 0, 0], 49.9, parent="p"), True),
        (sphere("c", [0, 0, 0], 50.1, parent="p"), False),
        (box("c", [10, 0, 0], [40, 50, 50], parent="p"), True),
        (box("c", [10.1, 0, 0], [40, 50, 50], parent="p"), False),
        (cylinder("c", [0, 0, 0], [1, 1, 0], 10, 100, parent="p"), True),
        (cylinder("c", [0, 0, 0], [1, 1, 0], 10, 150, parent="p"), False),
    ],
)
def test_nesting_inside_a_box(child, inside):
    _, (parent, c) = R.parse(doc(box("p", [0, 0, 0], [50, 50, 50]), child))
    assert R.nested(c, parent) is inside


@pytest.mark.parametrize(
    "child, inside",
    [
        (sphere("c", [10, 0, 0], 39.9, parent="p"), True),
        (sphere("c", [10, 0, 0], 40.1, parent="p"), False),
        (box("c", [0, 0, 0], [28, 28, 28], parent="p"), True),  # corner at 48.5
        (box("c", [0, 0, 0], [29, 29, 29], parent="p"), False),  # corner at 50.2
        (cylinder("c", [0, 0, 0], [0, 0, 1], 30, 79, parent="p"), True),  # rim 49.6
        (cylinder("c", [0, 0, 0], [0, 0, 1], 30, 81, parent="p"), False),  # rim 50.4
    ],
)
def test_nesting_inside_a_sphere(child, inside):
    _, (parent, c) = R.parse(doc(sphere("p", [0, 0, 0], 50), child))
    assert R.nested(c, parent) is inside


@pytest.mark.parametrize(
    "child, inside",
    [
        (sphere("c", [0, 0, 40], 9.9, parent="p"), True),  # axial room 50 - 40
        (sphere("c", [0, 0, 41], 9.9, parent="p"), False),
        (sphere("c", [0, 10, 0], 9.9, parent="p"), True),  # radial room 20 - 10
        (sphere("c", [0, 10.2, 0], 9.9, parent="p"), False),
        (box("c", [0, 0, 0], [14, 14, 50], parent="p"), True),  # corner radius 19.8
        (box("c", [0, 0, 0], [14.2, 14.2, 50], parent="p"), False),  # 20.08
    ],
)
def test_nesting_inside_a_cylinder(child, inside):
    _, (parent, c) = R.parse(doc(cylinder("p", [0, 0, 0], [0, 0, 1], 20, 100), child))
    assert R.nested(c, parent) is inside


def test_a_child_outside_its_parent_is_refused_by_name():
    _, regions = R.parse(
        doc(
            sphere("parent", [0, 0, 0], 10),
            sphere("child", [9, 0, 0], 5, parent="parent"),
        )
    )
    with pytest.raises(
        R.RegionsError, match="'child' is not inside its parent 'parent'"
    ):
        R.check_nesting(regions)


# --- the paint plan -----------------------------------------------------------


def painted(regions, points):
    """
    What the multislice path draws: region i wherever its shape is and no shape
    it subtracts is. Returns, per point, the list of regions drawn there.

    """
    plan = R.paint_plan(regions)
    by_id = {r.id: r for r in regions}
    drawn = []
    shapes = {r.id: R.contains(r, points) for r in regions}
    for r in regions:
        mask = shapes[r.id].copy()
        for other in plan[r.id]:
            mask &= ~shapes[other]
        drawn.append(mask)
    drawn = np.array(drawn)
    return [list(np.flatnonzero(drawn[:, k])) for k in range(len(points))], by_id


def test_the_paint_plan_draws_exactly_the_owner_at_every_point():
    _, regions = R.parse(
        doc(
            box("slab", [0, 0, 0], [100, 100, 40]),
            sphere("cell", [-30, 0, 0], 35, parent="slab"),
            sphere("nucleus", [-30, 0, 0], 15, parent="cell"),
            box("film", [0, 0, -30], [100, 100, 5], priority=2),
            cylinder("hole", [40, 0, -30], [0, 0, 1], 20, 10, priority=3),
            sphere("bead", [40, 40, 0], 8, priority=1),
            sphere("overlap", [-10, 0, 0], 30),
        )
    )
    R.check_nesting(regions)
    points = np.random.default_rng(1).uniform(-110, 110, size=(200000, 3))
    points[:, 2] /= 2
    owner = R.owner(regions, points)
    drawn, _ = painted(regions, points)
    for k in range(len(points)):
        assert drawn[k] == ([] if owner[k] < 0 else [owner[k]])
    # Every region owns some of the sample points, so all of them were tested
    assert set(owner) == set(range(-1, len(regions)))


# --- configuration and storage ------------------------------------------------


def test_a_material_is_the_ice_parameters_with_two_constants():
    m = parakeet.config.Material(density=1.88, mean_constant=200.0)
    assert m.representation == "grf"
    assert m.m2 == pytest.approx(1 / 2.88)  # the ice spectrum unless given
    assert m.variance_constant is None
    with pytest.raises(Exception):
        parakeet.config.Material(representation="atoms")
    with pytest.raises(Exception):
        parakeet.config.Material(mean_constant=-1)


def test_the_regions_document_is_stored_in_the_sample(tmp_path):
    regions_file = tmp_path / "regions.json"
    regions_file.write_text(
        json.dumps(
            doc(
                box("film", [500, 500, 500], [400, 400, 60], material="carbon"),
                sphere("bead", [500, 500, 500], 30, material="gold", parent="film"),
                medium="vacuum",
            )
        )
    )
    config = parakeet.config.Sample(
        box=(1000, 1000, 1000),
        centre=(500, 500, 500),
        regions={"file": str(regions_file)},
    )
    parakeet.sample.new(config, str(tmp_path / "sample.h5"))
    sample = parakeet.sample.load(str(tmp_path / "sample.h5"))
    medium, regions = R.parse(sample.regions)
    assert medium == "vacuum"
    assert [r.id for r in regions] == ["film", "bead"]


def test_a_sample_without_regions_has_none(tmp_path):
    parakeet.sample.new(parakeet.config.Sample(), str(tmp_path / "sample.h5"))
    assert parakeet.sample.load(str(tmp_path / "sample.h5")).regions is None


def test_the_sample_refuses_a_child_outside_its_parent(tmp_path):
    config = parakeet.config.Sample(
        regions={
            "document": doc(
                sphere("parent", [500, 500, 500], 10),
                sphere("child", [509, 500, 500], 5, parent="parent"),
            )
        }
    )
    with pytest.raises(R.RegionsError, match="not inside its parent"):
        parakeet.sample.new(config, str(tmp_path / "sample.h5"))


def test_regions_need_exactly_one_source(tmp_path):
    with pytest.raises(R.RegionsError, match="exactly one of file and document"):
        parakeet.sample.new(
            parakeet.config.Sample(regions={}), str(tmp_path / "sample.h5")
        )
