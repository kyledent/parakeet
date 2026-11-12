#
# test_keyed_grf.py
#
# The Parakeet side of the engine's keyed random field: region keys that follow
# the region, and the cull of regions no column of the image can see. CPU only.
#
import math

import numpy as np

import parakeet.sample._regions as R
from parakeet.simulate.engine import MEDIUM_KEY_ID, region_key, regions_in_columns


def doc(*regions):
    return {"schema": R.SCHEMA, "medium": "vacuum", "regions": list(regions)}


def sphere(id, x, y, z, r=50.0):
    return dict(id=id, kind="sphere", material="water", centre_A=[x, y, z], radius_A=r)


def test_region_keys_follow_the_id():
    assert region_key("o12.cell") == region_key("o12.cell")
    keys = {region_key(i) for i in ("o12.cell", "o13.cell", "o12.cells", MEDIUM_KEY_ID)}
    assert len(keys) == 4
    assert all(0 <= k < 2**64 for k in keys)


def test_regions_outside_the_columns_are_dropped_in_document_order():
    _, regions = R.parse(doc(sphere("a", 100, 100, 0), sphere("far", 5000, 100, 0), sphere("b", 300, 300, 0)))
    kept = regions_in_columns(regions, (200.0, 200.0, 0.0), (0, 0, 0), lambda p: np.asarray(p, float), (400, 400))
    assert [r.id for r in kept] == ["a", "b"]


def test_the_cull_follows_the_tilt():
    # 900 A above the centre along z: at 0 deg over the centre; at 60 deg about y
    # it swings 779 A along x, out of a 400 A field
    _, regions = R.parse(doc(sphere("up", 200, 200, 900, r=20)))
    place = lambda p: np.asarray(p, float)  # noqa: E731
    flat = regions_in_columns(regions, (200.0, 200.0, 0.0), (0, 0, 0), place, (400, 400))
    tilted = regions_in_columns(regions, (200.0, 200.0, 0.0), (0, math.radians(60), 0), place, (400, 400))
    assert [r.id for r in flat] == ["up"] and tilted == []


def test_a_region_touching_the_columns_is_kept():
    _, regions = R.parse(doc(sphere("edge", 440, 200, 0, r=50)))
    kept = regions_in_columns(regions, (200.0, 200.0, 0.0), (0, 0, 0), lambda p: np.asarray(p, float), (400, 400))
    assert [r.id for r in kept] == ["edge"]
