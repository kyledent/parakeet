#
# parakeet.sample._regions.py
#
# Copyright (C) 2026 Diamond Light Source and Rosalind Franklin Institute
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
"""
Regions of material in a sample: the ``parakeet-regions/1`` format.

A regions document lists pieces of geometry, each filled with one material::

    {
      "schema": "parakeet-regions/1",
      "medium": "water",
      "regions": [
        {"id": "film", "kind": "box", "material": "carbon",
         "centre_A": [500, 500, 300], "half_lengths_A": [500, 500, 60]},
        {"id": "hole", "kind": "cylinder", "material": "water", "parent": "film",
         "centre_A": [500, 500, 300], "axis": [0, 0, 1],
         "radius_A": 200, "length_A": 120}
      ]
    }

Lengths are in Angstroms, in the sample frame (the frame of ``sample.box``,
``sample.centre`` and the atom coordinates). The kinds are:

    box       centre_A, half_lengths_A; axis aligned
    sphere    centre_A, radius_A
    cylinder  centre_A, axis (any direction), radius_A, length_A

Every region has an ``id``, a ``kind`` and a ``material`` (a name from
``simulation.materials``, ``"water"`` for the ice parameters, or
``"vacuum"``), and may have a ``parent`` (another region's id) and an integer
``priority`` (default 0).

**Which region owns a point.** A point is in a region if it is inside the
region's shape and inside its parent. Of the regions that contain a point, the
one with the highest priority owns it; a tie goes to the deepest region in the
tree, and a tie on both to the one listed later. The owner's material fills the
point, with nothing added from the others. The ``medium`` fills
``sample.shape`` wherever no region owns a point, and ``"vacuum"`` there means
nothing does. Every point of the specimen therefore has exactly one material.

`owner()` is the reference for this, on the CPU. The multislice path paints
the same partition by giving each region the shapes of every region that
outranks it to subtract. That is exact only when each child lies inside its
parent, so `check_nesting()` refuses a child that does not: clipping a child
to its parent needs a per-slice label raster, which the engine does not have
yet.

"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

SCHEMA = "parakeet-regions/1"
VACUUM = "vacuum"
WATER = "water"
KINDS = ("box", "sphere", "cylinder")

# The geometry fields of each kind
GEOMETRY = {
    "box": ("centre_A", "half_lengths_A"),
    "sphere": ("centre_A", "radius_A"),
    "cylinder": ("centre_A", "axis", "radius_A", "length_A"),
}

# How far a child may poke out of its parent before it counts (A)
NESTING_TOLERANCE_A = 1e-6

# Points on each rim of a cylinder used to test its nesting. A cylinder's
# convex hull is spanned by its two rims; with this many points a rim is
# missed by at most r * (1 - cos(pi / n)), 1e-5 r
CYLINDER_RIM_POINTS = 720


class RegionsError(ValueError):
    """
    A regions document that cannot be rendered

    """


@dataclass
class Region:
    """
    One region of a regions document

    """

    id: str
    kind: str
    material: str
    priority: int
    parent: Optional[str]
    geometry: dict
    index: int
    depth: int = 0
    ancestors: List[str] = field(default_factory=list)

    @property
    def rank(self):
        """
        The key that decides ownership: higher wins

        """
        return (self.priority, self.depth, self.index)


def _vector(value, name, region_id, length=3):
    try:
        v = np.asarray(value, dtype="float64")
    except (TypeError, ValueError):
        raise RegionsError("Region %r: %s is not numeric" % (region_id, name))
    if v.shape != (length,) or not np.all(np.isfinite(v)):
        raise RegionsError(
            "Region %r: %s must be %d finite numbers" % (region_id, name, length)
        )
    return v


def _positive(value, name, region_id):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise RegionsError("Region %r: %s is not a number" % (region_id, name))
    if not np.isfinite(v) or v <= 0:
        raise RegionsError("Region %r: %s must be positive" % (region_id, name))
    return v


def _geometry(entry, region_id):
    kind = entry["kind"]
    missing = [k for k in GEOMETRY[kind] if k not in entry]
    if missing:
        raise RegionsError(
            "Region %r (%s) is missing %s" % (region_id, kind, ", ".join(missing))
        )
    g = {"centre_A": _vector(entry["centre_A"], "centre_A", region_id)}
    if kind == "box":
        h = _vector(entry["half_lengths_A"], "half_lengths_A", region_id)
        if np.any(h <= 0):
            raise RegionsError("Region %r: half_lengths_A must be positive" % region_id)
        if entry.get("axes") is not None and not np.allclose(entry["axes"], np.eye(3)):
            raise RegionsError(
                "Region %r: a rotated box is not supported yet; give an axis "
                "aligned box (no axes, or the identity)" % region_id
            )
        g["half_lengths_A"] = h
    elif kind == "sphere":
        g["radius_A"] = _positive(entry["radius_A"], "radius_A", region_id)
    elif kind == "cylinder":
        axis = _vector(entry["axis"], "axis", region_id)
        norm = np.linalg.norm(axis)
        if norm == 0:
            raise RegionsError("Region %r: axis must not be zero" % region_id)
        g["axis"] = axis / norm
        g["radius_A"] = _positive(entry["radius_A"], "radius_A", region_id)
        g["length_A"] = _positive(entry["length_A"], "length_A", region_id)
    return g


def load(source):
    """
    Read a regions document

    Args:
        source: A path to a JSON file, a JSON string, or a dict

    Returns:
        dict: The document

    """
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
        with open(source) as f:
            return json.load(f)
    return json.loads(source)


def parse(document, materials=None):
    """
    Validate a regions document

    Args:
        document: The document (see `load`)
        materials: The material names that may be used, or None to skip the
            check (the sample stage does not know them; the simulation stage
            does). "water" and "vacuum" are always allowed.

    Returns:
        tuple: (medium, list of Region in document order)

    Raises:
        RegionsError: If the document is not a valid parakeet-regions/1

    """
    document = load(document)
    if document.get("schema") != SCHEMA:
        raise RegionsError(
            "Not a %s document (schema %r)" % (SCHEMA, document.get("schema"))
        )
    entries = document.get("regions", [])
    if not isinstance(entries, list):
        raise RegionsError("regions must be a list")
    allowed = None
    if materials is not None:
        allowed = set(materials) | {WATER, VACUUM}

    medium = document.get("medium", WATER)
    if allowed is not None and medium not in allowed:
        raise RegionsError("The medium %r is not a known material" % medium)

    regions = []
    ids = set()
    for index, entry in enumerate(entries):
        region_id = entry.get("id")
        if not isinstance(region_id, str) or not region_id:
            raise RegionsError("Region %d has no id" % index)
        if region_id in ids:
            raise RegionsError("Region id %r is used twice" % region_id)
        ids.add(region_id)
        kind = entry.get("kind")
        if kind not in KINDS:
            raise RegionsError(
                "Region %r: kind %r is not one of %s" % (region_id, kind, KINDS)
            )
        material = entry.get("material")
        if not isinstance(material, str):
            raise RegionsError("Region %r has no material" % region_id)
        if allowed is not None and material not in allowed:
            raise RegionsError(
                "Region %r: material %r is not a known material" % (region_id, material)
            )
        priority = entry.get("priority", 0)
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise RegionsError("Region %r: priority must be an integer" % region_id)
        regions.append(
            Region(
                id=region_id,
                kind=kind,
                material=material,
                priority=priority,
                parent=entry.get("parent"),
                geometry=_geometry(entry, region_id),
                index=index,
            )
        )

    # Resolve the tree: depth and ancestors, refusing unknown parents and cycles
    by_id = {r.id: r for r in regions}
    for r in regions:
        chain = []
        parent = r.parent
        while parent is not None:
            if parent not in by_id:
                raise RegionsError(
                    "Region %r: parent %r does not exist" % (r.id, parent)
                )
            if parent == r.id or parent in chain:
                raise RegionsError("Region %r is its own ancestor" % r.id)
            chain.append(parent)
            parent = by_id[parent].parent
        r.ancestors = chain
        r.depth = len(chain)
    return medium, regions


def contains(region, points):
    """
    Whether points are inside a region's own shape (its parent is not
    considered)

    Boxes are half open, [centre - half, centre + half) on each axis, like the
    sample's cube and cuboid shapes; a sphere is |p - c| < r; a cylinder is
    0 <= t < length along its axis from the start and < r from it.

    Args:
        region: A Region
        points: An (N, 3) array, A

    Returns:
        A boolean array of length N

    """
    p = np.asarray(points, dtype="float64").reshape(-1, 3)
    g = region.geometry
    d = p - g["centre_A"]
    if region.kind == "box":
        h = g["half_lengths_A"]
        return np.all((d >= -h) & (d < h), axis=1)
    if region.kind == "sphere":
        return np.einsum("ij,ij->i", d, d) < g["radius_A"] ** 2
    if region.kind == "cylinder":
        a = g["axis"]
        t = d @ a + g["length_A"] / 2
        radial = d - np.outer(d @ a, a)
        return (
            (t >= 0)
            & (t < g["length_A"])
            & (np.einsum("ij,ij->i", radial, radial) < g["radius_A"] ** 2)
        )
    raise RegionsError("Unknown kind %r" % region.kind)


def owner(regions, points):
    """
    Which region owns each point: the reference for the partition

    Args:
        regions: The list of Region from `parse`
        points: An (N, 3) array, A

    Returns:
        An int array of length N: the index into `regions` of the owner, or -1
        where no region contains the point (the medium, if inside the sample
        shape; otherwise nothing)

    """
    p = np.asarray(points, dtype="float64").reshape(-1, 3)
    own = {r.id: contains(r, p) for r in regions}
    result = np.full(len(p), -1, dtype=int)
    # In ascending rank, so a higher ranked region overwrites a lower one
    for r in sorted(regions, key=lambda r: r.rank):
        inside = np.logical_and.reduce([own[r.id]] + [own[a] for a in r.ancestors])
        result[inside] = r.index
    return result


def _farthest_points(region):
    """
    Points whose convex hull is the region's shape (a sphere is handled by
    eroding the parent instead)

    """
    g = region.geometry
    c = g["centre_A"]
    if region.kind == "box":
        h = g["half_lengths_A"]
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
        return c + signs * h
    if region.kind == "cylinder":
        a = g["axis"]
        u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(a, u)
        phi = np.linspace(0, 2 * np.pi, CYLINDER_RIM_POINTS, endpoint=False)
        rim = g["radius_A"] * (np.outer(np.cos(phi), u) + np.outer(np.sin(phi), v))
        half = a * g["length_A"] / 2
        return np.concatenate([c + half + rim, c - half + rim])
    raise RegionsError("No hull points for %r" % region.kind)


def _inside_eroded(parent, points, margin):
    """
    Whether points lie inside the parent's shape shrunk by margin (closed)

    """
    g = parent.geometry
    d = np.asarray(points, dtype="float64").reshape(-1, 3) - g["centre_A"]
    tol = NESTING_TOLERANCE_A
    if parent.kind == "box":
        return np.all(np.abs(d) <= g["half_lengths_A"] - margin + tol, axis=1)
    if parent.kind == "sphere":
        return np.linalg.norm(d, axis=1) <= g["radius_A"] - margin + tol
    if parent.kind == "cylinder":
        a = g["axis"]
        t = d @ a
        radial = np.linalg.norm(d - np.outer(t, a), axis=1)
        return (np.abs(t) <= g["length_A"] / 2 - margin + tol) & (
            radial <= g["radius_A"] - margin + tol
        )
    raise RegionsError("Unknown kind %r" % parent.kind)


def nested(child, parent):
    """
    Whether a child's shape lies inside its parent's shape

    Every kind is convex, so a box or cylinder is inside when the points
    spanning its hull are (corners; rims, sampled), and a sphere when its centre
    is inside the parent shrunk by its radius.

    """
    if child.kind == "sphere":
        points = child.geometry["centre_A"][None, :]
        margin = child.geometry["radius_A"]
    else:
        points = _farthest_points(child)
        margin = 0.0
    return bool(np.all(_inside_eroded(parent, points, margin)))


def check_nesting(regions):
    """
    Refuse a child whose shape pokes out of its parent's

    Raises:
        RegionsError: Naming the first such child

    """
    by_id = {r.id: r for r in regions}
    for r in regions:
        if r.parent is not None and not nested(r, by_id[r.parent]):
            raise RegionsError(
                "Region %r is not inside its parent %r. Clipping a child to its "
                "parent is not supported yet: make the child's shape fit "
                "inside the parent's" % (r.id, r.parent)
            )


def paint_plan(regions):
    """
    For each region, the regions whose shapes it must subtract

    A region's shape less every region that outranks it is exactly the set of
    points it owns, provided every child lies inside its parent
    (`check_nesting`).

    Returns:
        dict: region id -> list of the ids of the regions that outrank it,
        in document order

    """
    return {r.id: [o.id for o in regions if o.rank > r.rank] for r in regions}


def to_json(medium, regions):
    """
    Write the document back (normalised: unit cylinder axes, defaults filled)

    """
    out = []
    for r in regions:
        entry = {
            "id": r.id,
            "kind": r.kind,
            "material": r.material,
            "priority": r.priority,
            "parent": r.parent,
        }
        for key, value in r.geometry.items():
            entry[key] = value.tolist() if isinstance(value, np.ndarray) else value
        out.append(entry)
    return json.dumps({"schema": SCHEMA, "medium": medium, "regions": out})


def materials_used(medium, regions) -> Dict[str, List[str]]:
    """
    The materials used, and by which regions ("<medium>" for the medium)

    """
    used: Dict[str, List[str]] = {}
    used.setdefault(medium, []).append("<medium>")
    for r in regions:
        used.setdefault(r.material, []).append(r.id)
    return used
