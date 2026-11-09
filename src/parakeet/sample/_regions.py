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

    box                centre_A, half_lengths_A, optional axes (a rotation:
                       its columns are the box's axes)
    sphere             centre_A, radius_A
    cylinder           centre_A, axis (any direction), radius_A, length_A
    ellipsoid          centre_A, semi_axes_A, optional axes, optional cuts
    ellipsoidal_shell  an ellipsoid with an inner surface: inner is
                       {"mode": "scale", "factor": f} (semi-axes x f) or
                       {"mode": "offset", "thickness_A": t} (semi-axes - t);
                       optional cut_offset_A moves every cut inward for the
                       inner surface
    slab               origin_A, normal, u_axis, thickness_A and a footprint:
                       {"kind": "rectangle", "half_u_A", "half_v_A"} or
                       {"kind": "disc", "centre_uv_A", "radius_A"}, with
                       optional "clips"

A cut, {"normal": n, "offset_A": b}, keeps the points p with
n . (p - centre_A) <= b: it is relative to the region's centre, so moving a
region moves its cuts. A slab's clip, {"normal_uv": n, "offset_A": b}, keeps
the points of the footprint with n . (u, v) <= b, where u runs along u_axis
and v along normal x u_axis from origin_A.

The multislice engine renders axis-aligned boxes, spheres and cylinders (and
an ellipsoid that is a sphere: equal semi-axes, no cuts). The other kinds are
parsed and checked, `contains()` and `owner()` handle them, and they are the
reference the engine's label raster is to be tested against; rendering them
is refused by name until it exists.

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

A document may set ``"containment": "clip"``: then a child that pokes out of
its parent is clipped to it rather than refused (see `containment`).

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
KINDS = ("box", "sphere", "cylinder", "ellipsoid", "ellipsoidal_shell", "slab")

# The geometry fields each kind must have
GEOMETRY = {
    "box": ("centre_A", "half_lengths_A"),
    "sphere": ("centre_A", "radius_A"),
    "cylinder": ("centre_A", "axis", "radius_A", "length_A"),
    "ellipsoid": ("centre_A", "semi_axes_A"),
    "ellipsoidal_shell": ("centre_A", "semi_axes_A", "inner"),
    "slab": ("origin_A", "normal", "u_axis", "thickness_A", "footprint"),
}

# How far a rotation's columns may be from orthonormal
ROTATION_TOLERANCE = 1e-6

# Sampled nesting checks (kinds without an exact test): points drawn in the
# child's bounding box, and the fewest that must land inside the child
NESTING_SAMPLES = 400000
NESTING_MIN_INSIDE = 2000

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


def _rotation(value, name, region_id):
    """A rotation matrix whose columns are a body's axes"""
    if value is None:
        return np.eye(3)
    try:
        R = np.asarray(value, dtype="float64")
    except (TypeError, ValueError):
        raise RegionsError("Region %r: %s is not numeric" % (region_id, name))
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise RegionsError("Region %r: %s must be a 3 x 3 matrix" % (region_id, name))
    if (
        not np.allclose(R.T @ R, np.eye(3), atol=ROTATION_TOLERANCE)
        or np.linalg.det(R) < 0
    ):
        raise RegionsError(
            "Region %r: %s must be a rotation (orthonormal columns, determinant +1)"
            % (region_id, name)
        )
    return R


def _unit(value, name, region_id, length=3):
    v = _vector(value, name, region_id, length)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise RegionsError("Region %r: %s must not be zero" % (region_id, name))
    return v / norm


def _cuts(value, region_id, key="cuts", normal="normal", length=3):
    cuts = []
    for k, c in enumerate(value or []):
        if not isinstance(c, dict) or normal not in c or "offset_A" not in c:
            raise RegionsError(
                "Region %r: %s[%d] needs %s and offset_A" % (region_id, key, k, normal)
            )
        offset = float(c["offset_A"])
        if not np.isfinite(offset):
            raise RegionsError(
                "Region %r: %s[%d] offset is not finite" % (region_id, key, k)
            )
        cuts.append(
            {
                normal: _unit(
                    c[normal], "%s[%d].%s" % (key, k, normal), region_id, length
                ),
                "offset_A": offset,
            }
        )
    return cuts


def _geometry(entry, region_id):
    kind = entry["kind"]
    missing = [k for k in GEOMETRY[kind] if k not in entry]
    if missing:
        raise RegionsError(
            "Region %r (%s) is missing %s" % (region_id, kind, ", ".join(missing))
        )
    if kind == "slab":
        return _slab(entry, region_id)
    g = {"centre_A": _vector(entry["centre_A"], "centre_A", region_id)}
    if kind == "box":
        h = _vector(entry["half_lengths_A"], "half_lengths_A", region_id)
        if np.any(h <= 0):
            raise RegionsError("Region %r: half_lengths_A must be positive" % region_id)
        g["half_lengths_A"] = h
        g["axes"] = _rotation(entry.get("axes"), "axes", region_id)
    elif kind == "sphere":
        g["radius_A"] = _positive(entry["radius_A"], "radius_A", region_id)
    elif kind == "cylinder":
        g["axis"] = _unit(entry["axis"], "axis", region_id)
        g["radius_A"] = _positive(entry["radius_A"], "radius_A", region_id)
        g["length_A"] = _positive(entry["length_A"], "length_A", region_id)
    else:  # ellipsoid, ellipsoidal_shell
        a = _vector(entry["semi_axes_A"], "semi_axes_A", region_id)
        if np.any(a <= 0):
            raise RegionsError("Region %r: semi_axes_A must be positive" % region_id)
        g["semi_axes_A"] = a
        g["axes"] = _rotation(entry.get("axes"), "axes", region_id)
        g["cuts"] = _cuts(entry.get("cuts"), region_id)
        if kind == "ellipsoidal_shell":
            inner = entry["inner"]
            mode = inner.get("mode") if isinstance(inner, dict) else None
            if mode == "scale":
                f = float(inner.get("factor", float("nan")))
                if not 0 < f < 1:
                    raise RegionsError(
                        "Region %r: inner factor must be in (0, 1)" % region_id
                    )
                g["inner"] = {"mode": "scale", "factor": f}
            elif mode == "offset":
                t = _positive(inner.get("thickness_A"), "inner thickness_A", region_id)
                if t >= a.min():
                    raise RegionsError(
                        "Region %r: inner thickness_A must be less than the smallest semi-axis"
                        % region_id
                    )
                g["inner"] = {"mode": "offset", "thickness_A": t}
            else:
                raise RegionsError(
                    "Region %r: inner must be {mode: scale, factor} or {mode: offset, thickness_A}"
                    % region_id
                )
            co = float(entry.get("cut_offset_A", 0.0))
            if not np.isfinite(co) or co < 0:
                raise RegionsError(
                    "Region %r: cut_offset_A must be finite and >= 0" % region_id
                )
            g["cut_offset_A"] = co
    return g


def _slab(entry, region_id):
    if entry.get("thickness_map") is not None:
        raise RegionsError(
            "Region %r: a slab's thickness_map is not supported yet; give a uniform thickness_A"
            % region_id
        )
    n = _unit(entry["normal"], "normal", region_id)
    u = _vector(entry["u_axis"], "u_axis", region_id)
    u = u - (u @ n) * n
    if np.linalg.norm(u) < 1e-9:
        raise RegionsError(
            "Region %r: u_axis must not be parallel to normal" % region_id
        )
    u = u / np.linalg.norm(u)
    g = {
        "origin_A": _vector(entry["origin_A"], "origin_A", region_id),
        "normal": n,
        "u_axis": u,
        "v_axis": np.cross(n, u),
        "thickness_A": _positive(entry["thickness_A"], "thickness_A", region_id),
    }
    fp = entry["footprint"]
    kind = fp.get("kind") if isinstance(fp, dict) else None
    if kind == "rectangle":
        g["footprint"] = {
            "kind": "rectangle",
            "half_u_A": _positive(fp.get("half_u_A"), "footprint half_u_A", region_id),
            "half_v_A": _positive(fp.get("half_v_A"), "footprint half_v_A", region_id),
        }
    elif kind == "disc":
        g["footprint"] = {
            "kind": "disc",
            "centre_uv_A": _vector(
                fp.get("centre_uv_A"), "footprint centre_uv_A", region_id, 2
            ),
            "radius_A": _positive(fp.get("radius_A"), "footprint radius_A", region_id),
        }
    else:
        raise RegionsError(
            "Region %r: footprint kind must be rectangle or disc" % region_id
        )
    g["footprint"]["clips"] = _cuts(fp.get("clips"), region_id, "clips", "normal_uv", 2)
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

    containment(document)  # checks the rule's name
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


def _ellipsoid_body(p, centre, R, a, cuts, cut_shift=0.0):
    d = p - centre
    local = d @ R
    inside = np.einsum("ij,ij->i", local / a, local / a) < 1
    for c in cuts:
        inside &= d @ c["normal"] <= c["offset_A"] - cut_shift
    return inside


def contains(region, points):
    """
    Whether points are inside a region's own shape (its parent is not
    considered)

    Boxes are half open, [-half, +half) along each of their axes, like the
    sample's cube and cuboid shapes; a sphere is |p - c| < r; a cylinder is
    0 <= t < length along its axis from the start and < r from it; an
    ellipsoid is sum (x_i / a_i)^2 < 1 in its own axes, inside every cut; a
    shell is its outer ellipsoid less its inner one; a slab is
    -t/2 <= (p - origin) . normal < t/2 over its footprint.

    Args:
        region: A Region
        points: An (N, 3) array, A

    Returns:
        A boolean array of length N

    """
    p = np.asarray(points, dtype="float64").reshape(-1, 3)
    g = region.geometry
    if region.kind == "slab":
        d = p - g["origin_A"]
        s = d @ g["normal"]
        u, v = d @ g["u_axis"], d @ g["v_axis"]
        t = g["thickness_A"]
        inside = (s >= -t / 2) & (s < t / 2)
        fp = g["footprint"]
        if fp["kind"] == "rectangle":
            inside &= (np.abs(u) < fp["half_u_A"]) & (np.abs(v) < fp["half_v_A"])
        else:
            cu, cv = fp["centre_uv_A"]
            inside &= (u - cu) ** 2 + (v - cv) ** 2 < fp["radius_A"] ** 2
        for c in fp["clips"]:
            inside &= u * c["normal_uv"][0] + v * c["normal_uv"][1] <= c["offset_A"]
        return inside
    d = p - g["centre_A"]
    if region.kind == "box":
        local = d @ g["axes"]
        h = g["half_lengths_A"]
        return np.all((local >= -h) & (local < h), axis=1)
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
    if region.kind in ("ellipsoid", "ellipsoidal_shell"):
        outer = _ellipsoid_body(
            p, g["centre_A"], g["axes"], g["semi_axes_A"], g["cuts"]
        )
        if region.kind == "ellipsoid":
            return outer
        inner = g["inner"]
        a_in = (
            g["semi_axes_A"] * inner["factor"]
            if inner["mode"] == "scale"
            else g["semi_axes_A"] - inner["thickness_A"]
        )
        return outer & ~_ellipsoid_body(
            p, g["centre_A"], g["axes"], a_in, g["cuts"], g["cut_offset_A"]
        )
    raise RegionsError("Unknown kind %r" % region.kind)


def _plane_crossings(p0, d, normals, offsets):
    """t where p0 + t d meets each plane n . x = b (NaN where parallel)"""
    nd = np.asarray(normals) @ d
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (np.asarray(offsets)[None, :] - p0 @ np.asarray(normals).T) / nd
    t[:, np.abs(nd) < 1e-15] = np.nan
    return t


def _quadric_crossings(u, v):
    """t where |u + t v|^2 = 1, per row of u (v shared); NaN where it misses"""
    a = v @ v
    b = 2 * (u @ v)
    c = np.einsum("ij,ij->i", u, u) - 1
    disc = b * b - 4 * a * c
    root = np.sqrt(np.where(disc >= 0, disc, np.nan))
    if a < 1e-30:
        return np.full((len(u), 2), np.nan)
    return np.stack([(-b - root) / (2 * a), (-b + root) / (2 * a)], axis=1)


def _ellipsoid_crossings(p0, d, centre, R, a, cuts, cut_shift=0.0):
    w = p0 - centre
    t = [_quadric_crossings((w @ R) / a, (d @ R) / a)]
    if cuts:
        normals = np.array([c["normal"] for c in cuts])
        offsets = np.array([c["offset_A"] - cut_shift for c in cuts]) + normals @ centre
        t.append(_plane_crossings(p0, d, normals, offsets))
    return np.concatenate(t, axis=1)


def boundary_crossings(region, p0, d):
    """
    Every t at which the rays p0 + t d (p0 (N, 3), d (3,) unit) cross a surface
    of the region's shape. Between consecutive crossings a ray is either inside
    the shape throughout or outside throughout, so these, with `contains` at
    one point per piece, give exact path lengths. Some crossings may lie where
    the ray is outside the shape anyway; they only split a piece in two.

    Returns:
        An (N, k) array of t, NaN where a surface is not met

    """
    p0 = np.asarray(p0, dtype="float64").reshape(-1, 3)
    d = np.asarray(d, dtype="float64")
    g = region.geometry
    if region.kind == "box":
        R, h, c = g["axes"], g["half_lengths_A"], g["centre_A"]
        normals = np.concatenate([R.T, R.T])
        offsets = np.concatenate([R.T @ c + h, R.T @ c - h])
        return _plane_crossings(p0, d, normals, offsets)
    if region.kind == "sphere":
        return _quadric_crossings(
            (p0 - g["centre_A"]) / g["radius_A"], d / g["radius_A"]
        )
    if region.kind == "cylinder":
        a, c, r, L = g["axis"], g["centre_A"], g["radius_A"], g["length_A"]
        caps = _plane_crossings(
            p0, d, np.array([a, a]), np.array([a @ c + L / 2, a @ c - L / 2])
        )
        w = p0 - c
        u = (w - np.outer(w @ a, a)) / r
        v = (d - (d @ a) * a) / r
        return np.concatenate([caps, _quadric_crossings(u, v)], axis=1)
    if region.kind in ("ellipsoid", "ellipsoidal_shell"):
        out = _ellipsoid_crossings(
            p0, d, g["centre_A"], g["axes"], g["semi_axes_A"], g["cuts"]
        )
        if region.kind == "ellipsoid":
            return out
        inner = g["inner"]
        a_in = (
            g["semi_axes_A"] * inner["factor"]
            if inner["mode"] == "scale"
            else g["semi_axes_A"] - inner["thickness_A"]
        )
        return np.concatenate(
            [
                out,
                _ellipsoid_crossings(
                    p0, d, g["centre_A"], g["axes"], a_in, g["cuts"], g["cut_offset_A"]
                ),
            ],
            axis=1,
        )
    if region.kind == "slab":
        o, n, u_ax, v_ax, t = (
            g["origin_A"],
            g["normal"],
            g["u_axis"],
            g["v_axis"],
            g["thickness_A"],
        )
        planes = [(n, n @ o + t / 2), (n, n @ o - t / 2)]
        fp = g["footprint"]
        if fp["kind"] == "rectangle":
            for ax, half in ((u_ax, fp["half_u_A"]), (v_ax, fp["half_v_A"])):
                planes += [(ax, ax @ o + half), (ax, ax @ o - half)]
        for c in fp["clips"]:
            m = c["normal_uv"][0] * u_ax + c["normal_uv"][1] * v_ax
            planes.append((m, c["offset_A"] + m @ o))
        out = [
            _plane_crossings(
                p0,
                d,
                np.array([q[0] for q in planes]),
                np.array([q[1] for q in planes]),
            )
        ]
        if fp["kind"] == "disc":
            cu, cv = fp["centre_uv_A"]
            r = fp["radius_A"]
            w = p0 - o
            uv = np.stack([w @ u_ax - cu, w @ v_ax - cv], axis=1) / r
            duv = np.array([d @ u_ax, d @ v_ax]) / r
            out.append(_quadric_crossings(uv, duv))
        return np.concatenate(out, axis=1)
    raise RegionsError("Unknown kind %r" % region.kind)


def thinnest(region):
    """
    The region's smallest dimension (A), whatever its orientation: a sphere's or
    cylinder's diameter, a box's smallest side, an ellipsoid's smallest
    diameter, a shell's wall, a slab's thickness. Cuts can make a region
    thinner still near them; that is not counted

    """
    g = region.geometry
    if region.kind == "sphere":
        return 2 * g["radius_A"]
    if region.kind == "cylinder":
        return min(2 * g["radius_A"], g["length_A"])
    if region.kind == "box":
        return 2 * float(np.min(g["half_lengths_A"]))
    if region.kind == "ellipsoid":
        return 2 * float(np.min(g["semi_axes_A"]))
    if region.kind == "ellipsoidal_shell":
        inner = g["inner"]
        if inner["mode"] == "offset":
            return inner["thickness_A"]
        return float(np.min(g["semi_axes_A"])) * (1 - inner["factor"])
    if region.kind == "slab":
        return g["thickness_A"]
    raise RegionsError("Unknown kind %r" % region.kind)


def bounds(region):
    """
    An axis-aligned box (lower, upper corners) containing a region's shape

    """
    g = region.geometry
    if region.kind == "slab":
        fp = g["footprint"]
        if fp["kind"] == "rectangle":
            uv = np.array(
                [
                    [su * fp["half_u_A"], sv * fp["half_v_A"]]
                    for su in (-1, 1)
                    for sv in (-1, 1)
                ]
            )
        else:
            phi = np.linspace(0, 2 * np.pi, 64, endpoint=False)
            r = fp["radius_A"] / np.cos(
                np.pi / 64
            )  # the polygon circumscribes the disc
            uv = fp["centre_uv_A"] + r * np.stack([np.cos(phi), np.sin(phi)], axis=1)
        pts = []
        for s in (-g["thickness_A"] / 2, g["thickness_A"] / 2):
            pts.append(
                g["origin_A"]
                + uv[:, :1] * g["u_axis"]
                + uv[:, 1:] * g["v_axis"]
                + s * g["normal"]
            )
        pts = np.concatenate(pts)
        return pts.min(axis=0), pts.max(axis=0)
    c = g["centre_A"]
    if region.kind == "box":
        e = np.abs(g["axes"]) @ g["half_lengths_A"]
    elif region.kind == "sphere":
        e = np.full(3, g["radius_A"])
    elif region.kind == "cylinder":
        a = g["axis"]
        e = np.abs(a) * g["length_A"] / 2 + g["radius_A"] * np.sqrt(
            np.clip(1 - a * a, 0, None)
        )
    else:
        e = np.sqrt((g["axes"] ** 2) @ (g["semi_axes_A"] ** 2))
    return c - e, c + e


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


EXACT_NESTING = ("box", "sphere", "cylinder")


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
        return c + (signs * h) @ g["axes"].T
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
        local = d @ g["axes"]
        return np.all(np.abs(local) <= g["half_lengths_A"] - margin + tol, axis=1)
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


def _nested_sampled(child, parent):
    """
    Nesting by sampling: points drawn uniformly in the child's bounding box
    (a fixed seed), and every one inside the child must be inside the parent.
    A sliver thinner than the sampling can be missed

    """
    lo, hi = bounds(child)
    rng = np.random.default_rng(0)
    p = lo + (hi - lo) * rng.random((NESTING_SAMPLES, 3))
    p = p[contains(child, p)]
    if len(p) < NESTING_MIN_INSIDE:
        raise RegionsError(
            "Region %r is too thin for its nesting in %r to be sampled"
            % (child.id, parent.id)
        )
    return bool(np.all(contains(parent, p)))


def nested(child, parent):
    """
    Whether a child's shape lies inside its parent's shape

    Boxes, spheres and cylinders are convex, so between them it is exact: a box
    or cylinder is inside when the points spanning its hull are (corners; rims,
    sampled), and a sphere when its centre is inside the parent shrunk by its
    radius. Any other pair is checked by sampling (`_nested_sampled`).

    """
    if child.kind not in EXACT_NESTING or parent.kind not in EXACT_NESTING:
        return _nested_sampled(child, parent)
    if child.kind == "sphere":
        points = child.geometry["centre_A"][None, :]
        margin = child.geometry["radius_A"]
    else:
        points = _farthest_points(child)
        margin = 0.0
    return bool(np.all(_inside_eroded(parent, points, margin)))


CONTAINMENT = ("refuse", "clip")


def containment(document):
    """
    The document's containment rule: "refuse" (the default) or "clip"

    With "refuse", a child whose shape pokes out of its parent's is an error.
    With "clip", it is allowed and clipped to its parent, as `owner()` and the
    path-length maps always do (a point is in a region only if it is also in
    the region's parent); only the multislice renderer, whose paint plan cannot
    clip, still refuses such a child until the engine has a label raster.

    """
    document = load(document)
    mode = document.get("containment", "refuse")
    if mode not in CONTAINMENT:
        raise RegionsError(
            "containment must be one of %s, not %r" % (CONTAINMENT, mode)
        )
    return mode


def outside_fraction(child, parent):
    """The fraction of a child's volume outside its parent (sampled, as in nesting)"""
    lo, hi = bounds(child)
    rng = np.random.default_rng(0)
    p = lo + (hi - lo) * rng.random((NESTING_SAMPLES, 3))
    p = p[contains(child, p)]
    return float(np.mean(~contains(parent, p))) if len(p) else 0.0


def check_nesting(regions, mode="refuse"):
    """
    Check that every child lies inside its parent

    Args:
        regions: The list of Region
        mode: "refuse" raises for the first child outside its parent; "clip"
            allows it, and returns what is clipped

    Returns:
        list: (child id, parent id, fraction outside) for each child outside
        its parent (only with "clip"; "refuse" raises instead)

    Raises:
        RegionsError: Naming the first such child, with "refuse"

    """
    by_id = {r.id: r for r in regions}
    clipped = []
    for r in regions:
        if r.parent is not None and not nested(r, by_id[r.parent]):
            if mode == "clip":
                clipped.append((r.id, r.parent, outside_fraction(r, by_id[r.parent])))
                continue
            raise RegionsError(
                "Region %r is not inside its parent %r. Clipping a child to its "
                "parent is not supported yet: make the child's shape fit "
                "inside the parent's" % (r.id, r.parent)
            )
    return clipped


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


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def to_json(medium, regions, containment="refuse"):
    """
    Write the document back (normalised: unit vectors, defaults filled)

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
            if key == "v_axis":
                continue  # derived from normal and u_axis
            entry[key] = _jsonable(value)
        out.append(entry)
    doc = {"schema": SCHEMA, "medium": medium, "regions": out}
    if containment != "refuse":
        doc["containment"] = containment
    return json.dumps(doc)


def engine_shape(region, convex=False):
    """
    The shape the multislice engine renders for a region: (kind, geometry) for
    an axis-aligned box, a sphere (or an ellipsoid that is one: equal
    semi-axes, no cuts) or a cylinder; and, when the engine has convex regions
    (convex=True), ("convex", convex_form(region)) for every other kind

    Raises:
        RegionsError: Naming a region the engine cannot render

    """
    g = region.geometry
    if region.kind == "box" and np.allclose(g["axes"], np.eye(3)):
        return "box", g
    if region.kind in ("sphere", "cylinder"):
        return region.kind, g
    if (
        region.kind == "ellipsoid"
        and not g["cuts"]
        and np.allclose(g["semi_axes_A"], g["semi_axes_A"][0])
    ):
        return "sphere", {
            "centre_A": g["centre_A"],
            "radius_A": float(g["semi_axes_A"][0]),
        }
    if convex:
        return "convex", convex_form(region)
    what = "rotated box" if region.kind == "box" else region.kind
    raise RegionsError(
        "Region %r (%s) cannot be rendered by this multislice engine: it needs the "
        "engine feature convex_regions. Axis-aligned boxes, spheres and cylinders "
        "can be rendered" % (region.id, what)
    )


def _ellipsoid_set(centre, R, a, cuts, cut_shift=0.0):
    """The convex set of an ellipsoid (axes R, semi-axes a) and its cuts"""
    A = R @ np.diag(1.0 / np.asarray(a, dtype=float) ** 2) @ R.T
    planes = [
        (np.asarray(c["normal"], float), float(c["offset_A"] - cut_shift + c["normal"] @ centre))
        for c in cuts
    ]
    return {"A": A, "centre": np.asarray(centre, float), "planes": planes}


def convex_form(region):
    """
    A region as the engine's convex set: an optional quadric
    (p - centre)^T A (p - centre) <= 1 intersected with half-spaces n . p <= d,
    in the sample frame, less an optional inner set of the same form (a shell)::

        {"A": 3 x 3 or None, "centre": (3,), "planes": [(n, d), ...],
         "inner": None or {"A", "centre", "planes"}}

    `convex_contains` tests points against it; it agrees with `contains`
    everywhere but on the surfaces.

    """
    g = region.geometry
    if region.kind == "box":
        c, h, R = g["centre_A"], g["half_lengths_A"], g["axes"]
        planes = []
        for k in range(3):
            a = R[:, k]
            planes += [(a, float(h[k] + a @ c)), (-a, float(h[k] - a @ c))]
        form = {"A": None, "centre": np.asarray(c, float), "planes": planes}
    elif region.kind in ("sphere", "ellipsoid", "ellipsoidal_shell"):
        if region.kind == "sphere":
            c, R, a, cuts = g["centre_A"], np.eye(3), [g["radius_A"]] * 3, []
        else:
            c, R, a, cuts = g["centre_A"], g["axes"], g["semi_axes_A"], g["cuts"]
        form = _ellipsoid_set(c, R, a, cuts)
        if region.kind == "ellipsoidal_shell":
            inner = g["inner"]
            a_in = (
                g["semi_axes_A"] * inner["factor"]
                if inner["mode"] == "scale"
                else g["semi_axes_A"] - inner["thickness_A"]
            )
            form["inner"] = _ellipsoid_set(c, R, a_in, cuts, g["cut_offset_A"])
    elif region.kind == "cylinder":
        c, a, L = np.asarray(g["centre_A"], float), g["axis"], g["length_A"]
        form = {
            "A": (np.eye(3) - np.outer(a, a)) / g["radius_A"] ** 2,
            "centre": c,
            "planes": [(a, float(L / 2 + a @ c)), (-a, float(L / 2 - a @ c))],
        }
    elif region.kind == "slab":
        o, n, u, v, t = (g["origin_A"], g["normal"], g["u_axis"], g["v_axis"], g["thickness_A"])
        planes = [(n, float(t / 2 + n @ o)), (-n, float(t / 2 - n @ o))]
        fp = g["footprint"]
        A, centre = None, np.asarray(o, float)
        if fp["kind"] == "rectangle":
            for ax, half in ((u, fp["half_u_A"]), (v, fp["half_v_A"])):
                planes += [(ax, float(half + ax @ o)), (-ax, float(half - ax @ o))]
        else:
            cu, cv = fp["centre_uv_A"]
            centre = o + cu * u + cv * v
            A = (np.eye(3) - np.outer(n, n)) / fp["radius_A"] ** 2
        for c in fp["clips"]:
            m = c["normal_uv"][0] * u + c["normal_uv"][1] * v
            planes.append((m, float(c["offset_A"] + m @ o)))
        form = {"A": A, "centre": centre, "planes": planes}
    else:
        raise RegionsError("Region %r (%s) has no convex form" % (region.id, region.kind))
    form.setdefault("inner", None)
    return form


def convex_contains(form, points):
    """Whether points are in a convex_form (the engine's semantics, on the CPU)"""
    p = np.asarray(points, dtype="float64").reshape(-1, 3)

    def inside(f):
        ok = np.ones(len(p), dtype=bool)
        if f["A"] is not None:
            d = p - f["centre"]
            ok &= np.einsum("ij,jk,ik->i", d, f["A"], d) <= 1
        for n, b in f["planes"]:
            ok &= p @ n <= b
        return ok

    result = inside(form)
    if form["inner"] is not None:
        result &= ~inside(form["inner"])
    return result


def materials_used(medium, regions) -> Dict[str, List[str]]:
    """
    The materials used, and by which regions ("<medium>" for the medium)

    """
    used: Dict[str, List[str]] = {}
    used.setdefault(medium, []).append("<medium>")
    for r in regions:
        used.setdefault(r.material, []).append(r.id)
    return used
