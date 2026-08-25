#!/usr/bin/env python3
"""
Check that a benchmark config asks for a field its particles can actually fit in.

Parakeet places particles by relaxing them apart with a non-overlap constraint
and a box margin. When the field is too crowded the relaxation runs out of
iterations and returns what it has, without complaining -- so particles end up
overlapping the box wall and are clipped in the image, which shows up as hard
edges and CTF ringing rather than as an error. This catches that before a run.

Usage:
    benchmark/check_config.py benchmark/configs/spa_70s.yaml
    benchmark/check_config.py benchmark/configs/*.yaml
"""

import argparse
import sys

import numpy as np

import parakeet.config
import parakeet.data
import parakeet.sample

# Particles need roughly this multiple of their own footprint each before a
# random placement stops fighting the walls and its neighbours.
COMFORTABLE = 3.0


def bounding_radius(pdb_id):
    """Radius of the sphere enclosing the structure about its centre of mass."""
    coords = (
        parakeet.sample.AtomData.from_gemmi_file(parakeet.data.get_pdb(pdb_id))
        .data[["x", "y", "z"]]
        .to_numpy()
    )
    return float(np.linalg.norm(coords - coords.mean(axis=0), axis=1).max())


def check(path):
    config = parakeet.config.load(path)
    sample = config.sample

    if not (sample.molecules and sample.molecules.pdb):
        print(f"{path}: no pdb molecules to check")
        return True

    box_x, box_y, box_z = sample.box
    ok = True
    for entry in sample.molecules.pdb:
        count = (
            entry.instances
            if isinstance(entry.instances, int)
            else len(entry.instances)
        )
        radius = bounding_radius(entry.id)
        footprint = (2 * radius) ** 2
        touching = box_x * box_y / footprint
        comfortable = touching / COMFORTABLE

        print(f"{path}")
        print(
            f"  {entry.id}: bounding radius {radius:.0f} A (diameter {2 * radius:.0f} A)"
        )
        print(
            f"  field {box_x:.0f} x {box_y:.0f} A holds {comfortable:.1f} comfortably, "
            f"{touching:.1f} touching; config asks for {count}"
        )

        if count > touching:
            print(f"  FAIL: {count} cannot fit -- they overlap even when touching")
            ok = False
        elif count > comfortable:
            print(f"  WARN: {count} is crowded; expect particles clipped at the walls")
            ok = False
        else:
            print("  ok")

        # z is the ice: a particle centred too near a face pokes out of the ice
        if 2 * radius > box_z:
            print(
                f"  WARN: particle ({2 * radius:.0f} A) is thicker than the ice ({box_z:.0f} A)"
            )
            ok = False

    if sample.fiducials:
        f = sample.fiducials
        print(f"  fiducials: {f.count} x {f.diameter:.0f} A, {f.placement}")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args()
    return 0 if all([check(c) for c in args.configs]) else 1


if __name__ == "__main__":
    sys.exit(main())
