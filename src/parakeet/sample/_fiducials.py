#
# parakeet.sample._fiducials.py
#
# Copyright (C) 2026 Diamond Light Source and Rosalind Franklin Institute
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import logging
import numpy as np
import pandas
import scipy.constants
import parakeet.config
from functools import singledispatch
from parakeet.sample import Sample
from parakeet.sample import AtomData


__all__ = ["add_fiducials"]


# Get the logger
logger = logging.getLogger(__name__)


# Gold material properties
AU_ATOMIC_NUMBER = 79
AU_MOLAR_MASS = 196.967  # g/mol
AU_DENSITY = 19.3  # g/cm^3
# Number density: atoms per A^3
AU_NUMBER_DENSITY = (AU_DENSITY * 1e-24 * scipy.constants.Avogadro) / AU_MOLAR_MASS


def _generate_sphere_atoms(radius, centre, n_atoms):
    """
    Generate uniformly distributed atom positions within a sphere.

    Uses rejection sampling: generate points in a cube, keep those inside
    the sphere. Acceptance ratio is pi/6 ~ 0.524.

    Args:
        radius: Sphere radius in Angstroms.
        centre: (x, y, z) centre position in Angstroms.
        n_atoms: Number of atoms to place.

    Returns:
        numpy array of shape (n_atoms, 3) with atom positions.

    """
    centre = np.array(centre, dtype="float32")
    collected = []
    remaining = n_atoms

    while remaining > 0:
        # Oversample by ~2x to account for rejection (sphere/cube ratio ~ 0.524)
        n_gen = int(remaining * 2.2) + 100
        xyz = np.random.uniform(-radius, radius, size=(n_gen, 3)).astype("float32")
        r_sq = np.sum(xyz**2, axis=1)
        inside = xyz[r_sq <= radius**2]

        take = min(len(inside), remaining)
        collected.append(inside[:take])
        remaining -= take

    positions = np.concatenate(collected, axis=0)
    positions += centre
    return positions


def _place_fiducial_centres(sample, config):
    """
    Determine fiducial centre positions based on placement mode.

    Fiducials are placed with non-overlap constraint (centre-to-centre
    distance >= diameter). Uses simple rejection: generate candidates,
    check pairwise distances against all accepted so far.

    Args:
        sample: The Sample object.
        config: A Fiducial config object.

    Returns:
        numpy array of shape (count, 3) with fiducial centre positions.

    """
    radius = config.diameter / 2.0
    count = config.count
    placement = config.placement

    # Get sample geometry
    shape = sample.shape
    centre = np.array(sample.centre, dtype="float64")

    if shape["type"] == "cube":
        length = shape["cube"]["length"]
        half = length / 2.0
        x_min, x_max = centre[0] - half, centre[0] + half
        y_min, y_max = centre[1] - half, centre[1] + half
        z_min, z_max = centre[2] - half, centre[2] + half
    elif shape["type"] == "cuboid":
        lx = shape["cuboid"]["length_x"]
        ly = shape["cuboid"]["length_y"]
        lz = shape["cuboid"]["length_z"]
        x_min, x_max = centre[0] - lx / 2.0, centre[0] + lx / 2.0
        y_min, y_max = centre[1] - ly / 2.0, centre[1] + ly / 2.0
        z_min, z_max = centre[2] - lz / 2.0, centre[2] + lz / 2.0
    elif shape["type"] == "cylinder":
        cyl = shape["cylinder"]
        cyl_radius = cyl["radius"]
        cyl_length = cyl["length"]
        # Cylinder axis is along y; x,z are radial
        x_min, x_max = centre[0] - cyl_radius, centre[0] + cyl_radius
        y_min, y_max = centre[1] - cyl_length / 2.0, centre[1] + cyl_length / 2.0
        z_min, z_max = centre[2] - cyl_radius, centre[2] + cyl_radius
    else:
        raise RuntimeError(f"Unknown shape type: {shape['type']}")

    # Clamp ranges so fiducial spheres stay inside sample bounds
    x_min_eff = x_min + radius
    x_max_eff = x_max - radius
    y_min_eff = y_min + radius
    y_max_eff = y_max - radius

    if x_max_eff <= x_min_eff or y_max_eff <= y_min_eff:
        raise RuntimeError(
            f"Sample too small for fiducials of diameter {config.diameter} A. "
            f"Available x range: {x_max - x_min} A, y range: {y_max - y_min} A"
        )

    # Determine z positions based on placement mode
    if placement == "surface":
        # Fiducials sit on top surface of ice
        z_positions_fn = lambda n: np.full(n, z_max - radius, dtype="float64")
    elif placement == "both_surfaces":
        # Alternate between top and bottom
        def z_positions_fn(n):
            z = np.empty(n, dtype="float64")
            z[0::2] = z_max - radius  # top
            z[1::2] = z_min + radius  # bottom
            return z

    elif placement == "volume":
        # Random z within volume
        z_min_eff = z_min + radius
        z_max_eff = z_max - radius
        if z_max_eff <= z_min_eff:
            raise RuntimeError(
                f"Sample too thin for fiducials of diameter {config.diameter} A. "
                f"Available z range: {z_max - z_min} A"
            )
        z_positions_fn = lambda n: np.random.uniform(z_min_eff, z_max_eff, size=n)
    else:
        raise RuntimeError(
            f"Unknown placement mode: '{placement}'. "
            f"Expected 'surface', 'both_surfaces', or 'volume'."
        )

    # Place fiducials with non-overlap rejection
    min_separation = config.diameter  # centre-to-centre >= diameter
    accepted = []
    max_attempts = count * 200

    z_candidates = z_positions_fn(count)

    for attempt in range(max_attempts):
        if len(accepted) >= count:
            break

        idx = len(accepted)
        x = np.random.uniform(x_min_eff, x_max_eff)
        y = np.random.uniform(y_min_eff, y_max_eff)
        z = z_candidates[idx] if placement != "volume" else z_positions_fn(1)[0]

        # For cylinder shape, check that (x, z) is inside the cylinder
        if shape["type"] == "cylinder":
            dx = x - centre[0]
            dz = z - centre[2]
            if dx**2 + dz**2 > (cyl_radius - radius) ** 2:
                continue

        candidate = np.array([x, y, z])

        # Check distance to all previously accepted fiducials
        if len(accepted) > 0:
            dists = np.sqrt(np.sum((np.array(accepted) - candidate) ** 2, axis=1))
            if np.any(dists < min_separation):
                continue

        accepted.append(candidate)

    if len(accepted) < count:
        logger.warning(
            f"Could only place {len(accepted)}/{count} fiducials without overlap "
            f"(diameter={config.diameter} A). Reduce count or increase sample size."
        )

    positions = np.array(accepted, dtype="float64")
    return positions


def _create_atom_data(atomic_number, coords, sigma=0.085, occupancy=1.0):
    """
    Create an AtomData-compatible DataFrame from atom positions.

    Args:
        atomic_number: Atomic number (79 for Au).
        coords: numpy array of shape (N, 3).
        sigma: Positional sigma (B-factor proxy).
        occupancy: Atom occupancy.

    Returns:
        pandas DataFrame with AtomData columns.

    """
    n = coords.shape[0]
    return pandas.DataFrame(
        {
            "atomic_number": np.full(n, atomic_number, dtype="uint8"),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "sigma": np.full(n, sigma, dtype="float32"),
            "occupancy": np.full(n, occupancy, dtype="float32"),
            "charge": np.zeros(n, dtype="int8"),
        }
    )


@singledispatch
def add_fiducials(config_file, sample_file: str) -> Sample:
    """
    Add gold fiducial nanoparticles to the sample.

    Args:
        config_file: The input config filename.
        sample_file: The sample filename.

    """
    config = parakeet.config.load(config_file)
    parakeet.config.show(config)
    logger.info(f"Writing sample to {sample_file}")
    return _add_fiducials_Config(config, sample_file)


@add_fiducials.register(parakeet.config.Config)
def _add_fiducials_Config(config: parakeet.config.Config, sample_file: str) -> Sample:
    """
    Add fiducials from a full Config object.
    """
    sample = Sample(sample_file, mode="r+")
    if config.sample.fiducials:
        sample = _add_fiducials_Fiducial(config.sample.fiducials, sample)
    else:
        logger.info("No fiducial parameters found in config")
    return sample


@add_fiducials.register(parakeet.config.Fiducial)
def _add_fiducials_Fiducial(config: parakeet.config.Fiducial, sample: Sample) -> Sample:
    """
    Add gold fiducial nanoparticles to the sample.

    Places spherical clusters of gold atoms at bulk density (19.3 g/cm^3)
    at positions determined by the placement mode. Each fiducial is a sphere
    of the specified diameter filled with randomly placed Au atoms.

    Args:
        config: The Fiducial config object.
        sample: The Sample object.

    Returns:
        The sample with fiducial atoms added.

    """
    radius = config.diameter / 2.0
    volume_per_fiducial = (4.0 / 3.0) * np.pi * radius**3
    atoms_per_fiducial = int(AU_NUMBER_DENSITY * volume_per_fiducial)

    logger.info(
        f"Fiducials: diameter={config.diameter} A ({config.diameter/10:.1f} nm), "
        f"count={config.count}, placement={config.placement}, "
        f"occupancy={config.occupancy}"
    )
    logger.info(
        f"Fiducials: {atoms_per_fiducial} Au atoms per fiducial "
        f"(density={AU_DENSITY} g/cm^3, volume={volume_per_fiducial:.0f} A^3)"
    )

    # Determine placement positions
    centres = _place_fiducial_centres(sample, config)
    n_placed = len(centres)

    if n_placed == 0:
        logger.warning("No fiducials could be placed")
        return sample

    logger.info(f"Fiducials: placing {n_placed} fiducials")

    # Generate and add atoms for each fiducial
    total_atoms = 0
    data_buffers = []
    occupancy = config.occupancy

    for i, centre in enumerate(centres):
        # Generate gold atom positions within sphere
        positions = _generate_sphere_atoms(radius, centre, atoms_per_fiducial)

        # Create atom data (occupancy > 1.0 boosts scattering to compensate
        # for GRF ice potential overlap inside the fiducial volume)
        df = _create_atom_data(AU_ATOMIC_NUMBER, positions, occupancy=occupancy)
        data_buffers.append(df)
        total_atoms += len(df)

        logger.info(
            f"    Fiducial {i+1}/{n_placed}: "
            f"centre=({centre[0]:.1f}, {centre[1]:.1f}, {centre[2]:.1f}) A, "
            f"{len(df)} atoms"
        )

    # Combine all fiducial atoms and add to sample
    all_data = pandas.concat(data_buffers, ignore_index=True)
    sample.add_atoms(AtomData(data=all_data))

    logger.info(
        f"Fiducials: placed {total_atoms} Au atoms total across {n_placed} fiducials"
    )
    return sample
