#
# parakeet.sample._support_film.py
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


__all__ = ["support_film"]


# Get the logger
logger = logging.getLogger(__name__)


# Material properties: (atomic_number, molar_mass_g_per_mol, density_g_per_cm3)
MATERIAL_PROPERTIES = {
    "carbon": (6, 12.011, 2.0),
    "gold": (79, 196.967, 19.3),
}


def _generate_support_film_atoms(sample, config, max_atoms=5_000_000):
    """
    Place explicit atoms in the annular foil region (outside hole).

    The foil sits at the top of the ice slab (highest z). Atoms are placed
    in the annular region outside the circular hole using rejection sampling.

    At true atomic density, a 1.6 um x 1.6 um x 20 nm carbon foil would
    contain ~2.9 billion atoms (55 GB RAM). Instead we use coarse-grained
    placement: fewer atoms with increased occupancy to preserve the total
    projected scattering potential. This produces correct contrast at
    low-to-moderate magnification (>= 5 A/px).

    Args:
        sample: The Sample object (HDF5-backed).
        config: A SupportFilm config object.
        max_atoms: Maximum number of atoms to place (default 5M, ~100 MB RAM).

    Returns:
        The sample object with foil atoms added.
    """
    if not config.enabled:
        return sample
    if config.material == "none":
        return sample

    # Get material properties
    Z, molar_mass, default_density = MATERIAL_PROPERTIES[config.material]
    density = config.density if config.density is not None else default_density

    # Sample geometry
    box = sample.containing_box
    box_x = box[1][0] - box[0][0]
    box_y = box[1][1] - box[0][1]
    box_z = box[1][2] - box[0][2]
    centre_x = (box[0][0] + box[1][0]) / 2.0
    centre_y = (box[0][1] + box[1][1]) / 2.0

    hole_radius = config.hole_diameter / 2.0
    foil_thickness = config.thickness

    # Foil z-range: top of sample
    z_max = box[1][2]
    z_min = z_max - foil_thickness

    # Number density (atoms per A^3)
    # density in g/cm^3 -> g/A^3: multiply by 1e-24
    number_density = (density * 1e-24 * scipy.constants.Avogadro) / molar_mass

    # Annular volume: total box area minus hole area, times foil thickness
    box_area = box_x * box_y
    hole_area = np.pi * hole_radius**2
    foil_area = max(box_area - hole_area, 0)
    foil_volume = foil_area * foil_thickness
    n_atoms_physical = int(number_density * foil_volume)

    if n_atoms_physical <= 0:
        logger.warning("Support film: no atoms to place (foil area is zero)")
        return sample

    # Coarse-grain: cap atom count, boost occupancy to conserve scattering
    if n_atoms_physical > max_atoms:
        occupancy_scale = n_atoms_physical / max_atoms
        n_atoms = max_atoms
        logger.info(
            f"Support film: coarse-graining {n_atoms_physical:,} -> {n_atoms:,} atoms "
            f"(occupancy x{occupancy_scale:.1f} to conserve projected potential)"
        )
    else:
        occupancy_scale = 1.0
        n_atoms = n_atoms_physical

    logger.info(
        f"Support film: placing {n_atoms:,} {config.material} atoms "
        f"(Z={Z}, density={density} g/cm^3, thickness={foil_thickness} A, "
        f"hole_diameter={config.hole_diameter} A, occupancy={occupancy_scale:.1f})"
    )

    # Generate atoms in annular region via rejection sampling
    # Acceptance ratio = foil_area / box_area
    acceptance_ratio = foil_area / box_area if box_area > 0 else 0
    batch_size = min(int(n_atoms / max(acceptance_ratio, 0.01) * 1.2), 50_000_000)

    hole_radius_sq = hole_radius**2
    collected = 0
    data_buffers = []

    while collected < n_atoms:
        # Generate candidate positions
        remaining = n_atoms - collected
        gen_size = min(
            int(remaining / max(acceptance_ratio, 0.01) * 1.5) + 1000, batch_size
        )

        x = np.random.uniform(box[0][0], box[1][0], size=gen_size).astype("float32")
        y = np.random.uniform(box[0][1], box[1][1], size=gen_size).astype("float32")
        z = np.random.uniform(z_min, z_max, size=gen_size).astype("float32")

        # Reject points inside hole
        r_sq = (x - centre_x) ** 2 + (y - centre_y) ** 2
        outside_hole = r_sq > hole_radius_sq

        x = x[outside_hole]
        y = y[outside_hole]
        z = z[outside_hole]

        # Take only what we need
        take = min(len(x), remaining)
        x = x[:take]
        y = y[:take]
        z = z[:take]

        if len(x) > 0:
            data_buffers.append(
                pandas.DataFrame(
                    {
                        "atomic_number": np.full(len(x), Z, dtype="uint8"),
                        "x": x,
                        "y": y,
                        "z": z,
                        "sigma": np.full(len(x), 0.085, dtype="float32"),
                        "occupancy": np.full(len(x), occupancy_scale, dtype="float32"),
                        "charge": np.zeros(len(x), dtype="int8"),
                        "group": np.full(len(x), -1, dtype="int32"),
                    }
                )
            )
            collected += len(x)

    # Combine and add to sample
    all_data = pandas.concat(data_buffers, ignore_index=True)
    sample.add_atoms(AtomData(data=all_data))

    logger.info(f"Support film: placed {collected:,} atoms")
    return sample


@singledispatch
def support_film(config_file, sample_file: str) -> Sample:
    """
    Add support film to the sample.

    Args:
        config_file: The input config filename.
        sample_file: The sample filename.
    """
    config = parakeet.config.load(config_file)
    parakeet.config.show(config)
    logger.info(f"Writing sample to {sample_file}")
    return _support_film_Config(config, sample_file)


@support_film.register(parakeet.config.Config)
def _support_film_Config(config: parakeet.config.Config, sample_file: str) -> Sample:
    """
    Add support film from a full Config object.
    """
    sample = Sample(sample_file, mode="r+")
    if config.sample.support_film:
        sample = _support_film_SupportFilm(config.sample.support_film, sample)
    else:
        logger.info("No support film parameters found in config")
    return sample


@support_film.register(parakeet.config.SupportFilm)
def _support_film_SupportFilm(
    config: parakeet.config.SupportFilm, sample: Sample
) -> Sample:
    """
    Add support film atoms to the sample.
    """
    return _generate_support_film_atoms(sample, config)
