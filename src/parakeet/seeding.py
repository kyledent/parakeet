#
# parakeet.seeding.py
#
# Copyright (C) 2026 Diamond Light Source and Rosalind Franklin Institute
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
"""
Seeds for the random parts of a simulation, derived from one number.

`simulation.seed` is a base seed. Each random stream is derived from it and
from where it is used, so that a value depends only on the configuration and
not on the order things ran in, or on which process ran them:

    grf_seed(base, image_number)                         the GRF ice of one image
    detector_rng(base, image_number, fraction_number)    the shot noise of one frame

The ice is keyed on the image, not the frame, so every fraction of a movie sees
the same ice. Each image of a tilt series gets its own: the ice model draws a
2D field per slice, which cannot be consistent across tilts whatever the seed,
so independent ice per tilt is the honest choice for this model, not a feature
of the physics.

A seed of None means "random", but the randomness is drawn here by
`resolve()` and recorded, so any run can be reproduced from its metadata.

"""
import numpy as np

# Stream identifiers: one per random process, so that no two share a sequence
GRF = 1
DETECTOR = 2

# The ice generator is a 31-bit linear congruential engine. It takes a seed
# modulo 2^31 - 1 and maps 0 to 1, so seeds are kept to 1 .. 2^31 - 2, where
# that folding never applies and two derived seeds cannot collide through it.
GRF_SEED_MIN = 1
GRF_SEED_MAX = 2**31 - 2

# A drawn base seed fits a signed 64-bit header field
BASE_SEED_MAX = 2**63 - 1


def draw_base_seed():
    """
    Draw a fresh base seed from the operating system's entropy

    Returns:
        An int in 0 .. 2^63 - 1

    """
    state = np.random.SeedSequence().generate_state(1, np.uint64)[0]
    return int(state >> np.uint64(1))


def resolve(seed):
    """
    The base seed to use: the given one, or a freshly drawn one

    Args:
        seed: The configured base seed, or None

    Returns:
        A non-negative int

    """
    if seed is None:
        return draw_base_seed()
    seed = int(seed)
    if seed < 0:
        raise ValueError("A seed must be non-negative, not %d" % seed)
    return seed


def grf_seed(base, image_number):
    """
    The seed of the GRF ice for one image

    Args:
        base: The base seed
        image_number: The image the ice belongs to (all its fractions share it)

    Returns:
        An int in GRF_SEED_MIN .. GRF_SEED_MAX

    """
    state = np.random.SeedSequence([int(base), GRF, int(image_number)]).generate_state(
        1, np.uint32
    )[0]
    return int(state) % (GRF_SEED_MAX - GRF_SEED_MIN + 1) + GRF_SEED_MIN


def detector_rng(base, image_number, fraction_number):
    """
    The random generator for the shot noise of one frame

    Args:
        base: The base seed
        image_number: The image
        fraction_number: The fraction of the image (the movie frame)

    Returns:
        A numpy Generator

    """
    return np.random.default_rng(
        [int(base), DETECTOR, int(image_number), int(fraction_number)]
    )
