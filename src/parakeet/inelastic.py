#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import numpy as np
import parakeet.landau
import scipy.signal
from math import sqrt, pi, cos, exp, log, ceil, floor, factorial, erf


# Inelastic mean free path for amorphous ice at 300 keV. Note this does not
# vary with beam energy, so results away from 300 keV carry that error.
MEAN_FREE_PATH = 3150  # A

# Low-loss plasmon of amorphous ice. Plural scattering is Poisson distributed
# over multiples of this (Egerton, EELS in the Electron Microscope). The peak is
# broad -- around 10 eV half width -- which matters because a slit comparable to
# that width does not cleanly include or exclude it.
ICE_PLASMON_ENERGY = 20.0  # eV
ICE_PLASMON_FWHM = 20.0  # eV

# Default zero-loss peak width if no source spread is given. The ZLP is not a
# delta function: it has the width of the source, so a slit narrower than a few
# eV starts cutting into it.
DEFAULT_ZLP_FWHM = 0.7  # eV, a Schottky X-FEG

# Used when a config selects zero-loss filtering without naming a slit width.
# The model means the filter is inserted; retracted is inelastic_model:
# unfiltered, not a missing width.
DEFAULT_SLIT_WIDTH = 20.0  # eV


def effective_thickness(shape, angle):
    """
    Compute the effective thickness

    """
    TINY = 1e-5
    if shape["type"] == "cube":
        D0 = shape["cube"]["length"]
        cos_angle = cos(pi * angle / 180.0)
        if abs(cos_angle) < TINY:
            cos_angle = TINY
        thickness = D0 / cos_angle
    elif shape["type"] == "cuboid":
        D0 = shape["cuboid"]["length_z"]
        cos_angle = cos(pi * angle / 180.0)
        if abs(cos_angle) < TINY:
            cos_angle = TINY
        thickness = D0 / cos_angle
    elif shape["type"] == "cylinder":
        thickness = shape["cylinder"]["radius"] * 2
    if isinstance(thickness, list):
        thickness = np.mean(thickness)
    return abs(thickness)


def zero_loss_fraction(shape, angle):
    """
    Compute the fraction of electrons in the zero loss peak.

    This is exp(-t/lambda), the n = 0 term of the Poisson plural scattering
    series. It is a property of the specimen, not of the filter: see
    filter_transmission for how much a given slit actually passes.

    """
    thickness = effective_thickness(shape, angle)
    return exp(-thickness / MEAN_FREE_PATH)


def _gaussian_fraction_in_window(centre, fwhm, low, high):
    """Fraction of a Gaussian peak lying inside [low, high]."""
    sigma = fwhm / (2 * sqrt(2 * log(2)))
    if sigma <= 0:
        return 1.0 if low <= centre <= high else 0.0
    z_lo = (low - centre) / (sigma * sqrt(2))
    z_hi = (high - centre) / (sigma * sqrt(2))
    return 0.5 * (erf(z_hi) - erf(z_lo))


def filter_transmission(
    shape,
    angle,
    slit_width=None,
    slit_centre=0.0,
    plasmon_energy=None,
    plasmon_fwhm=None,
    zlp_fwhm=None,
    n_max=10,
):
    """
    Compute the fraction of electrons passed by an energy filter slit.

    Note this is the fraction TRANSMITTED, which is not the same as the
    zero-loss fraction: a retracted filter transmits everything while the
    zero-loss fraction is still exp(-t/lambda). See zero_loss_fraction for
    that quantity.

    The low-loss spectrum is modelled as a zero-loss peak plus plasmon
    replicas. Plural scattering is Poisson distributed, so the weight of the
    n-th replica is

        P(n) = (t/lambda)^n exp(-t/lambda) / n!

    at an energy loss of n * plasmon_energy. Each peak has a finite width: the
    zero-loss peak has the width of the source, and the plasmon is broad, about
    20 eV FWHM in vitreous ice, broadening as sqrt(n) for multiple losses. The
    transmitted fraction is the sum of each peak's weight times the part of it
    lying inside the slit.

    Giving finite widths matters at both ends of the range. A slit of a few eV
    or less cuts into the zero-loss peak itself, which a delta-function model
    would miss. And because the plasmon is 20 eV wide, a slit of comparable
    width neither cleanly excludes nor cleanly includes it: a 20 eV slit on the
    zero-loss peak still clips the plasmon's leading edge.

    Offsetting `slit_centre` to the plasmon energy gives plasmon imaging: the
    zero-loss peak falls outside the slit and the image is formed from
    inelastically scattered electrons.

    Args:
        shape: The sample shape
        angle: The tilt angle (degrees)
        slit_width: The slit width (eV). None means no slit -- the filter is
            retracted and everything is transmitted, returning 1.0. For a
            filter passing only the zero-loss peak, give a width small compared
            with the plasmon energy, which is what any real slit is.
        slit_centre: The centre of the slit in energy loss (eV). 0 is a
            zero-loss filter; plasmon_energy gives plasmon imaging
        plasmon_energy: The plasmon energy (eV), defaulting to amorphous ice
        plasmon_fwhm: The plasmon width (eV), defaulting to amorphous ice
        zlp_fwhm: The zero-loss peak FWHM (eV), i.e. the source energy spread
        n_max: The highest plasmon order to include

    Returns:
        float: The fraction of electrons passed by the filter

    """
    thickness = effective_thickness(shape, angle)
    return filter_transmission_tau(
        thickness / MEAN_FREE_PATH,
        slit_width=slit_width,
        slit_centre=slit_centre,
        plasmon_energy=plasmon_energy,
        plasmon_fwhm=plasmon_fwhm,
        zlp_fwhm=zlp_fwhm,
        n_max=n_max,
    )


def filter_transmission_tau(
    t_over_lambda,
    slit_width=None,
    slit_centre=0.0,
    plasmon_energy=None,
    plasmon_fwhm=None,
    zlp_fwhm=None,
    n_max=10,
):
    """
    filter_transmission for a given optical depth t/lambda: a number, or an
    array of them (one per pixel), with the same model and arguments

    """
    # No slit means the filter is retracted: nothing is removed.
    if slit_width is None or slit_width <= 0:
        return (
            np.ones_like(t_over_lambda, dtype="float64")
            if np.ndim(t_over_lambda)
            else 1.0
        )
    t_over_lambda = np.asarray(t_over_lambda, dtype="float64")

    if plasmon_energy is None:
        plasmon_energy = ICE_PLASMON_ENERGY
    if plasmon_fwhm is None:
        plasmon_fwhm = ICE_PLASMON_FWHM
    if zlp_fwhm is None:
        zlp_fwhm = DEFAULT_ZLP_FWHM

    low = slit_centre - slit_width / 2.0
    high = slit_centre + slit_width / 2.0

    total = np.zeros_like(t_over_lambda)
    for n in range(n_max + 1):
        weight = t_over_lambda**n * np.exp(-t_over_lambda) / factorial(n)
        if n == 0:
            width = zlp_fwhm
        else:
            # n independent losses convolve, so the width grows as sqrt(n)
            width = plasmon_fwhm * sqrt(n)
        total += weight * _gaussian_fraction_in_window(
            n * plasmon_energy, width, low, high
        )
    return float(total) if total.ndim == 0 else total


def path_length_maps(
    regions,
    shape,
    centre,
    orientation,
    position,
    origin,
    nx,
    ny,
    margin,
    pixel_size,
    stride=8,
    step=2.0,
):
    """
    The path length (A) along the beam through each material, for every pixel
    of an image of a sample with regions of material

    The rays run along the beam (lab z) through the pixel positions the exit
    wave uses (pixel k of the grid, margin included, at origin + (k - margin)
    pixel_size), into the sample through the frame's rotation about the sample
    centre and its position, as the atoms are placed. Each ray is sampled every
    `step` A and each sample takes the material of the region that owns it
    (parakeet.sample._regions.owner), or the medium inside the sample shape;
    so a path is exact to about one step per boundary it crosses. The maps are
    computed on every `stride`-th pixel and interpolated (linearly) to the
    rest, except where interpolation would lose detail, whose every pixel is
    traced: the cells where the rays at the four corners disagree (a boundary
    crosses them), and the pixels under any region narrower than two cells in
    projection (which could fall between corner rays unseen). Edges and small
    features keep the full resolution.

    Args:
        regions: The (medium, regions) from parakeet.sample._regions.parse
        shape, centre: The sample shape and centre
        orientation, position: The frame's rotation vector and position
        origin: The detector origin (x, y), A
        nx, ny, margin: The image size and margin (pixels)
        pixel_size: A
        stride: Pixels between computed rays
        step: Sample spacing along a ray (A)

    Returns:
        dict: material name -> (ny + 2 margin, nx + 2 margin) array of A
        (vacuum is left out)

    """
    from scipy.ndimage import map_coordinates
    from scipy.spatial.transform import Rotation

    import parakeet.sample
    import parakeet.sample._regions as R

    medium, regs = regions
    gx, gy = nx + 2 * margin, ny + 2 * margin
    kx = np.unique(np.arange(0, gx + stride - 1, stride).clip(max=gx - 1))
    ky = np.unique(np.arange(0, gy + stride - 1, stride).clip(max=gy - 1))
    rot = Rotation.from_rotvec(orientation)
    centre = np.asarray(centre, dtype="float64")
    position = np.asarray(position, dtype="float64")

    # The z range: everything that can hold material, placed as the atoms are
    lo, hi = parakeet.sample.shape_bounding_box(centre, shape)
    boxes = [(np.asarray(lo, float), np.asarray(hi, float))] + [
        R.bounds(r) for r in regs
    ]
    corners = np.array(
        [
            [b[i][0], b[j][1], b[k][2]]
            for b in boxes
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ]
    )
    lab = rot.apply(corners - centre) + centre - position
    z = np.arange(lab[:, 2].min(), lab[:, 2].max() + step, step)

    materials = sorted(({r.material for r in regs} | {medium}) - {VACUUM_NAME})

    def trace(px_x, px_y):
        """Path lengths (len(px), materials) for rays through pixels (k_x, k_y)"""
        out = np.zeros((len(px_x), len(materials)))
        for start in range(0, len(px_x), 4096):
            sx = origin[0] + (px_x[start : start + 4096] - margin) * pixel_size
            sy = origin[1] + (px_y[start : start + 4096] - margin) * pixel_size
            pts = np.stack(
                np.broadcast_arrays(sx[:, None], sy[:, None], z[None, :]), axis=-1
            )
            pts = pts.reshape(-1, 3)
            sample_pts = rot.inv().apply(pts + position - centre) + centre
            owner = R.owner(regs, sample_pts) if regs else np.full(len(sample_pts), -1)
            index = np.full(len(sample_pts), -1)
            for i, r in enumerate(regs):
                if r.material != VACUUM_NAME:
                    index[owner == i] = materials.index(r.material)
            if medium != VACUUM_NAME:
                inside = parakeet.sample.shape_contains(sample_pts, centre, shape)
                index[(owner < 0) & inside] = materials.index(medium)
            index = index.reshape(-1, len(z))
            for m in range(len(materials)):
                out[start : start + len(sx), m] = (index == m).sum(axis=1) * step
        return out

    # The coarse rays, interpolated onto every pixel
    cy, cx = np.meshgrid(ky, kx, indexing="ij")
    coarse = trace(cx.ravel(), cy.ravel()).reshape(len(ky), len(kx), len(materials))
    fy = np.interp(np.arange(gy), ky, np.arange(len(ky)))
    fx = np.interp(np.arange(gx), kx, np.arange(len(kx)))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    maps = np.stack(
        [
            map_coordinates(coarse[..., m], [yy, xx], order=1, mode="nearest")
            for m in range(len(materials))
        ],
        axis=-1,
    )

    # Where the four rays at a cell's corners disagree, a boundary crosses the
    # cell and interpolation would smear it: trace every pixel of such a cell
    if len(ky) > 1 and len(kx) > 1 and materials:
        corners = np.stack(
            [coarse[:-1, :-1], coarse[:-1, 1:], coarse[1:, :-1], coarse[1:, 1:]]
        )
        spread = (corners.max(axis=0) - corners.min(axis=0)).max(axis=-1)
        refine = np.zeros((gy, gx), bool)
        for j, i in zip(*np.nonzero(spread > step / 2)):
            refine[ky[j] : ky[j + 1] + 1, kx[i] : kx[i + 1] + 1] = True
        # A region narrower than two cells can fall between all four corner
        # rays, which then agree: trace every pixel its projection touches
        cell = stride * pixel_size
        for r in regs:
            b = R.bounds(r)
            box = np.array(
                [
                    [b[i][0], b[j][1], b[k][2]]
                    for i in (0, 1)
                    for j in (0, 1)
                    for k in (0, 1)
                ]
            )
            pl = rot.apply(box - centre) + centre - position
            x0, y0 = pl[:, :2].min(axis=0)
            x1, y1 = pl[:, :2].max(axis=0)
            if min(x1 - x0, y1 - y0) < 2 * cell:
                i0 = int(np.floor((x0 - origin[0]) / pixel_size)) + margin - 1
                i1 = int(np.ceil((x1 - origin[0]) / pixel_size)) + margin + 1
                j0 = int(np.floor((y0 - origin[1]) / pixel_size)) + margin - 1
                j1 = int(np.ceil((y1 - origin[1]) / pixel_size)) + margin + 1
                refine[
                    max(j0, 0) : max(min(j1, gy), 0), max(i0, 0) : max(min(i1, gx), 0)
                ] = True
        if refine.any():
            ry, rx = np.nonzero(refine)
            maps[ry, rx] = trace(rx, ry)
    return {m: maps[..., k] for k, m in enumerate(materials)}


def optical_depth_map(regions, materials_mfp, *args, **kwargs):
    """
    The inelastic optical depth t/lambda of every pixel of an image of a sample
    with regions of material: the sum over materials of the path length
    through each (`path_length_maps`, which takes the remaining arguments)
    divided by its inelastic mean free path

    Args:
        regions: The (medium, regions) from parakeet.sample._regions.parse
        materials_mfp: Material name -> inelastic mean free path (A)

    Returns:
        A (ny + 2 margin, nx + 2 margin) array

    """
    paths = path_length_maps(regions, *args, **kwargs)
    tau = None
    for material, length in paths.items():
        t = length / materials_mfp[material]
        tau = t if tau is None else tau + t
    if tau is None:  # nothing but vacuum
        nx, ny, margin = args[5], args[6], args[7]
        tau = np.zeros((ny + 2 * margin, nx + 2 * margin))
    return tau


VACUUM_NAME = "vacuum"


def mp_loss_fraction(shape, angle):
    """
    Compute the inelastic fraction

    """
    thickness = effective_thickness(shape, angle)
    electron_fraction = exp(-thickness / MEAN_FREE_PATH)
    return 1.0 - electron_fraction


def fraction_of_electrons(shape, angle, model=None):
    """
    Compute the fraction of electrons

    """
    if model is None:
        fraction = 1.0
    elif model == "zero_loss":
        fraction = zero_loss_fraction(shape, angle)
    elif model == "mp_loss":
        fraction = mp_loss_fraction(shape, angle)
    elif model == "unfiltered":
        fraction = 1.0
    elif model == "cc_corrected":
        fraction = 1.0
    return fraction


def most_probable_loss(energy, shape, angle):
    """
    Compute the MPL peak and sigma

    Params:
        energy (float): Beam energy in keV

    Returns:
        tuple: (peak, sigma) of the energy loss distribution (eV)

    """
    thickness = effective_thickness(shape, angle)
    thickness = min(thickness, 100000)  # Maximum 10 um - to avoid issues at high tilt
    peak, fwhm = parakeet.landau.mpl_and_fwhm(energy, thickness)
    return peak, fwhm / (2 * sqrt(2 * log(2)))


class EnergyFilterOptimizer(object):
    """
    A simple class to find the optimal energy filter placement when trying to
    take into account the inelastic scattering component

    """

    def __init__(self, energy_spread=0.8, dE_min=-10, dE_max=200, dE_step=0.01):
        """
        Initialise the class

        Params:
            energy_spread (float): The energy spread (eV)
            dE_min (float): The minimum energy loss (eV)
            dE_max (float): The maximum energy loss (eV)
            dE_step (float): The energy loss step (eV)

        """

        # Save the energy spread
        self.energy_spread = energy_spread  # eV

        # Save an instance of the landau distribution class
        self.landau = parakeet.landau.Landau()

        # Set the energy losses to consider
        self.dE_min = dE_min
        self.dE_max = dE_max
        self.dE_step = dE_step

    def __call__(self, energy, thickness, filter_width=None):
        """
        Compute the optimum position for the energy filter

        Params:
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)
            filter_width (float): The energy filter width (eV)

        Returns:
            float: The optimal position to maximum number of electrons (eV)

        """

        # Check the input
        assert energy > 0
        assert thickness > 0

        # The energy loss distribution
        dE, distribution = self.energy_loss_distribution(energy, thickness)

        # Compute optimum given the position and filter width
        if filter_width is None:
            position = np.sum(dE * distribution) / np.sum(dE)
        else:
            size = len(distribution)
            kernel_size = filter_width / self.dE_step
            ks = kernel_size / 2
            kx = np.arange(-size // 2, size // 2 + 1)
            kernel = np.exp(-0.5 * (kx / ks) ** 80)
            kernel = kernel / np.sum(kernel)
            num = scipy.signal.fftconvolve(distribution, kernel, mode="same")
            position = dE[np.argmax(num)]

        # Return the filter position
        return position

    def elastic_fraction(self, energy, thickness):
        """
        Compute the elastic electron fraction

        Params:
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)

        Returns:
            float: The elastic electron fraction

        """
        # Compute the fractions for the zero loss and energy losses
        return exp(-thickness / MEAN_FREE_PATH)

    def energy_loss_distribution(self, energy, thickness):
        """
        Compute the energy loss distribution

        Params:
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)

        Returns:
            (array, array): dE (ev) and the energy loss distribution

        """

        # The energy losses to consider
        dE = np.arange(self.dE_min, self.dE_max, self.dE_step, dtype="float64")

        # The energy loss distribution
        energy_loss_distribution = self.landau(dE, energy, thickness)
        sum_ELD = np.sum(energy_loss_distribution)
        if sum_ELD > 0:
            energy_loss_distribution /= self.dE_step * sum_ELD

        # The zero loss distribution
        zero_loss_distribution = (1.0 / sqrt(pi * self.energy_spread**2)) * np.exp(
            -(dE**2) / self.energy_spread**2
        )

        # Compute the fractions for the zero loss and energy losses
        elastic_fraction = self.elastic_fraction(energy, thickness)

        # Return the distribution
        distribution = (
            elastic_fraction * zero_loss_distribution
            + (1 - elastic_fraction) * energy_loss_distribution
        )
        return dE, distribution

    def compute_elastic_component(self, energy, thickness, position, filter_width):
        """
        Compute the elastic fraction and energy spread

        Params:
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)
            position (float): The filter position (eV)
            filter_width (float): The energy filter width (eV)

        Returns:
            float, float: The electron fraction and energy spread (eV)

        """

        # The energy losses to consider
        dE = np.arange(self.dE_min, self.dE_max, self.dE_step)

        # The zero loss distribution
        P = (1.0 / sqrt(pi * self.energy_spread**2)) * np.exp(
            -(dE**2) / self.energy_spread**2
        )
        C = self.dE_step * np.cumsum(P)

        # Compute the fractions for the zero loss and energy losses
        fraction = self.elastic_fraction(energy, thickness)

        # Compute the number of electrons and the sigma
        if filter_width is not None:
            dE0 = position - filter_width / 2.0
            dE1 = position + filter_width / 2.0
            x0 = int(floor((dE0 - self.dE_min) / self.dE_step))
            x1 = int(ceil((dE1 - self.dE_min) / self.dE_step))
            x0 = max(x0, 0)
            x1 = min(x1, len(P) - 1)
            assert x1 > x0
            fraction *= C[x1] - C[x0]
            P = P[x0:x1]
            dE = dE[x0:x1]

        # Compute the spread
        if len(P) > 0 and np.sum(P) > 0:
            dE_mean = np.sum(P * dE) / np.sum(P)
            spread = sqrt(np.sum(P * (dE - dE_mean) ** 2) / np.sum(P)) * sqrt(2)
        else:
            spread = 0

        # Return the fraction and spread
        return fraction, spread

    def compute_inelastic_component(self, energy, thickness, position, filter_width):
        """
        Compute the inelastic fraction and energy spread

        Params:
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)
            position (float): The filter position (eV)
            filter_width (float): The energy filter width (eV)

        Returns:
            float, float: The electron fraction and energy spread (eV)

        """

        # The energy losses to consider
        dE = np.arange(self.dE_min, self.dE_max, self.dE_step)

        # The energy loss distribution
        P = self.landau(dE, energy, thickness)
        sum_P = np.sum(P)
        if sum_P > 0:
            P /= self.dE_step * sum_P
        C = np.cumsum(P) * self.dE_step

        # Compute the fractions for the zero loss and energy losses
        fraction = 1 - self.elastic_fraction(energy, thickness)

        # Compute the number of electrons and the sigma
        if filter_width is not None:
            dE0 = position - filter_width / 2.0
            dE1 = position + filter_width / 2.0
            x0 = int(floor((dE0 - self.dE_min) / self.dE_step))
            x1 = int(ceil((dE1 - self.dE_min) / self.dE_step))
            x0 = max(x0, 0)
            x1 = min(x1, len(P) - 1)
            assert x1 > x0
            fraction *= C[x1] - C[x0]
            P = P[x0:x1]
            dE = dE[x0:x1]

        # Compute the spread
        if len(P) > 0 and np.sum(P) > 0:
            peak, fwhm = parakeet.landau.mpl_and_fwhm(energy / 1000, thickness)
            sigma = fwhm / (2 * sqrt(2 * log(2)))

            # This is a hack to compute the variance because the heavy tail of
            # the landau distribution makes the variance undefined and this
            # causes problems for large filter sizes. There is probably a
            # better way to do this so maybe should look at fixing.
            P *= np.exp(-0.5 * (dE - peak) ** 2 / (2 * sigma) ** 2)

            P /= np.sum(P)
            dE_mean = np.sum(P * dE) / np.sum(P)
            spread = sqrt(np.sum(P * (dE - dE_mean) ** 2) / np.sum(P)) * sqrt(2)
        else:
            spread = 0

        # Return the fraction and spread
        return fraction, spread


def get_energy_bins(
    energy,
    thickness,
    energy_spread=0.8,
    filter_energy=None,
    filter_width=None,
    dE_max=200,
    dE_step=5,
):
    """
    Get some energy bins with weights
    """

    # Ensure min and max such that when we split the mean dE will be at
    # sensible locations (i.e. we will have one at zero)
    dE_step_sub = 0.01
    assert dE_step < 10
    dE_min = -dE_step * (0.5 + int(10 / dE_step))
    dE_max = dE_step * (0.5 + int(dE_max / dE_step))
    assert dE_min < 0
    assert dE_max > dE_min

    # Check the filter width and step size
    if filter_energy is not None and filter_width is not None:
        # Adjust the step size
        num_step = int(filter_width / dE_step)
        dE_step = filter_width / num_step

        # The min and max energies
        filter_min = filter_energy - filter_width / 2.0
        filter_max = filter_energy + filter_width / 2.0

        # Set the bins
        bins = []
        for i in range(num_step):
            E1 = filter_min + i * dE_step
            E2 = E1 + dE_step
            bins.append((E1, E2))

    else:
        # Number of steps
        num_step = int((dE_max - dE_min) / dE_step)

        bins = []
        for i in range(num_step):
            E1 = dE_min + i * dE_step
            E2 = E1 + dE_step
            bins.append((E1, E2))

    # Get the distribution of energy losses
    optimizer = EnergyFilterOptimizer(
        energy_spread,
        dE_min=dE_min + dE_step_sub / 2.0,
        dE_max=dE_max + dE_step_sub / 2.0,
        dE_step=dE_step_sub,
    )
    dE, distribution = optimizer.energy_loss_distribution(energy, thickness)
    distribution /= np.sum(distribution)

    # The maximum spread
    dE_spread_max = sqrt(dE_step**2 / 12) * sqrt(2)

    # Loop over the subdivisions and compute mean energy, spread and total
    # weight. For each bin we take the distribution and compute the weighted
    # mean energy loss and the weighted variance as the energy spread. The mean
    # will always be within the energy bin and the variance will always be > 0
    # and < the variance of the uniform distribution within the bin.
    TINY = 1e-5
    nbins = len(bins)
    bin_energy = np.zeros(nbins)
    bin_spread = np.zeros(nbins)
    bin_weight = np.zeros(nbins)
    for i in range(nbins):
        E1, E2 = bins[i]
        select = (dE >= E1) & (dE < E2)
        dE_sub = dE[select]
        P_sub = distribution[select]
        dE_mean = np.mean(dE_sub)
        P_tot = np.sum(P_sub)
        if P_tot > 1e-7:
            P_sub = P_sub / P_tot
            dE_mean = np.sum(P_sub * dE_sub)
            dE_spread = np.sum(P_sub * ((dE_sub - dE_mean) ** 2))
            dE_spread = sqrt(dE_spread) * sqrt(2)
        else:
            dE_spread = dE_spread_max
        fudge = np.sqrt((dE_step_sub**2 / 12) * len(P_sub))
        assert (E2 - E1) <= (dE_step + TINY)
        assert dE_mean >= E1
        assert dE_mean <= E2
        assert dE_spread >= 0
        # assert dE_spread <= (dE_spread_max + fudge)
        bin_energy[i] = energy - dE_mean
        bin_weight[i] = P_tot
        bin_spread[i] = dE_spread

    # Return the bins and weights
    return bin_energy, bin_spread, bin_weight
