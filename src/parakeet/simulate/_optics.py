#
# parakeet.simulate.optics.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#

import copy
import logging
import numpy as np
import parakeet.config
import parakeet.dqe
import parakeet.freeze
import parakeet.futures
import parakeet.inelastic
import parakeet.io
import parakeet.sample
from parakeet.config import Device
from parakeet.microscope import Microscope
from parakeet.scan import Scan
from functools import singledispatch
from parakeet.simulate.simulation import Simulation
import multem
from parakeet.simulate.engine import SimulationEngine, defocus_spread
from parakeet.simulate._fft_backend import fft2, ifft2, from_numpy, to_numpy
from parakeet.simulate._fast_ctf import compute_ctf as fast_ctf
from parakeet.microscope import Microscope
from parakeet.scan import Scan


__all__ = ["optics"]


# Get the logger
logger = logging.getLogger(__name__)


class OpticsImageSimulator(object):
    """
    A class to do the actual simulation

    The simulation is structured this way because the input data to the
    simulation is large enough that it makes an overhead to creating the
    individual processes.

    """

    def __init__(
        self,
        microscope=None,
        exit_wave=None,
        scan=None,
        simulation=None,
        sample=None,
        device="gpu",
        gpu_id=None,
    ):
        self.microscope = microscope
        self.exit_wave = exit_wave
        self.scan = scan
        self.simulation = simulation
        self.sample = sample
        self.device = device
        self.gpu_id = gpu_id

    def optical_depth(self, index, margin):
        """
        The per-pixel inelastic optical depth of frame index, for a sample with
        regions (parakeet.inelastic.optical_depth_map)

        """
        import parakeet.config
        import parakeet.sample._new
        import parakeet.sample._regions

        document = parakeet.sample._new.regions_document(
            parakeet.config.SampleRegions(**self.sample["regions"])
        )
        regions = parakeet.sample._regions.parse(document)
        default = parakeet.inelastic.MEAN_FREE_PATH
        ice = self.simulation.get("ice_parameters") or {}
        mfp = {"water": ice.get("inelastic_mfp_A") or default}
        for name, m in (self.simulation.get("materials") or {}).items():
            mfp[name] = m.get("inelastic_mfp_A") or default
        detector = self.microscope.detector
        return parakeet.inelastic.optical_depth_map(
            regions,
            mfp,
            self.sample["shape"],
            self.sample["centre"],
            self.scan.orientation[index],
            self.scan.position[index],
            detector.origin,
            detector.nx,
            detector.ny,
            margin,
            detector.pixel_size,
            stride=self.simulation.get("inelastic_map_stride", 8),
        )

    def __call__(self, index):
        """
        Simulate a single frame

        Args:
            simulation (object): The simulation object
            index (int): The frame number

        Returns:
            tuple: (angle, image)

        """

        # Get the rotation angle
        angle = self.scan.angles[index]
        defocus_offset = self.scan.defocus_offset[index]

        # Check the angle and position
        assert abs(angle - self.exit_wave.header[index]["tilt_alpha"]) < 1e7

        # The field of view
        nx = self.microscope.detector.nx
        ny = self.microscope.detector.ny
        pixel_size = self.microscope.detector.pixel_size
        x_fov = nx * pixel_size
        y_fov = ny * pixel_size
        margin = self.simulation["margin"]
        offset = margin * pixel_size

        # CTF grid dimensions (detector + margin)
        ctf_ny = ny + 2 * margin
        ctf_nx = nx + 2 * margin
        ctf_shape = (ctf_ny, ctf_nx)

        # Pre-compute frequency grids (shared across all CTF evaluations this frame)
        qy = np.fft.fftfreq(ctf_ny, d=pixel_size)
        qx = np.fft.fftfreq(ctf_nx, d=pixel_size)
        q2_grid = qy[:, None] ** 2 + qx[None, :] ** 2
        q_grid = np.sqrt(q2_grid)

        # Pre-compute coherence parameters (shared across energy bins)
        Cc_A = self.microscope.lens.c_c * 1e7  # mm -> A
        ds = defocus_spread(
            Cc_A,
            self.microscope.beam.energy_spread,
            self.microscope.lens.current_spread,
            self.microscope.beam.acceleration_voltage_spread,
        )
        sigma_df = multem.iehwgd_to_sigma(ds)
        sigma_source = multem.mrad_to_sigma(
            self.microscope.beam.energy,
            self.microscope.beam.illumination_semiangle,
        )

        # Pre-compute aperture mask and phase plate shift (constant across energy bins)
        aperture_mask = None
        if self.microscope.objective_aperture_cutoff_freq is not None:
            aperture_mask = q_grid < self.microscope.objective_aperture_cutoff_freq

        phase_plate_shift = None
        if self.microscope.phase_plate.use:
            phase_plate_shift = parakeet.simulate.phase_plate.compute_phase_shift(
                ctf_shape,
                pixel_size,
                self.microscope.phase_plate.phase_shift,
                self.microscope.phase_plate.radius,
            )

        def _compute_ctf_and_apply(
            psi_fft, energy_kev, c_10, c_30_mm, sigma_df_val, sigma_src_val
        ):
            """Compute CTF via fast pure-Python path and apply to FFT'd exit wave."""
            ctf = fast_ctf(
                ctf_shape,
                pixel_size,
                energy_kev,
                c_10,
                c_30_mm,
                sigma_df_val,
                sigma_src_val,
                q2=q2_grid,
                q=q_grid,
            )
            if aperture_mask is not None:
                ctf = ctf * aperture_mask
            if phase_plate_shift is not None:
                ctf = ctf * phase_plate_shift
            psi_out = to_numpy(ifft2(psi_fft * from_numpy(ctf)))
            return np.abs(psi_out) ** 2

        # Get the specimen atoms
        logger.info(f"Simulating image {index+1}")

        # Set the input wave
        psi = self.exit_wave.data[index]

        microscope = copy.deepcopy(self.microscope)

        # Get the defocus
        defocus = microscope.lens.c_10 + defocus_offset

        # If we do CC correction then set spherical aberration and chromatic
        # aberration to zero
        shape = self.sample["shape"]
        energy_shift = 0

        # Pre-compute forward FFT (shared across all CTF evaluations)
        psi_fft = fft2(psi)

        if self.simulation["inelastic_model"] is None:
            # If no inelastic model just calculate image as normal
            image = _compute_ctf_and_apply(
                psi_fft,
                microscope.beam.energy,
                defocus,
                microscope.lens.c_30,
                sigma_df,
                sigma_source,
            )
            electron_fraction = 1.0

        elif self.simulation["inelastic_model"] == "zero_loss":
            # Compute the image
            image = _compute_ctf_and_apply(
                psi_fft,
                microscope.beam.energy,
                defocus,
                microscope.lens.c_30,
                sigma_df,
                sigma_source,
            )

            # Calculate the fraction of electrons passed by the filter. The
            # slit width is read here as well as in the mp_loss branch, and the
            # zero-loss peak is given the width of the source, so a narrow slit
            # cuts into it. A zero_loss model means the filter is inserted, so
            # a config that names no width gets a typical one rather than the
            # retracted case, which is inelastic_model: unfiltered.
            slit_width = self.simulation["mp_loss_width"]
            if slit_width is None:
                slit_width = parakeet.inelastic.DEFAULT_SLIT_WIDTH
            zlp_fwhm = microscope.beam.energy_spread * microscope.beam.energy * 1000.0
            if (self.sample or {}).get("regions"):
                # Each pixel's path through each material, not one thickness
                electron_fraction = parakeet.inelastic.filter_transmission_tau(
                    self.optical_depth(index, margin),
                    slit_width=slit_width,
                    zlp_fwhm=zlp_fwhm,
                )
            else:
                electron_fraction = parakeet.inelastic.filter_transmission(
                    shape,
                    angle,
                    slit_width=slit_width,
                    zlp_fwhm=zlp_fwhm,
                )

            # Scale the image by the fraction of electrons
            image *= electron_fraction

        else:
            if (self.sample or {}).get("regions"):
                logger.warning(
                    "The %s model uses the sample shape's thickness, not the "
                    "regions' materials (only zero_loss traces them)"
                    % self.simulation["inelastic_model"]
                )
            # Get the effective thickness
            thickness = parakeet.inelastic.effective_thickness(shape, angle)  # A
            if self.simulation["inelastic_model"] == "unfiltered":
                # Get the energy bins
                bin_energy, bin_spread, bin_weight = parakeet.inelastic.get_energy_bins(
                    energy=microscope.beam.energy * 1000,  # eV
                    thickness=thickness,
                    energy_spread=microscope.beam.energy_spread
                    * microscope.beam.energy
                    * 1000,  # dE
                )

            elif self.simulation["inelastic_model"] == "cc_corrected":
                # Get the energy bins
                bin_energy, bin_spread, bin_weight = parakeet.inelastic.get_energy_bins(
                    energy=microscope.beam.energy * 1000,  # eV
                    thickness=thickness,
                    energy_spread=microscope.beam.energy_spread
                    * microscope.beam.energy
                    * 1000,  # dE
                )

                # Set the Cs and CC to zero
                microscope.lens.c_30 = 0
                microscope.lens.c_c = 0

            elif self.simulation["inelastic_model"] == "mp_loss":
                # Set the filter width
                filter_width = self.simulation["mp_loss_width"]  # eV

                # Make optimizer
                optimizer = parakeet.inelastic.EnergyFilterOptimizer(
                    dE_min=-60, dE_max=200
                )
                assert self.simulation["mp_loss_position"] in ["peak", "optimal"]

                # Compute the energy and spread of the plasmon peak
                if self.simulation["mp_loss_position"] != "peak":
                    peak = optimizer(
                        microscope.beam.energy, thickness, filter_width=filter_width
                    )
                else:
                    peak, sigma = parakeet.inelastic.most_probable_loss(
                        microscope.beam.energy, shape, angle
                    )  # eV

                # Set a maximum peak energy loss at 10% of beam energy
                peak = min(peak, microscope.beam.energy * 1000 * 0.1)  # eV

                # Get the energy bins
                bin_energy, bin_spread, bin_weight = parakeet.inelastic.get_energy_bins(
                    energy=microscope.beam.energy * 1000,  # eV
                    thickness=thickness,
                    energy_spread=microscope.beam.energy_spread
                    * microscope.beam.energy
                    * 1000,  # dE
                    filter_energy=peak,
                    filter_width=filter_width,
                )

            else:
                raise RuntimeError("Unknown inelastic model")

            # Get the threshold to exclude bins that don't contribute much
            threshold = min(0.01 / len(bin_energy), max(bin_weight))
            logger.debug("Threshold weight: %f", threshold)

            # Select based on threshold
            selection = bin_weight >= threshold
            bin_energy = bin_energy[selection]
            bin_spread = bin_spread[selection]
            bin_weight = bin_weight[selection]

            # Get the basic energy and defocus
            energy0 = microscope.beam.energy
            defocus0 = microscope.lens.c_10

            # Energy and energy spread
            energy1 = bin_energy / 1000.0  # keV
            energy_spread1 = bin_spread / bin_energy  # dE / E

            # Compute the defocus at this point
            # Energy loss is positive.
            # Energy loss results in over focus which is also positive
            c_c_A = microscope.lens.c_c * 1e7  # A
            dE_E = (energy0 - energy1) / energy0
            defocus1 = defocus0 + c_c_A * dE_E  # A

            # Adjust defocus to mean
            # defocus_mean = np.average(defocus1, weights=bin_weight)
            # defocus1 = defocus1 + (defocus0 - defocus_mean)

            # Use Cs from microscope (may be zeroed for cc_corrected model)
            c_30_for_bins = microscope.lens.c_30

            # Loop through all energies using fast pure-Python CTF
            image = None
            for energy, energy_spread, defocus, weight in zip(
                energy1, energy_spread1, defocus1, bin_weight
            ):
                logger.debug(
                    "Energy: %f eV; Energy spread: %f eV; Weight: %f; Defocus: %f",
                    energy * 1000,
                    energy_spread * energy * 1000,
                    weight,
                    defocus,
                )

                # Compute per-bin defocus spread sigma
                Cc_A_bin = microscope.lens.c_c * 1e7
                ds_bin = defocus_spread(
                    Cc_A_bin,
                    energy_spread,
                    microscope.lens.current_spread,
                    microscope.beam.acceleration_voltage_spread,
                )
                sigma_df_bin = multem.iehwgd_to_sigma(ds_bin)
                sigma_src_bin = multem.mrad_to_sigma(
                    energy, microscope.beam.illumination_semiangle
                )

                image_n = weight * _compute_ctf_and_apply(
                    psi_fft,
                    energy,
                    defocus,
                    c_30_for_bins,
                    sigma_df_bin,
                    sigma_src_bin,
                )

                # Accumulate
                if image is None:
                    image = image_n
                else:
                    image += image_n

            # Compute the electron fraction
            electron_fraction = np.sum(bin_weight)

        logger.debug("Electron fraction = %.2f", float(np.mean(electron_fraction)))

        # Remove margin
        j0 = margin
        i0 = margin
        j1 = image.shape[0] - margin
        i1 = image.shape[1] - margin
        assert margin >= 0
        assert i1 > i0
        assert j1 > j0
        image = image[j0:j1, i0:i1]
        logger.info(
            "Ideal image min/mean/max: %f/%f/%f"
            % (np.min(image), np.mean(image), np.max(image))
        )

        # Set the metadata
        metadata = np.asarray(self.exit_wave.header[index])
        metadata["c_10"] = defocus
        metadata["c_12"] = self.microscope.lens.c_12
        metadata["c_21"] = self.microscope.lens.c_21
        metadata["c_23"] = self.microscope.lens.c_23
        metadata["c_30"] = self.microscope.lens.c_30
        metadata["c_32"] = self.microscope.lens.c_32
        metadata["c_34"] = self.microscope.lens.c_34
        metadata["c_41"] = self.microscope.lens.c_41
        metadata["c_43"] = self.microscope.lens.c_43
        metadata["c_45"] = self.microscope.lens.c_45
        metadata["c_50"] = self.microscope.lens.c_50
        metadata["c_52"] = self.microscope.lens.c_52
        metadata["c_54"] = self.microscope.lens.c_54
        metadata["c_56"] = self.microscope.lens.c_56
        metadata["phi_12"] = self.microscope.lens.phi_12
        metadata["phi_21"] = self.microscope.lens.phi_21
        metadata["phi_23"] = self.microscope.lens.phi_23
        metadata["phi_32"] = self.microscope.lens.phi_32
        metadata["phi_34"] = self.microscope.lens.phi_34
        metadata["phi_41"] = self.microscope.lens.phi_41
        metadata["phi_43"] = self.microscope.lens.phi_43
        metadata["phi_45"] = self.microscope.lens.phi_45
        metadata["phi_52"] = self.microscope.lens.phi_52
        metadata["phi_54"] = self.microscope.lens.phi_54
        metadata["phi_56"] = self.microscope.lens.phi_56
        metadata["c_c"] = self.microscope.lens.c_c
        metadata["current_spread"] = self.microscope.lens.current_spread
        metadata["illumination_semiangle"] = self.microscope.beam.illumination_semiangle
        metadata["acceleration_voltage_spread"] = (
            self.microscope.beam.acceleration_voltage_spread
        )
        metadata["energy_spread"] = self.microscope.beam.energy_spread
        metadata["phase_plate"] = self.microscope.phase_plate
        metadata["inelastic_model"] = self.simulation["inelastic_model"]
        metadata["slit_inserted"] = self.simulation["inelastic_model"] in [
            "zero_loss",
            "mp_loss",
        ]
        metadata["slit_width"] = self.simulation["mp_loss_width"]
        metadata["energy_shift"] = energy_shift

        # Compute the image scaled with Poisson noise
        return (index, image, metadata)


def simulation_factory(
    microscope: Microscope,
    exit_wave: object,
    scan: Scan,
    simulation: dict = None,
    sample: dict = None,
    multiprocessing: dict = None,
) -> Simulation:
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        exit_wave (object): The exit_wave object
        scan (object): The scan object
        simulation (object): The simulation parameters
        multiprocessing (object): The multiprocessing parameters

    Returns:
        object: The simulation object

    """

    # Check multiprocessing settings
    if multiprocessing is None:
        multiprocessing = {"device": "gpu", "nproc": 1, "gpu_id": 0}
    else:
        assert multiprocessing["nproc"] in [None, 1]
        assert len(multiprocessing["gpu_id"]) == 1

    # Create the simulation
    return Simulation(
        image_size=(microscope.detector.nx, microscope.detector.ny),
        pixel_size=microscope.detector.pixel_size,
        scan=scan,
        simulate_image=OpticsImageSimulator(
            microscope=microscope,
            exit_wave=exit_wave,
            scan=scan,
            simulation=simulation,
            sample=sample,
            device=multiprocessing["device"],
            gpu_id=multiprocessing["gpu_id"][0],
        ),
    )


@singledispatch
def optics(
    config_file,
    exit_wave_file: str,
    optics_file: str,
    device: Device = None,
    nproc: int = None,
    gpu_id: list = None,
    dtype: str = None,
):
    """
    Simulate the optics

    Args:
        config_file: The input config filename
        exit_wave_file: The input exit wave filename
        optics_file: The output optics filename
        device: The device to run on (cpu or gpu)
        nproc: The number of processes
        gpu_id: The list of gpu ids
        dtype: Output data type (overrides config). One of float16, float32, int16, uint16.

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Set the command line args in a dict
    if device is not None:
        config.multiprocessing.device = device
    if nproc is not None:
        config.multiprocessing.nproc = nproc
    if gpu_id is not None:
        config.multiprocessing.gpu_id = gpu_id

    # CLI dtype overrides config
    if dtype is not None:
        config.output.dtype = dtype

    # Print some options
    parakeet.config.show(config)

    # Do the work
    _optics_Config(config, exit_wave_file, optics_file)


@optics.register(parakeet.config.Config)
def _optics_Config(
    config: parakeet.config.Config, exit_wave_file: str, optics_file: str
):
    """
    Simulate the optics

    Args:
        config: The input config
        exit_wave_file: The input exit wave filename
        optics_file: The output optics filename

    """

    # Resolve output dtype
    output_dtype = np.dtype(config.output.dtype)

    # Create the microscope
    microscope = parakeet.microscope.new(config.microscope)

    # Create the exit wave data
    logger.info(f"Loading sample from {exit_wave_file}")
    exit_wave = parakeet.io.open(exit_wave_file)

    # Create the scan
    scan = exit_wave.header.scan

    # Override the defocus_offset
    scan_new = parakeet.scan.new(**config.scan.model_dump())
    scan.data["defocus_offset"] = scan_new.defocus_offset

    # Create the simulation
    simulation = simulation_factory(
        microscope,
        exit_wave,
        scan,
        simulation=config.simulation.model_dump(),
        sample=config.sample.model_dump(),
        multiprocessing=config.multiprocessing.model_dump(),
    )

    # Create the writer
    logger.info(f"Opening file: {optics_file} (dtype={output_dtype})")
    writer = parakeet.io.new(
        optics_file,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=output_dtype,
        intensity_range=config.output.intensity_range,
    )

    # Propagate the particle positions
    writer.particle_positions = exit_wave.particle_positions

    # Run the simulation
    simulation.run(writer)
