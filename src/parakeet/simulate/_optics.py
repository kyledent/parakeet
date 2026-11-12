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
import json
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
try:  # optional at import: parakeet.sample.check_regions must import without CUDA
    import multem
except Exception:  # ImportError, or the extension failing to load (no libcuda)
    multem = None
from parakeet.simulate.engine import SimulationEngine, defocus_spread
from parakeet.simulate._fft_backend import fft2, ifft2, from_numpy, to_numpy
from parakeet.simulate._fast_ctf import compute_ctf as fast_ctf
from parakeet.microscope import Microscope
from parakeet.scan import Scan


__all__ = ["optics"]


# Get the logger
logger = logging.getLogger(__name__)


def zero_loss_record(mean_free_paths, path_maps, slit_width, zlp_fwhm, fraction):
    """
    What the zero-loss filter did to a frame of a sample with regions: its
    model and settings, and per material on the frame's paths the inelastic
    mean free path it used and whether the material set it

    Args:
        mean_free_paths: {material: (A, "set" | "default")}
        path_maps: {material: path length map (A)}
        slit_width: The slit (eV)
        zlp_fwhm: The zero-loss peak's width (eV)
        fraction: The electron fraction passed (a map)

    """
    fraction = np.asarray(fraction, dtype="float64")
    return {
        "model": (
            "inelastic.filter_transmission_tau: Poisson multiples of a "
            "%g eV plasmon (FWHM %g eV x sqrt(n)) and the zero-loss peak, each "
            "weighted by its Gaussian share inside the slit"
            % (
                parakeet.inelastic.ICE_PLASMON_ENERGY,
                parakeet.inelastic.ICE_PLASMON_FWHM,
            )
        ),
        "slit_eV": float(slit_width),
        "zlp_fwhm_eV": float(zlp_fwhm),
        "default_mfp_A": float(parakeet.inelastic.MEAN_FREE_PATH),
        "materials": {
            name: {
                "inelastic_mfp_A": mean_free_paths[name][0],
                "source": mean_free_paths[name][1],
                "mean_path_A": float(np.mean(length)),
            }
            for name, length in path_maps.items()
        },
        "electron_fraction": {
            "mean": float(fraction.mean()),
            "min": float(fraction.min()),
            "max": float(fraction.max()),
        },
    }


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

    def path_maps(self, index, margin):
        """
        The path length (A) through each material of every pixel of frame
        index, for a sample with regions (parakeet.inelastic.path_length_maps),
        computed once per frame and shared by the zero-loss filter and the
        elastic-out term

        """
        cache = getattr(self, "_path_maps", None)
        if cache is not None and cache[0] == (index, margin):
            return cache[1]
        import parakeet.config
        import parakeet.sample._new
        import parakeet.sample._regions

        document = parakeet.sample._new.regions_document(
            parakeet.config.SampleRegions(**self.sample["regions"])
        )
        regions = parakeet.sample._regions.parse(document)
        detector = self.microscope.detector
        maps = parakeet.inelastic.path_length_maps(
            regions,
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
            # The elastic-out term needs edges exact: trace every crossing
            # analytically. Without it the paths keep their sampled form
            exact=bool(self.simulation.get("elastic_out")),
            workers=int(self.simulation.get("inelastic_map_workers") or 1),
        )
        self._path_maps = ((index, margin), maps)
        return maps

    def inelastic_mean_free_paths(self):
        """
        Each material's inelastic mean free path (A) and where it came from:
        {name: (value, "set" | "default")}, "set" when the material (or, for
        water, the ice parameters) gives inelastic_mfp_A, "default" when it is
        parakeet.inelastic.MEAN_FREE_PATH (amorphous ice at 300 keV)

        """
        default = parakeet.inelastic.MEAN_FREE_PATH
        ice = self.simulation.get("ice_parameters") or {}
        own = {"water": ice.get("inelastic_mfp_A")}
        for name, m in (self.simulation.get("materials") or {}).items():
            own[name] = m.get("inelastic_mfp_A")
        return {
            name: (float(v), "set") if v else (float(default), "default")
            for name, v in own.items()
        }

    def optical_depth(self, index, margin):
        """
        The per-pixel inelastic optical depth of frame index, for a sample with
        regions: each material's path over its inelastic mean free path
        (parakeet.inelastic.optical_depth_map)

        """
        mfp = {k: v for k, (v, _) in self.inelastic_mean_free_paths().items()}
        detector = self.microscope.detector
        tau = np.zeros((detector.ny + 2 * margin, detector.nx + 2 * margin))
        for material, length in self.path_maps(index, margin).items():
            tau += length / mfp[material]
        return tau

    def elastic_out(self, index, margin):
        """
        The elastic-out optical depth of frame index (parakeet.elastic): each
        material's path times the inverse mean free path of the scattering the
        band-limited grid cannot hold, lost outside the objective aperture and
        (halo off) inside it as well

        Returns:
            (tau, record): the (ny + 2 margin, nx + 2 margin) optical depth, to
            be applied as exp(-tau / 2) on the exit wave's amplitude, and the
            constants used, for provenance

        """
        import parakeet.elastic
        import parakeet.simulate.engine

        settings = self.simulation["elastic_out"]
        if not (self.sample or {}).get("regions"):
            raise ValueError(
                "simulation.elastic_out needs a sample with regions: it applies "
                "each material's path through every pixel"
            )
        band_limited = parakeet.simulate.engine.engine_band_limited()
        if not band_limited:
            raise ValueError(
                "simulation.elastic_out is valid only with the band-limited "
                "potential (MULTEM_BANDLIMITED_POTENTIAL=1): without it the "
                "potential above the grid's Nyquist frequency aliases back into "
                "the grid, the multislice already scatters part of that flux, and "
                "the term would count it twice"
            )
        detector = self.microscope.detector
        energy = self.microscope.beam.energy
        lam = parakeet.elastic.wavelength(energy)
        q_acc = self.microscope.objective_aperture_cutoff_freq
        q_acc = np.inf if q_acc is None else float(q_acc)
        ice = self.simulation.get("ice_parameters") or {}
        densities = {
            "water": ice.get("number_density_A3")
            or parakeet.elastic.water_number_density(ice.get("density", 0.94))
        }
        for name, m in (self.simulation.get("materials") or {}).items():
            densities[name] = m.get("number_density_A3")
        paths = self.path_maps(index, margin)
        tau = np.zeros((detector.ny + 2 * margin, detector.nx + 2 * margin))
        constants = {}
        uncovered = []
        bands = {}
        for material, length in paths.items():
            n = densities.get(material)
            if n is None:
                uncovered.append(material)
                continue
            mu_lost, mu_band = parakeet.elastic.missing_cross_sections(
                n, energy, detector.pixel_size, q_acc, settings["model"]
            )
            # The band is removed from the wave too; with the halo on it comes
            # back, displaced, as intensity
            mu = mu_lost + mu_band
            tau += length * mu
            if mu_band > 0:
                bands[material] = (length * mu_band, n)
            constants[material] = {
                "lambda_lost_A": 1 / mu_lost if mu_lost > 0 else None,
                "lambda_band_A": 1 / mu_band if mu_band > 0 else None,
                "lambda_applied_A": 1 / mu if mu > 0 else None,
                "tail_fraction": parakeet.elastic.tail_fraction(
                    n,
                    energy,
                    min(q_acc, parakeet.elastic.grid_q(detector.pixel_size)),
                    settings["model"],
                ),
            }
        if uncovered:
            logger.warning(
                "elastic_out: no number_density_A3 for %s, so no elastic-out term "
                "for %s; the render under-counts their loss"
                % (", ".join(uncovered), "it" if len(uncovered) == 1 else "them")
            )
        record = {
            "model": settings["model"],
            "halo": settings["halo"],
            "halo_grid": settings.get("halo_grid", "field"),
            "regime": "band-limited potential (MULTEM_BANDLIMITED_POTENTIAL=1)",
            "energy_keV": energy,
            "wavelength_A": lam,
            "pixel_size_A": detector.pixel_size,
            "q_N": parakeet.elastic.grid_q(detector.pixel_size),
            "theta_g_mrad": lam * parakeet.elastic.grid_q(detector.pixel_size) * 1e3,
            "q_acc": None if not np.isfinite(q_acc) else q_acc,
            "beta_mrad": None if not np.isfinite(q_acc) else lam * q_acc * 1e3,
            "materials": constants,
            "materials_without_composition": uncovered,
            "paths": "exact (every crossing traced; shared with zero_loss)",
            "approximations": [
                "independent atoms beyond what the grid draws (S(q) = 1)",
                "the material's random field taken to draw its own scattering "
                "within the grid (W^2 dsigma below q_N)",
                (
                    "halo on: scattering inside the aperture but outside the "
                    "grid returns as rings of radius lambda q |c_10 + C_s "
                    "lambda^2 q^2| about its pixel (geometric optics, one "
                    "defocus per image), incoherently"
                    if settings["halo"]
                    else "halo off: scattering inside the aperture but outside "
                    "the grid is removed as if lost"
                ),
                "single kinematic scattering; losses multiply",
            ],
        }
        self._elastic_bands = bands
        return tau, record

    def elastic_halo(self, index, intensity, tau, electron_fraction, defocus):
        """
        The band's electrons, put back as intensity (parakeet.elastic.halo):
        what each pixel's band removed from the wave, spread over the rings
        each material's scattering angles give, after the energy filter's
        share at the pixel they left

        Args:
            intensity: |psi|^2 of the exit wave before the term
            tau: The term's optical depth
            electron_fraction: The filter's share (a scalar or a map)
            defocus: The frame's c_10 (A)

        Returns:
            (halo, record)

        """
        import parakeet.elastic

        detector = self.microscope.detector
        energy = self.microscope.beam.energy
        q_acc = self.microscope.objective_aperture_cutoff_freq
        q_acc = np.inf if q_acc is None else float(q_acc)
        c_30_A = self.microscope.lens.c_30 * 1e7
        removed = intensity * -np.expm1(-tau)
        share = np.divide(removed, tau, out=np.zeros_like(tau), where=tau > 0)
        share = share * electron_fraction
        total = np.zeros_like(tau)
        # The coarse grid anchored to the detector field (the default), or to
        # the canvas corner as renders before it
        on_field = self.simulation["elastic_out"].get("halo_grid", "field") == "field"
        margin = int(self.simulation["margin"])
        record = {
            "defocus_A": float(defocus),
            "grid": "field" if on_field else "canvas",
            "materials": {},
        }
        for material, (band, n) in self._elastic_bands.items():
            r, w, rings = parakeet.elastic.band_rings(
                n,
                energy,
                detector.pixel_size,
                q_acc,
                defocus,
                c_30_A,
                self.simulation["elastic_out"]["model"],
            )
            if on_field:
                h, info = parakeet.elastic.halo(
                    share * band,
                    detector.pixel_size,
                    r,
                    w,
                    anchor=(margin, margin),
                    field=(detector.ny, detector.nx),
                )
            else:
                h, info = parakeet.elastic.halo(share * band, detector.pixel_size, r, w)
            total += h * (1 - rings["tail_share"])
            record["materials"][material] = dict(rings, **info)
        # One defocus per image: across a tilted field the defocus changes by
        # the field's half width times tan(tilt), which moves every ring by
        # that share of its radius at the field's edge
        angle = float(self.scan.angles[index])
        half = 0.5 * max(detector.nx, detector.ny) * detector.pixel_size
        record["tilt_deg"] = angle
        record["ring_radius_error_at_field_edge"] = (
            half * abs(np.tan(np.radians(angle))) / abs(defocus) if defocus else None
        )
        record["mean_halo"] = float(total.mean())
        if on_field:
            reach = [m.get("reach_px") for m in record["materials"].values()]
            reach = [x for x in reach if x is not None]
            record["reach_px"] = max(reach) if reach else None
            record["margin_covers_reach"] = (
                margin >= record["reach_px"] if reach else True
            )
            if reach and margin < record["reach_px"]:
                logger.warning(
                    "elastic_out halo: the margin (%d px) is less than the halo's "
                    "reach (%.1f px): field pixels within %.1f px of its edge miss "
                    "halo from specimen beyond the canvas"
                    % (margin, record["reach_px"], record["reach_px"] - margin)
                )
        return total, record

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

        # The elastic scattering the grid cannot hold leaves the wave: an
        # amplitude factor from each pixel's path through each material
        self.elastic_out_record = None
        self.zero_loss_record = None
        intensity_before = None
        if self.simulation.get("elastic_out"):
            tau_el, record = self.elastic_out(index, margin)
            if tau_el.shape != psi.shape:
                raise RuntimeError(
                    "elastic_out: path maps %s do not match the exit wave %s"
                    % (tau_el.shape, psi.shape)
                )
            if self.simulation["elastic_out"]["halo"]:
                intensity_before = np.abs(psi) ** 2
            psi = psi * np.exp(-0.5 * tau_el).astype(psi.dtype)
            record["frame"] = int(index)
            record["mean_transmission"] = float(np.mean(np.exp(-tau_el)))
            self.elastic_out_record = record
            logger.info("elastic_out: %s" % json.dumps(record))

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
                record = zero_loss_record(
                    self.inelastic_mean_free_paths(),
                    self.path_maps(index, margin),
                    slit_width,
                    zlp_fwhm,
                    electron_fraction,
                )
                record["frame"] = int(index)
                self.zero_loss_record = record
                if self.elastic_out_record is not None:
                    self.elastic_out_record["zero_loss"] = record
                logger.info("zero_loss: %s" % json.dumps(record))
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

        # The band's electrons land again, displaced, after the filter's share
        if self.elastic_out_record is not None and intensity_before is not None:
            halo, halo_record = self.elastic_halo(
                index, intensity_before, tau_el, electron_fraction, defocus
            )
            image = image + halo
            self.elastic_out_record["halo"] = halo_record
            logger.info("elastic_out halo: %s" % json.dumps(halo_record))

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
