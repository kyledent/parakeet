#!/usr/bin/env python3
"""
Generate a Parakeet config from acquisition parameters.

Takes the things you would set at a microscope -- accelerating voltage,
detector, objective aperture, energy filter slit, ice thickness, total dose and
dose rate -- and writes the Parakeet YAML that corresponds to them, doing the
conversions in between (aperture diameter to cutoff frequency, dose and dose
rate to exposure time, filter state to inelastic model).

It also tells you where the simulation stops matching the parameters you gave
it. Several of Parakeet's models are calibrated at 300 keV and do not vary with
the voltage you set, and the energy filter slit width is read by only one of
the four inelastic models. Those are reported as warnings rather than silently
absorbed, and repeated in a header comment in the generated file.

Examples:
    # 300 keV Krios, K3, 100 nm ice, 20 eV slit in, single exposure
    benchmark/make_config.py -o spa.yaml

    # Falcon 4i, slit retracted, thicker ice, tilt series
    benchmark/make_config.py --detector falcon4i --slit retracted \\
        --ice 150 --mode tilt-series -o ts.yaml

    # 200 keV Glacios, MPL-centred filter so the slit width is actually used
    benchmark/make_config.py --kv 200 --filter-model mpl --slit-width 20 \\
        -o glacios.yaml
"""

import argparse
import datetime
import math
import os
import shlex
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import parakeet.mtf  # noqa: E402

# --- detectors ---------------------------------------------------------------
# nx, ny, a typical specimen-level pixel size, and the MTF curve to use at each
# voltage. The DQE remains a single table measured at 300 keV and is not
# selected per detector; the MTF now is (see --dqe and --mtf below).
#
# `mtf` maps accelerating voltage in kV to a parakeet.mtf preset name. There is
# deliberately no interpolation between voltages and no substitution of a
# neighbouring one: the 200 and 300 kV curves for the same detector differ by
# more at high frequency than at low, so a curve borrowed from the wrong voltage
# is wrong in a way that grows with resolution. A voltage with no entry gets no
# MTF and a warning saying so.
DETECTORS = {
    "k3": {
        "nx": 5760,
        "ny": 4092,
        "pixel_size": 1.05,
        "description": "Gatan K3, 5760 x 4092, counting mode",
        "modes": {
            "standard": {200: "k3 standard 200kV", 300: "k3 standard 300kV"},
            "cds": {200: "k3 CDS 200kV", 300: "k3 CDS 300kV"},
        },
        "default_mode": "standard",
        "mtf_note": (
            "Gatan's own measurements, one request form each from their K3 "
            "camera page. They are not distributed with this code, so the "
            "profile names the curve and it is found on the search path."
        ),
    },
    "falcon4i": {
        "nx": 4096,
        "ny": 4096,
        "pixel_size": 1.20,
        "description": "Thermo Falcon 4i, 4096 x 4096, EER",
        "modes": {
            "counting": {200: "falcon4EC_200kV", 300: "falcon4EC_300kV"},
        },
        "default_mode": "counting",
        "mtf_note": (
            "Falcon 4EC curves, used as the closest published proxy for the 4i. "
            "Same sensor generation, not the same part."
        ),
    },
}

# Every readout mode any detector offers, for --readout
READOUT_MODES = sorted({m for d in DETECTORS.values() for m in d["modes"]})

# --- microscopes -------------------------------------------------------------
# Voltage, Cs, Cc and source energy spread per instrument. Cs and Cc are the
# standard uncorrected Thermo objective values; dE is the nominal source spread,
# converted to the dE/E that Parakeet wants. These are starting points, not
# calibrations of any individual column -- override with --kv/--cs/--cc.
#
# Deliberately NOT emitted as Parakeet's own `microscope.model` field: setting
# that to "krios" or "talos" makes microscope.new() overwrite the beam energy
# with 300 or 200 regardless of what the config says (src/parakeet/microscope.py),
# which silently discards --kv. Explicit values avoid that.
EMITTERS = {
    # Nominal source energy spread. Thermo quote 0.7 eV for the X-FEG and
    # <=0.3 eV for the E-CFEG; 0.3-0.4 eV and 0.8 eV are also reported for cold
    # and Schottky sources in the literature. Override with --energy-spread-ev.
    "schottky": {"dE_eV": 0.7, "description": "X-FEG (Schottky)"},
    "cold": {"dE_eV": 0.3, "description": "E-CFEG (cold FEG)"},
}

MICROSCOPES = {
    # Cs and Cc are the standard uncorrected Thermo objective values. The
    # objective focal length matters because it sets what a physical aperture
    # diameter corresponds to in spatial frequency, and it is NOT the same
    # across the range: sourced values are marked, inferred ones flagged.
    "krios": {
        "kv": 300,
        "kv_options": [300, 200, 100],
        "cs": 2.7,
        "cc": 2.7,
        "focal_length_mm": 3.5,
        "focal_length_sourced": False,
        "emitters": ["schottky", "cold"],
        "default_emitter": "schottky",
        "max_tilt": 70,
        "objective": "C-TWIN, 11 mm gap",
        "description": "Thermo Titan Krios",
    },
    "glacios": {
        "kv": 200,
        "kv_options": [200, 120, 80],
        "cs": 2.7,
        "cc": 2.7,
        "focal_length_mm": 3.4,
        "focal_length_sourced": True,
        "emitters": ["schottky", "cold"],
        "default_emitter": "schottky",
        "max_tilt": 70,
        "objective": "C-TWIN, 11 mm gap",
        "description": "Thermo Glacios 2",
    },
    "talos": {
        "kv": 200,
        "kv_options": [200, 120],
        "cs": 2.7,
        "cc": 2.7,
        "focal_length_mm": 3.5,
        "focal_length_sourced": False,
        "emitters": ["schottky"],
        "default_emitter": "schottky",
        "max_tilt": 70,
        "objective": "C-TWIN class",
        "objective": "C-TWIN class",
        "objective": "C-TWIN class",
        "description": "Thermo Talos (Arctica 200 kV / L120C 120 kV)",
    },
    "tundra": {
        "kv": 100,
        "kv_options": [100],
        # SP-TWIN, developed for the Tundra specifically to cut spherical and
        # chromatic aberration at 100 kV. NOT the C-TWIN's 2.7/2.7.
        "cs": 1.6,
        "cc": 1.7,
        # Thermo state the Tundra objective arrangement gives a 2.3 mm focal
        # length, chosen to reduce chromatic aberration at 100 kV. A physical
        # aperture therefore cuts at a HIGHER spatial frequency than the same
        # aperture on a Krios.
        "focal_length_mm": 2.3,
        "focal_length_sourced": True,
        # The Tundra ships with an X-FEG. It is not a cold-FEG instrument.
        "emitters": ["schottky"],
        "default_emitter": "schottky",
        # +/-15 degrees only: this is a single-particle instrument, not a
        # tomography one.
        "max_tilt": 15,
        "objective": "SP-TWIN, 7 mm gap",
        "description": "Thermo Tundra, 100 kV, X-FEG",
    },
}

# The DQE table is indexed by electrons per pixel per second over this range and
# clamps outside it, so a dose rate beyond these is indistinguishable from the
# endpoint (src/parakeet/dqe.py).
DQE_EPS_RANGE = (1.5, 5.0)

# Slit state -> Parakeet inelastic model.
FILTER_MODELS = {
    "zero-loss": "zero_loss",
    "mpl": "mp_loss",
    "cc-corrected": "cc_corrected",
    "none": None,
}

_M0C2 = 510.99895  # keV
_HC = 12.3984198  # keV.A


def electron_wavelength(kv):
    """Relativistic electron wavelength in Angstroms."""
    return _HC / math.sqrt(kv * (2 * _M0C2 + kv))


def resolve_mtf(args, det):
    """
    Which MTF curve this config should name, if any.

    An explicit --mtf wins and is passed through untouched, so a path to a curve
    obtained from a vendor works as well as a preset name. Otherwise the
    detector's own curve for this exact voltage is used, or none.

    """
    if args.no_mtf:
        return None
    if args.mtf:
        return args.mtf
    mode = args.readout or det["default_mode"]
    if mode not in det["modes"]:
        raise SystemExit(
            f"--detector {args.detector} has no readout mode {mode!r}; "
            f"it offers {', '.join(sorted(det['modes']))}"
        )
    return det["modes"][mode].get(int(round(args.kv)))


def aperture_cutoff_frequency(diameter_um, focal_length_mm, kv):
    """
    Spatial frequency (1/A) passed by an objective aperture.

    The aperture subtends a semi-angle of r/f at the back focal plane, and a
    scattering angle theta corresponds to a spatial frequency theta/lambda.
    """
    radius_mm = (diameter_um * 1e-3) / 2.0
    semi_angle = radius_mm / focal_length_mm  # radians
    return semi_angle / electron_wavelength(kv)  # 1/A


def exposure_time(dose_per_image, dose_rate_e_px_s, pixel_size_A):
    """
    Seconds needed to deliver dose_per_image (e/A^2) at dose_rate (e/px/s).

    The rate is quoted per detector pixel, so it converts to e/A^2/s by
    dividing by the pixel area at the specimen.
    """
    if dose_rate_e_px_s <= 0:
        return None
    rate_per_A2 = dose_rate_e_px_s / (pixel_size_A**2)
    return dose_per_image / rate_per_A2


def build(args, warnings):
    det = DETECTORS[args.detector]
    mtf = resolve_mtf(args, det)
    pixel_size = args.pixel_size if args.pixel_size else det["pixel_size"]
    ice_A = args.ice * 10.0
    field_A = args.field if args.field else None

    # --- scan -----------------------------------------------------------------
    if args.mode == "spa":
        num_images = 1
        scan = {"mode": "still", "num_images": 1, "start_angle": 0, "step_angle": 0}
    else:
        num_images = args.tilts
        scan = {
            "mode": "dose_symmetric",
            "num_images": num_images,
            "start_angle": -args.tilt_range,
            "step_angle": args.tilt_step,
            "axis": [0, 1, 0],
        }
    dose_per_image = args.total_dose / num_images

    # --- energy filter --------------------------------------------------------
    if args.slit == "retracted":
        inelastic_model = "unfiltered"
        if args.filter_model != "zero-loss":
            warnings.append(
                f"--slit retracted overrides --filter-model {args.filter_model}: "
                "the config uses 'unfiltered'"
            )
        if args.slit_width is not None:
            warnings.append("--slit-width is ignored with the slit retracted")
    else:
        inelastic_model = FILTER_MODELS[args.filter_model]
        if args.filter_model == "zero-loss" and args.slit_width is not None:
            warnings.append(
                f"--slit-width {args.slit_width} eV has NO EFFECT with "
                "--filter-model zero-loss: that model scales the image by "
                "exp(-t/lambda) and never reads the slit width. Use "
                "--filter-model mpl for the width to matter."
            )
        if args.filter_model == "mpl" and args.slit_width is None:
            warnings.append("--filter-model mpl needs --slit-width; using 20 eV")

    slit_width = args.slit_width if args.slit_width is not None else 20.0

    # --- optics ---------------------------------------------------------------
    microscope = {
        "beam": {
            "energy": args.kv,
            "electrons_per_angstrom": round(dose_per_image, 4),
            "energy_spread": args.energy_spread,
            "acceleration_voltage_spread": 0.8e-6,
            "illumination_semiangle": args.semiangle,
        },
        "lens": {"c_10": args.defocus, "c_30": args.cs, "c_c": args.cc},
        "detector": {
            "nx": det["nx"] if field_A is None else int(field_A / pixel_size),
            "ny": det["ny"] if field_A is None else int(field_A / pixel_size),
            "pixel_size": pixel_size,
            "dqe": args.dqe,
            "mtf": mtf,
        },
    }
    if args.aperture:
        microscope["objective_aperture_cutoff_freq"] = round(
            aperture_cutoff_frequency(
                args.aperture, args.objective_focal_length, args.kv
            ),
            4,
        )

    nx = microscope["detector"]["nx"]
    ny = microscope["detector"]["ny"]
    box = [round(nx * pixel_size, 1), round(ny * pixel_size, 1), ice_A]

    sample = {
        "box": box,
        "centre": [box[0] / 2, box[1] / 2, box[2] / 2],
        "shape": {
            "type": "cuboid",
            "cuboid": {"length_x": box[0], "length_y": box[1], "length_z": box[2]},
        },
        "molecules": {"pdb": [{"id": args.pdb, "instances": args.particles}]},
    }
    if args.fiducials:
        sample["fiducials"] = {
            "diameter": float(args.fiducial_diameter),
            "count": args.fiducials,
            "placement": args.fiducial_placement,
            "occupancy": 1.0,
        }

    simulation = {"ice": True, "slice_thickness": 3.0, "margin": 100}
    if inelastic_model is not None:
        simulation["inelastic_model"] = inelastic_model
    if inelastic_model == "mp_loss":
        simulation["mp_loss_width"] = slit_width
        simulation["mp_loss_position"] = args.slit_position

    t = exposure_time(dose_per_image, args.dose_rate, pixel_size)
    if t is not None:
        scan["exposure_time"] = round(t, 4)

    return (
        {
            "microscope": microscope,
            "sample": sample,
            "scan": scan,
            "simulation": simulation,
            "output": {"dtype": args.dtype},
            "multiprocessing": {"device": "gpu", "nproc": 1, "gpu_id": [args.gpu]},
        },
        dose_per_image,
        t,
        pixel_size,
    )


def collect_model_caveats(args, warnings):
    """Where the simulation stops tracking the parameters that were asked for."""
    scope = MICROSCOPES[args.microscope]
    if args.kv not in scope["kv_options"]:
        warnings.append(
            f"--kv {args.kv:.0f} is not a voltage {args.microscope} normally offers "
            f"({', '.join(str(v) for v in scope['kv_options'])} keV). Cs, Cc and the "
            "source spread are still the preset's."
        )

    if args.emitter not in scope["emitters"]:
        warnings.append(
            f"the {args.microscope} is not offered with a {args.emitter} emitter "
            f"({'/'.join(scope['emitters'])} only); using its "
            f"{EMITTERS[args.emitter]['dE_eV']} eV spread anyway because you asked"
        )
    if not scope["focal_length_sourced"]:
        warnings.append(
            f"the objective focal length for {args.microscope} "
            f"({args.objective_focal_length} mm) is a widely used figure rather than "
            "one taken from a specification sheet; it only affects the conversion "
            "from aperture diameter to cutoff frequency. Set "
            "--objective-focal-length if you know yours."
        )
    if args.mode == "tilt-series" and args.tilt_range > scope["max_tilt"]:
        warnings.append(
            f"the {args.microscope} stage reaches +/-{scope['max_tilt']} degrees, but "
            f"--tilt-range is {args.tilt_range:.0f}. The simulation will happily "
            "produce tilts the instrument cannot."
        )

    # Dose rate reaches exactly one thing: the DQE curve lookup.
    if not args.dqe:
        warnings.append(
            f"--dose-rate {args.dose_rate} has NO EFFECT without --dqe. It sets "
            "scan.exposure_time, and the only physics that reads it is the DQE "
            "curve lookup; shot noise is set by the dose, not the rate."
        )
    else:
        lo, hi = DQE_EPS_RANGE
        eps = args.dose_rate
        if eps < lo or eps > hi:
            warnings.append(
                f"--dose-rate {eps} e/px/s is outside the DQE table's range "
                f"({lo}-{hi} e/px/s), so it is clamped to the nearest endpoint: any "
                f"rate {'below ' + str(lo) if eps < lo else 'above ' + str(hi)} gives "
                "identical output."
            )
    if args.kv != 300:
        warnings.append(
            f"--kv {args.kv}: the inelastic mean free path is hardcoded to 3150 A "
            "for amorphous ice at 300 keV and does not vary with voltage, so the "
            "elastic/inelastic split will be that of a 300 keV beam. The Landau "
            "energy-loss distribution IS voltage dependent."
        )
        if args.dqe:
            warnings.append(
                f"--kv {args.kv} with --dqe: the DQE curve is a single table "
                "measured at 300 keV and is not interpolated over voltage."
            )
    if args.dqe:
        warnings.append(
            f"--dqe with --detector {args.detector}: the DQE is a single table, "
            "not a per-detector curve. The detector choice sets the array size, "
            "the pixel size and the MTF; the DQE is the same for both."
        )

    det = DETECTORS[args.detector]
    mtf = resolve_mtf(args, det)
    if args.no_mtf:
        warnings.append(
            "--no-mtf: the detector transfers signal perfectly to Nyquist. "
            "Real images are blurred by charge spreading; comparisons against "
            "them will show more high-frequency signal than any detector gives."
        )
    elif args.mtf:
        pass
    elif mtf is None:
        warnings.append(
            f"--detector {args.detector} at {args.kv:g} kV has no MTF curve, so "
            "none is applied and signal transfers perfectly to Nyquist. "
            + det["mtf_note"]
            + " Supply one with --mtf <preset|file.star>."
        )
    else:
        mode = args.readout or det["default_mode"]
        try:
            found = parakeet.mtf.MTF.resolve(mtf)
            where = getattr(found, "source", mtf)
            warnings.append(
                f"--detector {args.detector} ({mode}) at {args.kv:g} kV uses "
                f"MTF '{mtf}' -> {where}. " + det["mtf_note"]
            )
        except ValueError as e:
            warnings.append(
                f"--detector {args.detector} ({mode}) at {args.kv:g} kV wants "
                f"MTF '{mtf}', which was NOT FOUND, so the config names it and "
                f"the simulation will fail until it is available. "
                + det["mtf_note"]
                + f" ({e})"
            )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    g = p.add_argument_group("beam and optics")
    g.add_argument(
        "--microscope",
        choices=sorted(MICROSCOPES),
        default="krios",
        help="instrument preset: sets voltage, Cs, Cc and source energy spread "
        "(default krios). Each can still be overridden individually",
    )
    g.add_argument(
        "--kv",
        type=float,
        default=None,
        help="accelerating voltage, keV (default: the microscope's)",
    )
    g.add_argument(
        "--emitter",
        choices=sorted(EMITTERS),
        default=None,
        help="field emitter (default: the microscope's). cold = E-CFEG (0.3 eV), "
        "schottky = X-FEG (0.7 eV)",
    )
    g.add_argument(
        "--energy-spread-ev",
        type=float,
        default=None,
        help="source energy spread in eV, overriding the emitter's nominal value",
    )
    g.add_argument("--cs", type=float, default=None, help="spherical aberration, mm")
    g.add_argument("--cc", type=float, default=None, help="chromatic aberration, mm")
    g.add_argument(
        "--defocus",
        type=float,
        default=-20000,
        help="defocus, A (negative = underfocus)",
    )
    g.add_argument(
        "--semiangle", type=float, default=0.02, help="illumination semiangle, mrad"
    )
    g.add_argument(
        "--energy-spread",
        type=float,
        default=None,
        help="dE/E (default: from the microscope's nominal source spread)",
    )

    g = p.add_argument_group("objective aperture")
    g.add_argument(
        "--aperture",
        type=float,
        default=100,
        help="aperture diameter, um (default 100; 0 = retracted)",
    )
    g.add_argument(
        "--objective-focal-length",
        type=float,
        default=None,
        help="objective focal length, mm (default: the microscope's)",
    )

    g = p.add_argument_group("energy filter")
    g.add_argument(
        "--slit",
        choices=["inserted", "retracted"],
        default="inserted",
        help="energy filter slit state (default inserted)",
    )
    g.add_argument(
        "--slit-width",
        type=float,
        default=None,
        help="slit width, eV. Only read by --filter-model mpl",
    )
    g.add_argument(
        "--filter-model",
        choices=sorted(FILTER_MODELS),
        default="zero-loss",
        help="how the filter is modelled when the slit is inserted "
        "(default zero-loss)",
    )
    g.add_argument(
        "--slit-position",
        choices=["peak", "optimal"],
        default="peak",
        help="where the mpl slit is centred (default peak)",
    )

    g = p.add_argument_group("detector and dose")
    g.add_argument("--detector", choices=sorted(DETECTORS), default="k3")
    g.add_argument(
        "--readout",
        choices=READOUT_MODES,
        default=None,
        help="detector readout mode; CDS is a different detector response "
        "rather than a processing option, so it has its own MTF curve",
    )
    g.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="specimen pixel size, A/px (default: the detector's typical value)",
    )
    g.add_argument(
        "--field",
        type=float,
        default=None,
        help="crop to a square field of this size in A, instead of the full detector",
    )
    g.add_argument(
        "--total-dose",
        type=float,
        default=40.0,
        help="total exposure over the whole acquisition, e/A^2 (default 40)",
    )
    g.add_argument(
        "--dose-rate",
        type=float,
        default=8.0,
        help="dose rate, e/px/s, used to derive the exposure time (default 8)",
    )
    g.add_argument("--dqe", action="store_true", help="apply the detector DQE model")
    g.add_argument(
        "--mtf",
        help=(
            "MTF to use instead of the detector's own: a parakeet.mtf preset "
            "name or the path to a RELION MTF star file"
        ),
    )
    g.add_argument(
        "--no-mtf",
        action="store_true",
        help="apply no MTF at all, giving a detector that transfers perfectly",
    )

    g = p.add_argument_group("specimen")
    g.add_argument(
        "--ice", type=float, default=100, help="ice thickness, nm (default 100)"
    )
    g.add_argument(
        "--pdb", default="4v5d", help="PDB id of the particle (default 4v5d, 70S)"
    )
    g.add_argument("--particles", type=int, default=14, help="number of particles")
    g.add_argument(
        "--fiducials", type=int, default=6, help="number of gold fiducials (0 for none)"
    )
    g.add_argument(
        "--fiducial-diameter", type=float, default=100.0, help="fiducial diameter, A"
    )
    g.add_argument(
        "--fiducial-placement",
        default="both_surfaces",
        choices=["surface", "both_surfaces", "volume"],
    )

    g = p.add_argument_group("acquisition")
    g.add_argument("--mode", choices=["spa", "tilt-series"], default="spa")
    g.add_argument(
        "--tilts", type=int, default=41, help="number of tilts (tilt-series)"
    )
    g.add_argument("--tilt-range", type=float, default=60, help="max tilt, degrees")
    g.add_argument("--tilt-step", type=float, default=3, help="tilt step, degrees")

    p.add_argument(
        "--dtype", default="float16", choices=["float16", "float32", "int16", "uint16"]
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("-o", "--output", default="-", help="output YAML (default stdout)")

    args = p.parse_args()
    if args.aperture == 0:
        args.aperture = None

    scope = MICROSCOPES[args.microscope]
    explicit = {a.lstrip("-").replace("-", "_") for a in sys.argv if a.startswith("--")}
    if "kv" not in explicit:
        args.kv = scope["kv"]
    if "cs" not in explicit:
        args.cs = scope["cs"]
    if "cc" not in explicit:
        args.cc = scope["cc"]
    if args.emitter is None:
        args.emitter = scope["default_emitter"]
    if "objective_focal_length" not in explicit:
        args.objective_focal_length = scope["focal_length_mm"]
    if args.energy_spread_ev is None:
        args.energy_spread_ev = EMITTERS[args.emitter]["dE_eV"]
    if "energy_spread" not in explicit:
        # dE/E from the source spread at this voltage
        args.energy_spread = args.energy_spread_ev / (args.kv * 1e3)

    warnings = []
    config, dose_per_image, t_exp, pixel_size = build(args, warnings)
    collect_model_caveats(args, warnings)

    try:
        import yaml
    except ImportError:
        print("pyyaml is required", file=sys.stderr)
        return 1

    header = [
        "# Parakeet config generated by benchmark/make_config.py",
        f"# {datetime.datetime.now().isoformat(timespec='seconds')}",
        "#",
        "#   " + " ".join(shlex.quote(a) for a in sys.argv),
        "#",
        f"# {MICROSCOPES[args.microscope]['description']} at {args.kv:.0f} keV, "
        f"{MICROSCOPES[args.microscope]['objective']} objective "
        f"(Cs {args.cs} mm, Cc {args.cc} mm, f {args.objective_focal_length} mm)",
        f"# {EMITTERS[args.emitter]['description']}, "
        f"{args.energy_spread_ev} eV source spread (dE/E = {args.energy_spread:.2e})",
        f"# {DETECTORS[args.detector]['description']}",
        f"# {args.ice:.0f} nm ice, {pixel_size} A/px",
        f"# slit {args.slit}" + _slit_note(args, config),
        f"# objective aperture "
        + (
            f"{args.aperture:.0f} um -> cutoff "
            f"{config['microscope'].get('objective_aperture_cutoff_freq')} 1/A"
            if args.aperture
            else "retracted"
        ),
        f"# dose {args.total_dose:.1f} e/A^2 total over {config['scan']['num_images']} image(s)"
        f" = {dose_per_image:.3f} e/A^2 each"
        + (f", {t_exp:.3f} s at {args.dose_rate:.1f} e/px/s" if t_exp else ""),
    ]
    if warnings:
        header += [
            "#",
            "# WARNINGS -- where the simulation does not follow these settings:",
        ]
        for w in warnings:
            for i, line in enumerate(_wrap(w)):
                header.append(("#   - " if i == 0 else "#     ") + line)
    header.append("")

    text = "\n".join(header) + yaml.safe_dump(
        config, sort_keys=False, default_flow_style=False
    )

    if args.output == "-":
        sys.stdout.write(text)
    else:
        with open(args.output, "w") as fh:
            fh.write(text)
        print(f"wrote {args.output}", file=sys.stderr)

    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def _slit_note(args, config):
    """Describe the filter honestly: say when the width is not used."""
    if args.slit == "retracted":
        return " (inelastic_model: unfiltered)"
    model = config["simulation"].get("inelastic_model")
    if model == "mp_loss":
        return (
            f", {config['simulation']['mp_loss_width']:.0f} eV slit centred on the "
            f"{args.slit_position} (inelastic_model: mp_loss)"
        )
    if model == "zero_loss":
        return (
            " (inelastic_model: zero_loss -- an ideal filter passing only the "
            "zero-loss peak; the slit WIDTH is not used by this model)"
        )
    if model is None:
        return " (no inelastic model: elastic scattering only)"
    return f" (inelastic_model: {model})"


def _wrap(text, width=74):
    import textwrap

    return textwrap.wrap(text, width=width)


if __name__ == "__main__":
    sys.exit(main())
