#
# parakeet.detector_profiles.py
#
# Copyright (C) 2026 Diamond Light Source and Rosalind Franklin Institute
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
"""
Named detector profiles, so that a configuration can switch detectors by name.

A profile gives the sensor's size in pixels, a typical specimen-level pixel
size, its readout modes, and, per mode, the MTF curve to use at each
accelerating voltage. The DQE is not part of a profile: parakeet has one DQE
table, measured at 300 keV.

`mtf` maps an accelerating voltage in kV to a parakeet.mtf curve name. There is
deliberately no interpolation between voltages and no substitution of a
neighbouring one: the 200 and 300 kV curves for the same detector differ by
more at high frequency than at low, so a curve borrowed from the wrong voltage
is wrong in a way that grows with resolution. A voltage with no entry gets no
MTF, and a warning says so.

    import parakeet.detector_profiles as dp

    dp.names()                              # ["falcon4i", "k3"]
    dp.mtf_for("k3", 300, mode="cds")       # "k3 CDS 300kV"
    warnings = dp.apply(config["microscope"]["detector"], "falcon4i", kv=300)

"""

from collections.abc import Mapping

__all__ = ["DETECTORS", "READOUT_MODES", "names", "profile", "mtf_for", "apply"]


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

# Every readout mode any detector offers
READOUT_MODES = sorted({m for d in DETECTORS.values() for m in d["modes"]})


def names():
    """The detector profile names."""
    return sorted(DETECTORS)


def profile(name):
    """
    The profile of a detector.

    Args:
        name (str): A detector profile name

    Returns:
        dict: The profile

    """
    try:
        return DETECTORS[name]
    except KeyError:
        raise ValueError(
            "No detector profile %r; the profiles are %s" % (name, ", ".join(names()))
        ) from None


def mtf_for(name, kv, mode=None):
    """
    The MTF curve name for a detector at this exact voltage, or None.

    Args:
        name (str): A detector profile name
        kv (float): The accelerating voltage in kV
        mode (str): The readout mode, or None for the detector's default

    Returns:
        str: The curve name, or None if the detector has none at this voltage

    """
    det = profile(name)
    mode = mode or det["default_mode"]
    if mode not in det["modes"]:
        raise ValueError(
            "Detector %r has no readout mode %r; it offers %s"
            % (name, mode, ", ".join(sorted(det["modes"])))
        )
    return det["modes"][mode].get(int(round(kv)))


def _set(target, key, value):
    if isinstance(target, Mapping) or hasattr(target, "__setitem__"):
        target[key] = value
    else:
        setattr(target, key, value)


def apply(detector, name, kv, mode=None, pixel_size=None):
    """
    Set a detector configuration from a named profile.

    Sets nx, ny, pixel_size and mtf on `detector`, which may be the
    `microscope.detector` dict of a config or a parakeet.config.Detector. Other
    fields (the DQE, the origin) are left alone.

    Args:
        detector: The detector configuration to update in place
        name (str): A detector profile name
        kv (float): The accelerating voltage in kV, which selects the MTF
        mode (str): The readout mode, or None for the detector's default
        pixel_size (float): The pixel size in A, or None for the profile's

    Returns:
        list: Warnings, as strings: the voltage has no curve, so no MTF is
            set; or the named curve cannot be found, so the simulation will
            fail until it is available

    """
    det = profile(name)
    mode = mode or det["default_mode"]
    mtf = mtf_for(name, kv, mode)
    _set(detector, "nx", det["nx"])
    _set(detector, "ny", det["ny"])
    _set(
        detector, "pixel_size", det["pixel_size"] if pixel_size is None else pixel_size
    )
    _set(detector, "mtf", mtf)

    warnings = []
    if mtf is None:
        warnings.append(
            "Detector %s at %g kV has no MTF curve, so none is applied and "
            "signal transfers perfectly to Nyquist. %s" % (name, kv, det["mtf_note"])
        )
    else:
        import parakeet.mtf

        try:
            parakeet.mtf.MTF.resolve(mtf)
        except ValueError as e:
            warnings.append(
                "Detector %s (%s) at %g kV wants MTF %r, which was NOT FOUND; the "
                "simulation will fail until it is available. %s (%s)"
                % (name, mode, kv, mtf, det["mtf_note"], e)
            )
    return warnings
