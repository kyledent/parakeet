#
# parakeet.engine_check.py
#
# Copyright (C) 2026 Diamond Light Source and Rosalind Franklin Institute
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
"""
Which multislice engine is loaded, and whether it has what a simulation needs.

This fork pins its engine in setup.py (PYTHON_MULTEM_COMMIT): one commit of the
fork's python-multem, whose MULTEM submodule is in turn one commit of the fork's
MULTEM. Nothing stops another python-multem being installed in its place (the
PyPI release, say, or an older build left on a path), and that engine runs
quietly and gives different numbers. The functions here say which engine was
imported and refuse to go on when a feature a configuration uses is missing.

    from parakeet.engine_check import require_engine_features
    require_engine_features("multi_region_masker", "ice_m0")

`engine_identity()` collects everything this module can establish without git:
the python-multem version and the commit it was built from (from the version
setuptools_scm wrote, "...+g<hash>"), which features exist, whether the
band-limited potential switch is on (Python layer only), and the CUDA architectures compiled into the
extension (read from its fat binary). The MULTEM commit is not recorded by the
build; it is the submodule commit of the python-multem commit.

"""
import importlib
import os
import re
import struct

# The features this fork uses, each with the test for it. Tests take the
# imported multem module.
FEATURES = {
    # Several implicit-material regions in one simulation, each with its own
    # random-field constants, with boolean subtraction.
    "multi_region_masker": lambda multem: hasattr(multem, "MultiRegionMasker"),
    # Mean and variance of the ice random field set per material (m0, v0).
    "ice_m0": lambda multem: hasattr(multem.IceParameters(), "m0"),
    # The switch for atomic potentials band-limited to the grid. This is the
    # Python layer only: it reads an environment variable and does not touch
    # the compiled extension, so it shows the package is recent enough, not
    # that the extension behind it was built with the band-limited tables. The
    # build's commit (engine_identity()["commit"]) is the identity check.
    "band_limited_toggle": lambda multem: hasattr(multem, "band_limited_potential"),
    # The band-limited potential's grid spacing, bound from the compiled
    # extension: unlike the toggle, it shows the extension itself is recent.
    "band_limit_spacing": lambda multem: hasattr(multem, "band_limit_spacing"),
    # A seed for the GRF ice, carried by the masker, so its field is reproducible.
    "grf_seed": lambda multem: hasattr(multem.Masker(), "set_seed"),
    # A multi-region masker region that only subtracts: vacuum, which has no
    # random field but must still cut its shape out of the regions below it.
    # A tabulated GRF power spectrum in place of the two Gaussians.
    "spectrum_table": lambda multem: hasattr(multem.IceParameters(), "spectrum_q"),
    "vacuum_regions": lambda multem: bool(
        getattr(
            getattr(multem, "MultiRegionMasker", None), "supports_vacuum_regions", False
        )
    ),
    # The projected potential of several regions of material: an overload of
    # compute_projected_potential that takes a MultiRegionMasker (it came with
    # ice_slice_statistics, the engine's own per-slice mean and sigma)
    "multi_region_potential": lambda multem: (
        hasattr(multem, "ice_slice_statistics")
        and "MultiRegionMasker" in (multem.compute_projected_potential.__doc__ or "")
    ),
}

PIN_HINT = (
    "Install the python-multem commit this fork pins in setup.py "
    "(PYTHON_MULTEM_COMMIT), and run tools/check_engine.py to see which engine "
    "is loaded."
)

# The fat binary nvcc embeds: a wrapper header, then entries, each with a kind
# (1 = PTX, 2 = ELF cubin) and the SM architecture it targets.
_FATBIN_MAGIC = struct.pack("<I", 0xBA55ED50)
_FATBIN_KINDS = {1: "ptx", 2: "cubin"}


class EngineFeatureError(RuntimeError):
    """
    The loaded multislice engine lacks a feature that is needed

    """


def import_multem():
    """
    Import the multislice engine

    Returns:
        The multem module, or None if it cannot be imported

    """
    try:
        return importlib.import_module("multem")
    except ImportError:
        return None


def engine_features(multem=None):
    """
    Which of FEATURES the engine has

    Args:
        multem: The multem module (imported if None)

    Returns:
        A dict, feature name -> bool. Every feature is False when multem
        cannot be imported.

    """
    if multem is None:
        multem = import_multem()
    result = {}
    for name, test in FEATURES.items():
        try:
            result[name] = multem is not None and bool(test(multem))
        except Exception:
            result[name] = False
    return result


def require_engine_features(*names, multem=None):
    """
    Raise unless the engine has every named feature

    Args:
        names: Feature names, keys of FEATURES
        multem: The multem module (imported if None)

    Raises:
        KeyError: A name is not a known feature
        EngineFeatureError: The engine cannot be imported, or lacks a feature

    """
    for name in names:
        if name not in FEATURES:
            raise KeyError(
                "Unknown engine feature %r; known: %s" % (name, sorted(FEATURES))
            )
    if multem is None:
        multem = import_multem()
    if multem is None:
        raise EngineFeatureError("Could not import multem. " + PIN_HINT)
    have = engine_features(multem)
    missing = [name for name in names if not have[name]]
    if missing:
        raise EngineFeatureError(
            "The loaded multem (%s, version %s) lacks %s. %s"
            % (
                getattr(multem, "__file__", "?"),
                getattr(multem, "__version__", "unknown"),
                ", ".join(missing),
                PIN_HINT,
            )
        )


def commit_from_version(version):
    """
    The commit a setuptools_scm version names, if it names one

    Args:
        version: A version string, e.g. "0.4.2.dev7+g50a3a34" or
            "0.4.2.dev7+g50a3a34.d20261106"

    Returns:
        The abbreviated commit hash, or None (a release version names none)

    """
    match = re.search(r"\+g([0-9a-f]{7,40})", version or "")
    return match.group(1) if match else None


def cuda_architectures(path):
    """
    The CUDA code compiled into a shared library

    Reads the fat binaries nvcc embeds. Compressed entries still carry their
    architecture in the entry header, so they are reported too.

    Args:
        path: The shared library

    Returns:
        A sorted list of (kind, arch) pairs, kind "ptx" or "cubin" (or
        "kind<n>" if unknown), arch e.g. "sm_89". Empty if there is no CUDA
        code or the file cannot be read.

    """
    try:
        with open(path, "rb") as infile:
            data = infile.read()
    except OSError:
        return []
    found = set()
    start = data.find(_FATBIN_MAGIC)
    while start >= 0:
        try:
            header_size = struct.unpack_from("<H", data, start + 6)[0]
            fatbin_size = struct.unpack_from("<Q", data, start + 8)[0]
            entry = start + header_size
            end = min(entry + fatbin_size, len(data))
            while entry + 32 <= end:
                kind, _, entry_header = struct.unpack_from("<HHI", data, entry)
                payload = struct.unpack_from("<Q", data, entry + 8)[0]
                arch = struct.unpack_from("<I", data, entry + 28)[0]
                if entry_header == 0:
                    break
                found.add((_FATBIN_KINDS.get(kind, "kind%d" % kind), "sm_%d" % arch))
                entry += entry_header + payload
        except struct.error:
            pass
        start = data.find(_FATBIN_MAGIC, start + 4)
    return sorted(found)


def _extension_path(multem):
    """
    The path of the compiled extension behind multem, if it can be found

    """
    try:
        ext = importlib.import_module("multem_ext")
        return getattr(ext, "__file__", None)
    except ImportError:
        return None


def engine_identity():
    """
    Everything that can be established about the loaded engine without git

    Returns:
        A dict with keys: importable, module_path, version, commit,
        extension_path, cuda_architectures, features,
        band_limited_toggle_on, gpu_available, gpus

    """
    multem = import_multem()
    identity = {
        "importable": multem is not None,
        "module_path": None,
        "version": None,
        "commit": None,
        "extension_path": None,
        "cuda_architectures": [],
        "features": engine_features(multem),
        "band_limited_toggle_on": None,
        "gpu_available": None,
        "gpus": None,
    }
    if multem is None:
        return identity
    identity["module_path"] = getattr(multem, "__file__", None)
    identity["version"] = getattr(multem, "__version__", None)
    identity["commit"] = commit_from_version(identity["version"])
    ext_path = _extension_path(multem)
    identity["extension_path"] = ext_path
    if ext_path:
        identity["cuda_architectures"] = [
            "%s %s" % (kind, arch) for kind, arch in cuda_architectures(ext_path)
        ]
    if identity["features"]["band_limited_toggle"]:
        identity["band_limited_toggle_on"] = bool(multem.band_limited_potential())
    for key, name in (
        ("gpu_available", "is_gpu_available"),
        ("gpus", "number_of_gpu_available"),
    ):
        function = getattr(multem, name, None)
        if function is not None:
            try:
                identity[key] = function()
            except Exception:
                pass
    identity["environment"] = {
        name: os.environ[name]
        for name in sorted(os.environ)
        if name.startswith("MULTEM_")
    }
    return identity
