#!/usr/bin/env python3
#
# check_engine.py
#
# Copyright (C) 2026 Diamond Light Source and Rosalind Franklin Institute
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
"""
Print which Parakeet and which multislice engine are loaded, and check the
engine against the commit this fork pins.

Run it first in any job whose numbers will be compared with another's:

    python tools/check_engine.py            # text; exit 1 if the engine is not the pinned one
    python tools/check_engine.py --json     # the same as JSON
    python tools/check_engine.py --no-fail  # report only

The Parakeet commit comes from git when the source tree has one, else from the
PARAKEET_COMMIT environment variable (set it when a job ships the tree without
git), else from the installed version. The pin is read from setup.py beside
this script. The engine is checked by the commit in python-multem's version,
which is present only when it was built from a git checkout that is not a
release tag. The MULTEM commit is not recorded by the build: it is the MULTEM
submodule commit of that python-multem commit.

Exit status: 0 when the loaded python-multem was built from the pinned commit
and has every feature in parakeet.engine_check.FEATURES; 1 otherwise (or always
0 with --no-fail).

"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def read_pin(setup_py):
    """
    The python-multem commit pinned in setup.py, or None

    """
    try:
        with open(setup_py) as infile:
            text = infile.read()
    except OSError:
        return None
    match = re.search(r'^PYTHON_MULTEM_COMMIT = "([0-9a-f]{40})"$', text, re.MULTILINE)
    return match.group(1) if match else None


def parakeet_commit():
    """
    The Parakeet commit, and where it came from

    """
    if shutil.which("git"):
        try:
            commit = subprocess.run(
                ["git", "-C", ROOT, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            return commit + ("+dirty" if dirty else ""), "git"
        except (OSError, subprocess.CalledProcessError):
            pass
    if os.environ.get("PARAKEET_COMMIT"):
        return os.environ["PARAKEET_COMMIT"], "PARAKEET_COMMIT"
    return None, None


def gpus():
    """
    The GPUs nvidia-smi reports: name and compute capability

    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def collect():
    """
    The full report, as a dict

    """
    import parakeet
    from parakeet.engine_check import engine_identity

    commit, source = parakeet_commit()
    pin = read_pin(os.path.join(ROOT, "setup.py"))
    engine = engine_identity()
    built = engine["commit"]
    if pin is None:
        pin_state = "unknown: no PYTHON_MULTEM_COMMIT in setup.py"
    elif built is None:
        pin_state = "unknown: the python-multem version names no commit"
    elif pin.startswith(built) or built.startswith(pin):
        pin_state = "match"
    else:
        pin_state = "MISMATCH"
    return {
        "parakeet": {
            "commit": commit,
            "commit_source": source,
            "version": getattr(parakeet, "__version__", None),
            "path": os.path.dirname(getattr(parakeet, "__file__", "") or ""),
        },
        "python_multem_pin": pin,
        "pin_state": pin_state,
        "engine": engine,
        "multem_commit": "not recorded by the build: the MULTEM submodule of python-multem "
        + (built or "?"),
        "gpus": gpus(),
        "python": sys.version.split()[0],
        "host_arch": platform.machine(),
    }


def ok(report):
    """
    Whether the report shows the pinned engine with every feature

    """
    engine = report["engine"]
    return (
        engine["importable"]
        and report["pin_state"] == "match"
        and all(engine["features"].values())
    )


def print_text(report):
    engine = report["engine"]
    p = report["parakeet"]
    lines = [
        "parakeet         %s (%s), version %s"
        % (p["commit"] or "commit unknown", p["commit_source"] or "-", p["version"]),
        "                 %s" % p["path"],
        "python-multem    version %s, built from %s"
        % (engine["version"], engine["commit"] or "an unknown commit"),
        "                 %s" % engine["module_path"],
        "pinned           %s: %s" % (report["python_multem_pin"], report["pin_state"]),
        "MULTEM           %s" % report["multem_commit"],
        "extension        %s" % engine["extension_path"],
        "CUDA code        %s"
        % (", ".join(engine["cuda_architectures"]) or "none found"),
        "features         %s"
        % ", ".join(
            "%s=%s" % (k, "yes" if v else "NO") for k, v in engine["features"].items()
        ),
        "band-limit sw.   %s (python layer only; the build commit is the check)"
        % engine["band_limited_toggle_on"],
        "engine GPUs      available=%s count=%s"
        % (engine["gpu_available"], engine["gpus"]),
        "MULTEM_* env     %s"
        % (
            " ".join("%s=%s" % kv for kv in engine.get("environment", {}).items())
            or "-"
        ),
        "nvidia-smi       %s"
        % ("; ".join(report["gpus"]) if report["gpus"] else "none"),
        "python           %s on %s" % (report["python"], report["host_arch"]),
        "verdict          %s" % ("OK" if ok(report) else "NOT the pinned engine"),
    ]
    print("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--no-fail", action="store_true", help="always exit 0")
    args = parser.parse_args(argv)
    report = collect()
    if args.json:
        report["ok"] = ok(report)
        print(json.dumps(report, indent=2, default=str))
    else:
        print_text(report)
    return 0 if (args.no_fail or ok(report)) else 1


if __name__ == "__main__":
    sys.exit(main())
