#
# parakeet.command_line.sample._check_regions.py
#
# Copyright (C) 2026 Diamond Light Source and Rosalind Franklin Institute
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
"""
parakeet.sample.check_regions: every reason the multislice renderer would
refuse a parakeet-regions/1 document, without rendering. It runs the
renderer's own code path (parakeet.sample._regions.render_plan) and exits
non-zero if there is any.
"""

import json
import sys
from argparse import ArgumentParser
from typing import List

import parakeet.sample._regions

__all__ = ["check_regions"]


def get_description():
    """
    Get the program description

    """
    return "Check a regions document as the multislice renderer would, without rendering"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.sample.check_regions parser

    """
    if parser is None:
        parser = ArgumentParser(description=get_description())
    parser.add_argument("regions", type=str, help="A parakeet-regions/1 JSON file")
    parser.add_argument(
        "--max-planes",
        type=int,
        default=None,
        dest="max_planes",
        help=(
            "The engine's half-spaces per convex set (default: the installed "
            "engine's multem.convex_max_planes(), else %d)"
            % parakeet.sample._regions.MAX_CONVEX_PLANES
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the findings as JSON"
    )
    return parser


def check_regions_impl(args):
    """
    Check the document and report every finding

    """
    if args.max_planes is not None:
        max_planes, source = args.max_planes, "--max-planes"
    else:
        max_planes, source = parakeet.sample._regions.engine_max_planes()
    findings = parakeet.sample._regions.check_regions(args.regions, max_planes=max_planes)
    if args.json:
        print(json.dumps(
            {"max_planes": max_planes, "max_planes_source": source, "findings": findings}, indent=1
        ))
    else:
        for f in findings:
            print("%s\t%s\t%s" % (f["region"], f["reason"], f["message"]))
        print(
            "%d finding%s (max_planes %d, %s)"
            % (len(findings), "" if len(findings) == 1 else "s", max_planes, source)
        )
    return 1 if findings else 0


def check_regions(args: List[str] = None):
    """
    Check a regions document as the renderer would

    """
    sys.exit(check_regions_impl(get_parser().parse_args(args=args)))
