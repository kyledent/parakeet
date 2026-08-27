#!/usr/bin/env python3
"""
foil_regimes.py -- the numbers behind the support-film methods design.

A foil model is wanted across four orders of magnitude in sampling, from
0.5 A/px where atoms matter to 10 nm/px where a whole grid square fits in a
frame. This prints what changes across that range: what is resolvable, what an
all-atom foil would cost, what the coarse-grained approximation does to the
noise, and how carbon and gold differ.

    validation/grf/foil_regimes.py [--markdown]
"""

import argparse

import numpy as np
import scipy.constants as sc

AVOGADRO = sc.Avogadro

# name -> (Z, molar mass g/mol, density g/cm^3, typical thickness nm)
MATERIALS = {
    "amorphous carbon": (6, 12.011, 2.0, 12),
    "gold": (79, 196.967, 19.3, 50),
    "vitreous ice": (7.42, 18.015, 0.94, 100),
}

HOLE_UM = 1.2  # a Quantifoil R1.2/1.3 hole
SQUARE_UM = 90.0  # a 400-mesh grid square
CALIBRATED_PIXEL_AREA = 4.0  # the Cv tables stop here (2 A/px)


def number_density(rho, molar_mass, per_unit=1):
    """Atoms per cubic Angstrom."""
    return rho * 1e-24 * AVOGADRO * per_unit / molar_mass


def malis_lambda_nm(Z, E0_keV=300.0, beta_mrad=10.0):
    """Inelastic mean free path, Malis/Cheng/Egerton 1988 parameterisation."""
    Em = 7.6 * Z**0.36
    F = (1 + E0_keV / 1022) / (1 + E0_keV / 511) ** 2
    return 106 * F * E0_keV / (Em * np.log(2 * beta_mrad * E0_keV / Em))


def table(rows, header, md):
    if md:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            print("| " + " | ".join(str(c) for c in r) + " |")
    else:
        w = [
            max(len(str(h)), *(len(str(r[i])) for r in rows))
            for i, h in enumerate(header)
        ]
        print("  " + "  ".join(str(h).rjust(x) for h, x in zip(header, w)))
        for r in rows:
            print("  " + "  ".join(str(c).rjust(x) for c, x in zip(r, w)))
    print()


def q1_what_is_resolvable(md):
    print("1. What a pixel can resolve, across the range a foil model is wanted for")
    rows = []
    for px in (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0):
        rows.append(
            [
                f"{px:g}",
                f"{px/10:g}",
                f"{2.9/px:.2f}",
                f"{HOLE_UM*1e4/px:.0f}",
                f"{SQUARE_UM*1e4/px:.0f}",
                "yes" if px * px <= CALIBRATED_PIXEL_AREA else "NO",
            ]
        )
    table(
        rows,
        [
            "A/px",
            "nm/px",
            "px per 2.9 A",
            f"px per {HOLE_UM} um hole",
            f"px per {SQUARE_UM:g} um square",
            "Cv calibrated",
        ],
        md,
    )
    print("  Above 2 A/px every pixel size gets the same Cv correction, because the")
    print("  tables clamp. At 10 nm/px the pixel area is 2500x beyond calibration.\n")


def q2_all_atom_cost(md):
    print("2. What an all-atom foil would cost, for a field containing one hole")
    side_A = 1.5e4
    rows = []
    for name, (Z, M, rho, _) in MATERIALS.items():
        if name == "vitreous ice":
            continue
        for t_nm in (12, 20, 50):
            n = number_density(rho, M) * side_A * side_A * t_nm * 10
            rows.append(
                [name, f"{t_nm} nm", f"{n/1e9:.1f} billion", f"{n*20/1024**3:.0f} GiB"]
            )
    table(rows, ["material", "thickness", "atoms", "AtomData at 20 B/atom"], md)


def q3_coarse_graining(md):
    print("3. What capping the atom count does to the background texture")
    cap = 5_000_000
    rows = []
    Z, M, rho, _ = MATERIALS["amorphous carbon"]
    for side_um in (0.5, 1.0, 1.6):
        n = number_density(rho, M) * (side_um * 1e4) ** 2 * 12 * 10
        k = n / cap
        rows.append(
            [
                f"{side_um} um",
                f"{n/1e9:.2f} billion",
                f"x{k:.0f}",
                f"x{k:.0f}",
                f"{k**0.5:.0f}x",
            ]
        )
    table(
        rows,
        [
            "field",
            "physical atoms",
            "occupancy boost",
            "variance inflation",
            "noise amplitude",
        ],
        md,
    )
    print("  Mean goes as N.w and variance as N.w^2, so holding the mean fixed")
    print("  inflates the variance by the coarse-graining factor.\n")


def q4_mass_thickness(md):
    print("4. Mass thickness, which is what low magnification actually images")
    rows = []
    ref = None
    for name, (Z, M, rho, t_nm) in MATERIALS.items():
        for t in (12, 20, 50) if name != "vitreous ice" else (100,):
            mt = rho * t * 1e-7 * 1e6  # ug/cm^2
            if name == "vitreous ice":
                ref = mt
            rows.append([name, f"{t} nm", f"{rho:g}", f"{mt:.1f}"])
    for r in rows:
        r.append(f"{float(r[3])/ref:.1f}x")
    table(
        rows, ["material", "thickness", "g/cm3", "rho.t (ug/cm2)", "vs 100 nm ice"], md
    )


def q5_scattering(md):
    print("5. Inelastic mean free path and how much of the beam a foil removes")
    rows = []
    for name, (Z, M, rho, _) in MATERIALS.items():
        lam = malis_lambda_nm(Z)
        ts = (12, 20, 50) if name != "vitreous ice" else (50, 100, 200)
        rows.append(
            [name, f"{Z:g}", f"{lam:.0f} nm"]
            + [f"{t/lam:.2f}" for t in ts]
            + [f"{100*(1-np.exp(-ts[-1]/lam)):.0f}%"]
        )
    table(
        rows,
        [
            "material",
            "Z",
            "lambda_inel",
            "t/lam (thin)",
            "(mid)",
            "(thick)",
            "scattered at thickest",
        ],
        md,
    )
    print("  Inelastic alone understates gold: its contrast at low magnification is")
    print("  dominated by ELASTIC scattering outside the objective aperture, which")
    print("  rises steeply with Z and is what makes a gold bar look opaque.\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    print("Support film: what changes across the magnification range")
    print("=" * 72 + "\n")
    for fn in (
        q1_what_is_resolvable,
        q2_all_atom_cost,
        q3_coarse_graining,
        q4_mass_thickness,
        q5_scattering,
    ):
        fn(a.markdown)


if __name__ == "__main__":
    main()
