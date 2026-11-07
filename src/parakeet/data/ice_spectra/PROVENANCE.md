# Ice spectrum presets: provenance

`vitreous_ice_md.json` and `liquid_water_md.json` are tabulated power spectra for the GRF ice model, selected by
`ice_parameters.preset` (or a material's `preset`). Neither is the default: without a preset, the ice keeps the
two-Gaussian spectrum Parakeet has always used.

## What the tables are

Each table holds the projected-potential power spectrum per water molecule, P(q), of a slab thick enough that
its shape no longer changes with thickness (the thick limit). The engine draws each slice of ice independently,
so the thick limit is the spectrum it needs.

- **Model:** TIP4P/2005 water, molecular dynamics. The projected potential uses independent-atom electron
  scattering factors.
- **Box:** 5 nm. A run in an 8 nm box agreed within ±0.03 at matched slab thicknesses.
- **Vitreous ice:** quenched from liquid at 300 K to 80 K at 1, 3 and 10 K/ns, the three spectra averaged. Its
  spectrum shows the two diffraction rings of vitreous ice, near 3.6 and 2.1 Å.
- **Liquid water:** 300 K, three runs averaged.
- **Slabs:** 50 Å (glass) and 49 Å (liquid). The thick limit is reached by 50 Å: from 50 to 79 Å the spectrum
  moves by at most 0.02 relative to the tabulated GRF.
- **Range:** q from about 0.01 to 1.99 Å⁻¹. Below the first knot P is held at its first value; above the last
  it is zero; between knots it is linear. Nothing in the box size resolves q below about 0.01 Å⁻¹.

## V0

`V0` is the table's own integral, ∫ P(q) 2πq dq with the rule above, which is what the engine normalises the
spectrum by. Giving the preset sets `variance_constant` to it unless one is given. The simulated power above the
last knot is left out of V0: the engine draws nothing there, and including it made every windowed variance 8 %
too high.

## Density

The simulated vitreous ice is slightly denser than low-density amorphous ice (0.96–0.97 against 0.94 g/cm³). The
spectrum is per molecule, and the engine multiplies by the number of molecules the configured density gives, so
the difference in density is carried by the configured density, not by the table.

## Validation

Plain ice rendered with each preset, projected slice by slice, against the simulated variance per molecule
windowed to the pixel's band:

| pixel size (Å) | vitreous ice: preset ÷ MD | liquid water: preset ÷ MD |
|---|---|---|
| 0.5–4 | 0.977–1.007 | 0.990–1.020 |

The in-band shape of the rendered spectrum follows the table to 0.12–0.25 % (rms of the log), which is the
noise of the measurement.

## Caveats

The water model is rigid, and the simulated quench is several orders of magnitude faster than plunge
freezing. The scattering factors are those of independent atoms, so bonding is not represented.
