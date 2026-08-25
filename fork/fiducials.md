# Gold fiducial nanoparticle simulation

Adds colloidal gold fiducial markers to the sample, so simulated tilt series can exercise a
fiducial-based alignment pipeline. Available as `parakeet sample add_fiducials` and as a
`sample.add_fiducials` step in `parakeet run`, which is skipped when the config carries no
fiducial parameters.

```yaml
sample:
  fiducials:
    diameter: 100        # Angstroms (100 = 10 nm)
    count: 20
    placement: surface   # surface | both_surfaces | volume
    occupancy: 1.0       # >1 boosts scattering to offset GRF ice overlap
```

Implemented in `src/parakeet/sample/_fiducials.py`, with the config model in `config.py` and the
command wired up in `command_line/sample/_fiducials.py`. Gold atoms are placed at bulk density
(19.3 g/cm^3) in spherical clusters, by rejection sampling within each sphere.

The sections below are the original design notes, kept because they record the physical
reasoning and the numbers behind the choices.

## Motivation

Gold fiducial markers (typically 6-10 nm colloidal gold) are routinely applied to cryo-EM grids to provide high-contrast reference points for tilt-series alignment. In IMOD/AreTomo/WarpTools alignment workflows, fiducials are tracked across tilt angles to solve the projection geometry. They appear as dense, high-contrast spheres in tomographic reconstructions.

Parakeet currently has no mechanism to simulate fiducial markers. Adding this capability enables:

1. **End-to-end pipeline validation** -- simulated tilt series with known fiducial positions allow testing fiducial-based alignment (IMOD `autofidseed`, AreTomo fiducial tracking) against ground truth
2. **Fiducial-based vs patch-tracking comparison** -- quantify alignment accuracy differences between fiducial tracking and patch tracking on identical data
3. **Realistic training data** -- simulated tomograms with fiducials improve neural network training for particle picking (fiducials are a common false positive source)
4. **Variable exposure validation** -- verify that high-contrast fiducials are correctly dose-weighted across the HybridExposure variable exposure tilt series (30/3 and 50/3 e-/A^2 schemes)

## Physical Properties of Gold Fiducial Markers

### Material

- **Element:** Gold (Au), Z = 79
- **Molar mass:** 196.967 g/mol
- **Bulk density:** 19.3 g/cm^3
- **FCC lattice parameter:** a = 4.078 A (not modelled -- see Approximations)
- **Number density:** (19.3 * 6.022e23) / 196.967 = 5.90e22 atoms/cm^3 = **0.059 atoms/A^3**

### Commercial Fiducial Sizes

| Nominal diameter | Radius (A) | Volume (A^3) | Au atoms |
|-----------------|-----------|-------------|----------|
| 5 nm | 25 | 65,450 | 3,861 |
| 6 nm | 30 | 113,097 | 6,673 |
| 10 nm | 50 | 523,599 | 30,892 |
| 15 nm | 75 | 1,767,146 | 104,262 |

At 20 fiducials of 10 nm, that is ~618k Au atoms -- well within Parakeet's atom budget (samples routinely contain millions of water molecules).

### Placement in Real Specimens

In practice, colloidal gold solution is applied to the grid either before or after sample vitrification. Fiducials typically:

- Adhere to one or both surfaces of the ice layer (top and/or bottom)
- Occasionally embed within the ice
- May cluster in groups of 2-5 particles
- Are distributed at variable density (10-50 per field of view at typical concentrations)
- Sometimes aggregate near the carbon/foil edge

For simulation, we support three placement modes:

1. **`surface`** -- fiducials rest on the top surface of the ice (z_max - radius). Most common in real specimens.
2. **`both_surfaces`** -- fiducials on both top and bottom surfaces. Used when gold is applied to both sides.
3. **`volume`** -- fiducials distributed randomly throughout the ice volume. Less common but occurs with some sample preparation methods.

## Current Parakeet Architecture

### Atom Addition API

All atom placement goes through `sample.add_atoms(AtomData(data=df))` where `df` is a pandas DataFrame:

```python
pandas.DataFrame({
    "atomic_number": np.full(n, 79, dtype="uint8"),   # Au = 79
    "x": x_coords,                                      # float32, Angstroms
    "y": y_coords,
    "z": z_coords,
    "sigma": np.full(n, 0.085, dtype="float32"),        # B-factor proxy
    "occupancy": np.full(n, 1.0, dtype="float32"),
    "charge": np.zeros(n, dtype="int8"),
})
```

### Pipeline Stages

The `parakeet run` pipeline currently executes:

1. `sample.new` -- create ice slab
2. `sample.add_molecules` -- place PDB/local molecules

Sputter coating and support film are standalone commands (`parakeet sample sputter`, `parakeet sample support_film`), not part of the `run` pipeline.

**Fiducials will be integrated into the `run` pipeline** as a new step `sample.add_fiducials` between `sample.add_molecules` and simulation. This ensures fiducials are always included when using `parakeet run`. They will also be available as a standalone command `parakeet sample add_fiducials`.

### Existing Precedents

| Module | Approach | Relevance |
|--------|----------|-----------|
| `_sputter.py` | Random atoms at bulk density in a flat layer | Density calculation, `create_atom_data` pattern |
| `_support_film.py` | Au/C atoms in annular region, rejection sampling | Gold material properties, coarse-graining |
| `_add_molecules.py` | Place PDB structures, delete overlapping ice | Ice displacement via `del_atoms`, position distribution |
| `distribute.py` | Elastic collision dynamics for non-overlapping placement | Could be used for fiducial spacing |

## Design notes

### Config Model

Add to `config.py`:

```python
class Fiducial(BaseModel):
    """
    A model to describe gold fiducial nanoparticles for tilt-series alignment
    """

    diameter: float = Field(
        100,
        description="Fiducial diameter in Angstroms (e.g. 60 for 6 nm, 100 for 10 nm)",
        gt=0,
    )

    count: int = Field(
        10,
        description="Number of fiducial particles to place",
        gt=0,
    )

    placement: str = Field(
        "surface",
        description=(
            "Where to place fiducials: "
            "'surface' (top of ice), "
            "'both_surfaces' (top and bottom), "
            "'volume' (random in ice volume)"
        ),
    )
```

Add to `Sample` model:

```python
class Sample(BaseModel):
    ...
    fiducials: Optional[Fiducial] = Field(
        None, description="Gold fiducial nanoparticle parameters."
    )
```

### YAML Config Example

```yaml
sample:
  box: [4000, 4000, 1200]
  centre: [2000, 2000, 600]
  shape:
    type: cuboid
    cuboid:
      length_x: 4000
      length_y: 4000
      length_z: 1200

  molecules:
    pdb:
      - id: "4v5d"
        instances: 50

  fiducials:
    diameter: 100    # 10 nm gold
    count: 15        # 15 fiducials
    placement: surface  # on top of ice
```

### Implementation (`sample/_fiducials.py`)

New file: `src/parakeet/sample/_fiducials.py`

#### Algorithm

For each fiducial:

1. **Choose position** based on placement mode:
   - `surface`: random (x, y) within sample, z = z_max - radius (sitting on top surface)
   - `both_surfaces`: alternate between z_max - radius and z_min + radius
   - `volume`: random (x, y, z) within sample bounds, clamped so sphere stays inside

2. **Check spacing** -- reject positions where the fiducial would overlap another fiducial (centre-to-centre distance < diameter). Use simple pairwise distance check (N is small, typically < 50).

3. **Generate gold atoms** within a sphere:
   - Compute number of atoms from bulk gold density and sphere volume
   - Generate random positions within the sphere using rejection sampling (uniform in cube, reject outside sphere) or the r^3 root method
   - Translate to fiducial centre position

4. **Add atoms** to sample via `sample.add_atoms(AtomData(data=df))`

#### Gold Atom Generation Within a Sphere

Random uniform placement at bulk density within a sphere:

```python
# Number of atoms
radius = diameter / 2.0
volume = (4.0 / 3.0) * np.pi * radius**3
n_atoms = int(AU_NUMBER_DENSITY * volume)

# Uniform random points in sphere (rejection sampling)
# Generate in cube [-r, r]^3, reject outside sphere
oversample = 2.0  # pi/6 acceptance ratio for sphere in cube
n_gen = int(n_atoms * oversample)
while collected < n_atoms:
    xyz = np.random.uniform(-radius, radius, size=(n_gen, 3))
    r_sq = np.sum(xyz**2, axis=1)
    inside = r_sq <= radius**2
    xyz = xyz[inside]
    # take what we need...

# Translate to fiducial centre
xyz += centre_position
```

### Pipeline Integration

#### `_run.py` Changes

Add `sample.add_fiducials` to the pipeline between molecule addition and simulation:

```python
sample_steps = ["sample.new", "sample.add_molecules", "sample.add_fiducials"]

# ... in execution:
if "sample.add_fiducials" in steps:
    sample = parakeet.sample.add_fiducials(config.sample, sample)
```

#### `command_line/_main.py` Changes

Register `parakeet sample add_fiducials` subcommand.

#### `sample/__init__.py` Changes

Add `from parakeet.sample._fiducials import *  # noqa`

### Ground Truth Output

Fiducial positions are stored in the sample HDF5 file. They can be extracted for ground truth comparison with alignment results. The positions array records the centre of each fiducial in Angstroms.

Each fiducial's atoms are added with `group=-1` (same as sputter/support_film atoms), distinguishing them from molecule atoms which have positive group IDs.

Future enhancement: store fiducial metadata (positions, diameters) as a dedicated HDF5 dataset for easy extraction without parsing atom coordinates.

## Approximations and Limitations

1. **No crystalline structure** -- real gold nanoparticles have FCC crystal structure (a = 4.078 A). We use random atom placement at bulk density instead. At cryo-ET resolution (typically >= 5 A/px), the difference is negligible -- individual atoms are not resolved, only the projected potential matters.

2. **No surface ligands** -- commercial colloidal gold is stabilised with citrate or tannic acid surface coatings (1-3 nm shell). These organic ligands are not modelled. They contribute minimal additional scattering compared to the gold core.

3. **No size distribution** -- real colloidal gold has a narrow size distribution (CV ~10-15%). All fiducials are modelled as identical spheres. A future enhancement could add a `diameter_sigma` parameter for log-normal size variation.

4. **No clustering** -- fiducial aggregation is not modelled. Each fiducial is placed independently with only overlap avoidance. A future enhancement could add a `cluster_probability` parameter.

5. **GRF ice overlap (contrast impact)** -- The GRF implicit ice Masker generates ice potential across the entire sample volume, including inside fiducial spheres. It has no mechanism to exclude regions. This adds ice potential on top of gold potential, reducing relative contrast. The `occupancy` config parameter (default 1.0) allows boosting gold scattering to compensate: values of 1.5-2.0 are recommended when using GRF ice. For explicit atomic ice, `del_atoms` could be added in future to properly displace water molecules from fiducial volumes.

6. **Objective aperture required for realistic contrast** -- In a real microscope, gold fiducials derive most of their contrast from amplitude contrast: gold scatters electrons to high angles, and the objective aperture blocks them. Without enabling the aperture (`outer_aper_ang: 50` for 100 um), fiducials will appear significantly fainter than in real micrographs. This is a Parakeet config issue, not a fiducial implementation issue.

7. **Uniform inelastic MFP** -- The inelastic scattering model uses ice MFP (3150 A at 300 keV) for all materials. Gold's MFP is ~100-200 A -- gold absorbs far more electrons. This per-material absorption contrast is not captured. The `occupancy` boost partially compensates.

## Atom Count Budget

For context on whether fiducials significantly impact simulation performance:

| Component | Typical count | Atoms |
|-----------|--------------|-------|
| 75 ribosomes (70S) | 75 * 200k | 15,000,000 |
| GRF ice (implicit) | -- | 0 |
| 15 fiducials (10 nm) | 15 * 31k | 465,000 |
| Sputter coating (5 nm Ir) | -- | ~200,000 |

Fiducials add <3% to the atom count. No performance concern.

## Files

| File | Change |
|------|--------|
| `src/parakeet/config.py` | Add `Fiducial` model, add `fiducials` field to `Sample` |
| `src/parakeet/sample/_fiducials.py` | **New file.** Core generation and placement logic |
| `src/parakeet/sample/__init__.py` | Import `_fiducials` |
| `src/parakeet/_run.py` | Add `sample.add_fiducials` step |
| `src/parakeet/command_line/_run.py` | Add `sample.add_fiducials` to `--steps` choices |
| `src/parakeet/command_line/sample/_fiducials.py` | **New file.** CLI subcommand |
| `src/parakeet/command_line/sample/__init__.py` | Import `_fiducials` |
| `src/parakeet/command_line/_main.py` | Register `add_fiducials` subcommand |

## References

- Iancu et al. (2006) "A comparison of liquid nitrogen and liquid helium as cryogens for electron cryotomography" J Struct Biol 153:231-240 -- fiducial alignment methodology
- Mastronarde (2005) "Automated electron microscope tomography using robust prediction of specimen movements" J Struct Biol 152:36-51 -- IMOD fiducial tracking
- Zheng et al. (2022) "AreTomo" J Struct Biol X 6:100068 -- fiducial-based vs patch-tracking alignment
