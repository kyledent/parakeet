# Support film and foil hole geometry

> **Status: the model exists but is not wired up.** `src/parakeet/sample/_support_film.py` and
> the `SupportFilm` config model are in the tree, and `parakeet.sample.support_film` is
> importable, but **nothing reads `config.sample.support_film`** — there is no CLI verb and no
> step in `parakeet run`. Setting it in a config has no effect today. It is committed in this
> state deliberately, so the gap is visible rather than implied.

Real specimens sit on a support film — holey carbon, gold foil, graphene oxide — which scatters
strongly wherever the beam passes through it. Upstream simulates a free-standing slab of ice,
which is adequate for the centre of a large hole and wrong near its edge.

```yaml
sample:
  support_film:
    enabled: true
    material: carbon      # carbon | gold | none
    thickness: 120        # Angstroms (12 nm, typical Quantifoil)
    hole_diameter: 12000  # Angstroms (1.2 um)
    density: null         # override g/cm^3; inferred from material if null
```

Multi-position acquisition across a hole, and the ice meniscus model, were proposed with this
and are **not** here; they live in the project's `parakeet_dev/design/SUPPORT_FILM_FUTURE.md`.

The sections below are the original design notes.

## Motivation

Real cryo-EM specimens exist on support films -- holey carbon (Quantifoil, C-flat), gold (UltrAuFoil), or graphene oxide. The film strongly scatters electrons and contributes to the image wherever the beam passes through it. Parakeet currently simulates specimens as free-standing slabs of ice and particles with no surrounding support, which is adequate for simulating the centre of a large hole but does not capture:

1. **Edge effects** at hole boundaries -- the abrupt transition from ice-only to ice+foil where contrast and thickness change dramatically
2. **Foil scattering background** -- carbon/gold film contributes phase and amplitude contrast that affects CTF fitting and particle picking
3. **Multi-position acquisition** -- real data collection images multiple positions across a hole, including near the edge; template matching and particle picking must handle varying background
4. **Thickness variation** -- ice meniscus is typically thinner at the centre and thicker at the edges near the foil

Simulating the support film is essential for realistic micrograph simulation, particularly for validating particle picking algorithms (pytom-match-pick, GAPSTOP-TM) and CTF estimation near hole edges.

## Current Behaviour

### Sputter Coating (existing)

Parakeet has a sputter coating model (`src/parakeet/sample/_sputter.py`) that places atoms of a specified element (C, Ir, Cr, Pt) at bulk density in a uniform layer above the sample. Key characteristics:

- Random atom placement (no overlap checking)
- Uniform layer across the entire sample area
- Only applied to the top surface (`WARNING - ONLY APPLYING TO TOP OF SAMPLE`, line 176)
- Number density computed from element properties (molar mass, bulk density, Avogadro's number)
- Atoms added to the sample's HDF5 atom table via `sample.add_atoms()`

This is the closest existing capability to support film simulation, but it produces a blanket coating rather than a foil-with-hole geometry.

### Implicit Ice (GRF model)

The Gaussian Random Field ice model (`simulation.ice = True`) generates a random potential field matching amorphous ice statistics during multislice propagation. It is controlled by `IceParameters` (config.py line 718):

- Two-Gaussian power spectrum: m1=0, m2=1/2.88, s1=0.731, s2=0.081, a1=0.199, a2=0.801
- Density: 0.94 g/cm^3
- Computed by MULTEM's `Masker` object -- no explicit atoms, just a potential contribution per slice

This is efficient (no millions of atoms) and physically reasonable (amorphous materials are well-described by their power spectrum and mean inner potential).

### Sample Geometry

Parakeet supports three sample shapes (config.py lines 55-130):
- **Cube** -- single length parameter
- **Cuboid** -- length_x, length_y, length_z
- **Cylinder** -- length, radius, axis, optional x/z offsets

None of these describe a "thin slab with a hole" geometry.

### MULTEM Masker API

The Masker object generates per-slice 2D binary masks for GRF ice placement:

- `multem.Masker(nx, ny, pixel_size)` -- creates masker
- `masker.set_ice_parameters(ice_params)` -- accepts **one** set of parameters
- `masker.set_cuboid(origin, size)` or `set_cylinder(...)` -- accepts **one** shape
- `masker.compute(z_start, z_end)` -- returns flattened int32 array (0/1 mask)
- `multem.simulate(sys_conf, input, masker)` -- C++ consumes masker internally

The Masker supports only a single shape and a single set of IceParameters. It cannot natively produce spatially varying densities (e.g. ice in one region, carbon in another) within a single simulation pass.

## Specimen Geometry

Real specimen preparation is **asymmetric**:

1. A thin support film (carbon or gold foil) with holes is the structural grid
2. A droplet of sample (protein in buffer) is applied to **one side** of the foil
3. Blotting removes excess liquid
4. Vitrification freezes the remaining thin film of buffer

The resulting geometry:

```
Side view (not to scale):

     ════════════╗     ╔════════════    ← Foil (top surface, 12-50 nm)
     ────────────┘     └────────────    ← Foil bottom
     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    ← Ice (extends BELOW foil, 50-200 nm)
     ░░░░░░░░░░░ OOO ░░░░░░░░░░░░░░    ← Particles suspended in ice
     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     ────────────────────────────────

     |←  foil  →|←hole→|← foil   →|
     |←──────── full sample area ────────→|

Top view:

     ┌──────────────────────────┐
     │  FOIL (carbon/gold)      │
     │     ┌────────────┐       │
     │     │            │       │
     │     │    HOLE    │       │
     │     │  (ice only)│       │
     │     │            │       │
     │     └────────────┘       │
     │                          │
     └──────────────────────────┘
```

**Critical for tilt-series:** The ice layer extends across the **full sample area**, including under the foil. At high tilt angles, the beam projects through the ice at oblique angles that extend well beyond the hole boundary. If ice were only placed inside the hole, the projected thickness would be incorrect at tilt -- the edges of the hole would show an abrupt density change that doesn't exist in reality. The ice must fill the full cuboid volume so that projected potential is correct at all tilt angles.

The foil sits on top of the ice in the annular region outside the hole. Both the foil (explicit atoms) and ice (GRF) contribute to the projected potential, just as particle atoms and GRF ice coexist additively in the hole region.

## Design notes

### 1. Support Film Config Model

Add a `SupportFilm` model to `config.py`:

```yaml
sample:
  support_film:
    enabled: true
    material: carbon      # carbon | gold | none
    thickness: 120        # Angstroms (12 nm typical for Quantifoil)
    hole_diameter: 12000  # Angstroms (1.2 um)
    hole_shape: circle    # circle (only option initially)
```

Material presets (auto-populated from `material` if density not overridden):

| Material | Element | Z | Molar mass (g/mol) | Density (g/cm^3) | Mean inner potential (V) |
|----------|---------|---|---------------------|-------------------|--------------------------|
| carbon (amorphous) | C | 6 | 12.01 | 2.0 | 10.7 |
| gold | Au | 79 | 196.97 | 19.3 | 30.8 |
| graphene_oxide | C/O | 6/8 | ~14 | ~1.8 | ~8 |

Pydantic model:

```python
class SupportFilmMaterial(str, Enum):
    carbon = "carbon"
    gold = "gold"
    none = "none"

class SupportFilm(BaseModel):
    enabled: bool = Field(False, description="Enable support film simulation")
    material: SupportFilmMaterial = Field(
        "carbon", description="Support film material"
    )
    thickness: float = Field(
        120, description="Film thickness in Angstroms (12 nm typical for Quantifoil)"
    )
    hole_diameter: float = Field(
        12000, description="Hole diameter in Angstroms (1.2 um = 12000 A)"
    )
    hole_shape: str = Field(
        "circle", description="Hole shape (circle)"
    )
    density: Optional[float] = Field(
        None, description="Override density (g/cm^3). Auto-set from material if None."
    )
```

Add to `Sample` model:

```python
class Sample(BaseModel):
    ...
    support_film: Optional[SupportFilm] = Field(
        None, description="Support film (holey carbon/gold) parameters"
    )
```

### 2. Implementation: GRF Ice (Full Area) + Explicit Foil Atoms

The implementation uses the existing GRF ice framework unchanged -- the Masker cuboid covers the **entire sample area** at ice density. The foil is then added as explicit atoms in the annular region, the same way particles are explicit atoms in the hole. Both coexist additively with the GRF ice potential.

**Why this works:** In Parakeet, particle atoms and GRF ice already coexist in the same volume with no steric exclusion. The GRF generates a random potential field for amorphous ice, and explicit atoms (proteins, etc.) add their potential on top. The foil atoms work identically -- they are explicit atoms placed in the annular region outside the hole, and their potential adds to the GRF ice potential in that region.

**Physical justification:** In reality, the foil region contains solid carbon/gold with no ice. But the GRF ice potential in that region is a minor perturbation compared to the foil's much higher scattering (carbon: 2.0 g/cm^3 vs ice: 0.94 g/cm^3; gold: 19.3 g/cm^3 vs ice: 0.94 g/cm^3). The dominant visual effect -- strong scattering from the foil outside the hole, weak scattering from ice inside the hole -- is captured correctly. For quantitative accuracy, the ice contribution in the foil region could be subtracted in a future refinement, but for realistic micrograph generation this approximation is adequate.

**Implementation steps:**

#### Step 1: GRF Ice Masker (no changes needed)

The existing Masker is set to a **cuboid covering the full sample area**:

```python
# engine.py masker() method -- existing behaviour, no modification needed
# The cuboid already covers the full sample area (box_x, box_y, box_z)
masker = multem.Masker(nx, ny, pixel_size)
masker.set_ice_parameters(ice_params)  # density=0.94, standard power spectrum
masker.set_cuboid(origin, (box_x, box_y, box_z))  # full sample area
```

GRF ice fills the entire cuboid volume. At all tilt angles, projected ice thickness is correct everywhere.

#### Step 2: Explicit Foil Atoms (new module)

Create `src/parakeet/sample/_support_film.py` that places carbon or gold atoms in the annular foil region, adapting the existing sputter coating approach from `_sputter.py`:

```python
def generate_support_film(sample, config):
    """Place explicit atoms in the annular foil region (outside hole)."""

    # Foil geometry
    hole_radius = config.hole_diameter / 2  # Angstroms
    centre_x = sample.box[0] / 2
    centre_y = sample.box[1] / 2

    # Foil z-range: top of ice slab
    # Ice fills [0, ice_thickness], foil sits on top at [ice_thickness, ice_thickness + foil_thickness]
    z_min = sample.box[2] - config.thickness  # top of specimen
    z_max = sample.box[2]

    # Material properties
    if config.material == "carbon":
        element, Z, molar_mass = "C", 6, 12.011
        density = config.density or 2.0  # g/cm^3
    elif config.material == "gold":
        element, Z, molar_mass = "Au", 79, 196.967
        density = config.density or 19.3  # g/cm^3

    # Number density (atoms per Angstrom^3)
    N_A = 6.022e23
    density_A3 = density * 1e-24  # g/cm^3 -> g/A^3
    number_density = (density_A3 * N_A) / molar_mass  # atoms/A^3

    # Annular volume: full box area minus hole, times foil thickness
    box_area = sample.box[0] * sample.box[1]  # A^2
    hole_area = np.pi * hole_radius**2  # A^2
    foil_area = box_area - hole_area  # A^2
    foil_volume = foil_area * config.thickness  # A^3

    n_atoms = int(number_density * foil_volume)

    # Generate random positions in annular region (rejection sampling)
    atoms = []
    while len(atoms) < n_atoms:
        batch = 2 * n_atoms  # oversample for rejection
        x = np.random.uniform(0, sample.box[0], batch)
        y = np.random.uniform(0, sample.box[1], batch)
        z = np.random.uniform(z_min, z_max, batch)

        # Reject points inside hole
        r_sq = (x - centre_x)**2 + (y - centre_y)**2
        outside_hole = r_sq > hole_radius**2

        valid = np.column_stack([x, y, z])[outside_hole]
        atoms.append(valid)

    coords = np.concatenate(atoms)[:n_atoms]

    # Add to sample HDF5 via sample.add_atoms()
    sample.add_atoms(
        atomic_number=np.full(n_atoms, Z, dtype=np.int32),
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        sigma=np.zeros(n_atoms),  # no B-factor
        occupancy=np.ones(n_atoms),
        charge=np.zeros(n_atoms, dtype=np.int32),
    )
```

#### Step 3: Particle Placement Constraint

Particles should only be placed **inside the hole**, not in the foil region. Modify `distribute_particles_uniformly()` in `sample/distribute.py` to reject candidate positions outside the hole:

```python
# In distribute.py, after generating candidate position:
if support_film is not None:
    r_sq = (x - centre_x)**2 + (y - centre_y)**2
    if r_sq > hole_radius_sq:
        continue  # reject -- particle would be in foil region
```

### Summary of Approach

```
┌─────────────────────────────────────────────────────┐
│                    FULL SAMPLE AREA                  │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  GRF Ice (full cuboid, density 0.94 g/cm^3) │    │
│  │  ═══════════╗         ╔═══════════════════   │    │
│  │  Explicit C ║  HOLE   ║ Explicit C atoms    │    │
│  │  atoms      ║(ice+    ║ (foil, 2.0 g/cm^3)  │    │
│  │  (foil)     ║particles║                      │    │
│  │  ═══════════╝         ╚═══════════════════   │    │
│  │                                              │    │
│  │  GRF ice continues below foil everywhere     │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  At tilt: beam sees correct ice thickness at all    │
│  angles because GRF fills full volume. Foil atoms   │
│  add scattering in annular region.                  │
└─────────────────────────────────────────────────────┘
```

**What each pixel sees in projection:**

| Region | 0 deg tilt | High tilt |
|--------|-----------|-----------|
| Hole centre | Ice + particles | Ice + particles (thicker projected) |
| Hole edge | Ice + particles | Ice + particles + possibly foil (beam enters foil at oblique angle) |
| Foil region | Ice + foil atoms | Ice + foil atoms (both thicker projected) |

**Files to modify:**
- `src/parakeet/config.py` -- add `SupportFilm` model, `SupportFilmMaterial` enum
- `src/parakeet/sample/distribute.py` -- constrain particle placement to hole region

**Files to create:**
- `src/parakeet/sample/_support_film.py` -- explicit atom placement in annular foil region

**Files to register:**
- `src/parakeet/sample/__init__.py` -- call `generate_support_film()` after particle placement
- `src/parakeet/command_line/sample/__init__.py` -- pass `support_film` config through CLI

## Atom Counts and Performance

**Foil atom counts (approximate):**

| Grid type | Material | Thickness | Hole diam | Film area | Atoms |
|-----------|----------|-----------|-----------|-----------|-------|
| Quantifoil R1.2/1.3 | Carbon | 12 nm | 1.2 um | 5.1 um^2 | ~12 M |
| Quantifoil R2/2 | Carbon | 12 nm | 2.0 um | 3.4 um^2 | ~8 M |
| UltrAuFoil R1.2/1.3 | Gold | 50 nm | 1.2 um | 5.1 um^2 | ~15 M |

These counts are significant. For carbon, the multislice cost scales with atom count, so foil atoms will increase simulation time. Mitigations:
- Only place atoms in the region visible to the detector (crop to FOV + margin)
- For gold, consider coarse-graining (fewer atoms with scaled B-factors)
- The GRF ice is unaffected -- it uses the same full-cuboid Masker regardless of whether foil atoms exist

**Full specimen size (Quantifoil R1.2/1.3, 2.5 um sample width):**

| Component | Atoms | sample.h5 impact |
|-----------|-------|-------------------|
| 75 ribosomes (4YBB) | ~15 M | ~600 MB |
| GRF ice (implicit) | 0 | 0 (computed on-the-fly) |
| Carbon foil (12 nm) | ~12 M | ~480 MB |
| **Total** | **~27 M** | **~1.1 GB** |

## Compatibility with Existing Functionality

- **Sputter coating**: The support film is separate from sputter coating. Sputter coating adds a thin layer on top of the specimen (e.g. Ir for tomography). Support film is the structural grid. Both can be used simultaneously.
- **Implicit ice (GRF)**: The GRF Masker is used unchanged -- full cuboid, standard ice parameters. Foil atoms are additive, exactly like particle atoms.
- **Sample shapes**: The cuboid shape should be used with support film, sized to encompass the hole + surrounding foil. The support film defines additional geometric constraints (annular region) within the cuboid.
- **Tilt-series mode**: Support film applies per-tilt like everything else. At high tilt angles, the projected foil thickness increases (1/cos effect), which is physically correct and automatically handled by multislice. GRF ice across the full area ensures correct projected thickness at all angles.

## Common Grid Types

Reference parameters for commercially available grids:

| Grid | Material | Film thickness | Hole diameter | Hole spacing | Mesh |
|------|----------|---------------|---------------|-------------|------|
| Quantifoil R1.2/1.3 | Amorphous C | 12 nm | 1.2 um | 1.3 um (centre-to-centre) | Cu 300 |
| Quantifoil R2/2 | Amorphous C | 12 nm | 2.0 um | 2.0 um | Cu 300 |
| C-flat CF-2/2 | Amorphous C | 20 nm | 2.0 um | 2.0 um | Cu 400 |
| UltrAuFoil R1.2/1.3 | Gold | 50 nm | 1.2 um | 1.3 um | Au 300 |
| UltrAuFoil R0.6/1.0 | Gold | 50 nm | 0.6 um | 1.0 um | Au 300 |
| Graphene oxide on Quantifoil | C + GO | 12 nm + 1-2 nm | 1.2 um | 1.3 um | Cu 300 |

## Implementation Priority

1. **Config model** (`SupportFilm`, `SupportFilmMaterial`) -- straightforward Pydantic addition
2. **Explicit foil atoms** -- adapt sputter code with circular hole mask, place in annular region
3. **Particle placement constraint** -- reject particles outside hole in `distribute.py`
4. **Multi-position mode** -- extend simulate_batch.py with detector offset grid
5. **Ice meniscus** -- future enhancement

## References

- Vulovic et al. (2013) Image formation modeling in cryo-electron microscopy. J Struct Biol 183: 19-32 (mean inner potentials)
- Russo & Passmore (2014) Electron microscopy: Ultrastable gold substrates for electron cryomicroscopy. Science 346: 1377-1380 (UltrAuFoil)
- Parkhurst et al. (2021) Parakeet: a digital twin software pipeline. Open Biology 11: 210160
