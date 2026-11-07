# GRF ice validation

The implicit ice model (`simulation.ice: true`) replaces water molecules with a random field
rescaled to the mean and variance of amorphous ice. These scripts check that substitution
against the thing it stands in for, using the fact that **Parakeet can build the same specimen
both ways** — explicit water molecules, or the random field — so the comparison needs no
molecular dynamics package, no literature value and no microscope.

```bash
validation/grf/moments.py  --outdir /scratch/$USER/grf   # mean and variance vs sampling
validation/grf/spectrum.py --outdir /scratch/$USER/grf   # where in frequency the difference is
```

Both build the same box twice — explicit water with the GRF off, an empty box with the GRF on —
project with `parakeet simulate potential`, and compare. Results land in `results/`.

> Both scripts force `simulation.margin: 0`. That was needed while the GRF slab sat
> `(margin + padding) × pixel_size` above the atoms in z; the ice now moves laterally only, as the
> atoms do, and `z_register.py` (below) shows it in a render.

```bash
validation/grf/z_register.py --outdir /scratch/$USER/grf    # is the ice where the shape is, in z?
validation/grf/edge_bands.py --outdir /scratch/$USER/grf    # the edge bands of upstream issue #87
```

`z_register.py` renders the potential of an empty ice cuboid at margins 0 and 100 and compares the
z-range MULTEM slices with the shape's. `edge_bands.py` renders the same ice twice, once as a shape
that ends inside the simulation grid (the field) and once as a shape covering the whole grid, and
measures the bands near the field's edges in the exit-wave phase and in the image.

**A property of the masker to know when reading either:** it tests each slice at its *midpoint*
against the half-open shape `[z0, z1)`. A slice whose midpoint falls exactly on a face is decided
by rounding, and the rounding moves with the masker's rotation centre, which moves laterally with
the margin. In `z_register.py`'s geometry the last slice (248–252 Å, midpoint 250 Å, the shape's top
face) carries ice at margin 0 and none at margin 100, with the same z-range computed in both. The
script reports such slices as boundary slices and does not count them against the register.

## With the band-limited potential (the engine default)

Measured with the pinned engine, `MULTEM_BANDLIMITED_POTENTIAL` on, against explicit water placed
at random and by hard-sphere packing (`--pack`, `sample.ice.pack`). The JSON is in `results/`
(`*_2026-11-07.json`). The same run with the band-limited potential off reproduces the older
numbers below to three or four figures (atomic mean 11.34 / 6.514 / 1.092 at 0.5 / 1 / 2 Å/px),
so the two sets can be compared directly.

### The mean no longer depends on sampling, and the GRF matches it

| sampling | atomic mean, random | atomic mean, packed | GRF mean | GRF ÷ random | GRF ÷ packed |
|---|---|---|---|---|---|
| 0.5 Å/px | 13.60 | 13.47 | 13.68 | 1.006 | 1.016 |
| 1.0 Å/px | 13.59 | 13.43 | 13.68 | 1.007 | 1.018 |
| 1.5 Å/px | 13.61 | 13.44 | 13.68 | 1.005 | 1.018 |
| 2.0 Å/px | 13.60 | 13.42 | 13.68 | 1.006 | 1.019 |
| 2.5 Å/px | 13.60 | 13.41 | 13.68 | 1.006 | 1.020 |

The steep sampling dependence below (11.3 → 1.09) belonged to MULTEM's atomic tables, and it has
gone: each atom now keeps its whole projected potential at any pixel size. The packed box's mean
is 1.3 % lower than the random box's. That is what the packer leaving some molecules unplaced
would give, but the molecules were not counted in this run.

### The variance is short, mostly at low frequency

Per-slice variance, GRF ÷ atomic (`moments.py`):

| sampling | vs random water | vs packed water |
|---|---|---|
| 0.5 Å/px | 0.70 | 0.87 |
| 1.0 Å/px | 0.47 | 0.83 |
| 1.5 Å/px | 0.21 | 0.52 |
| 2.0 Å/px | 0.16 | 0.42 |
| 2.5 Å/px | 0.15 | 0.41 |

Projected through 200 Å (`spectrum.py`, 300 Å box), GRF ÷ atomic:

| | 1.0 Å/px random | 1.0 Å/px packed | 1.5 Å/px random | 1.5 Å/px packed |
|---|---|---|---|---|
| variance | 0.55 | 1.05 | 0.23 | 0.61 |
| power below 0.1 Å⁻¹ (> 10 Å) | 0.13 | 0.34 | 0.14 | 0.39 |
| power above 0.3 Å⁻¹ (< 3.3 Å) | 2.0 | 2.0 | 4.3 | 4.5 |

At 1.5 Å/px against packed water, the power ratio runs from 0.35–0.47 above 10 Å, through 0.53–0.63
at 5–7 Å and about 1 at 4 Å, to 1.4–2.1 at 3.2–3.7 Å.

- **Random placement is not a fair reference at low frequency.** Overlapping, uncorrelated molecules
  carry ideal-gas number fluctuations, which a liquid suppresses. The random arm's power above 10 Å
  is 2–3× the packed arm's (1.5 Å/px: 920–1,020 against 290–330).
- **Hard-sphere packing is not a fair reference either.** An equilibrated water model gives far less
  variance than the packed box. A molecular-dynamics reference (TIP4P/2005, liquid at 300 K and
  quenched glasses, analysed with MULTEM's own scattering factors under the band-limited potential;
  kept with the project's development notes) gives 0.84–1.01 × the GRF's variance per molecule at
  1.5 Å/px, depending on the slab thickness (49 to 10 Å). The same analysis of Parakeet's packed
  boxes gives 2.0–2.2 × the GRF, and shows why: the packer is strongly ordered, close to a lattice
  rather than a liquid. Its first oxygen–oxygen peak g_OO is ~27 at 2.71 Å, against ~3 for liquid
  water and ~8 for the glasses, and its low-frequency structure factor is 0.32–0.40, against
  0.03–0.09 for water.
- **So the shortfalls in these tables measure the explicit-water arms, not the GRF.** Against an
  equilibrated liquid, the GRF is within about 16 % at 1.5 Å/px. The same reference shows the GRF
  **overstating** the noise at coarse pixels in thick ice (0.70 × at 3 Å/px in 49 Å slabs), because
  an independent field per slice cannot reproduce water's suppression of long-range fluctuations.

## Before the band-limited potential (2026-08-26)

> Measured before the engine band-limited the atomic potential. Set
> `MULTEM_BANDLIMITED_POTENTIAL=0` to reproduce these numbers (they reproduce: see above). The
> reference arm placed water at random, which inflates its low-frequency power; see the section
> above before reading the low-frequency ratios here.

### The mean is right, including its pixel-size dependence

`Cv_mean` scales the target mean steeply with pixel area — 1.0 at fine sampling, 0.488 at
1 Å/px, 0.083 at 2 Å/px. That looks wrong, because the mean projected potential of a real medium
does not depend on how finely you sample it. Measured, it is not wrong:

| sampling | atomic mean | GRF mean | ratio |
|---|---|---|---|
| 0.5 Å/px | 11.343 | 11.471 | 1.011 |
| 1.0 Å/px | 6.515 | 6.667 | 1.023 |
| 2.0 Å/px | 1.092 | 1.133 | 1.037 |

The **atomic** path carries the same dependence, and the GRF tracks it to within 2% across a
fourfold change in pixel size. The sampling dependence belongs to MULTEM's projected-potential
representation; `Cv_mean` exists to keep implicit ice indistinguishable from explicit water
inside the simulator, and it succeeds.

One consequence worth recording: the potential array is therefore **not a potential in volts at
a fixed calibration** — its absolute scale moves with sampling — so a mean inner potential cannot
be read off it without first establishing what its units are.

### The variance is short, and the reason is the spectrum shape

| sampling | atomic variance | GRF variance | GRF shortfall |
|---|---|---|---|
| 0.5 Å/px | 539.4 | 419.9 | 22% |
| 1.0 Å/px | 100.0 | 63.3 | 37% |
| 2.0 Å/px | 1.17 | 0.39 | 67% |

The natural reading is that the amplitude is too low and `V0` or `Cv_var` needs rescaling.
`spectrum.py` shows that reading is wrong. The deficit is not flat — the GRF has **far too little
power at low frequency and too much at high frequency**, and the variance comes out low only
because the low-frequency deficit is the larger of the two:

| spatial frequency | spacing | GRF / atomic power |
|---|---|---|
| 0.036 Å⁻¹ | 28 Å | **0.070** |
| 0.098 Å⁻¹ | 10 Å | 0.083 |
| 0.160 Å⁻¹ | 6.2 Å | 0.123 |
| 0.222 Å⁻¹ | 4.5 Å | 0.314 |
| 0.295 Å⁻¹ | 3.4 Å | 1.00 — crossover |
| 0.347 Å⁻¹ | 2.9 Å | 1.59 |
| 0.419 Å⁻¹ | 2.4 Å | **2.18** |

Explicit water has a spectrum that decays monotonically from low frequency. The GRF has a
**band-pass** spectrum peaking near 0.35–0.42 Å⁻¹ — which is where the config's second Gaussian
sits, `m2 = 1/2.88 Å⁻¹`, carrying amplitude `a2 = 0.801` against the broad component's
`a1 = 0.199`.

**So the two-Gaussian parameterisation has its weight in the wrong place.** It puts four fifths
of the amplitude into the water structure-factor peak and a fifth into the broad background;
explicit water wants that roughly reversed, or wants the broad component far wider.

### Why it matters more than the variance number suggests

Across **3–10 Å — the band where particles carry structure — the simulated background is 2.2×
too weak**. Simulated ice is a quieter background than real molecules at the same density,
precisely where a picker or a denoiser does its work. Anything trained or benchmarked on this
background will look better than it should, and the error is not a uniform contrast offset that
a normalisation would absorb: it is a tilt across frequency.

## What to fix, and what not to

- **Do not rescale `V0` or `Cv_var`.** A single factor cannot fix a ratio that runs from 0.07 to
  2.18 across the band; it would trade a low-frequency deficit for a worse high-frequency excess.
- **Refit the spectrum, if at all, to an equilibrated liquid, not to either explicit-water arm.**
  `a1, a2, m1, m2, s1, s2` in `IceParameters` are the parameters that control the shape. Neither
  `spectrum.py`'s random nor its packed arm has a liquid's structure, so neither is the target. That is a
  Parakeet-side config change, not a C++ one, so it can be tried immediately.
- **Then re-run `moments.py`.** If the shape is right, the variance should follow without
  touching the amplitude constants.

## Limits of this comparison

The reference is **MULTEM's explicit water**, not reality. It establishes internal consistency —
whether the fast path reproduces the slow one — and nothing more. Whether either matches a real
micrograph is a separate question, and the protocol for it is in the project's development notes
(`ICE_EXPERIMENTAL_VERIFICATION.md`, experiments V1 and V2).

The 2026-08-26 explicit-water arm placed molecules at random. **That affects the low-frequency
comparison as well as the high**: random placement has the number fluctuations of an ideal gas,
which a liquid suppresses, and its power above 10 Å is 2–3× that of hard-sphere-packed water
(`--pack`, measured 2026-11-07; section above). Packing is not a fair reference either: it carries
about twice the variance an equilibrated water model gives. Both arms test internal consistency;
neither is the target for the GRF's constants.
