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

> **`simulation.margin: 0` is forced by both scripts and must stay that way.** The GRF slab is
> z-offset from the atoms by `margin × pixel_size`, so any other value silently compares two
> different regions.

## What the comparison shows

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
- **Refit the spectrum.** `a1, a2, m1, m2, s1, s2` in `IceParameters` are the parameters that
  control the shape, and `spectrum.py`'s atomic arm is the target to fit them to. That is a
  Parakeet-side config change, not a C++ one, so it can be tried immediately.
- **Then re-run `moments.py`.** If the shape is right, the variance should follow without
  touching the amplitude constants.

## Limits of this comparison

The reference is **MULTEM's explicit water**, not reality. It establishes internal consistency —
whether the fast path reproduces the slow one — and nothing more. Whether either matches a real
micrograph is a separate question, and the protocol for it is in the project's development notes
(`ICE_EXPERIMENTAL_VERIFICATION.md`, experiments V1 and V2).

The explicit-water arm also uses randomly placed molecules rather than a hard-sphere-packed or
equilibrated configuration, so its own short-range order is only as good as that placement. The
low-frequency comparison is unaffected by this; the high-frequency comparison near 2.88 Å is the
part to treat with most caution, and is worth repeating with `pack=True`.
