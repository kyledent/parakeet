# Benchmark: 70S ribosomes in 100 nm ice

A matched pair of simulations of the same specimen — *E. coli* 70S ribosomes suspended in
100 nm of vitreous ice with gold fiducials — imaged two ways: as a single untilted exposure,
and as a dose-symmetric tilt series.

The point of the pairing is that the specimen is identical, so anything that differs between
the two outputs is attributable to the acquisition rather than to the sample. That makes this
useful for more than timing: comparing what a single 40 e/Å² exposure recovers against what
123 e/Å² spread over 41 tilts recovers, on a specimen whose ground truth is known exactly.

## Before the first run

```bash
benchmark/check_config.py benchmark/configs/*.yaml
```

Parakeet places particles by relaxing them apart under a non-overlap constraint and a box
margin. When the field is too crowded the relaxation runs out of iterations and returns what it
has **without complaining**, so particles end up straddling the box wall and are clipped — which
appears as hard edges and CTF ringing in the image rather than as an error. `check_config.py`
compares the requested particle count against the structure's measured bounding radius and says
so first. It is worth running after any edit to a config's field size or particle count; the
shipped configs pass.

## Quick start

```bash
# smallest, run this first -- 512x512, 3 ribosomes, one image
benchmark/run_benchmark.sh spa_70s_small

# the tilt-series equivalent, 9 tilts
benchmark/run_benchmark.sh tilt_series_70s_small

# the full-size versions
benchmark/run_benchmark.sh spa_70s          --outdir /scratch/$USER/bench
benchmark/run_benchmark.sh tilt_series_70s  --outdir /scratch/$USER/bench

# collect what the runs cost
benchmark/summarise.py /scratch/$USER/bench
benchmark/summarise.py /scratch/$USER/bench --markdown   # for pasting into a write-up
```

Each run writes `image.mrc` plus a `result.json` recording the per-stage wall clock, the
specimen parameters and the GPU it ran on. **Quote numbers from `summarise.py`, not from this
README** — that is the only way they stay true when a config, a card or the code changes.

Or drive the config directly, without the wrapper:

```bash
parakeet run -c benchmark/configs/spa_70s_small.yaml \
    -s sample.h5 -e exit_wave.h5 -o optics.h5 -i image.mrc
```

The wrapper exists only to run the stages separately so each can be timed; `parakeet run`
does the same work in one call.

## Generating a config from acquisition parameters

The four shipped configs are a fixed pair. `make_config.py` writes new ones from the parameters
you would actually set at a microscope, and does the conversions in between — aperture diameter
to cutoff frequency, total dose and dose rate to per-image dose and exposure time, filter state
to inelastic model:

```bash
# 300 keV, K3, 100 nm ice, 100 um aperture, 20 eV slit in, 40 e/A^2 at 8 e/px/s
benchmark/make_config.py -o my_spa.yaml

# Falcon 4i, slit retracted, 150 nm ice, 41-tilt series
benchmark/make_config.py --detector falcon4i --slit retracted --ice 150 \
    --mode tilt-series --tilts 41 -o my_ts.yaml

# 200 keV, MPL-centred filter so the slit width actually does something
benchmark/make_config.py --kv 200 --filter-model mpl --slit-width 20 \
    --aperture 70 -o glacios.yaml
```

`--microscope {krios,glacios,talos,tundra}` sets voltage, Cs, Cc and source energy spread
together; any of them can still be overridden individually:

| preset | kV | Cs | Cc | source dE |
|---|---|---|---|---|
| `krios` | 300 (also 200, 100) | 2.7 mm | 2.7 mm | 0.9 eV |
| `glacios` | 200 | 2.7 mm | 2.7 mm | 0.9 eV |
| `talos` | 200 (Arctica), 120 (L120C) | 2.7 mm | 2.7 mm | 0.9 eV |
| `tundra` | 100 | 2.7 mm | 2.7 mm | 1.0 eV |

These are the standard uncorrected Thermo objective values and nominal source spreads — starting
points, not a calibration of any individual column. Note the generator deliberately does **not**
emit Parakeet's own `microscope.model` field: setting that to `krios` or `talos` makes
`microscope.new()` overwrite the beam energy with 300 or 200 whatever the config says, silently
discarding your `--kv`.

`--help` lists everything else. The parameters it covers: `--kv`, `--detector {k3,falcon4i}`,
`--pixel-size`, `--ice` (nm), `--aperture` (µm diameter, 0 to retract), `--slit
{inserted,retracted}`, `--slit-width` (eV), `--filter-model
{zero-loss,mpl,cc-corrected,none}`, `--total-dose` (e/Å²), `--dose-rate` (e/px/s), `--dqe`,
`--mode {spa,tilt-series}` and the specimen options.

**It also tells you when the simulation will not follow what you asked for**, as warnings on
stderr and in a header comment in the generated file. There are three worth knowing about
before you trust an output:

| You set | What actually happens |
|---|---|
| `--slit-width` with the default `--filter-model zero-loss` | **Nothing.** That model scales the image by `exp(−t/λ)` and never reads the width; a 10 eV and a 40 eV slit give bit-identical images. Use `--filter-model mpl` for the width to matter |
| `--kv` other than 300 | The Landau loss distribution follows the voltage, but the **inelastic mean free path is hardcoded at 3150 Å for 300 keV**, so the elastic/inelastic split is that of a 300 keV beam |
| `--detector` with `--dqe` | Parakeet has **one** DQE curve, measured at 300 keV, not per-detector presets. The detector choice sets array size and pixel size only — noise transfer is identical for K3 and Falcon 4i |
| `--dose-rate` without `--dqe` | **Nothing.** The rate sets `scan.exposure_time`, and the only physics reading it is the DQE lookup; shot noise follows the dose, not the rate |
| `--dose-rate` outside 1.5–5 e/px/s | Clamped. The DQE table has exactly two rate columns, so any realistic counting rate sits at the 5 e/px/s endpoint. There is no coincidence loss or counting saturation modelled anywhere |

None of this is a defect in the generator; it is what the simulator does, established by reading
and measuring the code. The full review is in the project's development notes as
`ENERGY_FILTER_REVIEW.md`, along with the scripts that measure it.

## The configs

| Config | Field | Particles | Acquisition | Total dose |
|---|---|---|---|---|
| `spa_70s.yaml` | 2048² px @ 1.0 Å | 14 ribosomes + 6 fiducials | one untilted image | 40 e/Å² |
| `spa_70s_small.yaml` | 1024² px | 3 + 2 | one untilted image | 40 e/Å² |
| `tilt_series_70s.yaml` | 2048² px | 14 + 6 | dose-symmetric, −60→+60°, 3° steps (41) | 123 e/Å² |
| `tilt_series_70s_small.yaml` | 1024² px | 3 + 2 | dose-symmetric, −60→+60°, 15° steps (9) | 27 e/Å² |

The `_small` variants exist so the first run is cheap and fits a modest card. They are the same
specimen and the same optics — only the field, the particle count and the tilt sampling change.

Particle counts are set from the structure's **measured** bounding radius (153 Å for 4V5D, so a
307 Å diameter) at roughly a third of touching density, which is what `check_config.py` checks.
An earlier draft of these configs asked for three ribosomes in a 512 Å field; that is over twice
touching density, the placement gave up, and particles were clipped at the walls.

## The specimen

- **Particles** — PDB **4V5D**, the *E. coli* 70S ribosome: 148,020 atoms, roughly
  215 × 252 × 265 Å. Parakeet fetches it by id on the first run and caches it. Instances are
  placed at random positions and orientations.
- **Ice** — 1000 Å (100 nm) of vitreous ice via the implicit GRF model, which is the sample
  box thickness in `sample.box[2]`.
- **Fiducials** — 10 nm colloidal gold, placed on **both ice surfaces**, as they are on a real
  grid. This uses the fork's `sample.fiducials` config and the `sample.add_fiducials` step.
- **Optics** — 300 kV, Cs 2.7 mm, Schottky FEG spread, zero-loss energy filtering. Defocus
  differs by regime: −2 µm for the single exposure, −3 µm for the series, matching normal
  practice for each.

**On the fiducials in the SPA config.** Gold beads are not part of a normal single-particle
experiment. They are there deliberately, for two reasons: they give a high-contrast object of
known size and position in the same field as the particles, which is useful for exercising
pickers and contrast normalisation; and keeping them makes the two configs differ *only* in
acquisition. If you want a conventional SPA field, delete the `sample.fiducials` block — the
`sample.add_fiducials` step then reports that there is nothing to do and moves on.

**On 100 nm ice for SPA.** That is thick for single particle work, where 30–60 nm is more
usual. It is specified here because the pairing requires it: the tilt series needs the
thickness, and the comparison only means something if both see the same specimen. Expect the
single exposure to look worse than a real SPA micrograph would, and for that to be the ice
rather than a mistake.

## Cost and sizing

`exit_wave` dominates — it is the multislice propagation, and everything after it is cheap by
comparison. That is why `run_benchmark.sh` times the stages separately rather than reporting
one total: a change that halves the optics stage barely moves the wall clock, and the summary
table shows you that directly.

Cost scales with field **area** and with the number of images, but **not linearly in the
number of images**: measured on the small pair, 9 tilts cost 35.8 s of exit wave against 7.4 s
for a single image — 4.8×, not 9×, because the per-image setup is amortised. Do not assume the
41-tilt config costs 41× the single exposure; run it and read the number.

If a run exhausts GPU memory, reduce `microscope.detector.nx`/`ny` and the matching
`sample.box`/`shape` entries together — they describe the same field and must agree.

## A caveat you will see immediately: ice edge artefacts

Run `spa_70s_small` and the image has a strong vertical band with symmetric fringes, plus
banding along the top and bottom edges. **This is not the configs and not your GPU.** Running
the identical sample with `simulation.ice: false` produces a completely clean image — same
particles, same fiducials, same optics — so the artefact belongs entirely to the implicit GRF
ice model.

This reproduces open upstream issue
[#87](https://github.com/rosalindfranklininstitute/parakeet/issues/87), *"Rad damage modelling
and oscillating bands at the edges when ice=True"*. These configs are a small, self-contained
case that triggers it in about half a minute, which is more useful than a prose description if
anyone wants to work on it.

What it means for using the benchmark:

- **Timing is unaffected** — the artefact costs nothing, so the numbers stand.
- **Particle-level work is mostly unaffected** away from the bands; the ribosomes and fiducials
  themselves are rendered correctly.
- **Anything sensitive to the background** — contrast normalisation, background estimation, a
  picker's false-positive rate near the edges — will be measuring the artefact as well as the
  specimen. Set `simulation.ice: false` to separate the two, at the cost of losing the realistic
  background entirely.

## Output

`image.mrc` is written as `float16` by default, which halves the file against `float32`
with no meaningful loss at these contrast levels. Override per run with `--dtype`.

If you want `int16`/`uint16` for a **tilt series**, you must also set `output.intensity_range`
in the config. Without it the write is refused, on purpose: frames are written one at a time,
so a per-frame range would rescale every tilt independently and destroy the intensity
relationship between them — which is exactly what a dose-varying series is carrying. See
`fork/OUTPUT_FORMATS.md`.
