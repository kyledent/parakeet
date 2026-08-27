# Detector MTF and DQE

Upstream Parakeet models the detector as **one DQE curve measured at 300 keV**, applied as a
Fourier-space multiplication onto the signal: `detector.dqe` is a boolean, there is no MTF at
all, no per-detector selection, and the `energy` argument to `dqe_table` is accepted and unused.
Choosing K3 or Falcon 4i changes the array size and the pixel size and nothing about the noise
transfer.

This fork keeps the DQE table and adds the MTF, applies each where it belongs, and fixes the
frequency axis. What that changes, and how it was checked, is in
[What is implemented here](#what-is-implemented-here) below. The survey that follows first is
what the decision was based on, and is left as it was written.

`mtf_survey.py` reports what real curves are available and what they say.

```bash
validation/detector/mtf_survey.py --relion-data ~/github_repos/relion/data
```

## The interchange format is already settled

RELION reads MTF as a two-column STAR file — `_rlnResolutionInversePixel`, `_rlnMtfValue`, with
q from 0 to 0.5 where 0.5 is the physical Nyquist — and **both Gatan and Thermo publish in it**.
A detector model here should read those files rather than embed its own tables, so that a curve
obtained from a vendor drops straight in.

## What is on disk

RELION ships seven curves. MTF at selected spatial frequencies, in inverse pixels:

| detector | MTF@0.1 | @0.25 | @0.4 | @0.49 |
|---|---|---|---|---|
| Gatan K2, 300 kV | 0.899 | 0.733 | 0.522 | 0.401 |
| Direct Electron DE-20, 300 kV | 0.826 | 0.570 | 0.333 | 0.216 |
| Thermo Falcon 3EC, 200 kV | 0.876 | 0.552 | 0.242 | 0.136 |
| Thermo Falcon 3EC, 300 kV | 0.900 | 0.572 | 0.251 | 0.153 |
| Thermo Falcon 4EC, 200 kV | 0.845 | 0.475 | 0.168 | 0.075 |
| Thermo Falcon 4EC, 300 kV | 0.892 | 0.535 | 0.213 | 0.095 |
| Thermo Falcon 2, 300 kV | 0.601 | 0.219 | 0.088 | 0.055 |

## Voltage dependence, which is the point of having more than one curve

| detector | q | 200 kV | 300 kV | ratio |
|---|---|---|---|---|
| Falcon 4EC | 0.10 | 0.845 | 0.892 | 0.947 |
| Falcon 4EC | 0.25 | 0.475 | 0.535 | 0.889 |
| Falcon 4EC | 0.40 | 0.168 | 0.213 | **0.789** |
| Falcon 3EC | 0.25 | 0.552 | 0.572 | 0.964 |
| Falcon 3EC | 0.40 | 0.242 | 0.251 | 0.962 |

Both detectors are worse at 200 kV, and **the deficit grows with spatial frequency** — lower
energy electrons spread further laterally in the sensor, widening the point spread and damping
high frequencies harder. Falcon 4 loses 21% of its MTF at 0.4 Nyquist going from 300 to 200 kV,
where Falcon 3 loses 4%.

Any per-detector model has to carry voltage with it. A single curve cannot serve a Krios and a
Glacios, and it certainly cannot serve a Tundra.

## MTF is not a figure of merit on its own

At 300 kV, Falcon 4EC has a **lower** MTF than Falcon 3EC at every frequency — 0.535 against
0.572 at quarter Nyquist. Falcon 4's 14 µm pixel is a deliberate trade: a larger pixel spreads
each event over a smaller fraction of a pixel and raises the single-event signal-to-noise, which
improves **DQE** while lowering MTF. A model that swapped in "the better detector" by MTF alone
would get the ranking backwards.

This is why the pairing matters: **MTF describes signal transfer, DQE describes signal-to-noise
transfer, and only the second says whether an image is more useful.** Parakeet currently has a
DQE and no MTF; the useful target is both, per detector, per voltage.

## What is missing

| curve | where it is |
|---|---|
| Gatan K3, 200 kV, standard | gatan.com K3 page, behind a request form |
| Gatan K3, 200 kV, **CDS** | as above |
| Gatan K3, 300 kV, standard | as above |
| Gatan K3, 300 kV, **CDS** | as above |
| Thermo Falcon 4i, any voltage | not published in STAR form; 4EC is the closest proxy |
| Anything at 100 kV | not shipped by RELION; Gatan's Alpine is the part designed for 100–200 kV |

**The K3 curves are the ones worth asking for.** Gatan publishes four — 200 and 300 kV, each in
standard and CDS mode — and CDS is a different detector response rather than a processing option,
so it needs its own curve. They sit behind a request form, which a person has to fill in.

K3 in CDS mode at 200 keV is a common working configuration, and is exactly one of the four.

## A caution before fitting anything to these

MTF measured on a counting detector is not a straightforward linear response. The counting
process is non-linear, so a measured MTF is strictly valid only at the very low dose rate it was
measured at, and coincidence loss changes the effective response at working rates.

Applying one MTF at all dose rates makes the same assumption Parakeet's single DQE curve already
makes. That is probably acceptable, but it should be stated rather than inherited silently —
particularly since Parakeet's DQE table *does* interpolate over dose rate, across a range of
1.5 to 5 e/px/s that clamps at both ends.


## What is implemented here

Three things were wrong or missing, and all three are visible in a measurement rather than in an
image.

### 1. The DQE was read at half the frequency it should have been

`parakeet.dqe.DQETable` tabulates against **fraction of Nyquist**, running 0 to 1.
`DQETable.dqe_fs` built its grid in **cycles per pixel**, where Nyquist is 0.5, and interpolated
one against the other. Every pixel therefore received the DQE belonging to half its frequency,
which raises the DQE everywhere and cannot be seen by eye. At Nyquist the table says 0.322 and
the code applied 0.628.

The grid now carries the factor of two, and the response is zero beyond Nyquist, where the table
has no data and the detector cannot respond. The corners of a square grid sit at √2 Nyquist and
were previously being given a DQE by extrapolation.

### 2. The DQE was applied where the MTF belongs

The old code multiplied the noiseless image by the DQE in Fourier space and *then* drew Poisson
noise. That attenuates the signal without touching the noise, so the delivered DQE — the thing
the number is defined to be — comes out as **DQE²/DQE(0)** rather than DQE. A detector specified
at 0.27 at Nyquist behaved like one at 0.09.

`parakeet.mtf.detect` does the three things a detector does, in order:

1. **Arrival.** Poisson at the delivered dose. This is the beam's noise, not the detector's.
2. **Charge spreading.** Blur by the MTF. This blurs the *realised counts*, so signal and noise
   are attenuated together and the signal-to-noise is unchanged — which is why a blur alone
   leaves DQE = 1, and why an MTF is not a DQE.
3. **The detector's own noise.** Whatever is left between an ideal detector and the measured
   DQE, added with the spectrum that makes the answer come out right:

       NPS_total(q) = MTF(q)² N / DQE(q)     so the added noise has power MTF²N(1/DQE − 1)

Passing neither curve reduces to plain Poisson, which is what Parakeet did before either
existed.

### 3. There was no MTF

`parakeet.mtf` reads RELION's two-column STAR format — the interchange format both Gatan and
Thermo publish in — so a curve obtained from a vendor drops straight in:

```yaml
microscope:
  detector:
    dqe: true
    mtf: falcon4EC_300kV          # a built-in preset
    mtf: /path/to/mtf_k3_200kV.star   # or any RELION MTF star file
```

Seven presets carry analytic fits to the curves RELION distributes, so a config can name a
detector without shipping a file. Each preset knows what it is and how good the fit is:

| preset | a | b | c | RMS residual | worst |
|---|---|---|---|---|---|
| `k2_300kV` | 0.7470 | 0.0192 | 0.9195 | 0.0065 | 0.0141 |
| `falcon2_300kV` | 0.3473 | 0.0938 | 0.2736 | 0.0144 | 0.0470 |
| `falcon3EC_200kV` | 0.3864 | 0.0834 | 0.9160 | 0.0042 | 0.0163 |
| `falcon3EC_300kV` | 0.3977 | 0.1392 | 0.9161 | 0.0046 | 0.0197 |
| `falcon4EC_200kV` | 0.3357 | 0.0666 | 0.9038 | 0.0029 | 0.0070 |
| `falcon4EC_300kV` | 0.3582 | 0.0699 | 0.9624 | 0.0038 | 0.0084 |
| `de20_300kV` | 0.5189 | 0.0735 | 0.7824 | 0.0071 | 0.0268 |

for `MTF(q) = sinc(q)·[c·exp(−(q/a)²) + (1−c)/(1+(q/b)²)]`, q in cycles per pixel. Reproduce
with `fit_mtf.py`; the fits are in [`results/mtf_fits.json`](results/mtf_fits.json). Falcon 2 is
the poorest fit at 0.014 RMS — it is a scintillator device and its MTF is a different shape from
the counting detectors'.

A named MTF that is neither a preset nor an existing file **raises** rather than falling back to
none, because an ignored MTF is indistinguishable in the output from a detector that has none.

### Checking it

```bash
validation/detector/delivered_dqe.py --repeats 24 --json results/delivered_dqe.json --plot dqe.png
```

The delivered DQE is measured the way a real one is, in two parts: noise power from a flat
field, signal transfer from a white random pattern averaged over many realisations. A flat field
alone is not enough — above DC it carries no signal, so a filter applied to the signal only is
invisible in it, which is exactly the bug in (2).

At 512², 24 fields, Falcon 4EC at 300 kV and 40 e/pixel:

| model | delivered DQE |
|---|---|
| ideal (Poisson only) | 1.000, within 0.011 |
| MTF only | 1.000, within 0.016 |
| MTF + DQE | tracks the tabulated curve, worst error **0.011** |
| legacy (DQE onto the signal) | worst error **0.200**, tracing DQE²/DQE(0) |

`tests/test_mtf.py` holds the same checks as assertions, including the two that would fail if
the curves were swapped: MTF alone must not change signal-to-noise, and MTF + DQE must reproduce
the table.

### What this does not do

- **One MTF at all dose rates.** Counting is non-linear, so a measured MTF is strictly valid
  only near the rate it was measured at. Parakeet's DQE table does interpolate over dose rate
  (1.5–5 e/px/s, clamped at both ends); the MTF does not. Stated rather than inherited silently.
- **No voltage interpolation.** Presets are named per voltage because interpolating between them
  does not work: fitting Falcon 3EC at 250 kV from the 200 and 300 kV curves misplaces the
  parameters by 2.7%, 56.4% and 1.0%, and rescaling in q leaves a residual of 0.045 against
  0.008 for a direct fit. Name the curve you have.
- **No K3 curve.** Still behind Gatan's request form. `falcon4EC_300kV` is the closest shipped
  proxy and is not the same detector.
