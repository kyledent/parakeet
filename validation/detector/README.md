# Detector MTF

Parakeet models the detector as **one DQE curve measured at 300 keV**, applied as a Fourier-space
multiplication: `detector.dqe` is a boolean, there is no MTF at all, no per-detector selection,
and the `energy` argument to `dqe_table` is accepted and unused. Choosing K3 or Falcon 4i changes
the array size and the pixel size and nothing about the noise transfer.

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

Note the immediate relevance: the V4 session is **K3 in CDS mode at 200 keV**, which is exactly
one of the four.

## A caution before fitting anything to these

MTF measured on a counting detector is not a straightforward linear response. The counting
process is non-linear, so a measured MTF is strictly valid only at the very low dose rate it was
measured at, and coincidence loss changes the effective response at working rates.

Applying one MTF at all dose rates makes the same assumption Parakeet's single DQE curve already
makes. That is probably acceptable, but it should be stated rather than inherited silently —
particularly since Parakeet's DQE table *does* interpolate over dose rate, across a range of
1.5 to 5 e/px/s that clamps at both ends.
