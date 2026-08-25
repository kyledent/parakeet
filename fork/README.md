# Fork design notes

What this fork changes, and why. One document per feature that exists in the code.

| Document | Feature | State |
|---|---|---|
| [OUTPUT_FORMATS.md](OUTPUT_FORMATS.md) | Configurable output data type (`float16` default, `float32`/`int16`/`uint16`) | shipped |
| [fiducials.md](fiducials.md) | Gold fiducial nanoparticles, `parakeet sample add_fiducials` | shipped |
| [support_film.md](support_film.md) | Holey carbon / gold foil support with a circular hole | **model present, not wired into the pipeline** |

The GPU FFT backend and the direct CTF evaluation are documented in the code itself —
see the module docstring of `src/parakeet/simulate/_fast_ctf.py` for how it relates to MULTEM
and where the two diverge, and `tests/test_fast_ctf.py` for the pinned behaviour.

Proposals for work that has **not** been done — detector MTF presets, energy-filter and
objective-aperture enhancements, the composite GRF masker, MDOC and MRC-stack export,
multi-position acquisition — are deliberately kept out of this repository, along with the
background research behind them. They live with the project's development notes.
