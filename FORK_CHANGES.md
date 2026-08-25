# Parakeet fork

Fork of [rosalindfranklininstitute/parakeet](https://github.com/rosalindfranklininstitute/parakeet).

Adapts Parakeet for variable-exposure tilt-series simulation, closer instrument fidelity, and
output that flows into downstream tomography processing without hand-built metadata.

## What this fork changes

| Change | State | Description |
|---|---|---|
| [Output data type](fork/OUTPUT_FORMATS.md) | shipped | `output.dtype` config plus a `--dtype` flag; `float16` default halves file size. Integer output uses one mapping for the whole file; multi-frame needs an explicit `intensity_range` |
| [Gold fiducials](fork/fiducials.md) | shipped | Colloidal gold markers at bulk density; `parakeet sample add_fiducials`, and a step in `parakeet run` |
| Direct CTF + GPU FFT | shipped | Optics stage evaluates the CTF without building a MULTEM engine per frame, and runs its FFTs through CuPy when available. A regime-bounded approximation of MULTEM's coupled envelope — see the `_fast_ctf` docstring and `tests/test_fast_ctf.py` |
| [Support film](fork/support_film.md) | **not wired up** | Model and config exist; nothing reads them yet |
| Variable per-tilt dose | not started | Native per-tilt dose in tilt-series mode |

Proposals for unbuilt work, and the research behind these changes, are kept with the project's
development notes rather than in this repository.

## Build

See [BUILD_NOTES.md](BUILD_NOTES.md) for source build instructions. Key issue: upstream PyPI packages (`python-multem`, `guanaco`) are source-only and must be compiled with the correct `CMAKE_CUDA_ARCHITECTURES` for your GPU.

- Local (Ubuntu 22.04, Pascal-class GPU): `bash install.sh`
- RHEL 8, Ada-class GPU: use the site install script under `parakeet_frontend/simulation/`

## Upstream

- Repository: https://github.com/rosalindfranklininstitute/parakeet
- Publication: Parkhurst et al., RFI/Diamond Light Source
- License: GPLv3
