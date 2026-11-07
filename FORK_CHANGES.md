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
| Pinned multislice engine | shipped | `setup.py` pins python-multem to one commit of [kyledent/python-multem](https://github.com/kyledent/python-multem), which pins [kyledent/MULTEM](https://github.com/kyledent/MULTEM) |
| Band-limited atomic potentials | shipped (in the engine) | MULTEM's per-atom tables lost most of each atom's projected potential at cryo-EM pixel sizes (oxygen keeps ~15% at 1.5 A/px), in a share that depends on the element. The pinned engine band-limits each atom's potential to the grid, so it is deposited in full at any pixel size, and the GRF ice follows. On by default; `MULTEM_BANDLIMITED_POTENTIAL=0` restores the old behaviour. See the python-multem README |
| GRF ice on the GPU | shipped (in the engine) | The engine's pipelined GPU slice loop skipped the external potential, so a GPU exit wave carried no GRF ice and no static B factor; the CPU path and the projected potential were unaffected. The pinned engine sends those cases through the original loop |
| Atom lists from arrays | shipped | `AtomData.to_multem` hands the atom columns to the engine as arrays (`AtomList.from_arrays`) instead of building one Python tuple per atom |
| Seedable ice | shipped (in the engine) | The engine's maskers take a random-field seed (`Masker.set_seed`, `MultiRegionMasker.set_seed`): with one, the same field and slicing give the same ice; the default stays a fresh field per simulation |
| Atom-type table cache | shipped (in the engine) | The engine builds its per-element potential tables once per process for each potential type, grid spacing and band limit, instead of on every simulation (`MULTEM_ATOM_TYPE_CACHE`, on by default; `0` turns it off). Results are unchanged |
| Ice set-up once per grid | shipped (in the engine) | With the band-limited potential on, the engine computed the ice's variance fraction afresh on every slice, about 1.5 ms each; it now keeps it per grid spacing and ice spectrum. Results are unchanged |
| Vacuum regions | shipped (in the engine) | A multi-region masker region with density 0 adds nothing but can still be subtracted by other regions, for holes and cracks (`MultiRegionMasker.supports_vacuum_regions`) |
| Tabulated ice spectrum | shipped (in the engine) | `IceParameters.set_spectrum_table(q, P)` replaces the ice's two-Gaussian power spectrum with a table (needs the band-limited potential); the single-ice path now also honours the `m0` and `v0` overrides |
| [Support film](fork/support_film.md) | **not wired up** | Model and config exist; nothing reads them yet |
| Variable per-tilt dose | not started | Native per-tilt dose in tilt-series mode |

Proposals for unbuilt work, and the research behind these changes, are kept with the project's
development notes rather than in this repository.

## Build

See [BUILD_NOTES.md](BUILD_NOTES.md) for source build instructions. Key issue: upstream PyPI packages (`python-multem`, `guanaco`) are source-only and must be compiled with the correct `CMAKE_CUDA_ARCHITECTURES` for your GPU.

- Local (Ubuntu 22.04, Pascal-class GPU): `bash install.sh`
- Other systems and GPUs: follow BUILD_NOTES.md, setting `CMAKE_CUDA_ARCHITECTURES` for the GPU

## Upstream

- Repository: https://github.com/rosalindfranklininstitute/parakeet
- Publication: Parkhurst et al., RFI/Diamond Light Source
- License: GPLv3
