# Output data type

Simulation output is written as `float16` by default in this fork, and the type is selectable
from the config or the command line. Upstream writes `float32` unconditionally.

## Why

A tilt series is written once and read many times. At K3 dimensions a 39-tilt series is 3.5 GB
in `float32`, and for most downstream use the extra precision is not recoverable from the shot
noise, so file size is the binding constraint rather than precision. Downstream tools (RELION,
WarpTools, IMOD) read `int16` and `float16` MRC, and counting detectors produce integer data
natively.

## Config

```yaml
output:
  dtype: float16   # float16 (default) | float32 | int16 | uint16
```

`Output` and `OutputDType` are defined in `src/parakeet/config.py`; `Config.output` carries them.

## Command line

`--dtype` is accepted by `parakeet run`, `parakeet simulate optics` and
`parakeet simulate image`, and overrides the config value:

```
parakeet simulate image -c config.yaml -o optics.h5 -i image.mrc --dtype float32
parakeet simulate image -c config.yaml -o optics.h5 -i image.mrc --dtype int16
```

It is **not** accepted by `simulate exit_wave`, which always writes complex data.

## Integer output

`float16` and `float32` are written as-is. `int16` and `uint16` go through
`IntegerRescalingProxy` (`src/parakeet/io.py`), which linearly maps the data onto the full
integer range so relative contrast survives the conversion.

Frames are written one at a time, so the mapping is **fixed for the whole file** rather than
recomputed per frame — otherwise every tilt would be normalised independently and intensities
could not be compared between them, which would erase exactly the difference a variable-exposure
series is built to carry.

```yaml
output:
  dtype: uint16
  intensity_range: [0.0, 4.0]   # the (min, max) mapped onto the integer range
```

- **With `intensity_range`** — that range is used for every frame. Because it lives in the config
  rather than in the data, the conversion can be inverted exactly. Values outside it clip.
- **Single-frame output without one** — the range is taken from that frame, which is unambiguous.
- **Multi-frame output without one** — writing **raises**. Set a range, or write `float16`
  (the default) or `float32`.

Pinned by `tests/test_output_dtype.py`, including a test that two frames of differing brightness
remain distinguishable through a round trip.

## Size Impact

Per-tilt image at 5760 x 4092 pixels:

| dtype | Bits/px | Size per image | vs float32 | 39-tilt series | 10 series |
|-------|---------|---------------|------------|----------------|-----------|
| float32 | 32 | 89.6 MB | 1.00x | 3.5 GB | 35.0 GB |
| **float16** | **16** | **44.8 MB** | **0.50x** | **1.7 GB** | **17.5 GB** |
| int16 | 16 | 44.8 MB | 0.50x | 1.7 GB | 17.5 GB |
| uint16 | 16 | 44.8 MB | 0.50x | 1.7 GB | 17.5 GB |
| int8 | 8 | 22.4 MB | 0.25x | 0.9 GB | 8.7 GB |

At K3 5760 x 4092 = 23,570,880 pixels per image.

**Default: `float16`** -- halves output size with negligible quality loss for typical cryo-EM contrast levels. MRC mode 12 (float16) is supported by RELION 4+, IMOD 4.11+, and mrcfile 1.4+. Upstream Parakeet default (`float32`) remains available via `--dtype float32`.


## Compatibility


- `float16` is the new default (fork only) -- halves disk usage
- Upstream Parakeet default (`float32`) available via `--dtype float32`
- `int16` MRC (mode 1) is universally supported by RELION, IMOD, WarpTools, ChimeraX
- `float16` MRC (mode 12) requires MRC 2014-aware readers (RELION 4+, IMOD 4.11+, mrcfile 1.4+)


## Not implemented

MRC stack output, MDOC metadata export and preserved TIFF bit depth were proposed alongside
this and have **not** been built. Those proposals live outside the repo, in the project's
`parakeet_dev/design/OUTPUT_FORMATS_FUTURE.md`.

## Upstream behaviour, for reference


### Output dtype

All simulate commands hardcode `dtype=np.float32` when creating the writer:

```python
# src/parakeet/simulate/_image.py:239-244
writer = parakeet.io.new(
    image_file,
    shape=simulation.shape,
    pixel_size=simulation.pixel_size,
    dtype=np.float32,       # <-- hardcoded
)
```

The `MrcFileWriter` (`src/parakeet/io.py:580`) accepts a dtype parameter and handles downcasting:

```python
# Existing downcasting logic (io.py:594-602)
if dtype == "int32":    dtype = np.dtype(np.int16)
elif dtype == "uint32": dtype = np.dtype(np.uint16)
elif dtype == "float64":dtype = np.dtype(np.float32)
elif dtype == "complex128": dtype = np.dtype(np.complex64)
```

MRC format modes supported by `mrcfile`:

| MRC mode | dtype | Bits/pixel | Use case |
|----------|-------|------------|----------|
| 0 | int8 | 8 | Masks, labels |
| 1 | int16 | 16 | Detector counting mode output |
| 2 | float32 | 32 | Current Parakeet default |
| 6 | uint16 | 16 | Unsigned counting mode |
| 12 | float16 | 16 | Half-precision (MRC 2014 ext) |

### Output file structure

Parakeet writes one file per simulation stage:
- `sample.h5` -- atom model
- `exit_wave.h5` -- complex exit wave (intermediate)
- `optics.h5` -- post-CTF (intermediate)
- `image.mrc` -- final detector image

For tilt series (`scan.mode = "tilt_series"` or `"dose_symmetric"`), all tilts are written as frames in a single file. For variable-dose simulation via `simulate_batch.py`, each tilt is a separate `image.mrc` in its own directory.

### NexusWriter dtype

The `NexusWriter` (`src/parakeet/io.py:753`) already accepts and respects the dtype parameter -- it creates the HDF5 dataset with whatever dtype is passed. No changes needed for NeXus output.

### ImageWriter dtype

The `ImageWriter` (`src/parakeet/io.py:837`) always rescales to `uint8` for PNG/JPEG/TIFF output (`io.py:890`). This is appropriate for preview images.

