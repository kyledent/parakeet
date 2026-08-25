# Parakeet Build Notes

## System (tested 2026-03-19)

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04.5 LTS (kernel 6.8.0-106-generic) |
| GPU | Pascal-class GPU (sm_61, 8 GB VRAM) |
| NVIDIA driver | 570.211.01 |
| CUDA toolkit | 12.8 (`/usr/local/cuda-12.8/`) |
| GCC | 11.4.0 (system) |
| Python | 3.10.20 (conda) |

## Build overview

Parakeet has three compiled components:

1. **python-multem** (PyPI, source-only) -- CUDA + FFTW multislice simulation engine. This is the hard one.
2. **guanaco** (PyPI, source-only) -- CUDA + FFTW reconstruction/denoising library.
3. **parakeet_ext** (in-repo) -- C++ pybind11 module containing `SpherePacker` for hard-sphere atom packing. No CUDA needed.

All three are source-only distributions with no pre-built wheels on PyPI. They require CUDA toolkit, FFTW3, and a C++ compiler at build time.

## Issues encountered and fixes

### 1. python-multem and guanaco fail to find CUDA during pip install

**Problem:** `pip install python-multem` downloads the source tarball and runs cmake, but cmake cannot find `nvcc` because CUDA is not on `$PATH` in the pip build environment.

**Fix:** Set `PATH=/usr/local/cuda-12.8/bin:$PATH` before running pip install, and set `CMAKE_CUDA_ARCHITECTURES` to match your GPU:
```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CMAKE_CUDA_ARCHITECTURES=61  # dev GPU = sm_61
```

### 2. python-multem fails with build isolation (cmake dependency loop)

**Problem:** pip's build isolation installs cmake into a temporary venv, but `python-multem`'s `pyproject.toml` lists cmake as both a build dependency AND something the isolated cmake needs to bootstrap. This circular dependency causes the build to fail with a cryptic error.

**Fix:** Install build dependencies first, then use `--no-build-isolation`:
```bash
pip install cmake ninja wheel numpy setuptools-scm
pip install python-multem --no-build-isolation
```

### 3. guanaco has the same CUDA/isolation issues

**Fix:** Same approach -- ensure CUDA is on PATH and install with `--no-build-isolation`.

### 4. parakeet pybind11 submodule not initialised

**Problem:** The cloned repo has an empty `pybind11/` directory (git submodule pointer only).

**Fix:**
```bash
cd parakeet/
git submodule update --init --recursive
```

### 5. CUDA architecture must be set explicitly

**Problem:** If `CMAKE_CUDA_ARCHITECTURES` is not set, cmake defaults to `OFF`, which either picks a wrong architecture or fails entirely.

**Fix:** Set via environment variable. Common values:

| GPU | Compute capability | CMAKE_CUDA_ARCHITECTURES |
|-----|-------------------|-------------------------|
| Pascal-class | 6.1 | 61 |
| RTX 2070/2080 | 7.5 | 75 |
| RTX 3070/3080 | 8.6 | 86 |
| RTX 4070/4080 | 8.9 | 89 |
| Ada-class | 8.9 | 89 |
| RTX 5070/5080 | 12.0 | 120 |

### 6. FFTW3 must be installed system-wide

**Problem:** Both python-multem and guanaco require FFTW3 development headers.

**Fix (Ubuntu/Debian):**
```bash
sudo apt install libfftw3-dev
```

## Quick install (from scratch)

```bash
# Prerequisites (Ubuntu)
sudo apt install libfftw3-dev build-essential

# Create conda env
conda create -n parakeet python=3.10 -y
conda activate parakeet

# Set CUDA and GPU arch
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CMAKE_CUDA_ARCHITECTURES=61  # adjust for your GPU

# Install build tools
pip install cmake ninja wheel numpy setuptools-scm

# Build CUDA dependencies from source
pip install python-multem --no-build-isolation
pip install guanaco --no-build-isolation

# Init submodules and install parakeet (editable)
cd parakeet/
git submodule update --init --recursive
pip install -e . --no-build-isolation
```

## Verification

```bash
conda activate parakeet
python -c "
import parakeet; print('parakeet', parakeet.__version__)
import multem; print('multem', multem.__version__)
from parakeet_ext import SpherePacker; print('SpherePacker OK')
"
# Should print:
# parakeet 0.6.8
# multem 0.4.1
# SpherePacker OK

# Full pipeline test:
parakeet config new -c /tmp/test_config.yaml
parakeet sample new -c /tmp/test_config.yaml -s /tmp/test_sample.h5
parakeet sample add_molecules -c /tmp/test_config.yaml -s /tmp/test_sample.h5
parakeet simulate exit_wave -c /tmp/test_config.yaml -s /tmp/test_sample.h5 -e /tmp/test_ew.h5
parakeet simulate optics -c /tmp/test_config.yaml -e /tmp/test_ew.h5 -o /tmp/test_op.h5
parakeet simulate image -c /tmp/test_config.yaml -o /tmp/test_op.h5 -i /tmp/test_image.mrc
```
