#!/usr/bin/env bash
#
# build_ubuntu22.sh -- Build Parakeet from source on an Ubuntu 22.04 workstation
#
# Workstation: Ubuntu 22.04, Pascal-class GPU (sm_61)
# CUDA:        12.8 (/usr/local/cuda-12.8/)
# GCC:         11.4.0 (system)
# Python:      3.10 (conda)
#
# This builds parakeet in editable mode so source changes (e.g. fork
# modifications to config.py, io.py, simulate/) take effect immediately
# without rebuilding. Only C++/CUDA components need a rebuild.
#
# Usage:
#   bash build_ubuntu22.sh         # full build (first time, ~5-10 min)
#   bash build_ubuntu22.sh --quick # python-only reinstall (skip CUDA, ~30s)
#
# After build:
#   conda activate parakeet
#   python -c "import parakeet; print(parakeet.__version__)"
#   cd parakeet_frontend/simulation && bash run_tests.sh --quick
#
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

CONDA_ENV_NAME="parakeet"
PYTHON_VERSION="3.10"
CUDA_PATH="/usr/local/cuda-12.8"
CUDA_ARCH="61"                    # dev GPU = sm_61

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAKEET_SRC="${SCRIPT_DIR}"

QUICK=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
fi

# ── Preflight ────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  Parakeet Build -- Ubuntu 22.04"
echo "============================================================"
echo ""
echo "  GPU:        dev GPU (sm_${CUDA_ARCH})"
echo "  CUDA:       ${CUDA_PATH}"
echo "  Source:     ${PARAKEET_SRC}"
echo "  Mode:       $($QUICK && echo 'quick (python-only)' || echo 'full (CUDA + python)')"
echo ""

# Check CUDA
if [ ! -x "${CUDA_PATH}/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at ${CUDA_PATH}/bin/nvcc"
    echo "       Install CUDA 12.8 or adjust CUDA_PATH."
    exit 1
fi

# Check FFTW
if ! ldconfig -p 2>/dev/null | grep -q libfftw3; then
    if [ ! -f /usr/include/fftw3.h ]; then
        echo "ERROR: FFTW3 not found. Install with:"
        echo "       sudo apt install libfftw3-dev"
        exit 1
    fi
fi

# Check conda
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found."
    exit 1
fi

# ── Environment ──────────────────────────────────────────────────────────────

export PATH="${CUDA_PATH}/bin:${PATH}"
export CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"

# Initialise conda
eval "$(conda shell.bash hook)"

# Create or activate env
if conda env list 2>/dev/null | grep -q "^${CONDA_ENV_NAME} "; then
    echo "  Conda env '${CONDA_ENV_NAME}' exists, activating..."
else
    echo "  Creating conda env '${CONDA_ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y
fi
conda activate "${CONDA_ENV_NAME}"

echo "  Python: $(python --version)"
echo "  nvcc:   $(nvcc --version | grep release | awk '{print $NF}')"
echo ""

# ── Build tools ──────────────────────────────────────────────────────────────

echo "=== Installing build tools ==="
pip install --upgrade pip --quiet
pip install cmake ninja wheel numpy "setuptools-scm>=6.0" pybind11 --quiet

# ── CUDA dependencies (skip with --quick) ────────────────────────────────────

if $QUICK; then
    echo ""
    echo "=== Skipping CUDA builds (--quick mode) ==="
    echo "    python-multem and guanaco assumed already installed."
    echo ""
else
    echo ""
    echo "=== Building python-multem (CUDA multislice, sm_${CUDA_ARCH}) ==="
    echo "    This compiles CUDA kernels -- takes 3-5 minutes..."
    pip install python-multem --no-build-isolation -v 2>&1 | tail -10

    echo ""
    echo "=== Building guanaco (CUDA reconstruction) ==="
    pip install guanaco --no-build-isolation -v 2>&1 | tail -10
fi

# ── Parakeet (editable) ─────────────────────────────────────────────────────

echo ""
echo "=== Installing parakeet (editable mode) ==="
cd "${PARAKEET_SRC}"

# Init pybind11 submodule if needed
if [ ! -f pybind11/CMakeLists.txt ]; then
    echo "    Initialising pybind11 submodule..."
    git submodule update --init --recursive
fi

pip install -e . --no-build-isolation 2>&1 | tail -10

# ── Runtime dependencies ─────────────────────────────────────────────────────

echo ""
echo "=== Installing runtime dependencies ==="
pip install scipy mrcfile gemmi h5py pyyaml pillow matplotlib --quiet

# ── Verify ───────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Verification"
echo "============================================================"
echo ""

python -c "
import sys
print(f'Python:       {sys.version.split()[0]}')

try:
    import parakeet
    print(f'parakeet:     {parakeet.__version__}')
except ImportError as e:
    print(f'parakeet:     FAILED ({e})')

try:
    import multem
    print(f'multem:       {multem.__version__}')
except ImportError as e:
    print(f'multem:       FAILED ({e})')

try:
    import guanaco
    print(f'guanaco:      {guanaco.__version__}')
except ImportError as e:
    print(f'guanaco:      FAILED ({e})')

try:
    from parakeet_ext import SpherePacker
    print('SpherePacker: OK')
except ImportError as e:
    print(f'SpherePacker: FAILED ({e})')

try:
    import numpy, scipy, mrcfile, h5py, yaml, gemmi
    print(f'numpy:        {numpy.__version__}')
    print(f'scipy:        {scipy.__version__}')
    print('mrcfile, h5py, yaml, gemmi: OK')
except ImportError as e:
    print(f'runtime deps: FAILED ({e})')

# Verify fork changes
try:
    from parakeet.config import Output, OutputDType
    o = Output()
    print(f'output.dtype: {o.dtype} (fork default)')
except Exception as e:
    print(f'fork config:  FAILED ({e})')

print()
print('All imports checked.')
"

echo ""
echo "============================================================"
echo "  Build complete"
echo "============================================================"
echo ""
echo "  To activate:"
echo "    conda activate ${CONDA_ENV_NAME}"
echo ""
echo "  To run tests:"
echo "    cd parakeet_frontend/simulation"
echo "    bash run_tests.sh --quick"
echo ""
echo "  Editable install: source changes to src/parakeet/*.py"
echo "  take effect immediately (no rebuild needed)."
echo ""
