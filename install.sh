#!/usr/bin/env bash
#
# install.sh -- Build and install Parakeet from source
#
# Tested on: Ubuntu 22.04, CUDA 12.8, Pascal-class GPU (sm_61), Python 3.10
# See BUILD_NOTES.md for detailed issue descriptions and fixes.
#
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
# Adjust these for your system:
CONDA_ENV_NAME="parakeet"
PYTHON_VERSION="3.10"
CUDA_PATH="/usr/local/cuda-12.8"

# Auto-detect GPU compute capability, or set manually
# Pascal=61, Turing=75, Ampere=86, Ada=89, Blackwell=120
if command -v nvidia-smi &>/dev/null; then
    _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    CUDA_ARCH="${CUDA_ARCH:-$_cc}"
else
    CUDA_ARCH="${CUDA_ARCH:-61}"
fi

# ── Preflight checks ────────────────────────────────────────────────────────
echo "=== Parakeet installer ==="
echo "CUDA path:    $CUDA_PATH"
echo "CUDA arch:    sm_$CUDA_ARCH"
echo "Conda env:    $CONDA_ENV_NAME"
echo "Python:       $PYTHON_VERSION"
echo ""

# Check CUDA
if [ ! -x "$CUDA_PATH/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at $CUDA_PATH/bin/nvcc"
    echo "Install CUDA toolkit or adjust CUDA_PATH at the top of this script."
    exit 1
fi

# Check FFTW
if ! ldconfig -p 2>/dev/null | grep -q libfftw3; then
    if [ ! -f /usr/include/fftw3.h ]; then
        echo "ERROR: FFTW3 development headers not found."
        echo "Install with: sudo apt install libfftw3-dev"
        exit 1
    fi
fi

# Check conda
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Anaconda or Miniconda first."
    exit 1
fi

# ── Set up environment ───────────────────────────────────────────────────────
export PATH="$CUDA_PATH/bin:$PATH"
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

# Initialise conda for this shell
eval "$(conda shell.bash hook)"

# Create conda env if it doesn't exist
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Conda env '$CONDA_ENV_NAME' already exists, activating..."
else
    echo "Creating conda env '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
fi
conda activate "$CONDA_ENV_NAME"

echo "Python: $(python --version)"
echo "nvcc:   $(nvcc --version | tail -1)"
echo ""

# ── Install build tools ─────────────────────────────────────────────────────
echo "=== Installing build tools ==="
pip install cmake ninja wheel numpy setuptools-scm

# ── Build CUDA dependencies ─────────────────────────────────────────────────
echo ""
echo "=== Building python-multem (CUDA multislice engine) ==="
echo "    This compiles CUDA kernels -- may take a few minutes..."
pip install python-multem --no-build-isolation

echo ""
echo "=== Building guanaco (CUDA reconstruction library) ==="
pip install guanaco --no-build-isolation

# ── Build and install parakeet ───────────────────────────────────────────────
echo ""
echo "=== Installing parakeet (editable mode) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Init pybind11 submodule
if [ ! -f pybind11/CMakeLists.txt ]; then
    echo "Initialising pybind11 submodule..."
    git submodule update --init --recursive
fi

pip install -e . --no-build-isolation

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
python -c "
import parakeet; print(f'parakeet {parakeet.__version__}')
import multem; print(f'multem   {multem.__version__}')
from parakeet_ext import SpherePacker; print('SpherePacker OK')
print()
print('All imports successful. Parakeet is ready.')
print('Activate with: conda activate $CONDA_ENV_NAME')
"
