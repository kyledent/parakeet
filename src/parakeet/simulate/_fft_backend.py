"""
FFT backend abstraction. Uses CuPy (GPU) if available, falls back to NumPy.

Set PARAKEET_NO_CUPY=1 to force NumPy backend (useful for benchmarking).

Usage:
    from parakeet.simulate._fft_backend import fft2, ifft2, fftshift, to_numpy

    result = ifft2(fft2(data) * kernel)
    numpy_result = to_numpy(result)
"""

import logging
import os

logger = logging.getLogger(__name__)

_use_cupy = not os.environ.get("PARAKEET_NO_CUPY", "")

if _use_cupy:
    try:
        import cupy as cp

        def fft2(x):
            return cp.fft.fft2(cp.asarray(x))

        def ifft2(x):
            return cp.fft.ifft2(x)

        def fftfreq(n, d=1.0):
            return cp.fft.fftfreq(n, d=d)

        def fftshift(x):
            return cp.fft.fftshift(cp.asarray(x))

        def to_numpy(x):
            return cp.asnumpy(x)

        def from_numpy(x):
            return cp.asarray(x)

        HAS_CUPY = True
        logger.debug("CuPy FFT backend enabled (GPU-accelerated)")

    except ImportError:
        _use_cupy = False

if not _use_cupy:
    import numpy as np

    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2
    fftfreq = np.fft.fftfreq
    fftshift = np.fft.fftshift
    to_numpy = lambda x: x
    from_numpy = lambda x: x

    HAS_CUPY = False
