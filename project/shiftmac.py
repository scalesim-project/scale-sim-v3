"""Compatibility imports for older project scripts."""

# The shifter arithmetic now lives in project/accuracy/shiftmac.py
from project.accuracy.shiftmac import (
    pot_quantize, apot_quantize, mitchell_mul, matmul,
)

if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    a = rng.normal(size=(64, 256))
    w = rng.normal(scale=0.05, size=(256, 64))
    ref = matmul(a, w)
    for pe in ("pot", "apot", "mitchell"):
        y = matmul(a, w, pe=pe)
        print(f"{pe:9s} relative error = {np.linalg.norm(y-ref)/np.linalg.norm(ref):.3f}")
