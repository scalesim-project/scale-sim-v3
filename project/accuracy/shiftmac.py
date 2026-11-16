"""Functional models of shift-based MAC arithmetic (Project 3 starter)."""
import numpy as np


def pot_quantize(w, bits=4):
    """Round each weight to a signed power of two or zero."""
    if not isinstance(bits, int) or bits < 2:
        raise ValueError("bits must be an integer of at least 2")
    w = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if w.size == 0:
        return w.copy()
    a = np.abs(w)
    if not np.any(a):
        return np.zeros_like(w)
    n_exp = 2 ** (bits - 1)                    # number of exponent levels
    k_max = np.floor(np.log2(a.max()))
    k_min = k_max - n_exp + 1
    k = np.clip(np.round(np.log2(np.maximum(a, 1e-30))), k_min, k_max)
    q = np.sign(w) * 2.0 ** k
    q[a < 2.0 ** k_min / 2] = 0.0              # too small -> zero
    return q


def apot_quantize(w, bits=4):
    """Approximate weights with a sum of two PoT terms."""
    w = np.asarray(w, dtype=float)
    q1 = pot_quantize(w, bits)
    q2 = pot_quantize(w - q1, bits)
    return q1 + q2


def mitchell_mul(a, b):
    """Approximate multiplication using the Mitchell method."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    s = np.sign(a) * np.sign(b)
    a, b = np.abs(a), np.abs(b)
    out = np.zeros(a.shape)
    nz = (a > 0) & (b > 0)
    ka, kb = np.floor(np.log2(a[nz])), np.floor(np.log2(b[nz]))
    xa, xb = a[nz] / 2 ** ka - 1, b[nz] / 2 ** kb - 1
    f = xa + xb
    out[nz] = np.where(f < 1, 2 ** (ka + kb) * (1 + f), 2 ** (ka + kb + 1) * f)
    return s * out


def matmul(A, W, pe="exact", bits=4):
    """Multiply two matrices using the chosen PE arithmetic."""
    A, W = np.asarray(A), np.asarray(W)
    if A.ndim != 2 or W.ndim != 2 or A.shape[1] != W.shape[0]:
        raise ValueError("matrix dimensions must be (M,K) and (K,N)")
    if pe in ("exact", "baseline"):
        return A @ W
    if pe == "pot":
        return A @ pot_quantize(W, bits)
    if pe == "apot":
        return A @ apot_quantize(W, bits)
    if pe == "mitchell":
        return mitchell_mul(A[:, :, None], W[None, :, :]).sum(axis=1)
    raise ValueError(pe)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.normal(size=(64, 256))
    W = rng.normal(scale=0.05, size=(256, 64))
    ref = A @ W
    for pe in ["pot", "apot", "mitchell"]:
        y = matmul(A, W, pe)
        print(f"{pe:9s} relative error = {np.linalg.norm(y - ref) / np.linalg.norm(ref):.3f}")
