"""Functional models of shift-based MAC arithmetic (Project 3 starter)."""
import numpy as np

def pot_quantize(w, bits=4):
    """Round each weight to sign * 2^k (power-of-two). bits = 1 sign + exponent bits."""
    n_exp = 2 ** (bits - 1)                       # number of exponent levels
    s = np.sign(w)
    a = np.abs(w)
    k_max = np.floor(np.log2(a.max()))
    k = np.clip(np.round(np.log2(np.maximum(a, 1e-30))), k_max - n_exp + 1, k_max)
    q = s * 2.0 ** k
    q[a < 2.0 ** (k_max - n_exp + 1) / 2] = 0.0   # too small -> zero
    return q

def apot_quantize(w, bits=4):
    """Additive power-of-two: sum of two PoT terms (greedy residual)."""
    q1 = pot_quantize(w, bits)
    q2 = pot_quantize(w - q1 + 1e-30, bits)
    return q1 + q2

def mitchell_mul(a, b):
    """Mitchell's logarithmic approximate multiply, elementwise (always <= exact)."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    s = np.sign(a) * np.sign(b)
    a, b = np.abs(a), np.abs(b)
    out = np.zeros(a.shape)
    nz = (a > 0) & (b > 0)
    ka, kb = np.floor(np.log2(a[nz])), np.floor(np.log2(b[nz]))
    xa, xb = a[nz] / 2 ** ka - 1, b[nz] / 2 ** kb - 1   # mantissa fractions in [0,1)
    f = xa + xb
    out[nz] = np.where(f < 1, 2 ** (ka + kb) * (1 + f), 2 ** (ka + kb + 1) * f)
    return s * out

def matmul(A, W, pe="exact", bits=4):
    """A (M,K) @ W (K,N) with the arithmetic of the chosen PE."""
    if pe == "exact":
        return A @ W
    if pe == "pot":
        return A @ pot_quantize(W, bits)        # shift = exact multiply by 2^k
    if pe == "apot":
        return A @ apot_quantize(W, bits)
    if pe == "mitchell":
        return mitchell_mul(A[:, :, None], W[None, :, :]).sum(axis=1)
    raise ValueError(pe)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.normal(size=(64, 256)); W = rng.normal(scale=0.05, size=(256, 64))
    ref = A @ W
    for pe in ["pot", "apot", "mitchell"]:
        y = matmul(A, W, pe)
        print(f"{pe:9s} relative error = {np.linalg.norm(y - ref) / np.linalg.norm(ref):.3f}")
