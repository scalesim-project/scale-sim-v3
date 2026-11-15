"""Energy post-processor for SCALE-Sim runs (Project 3 starter).

Usage: python energy.py <run_dir> <topology.csv> [gemm|conv] [pe_type]
Energy = actual MACs x E_mac(pe) + SRAM bits x E_sram_bit + DRAM bits x E_dram_bit.
pe_type: baseline | pot | apot | mitchell | bitserial (see costs.md).
"""
import sys, csv, math
import pandas as pd

# ---- Per-operation costs, 45 nm. Every number and its source is in costs.md.
# PE type: (energy per MAC in pJ, weight bits stored in SRAM/DRAM)
PE = {
    "baseline":  (0.300, 8),   # INT8 mult 0.2 + 32b accumulate add 0.1
    "pot":       (0.157, 4),   # INT16 shift 0.057 + 32b add 0.1; 4-bit PoT weights
    "apot":      (0.264, 8),   # 2 shifts + 16b add + 32b add (two 4-bit PoT terms)
    "mitchell":  (0.252, 8),   # 8b mult x 0.759 (Kim 2020 ratio) + 32b add
    "bitserial": (0.500, 8),   # 8 x 16b add + 32b add
}
WORD_BITS = 8                  # activation / output word size
E_SRAM_BIT = 5.0 / 32          # 32-bit SRAM read = 5 pJ  -> 0.156 pJ/bit
E_DRAM_BIT = 640.0 / 32        # 32-bit DRAM read = 640 pJ -> 20 pJ/bit

def layer_macs(row, fmt):
    v = [int(x) for x in row[1:] if x.strip()]
    if fmt == "gemm":                       # Layer, M, N, K
        m, n, k = v[:3]
        return m * n * k
    ih, iw, fh, fw, c, nf, s = v[:7]        # conv layer
    oh, ow = (ih - fh) // s + 1, (iw - fw) // s + 1
    return oh * ow * fh * fw * c * nf

def compute(run_dir, topo, fmt="conv", pe="baseline"):
    """Return a per-layer DataFrame of cycles, MACs and energy (pJ) for one run."""
    with open(topo) as f:
        rows = [r for r in csv.reader(f)][1:]
    macs = [layer_macs(r, fmt) for r in rows if len(r) > 1]
    acc = pd.read_csv(f"{run_dir}/DETAILED_ACCESS_REPORT.csv", skipinitialspace=True)
    comp = pd.read_csv(f"{run_dir}/COMPUTE_REPORT.csv", skipinitialspace=True)
    e_mac, w_bits = PE[pe]
    # SCALE-Sim counts words; filter words shrink when weights use fewer bits
    sram_bits = (acc["SRAM IFMAP Reads"] + acc["SRAM OFMAP Writes"]) * WORD_BITS \
        + acc["SRAM Filter Reads"] * w_bits
    dram_bits = (acc["DRAM IFMAP Reads"] + acc["DRAM OFMAP Writes"]) * WORD_BITS \
        + acc["DRAM Filter Reads"] * w_bits
    out = pd.DataFrame({
        "layer": comp["LayerID"], "cycles": comp["Total Cycles"], "macs": macs,
        "E_mac_pJ": [m * e_mac for m in macs],
        "E_sram_pJ": sram_bits * E_SRAM_BIT, "E_dram_pJ": dram_bits * E_DRAM_BIT})
    out["E_total_pJ"] = out[["E_mac_pJ", "E_sram_pJ", "E_dram_pJ"]].sum(axis=1)
    out["EDP"] = out["E_total_pJ"] * out["cycles"]
    return out

def main(run_dir, topo, fmt="conv", pe="baseline"):
    out = compute(run_dir, topo, fmt, pe)
    print(out.to_string(index=False))
    t = out.sum(numeric_only=True)
    print(f"\nTOTAL ({pe}): {t.E_total_pJ/1e6:.3f} uJ, MAC share {t.E_mac_pJ/t.E_total_pJ:.1%}, "
          f"cycles {int(t.cycles)}")

if __name__ == "__main__":
    main(*sys.argv[1:])
