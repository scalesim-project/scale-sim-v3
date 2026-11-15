"""Energy post-processor for SCALE-Sim runs (Project 3 starter).

Usage: python energy.py <run_dir> <topology.csv> [gemm|conv] [pe_type]
Energy = actual MACs x E_mac(pe) + SRAM accesses x E_sram + DRAM accesses x E_dram.
"""
import sys, csv, math
import pandas as pd

# ---- PLACEHOLDER energies (pJ per op / per word). Replace with cited values
# ---- (e.g. Horowitz ISSCC 2014 table) and sweep the shifter ones.
E_MAC = {"baseline": 1.0, "pot": 0.3, "apot": 0.45, "mitchell": 0.5, "bitserial": 0.25}
E_SRAM = 5.0      # per word read/write
E_DRAM = 100.0    # per word read/write

def layer_macs(row, fmt):
    v = [int(x) for x in row[1:] if x.strip()]
    if fmt == "gemm":                       # Layer, M, N, K
        m, n, k = v[:3]
        return m * n * k
    ih, iw, fh, fw, c, nf, s = v[:7]        # conv layer
    oh, ow = (ih - fh) // s + 1, (iw - fw) // s + 1
    return oh * ow * fh * fw * c * nf

def main(run_dir, topo, fmt="conv", pe="baseline"):
    with open(topo) as f:
        rows = [r for r in csv.reader(f)][1:]
    macs = [layer_macs(r, fmt) for r in rows if len(r) > 1]
    acc = pd.read_csv(f"{run_dir}/DETAILED_ACCESS_REPORT.csv", skipinitialspace=True)
    comp = pd.read_csv(f"{run_dir}/COMPUTE_REPORT.csv", skipinitialspace=True)
    sram = acc[["SRAM IFMAP Reads", "SRAM Filter Reads", "SRAM OFMAP Writes"]].sum(axis=1)
    dram = acc[["DRAM IFMAP Reads", "DRAM Filter Reads", "DRAM OFMAP Writes"]].sum(axis=1)
    out = pd.DataFrame({
        "layer": comp["LayerID"], "cycles": comp["Total Cycles"], "macs": macs,
        "E_mac_pJ": [m * E_MAC[pe] for m in macs],
        "E_sram_pJ": sram * E_SRAM, "E_dram_pJ": dram * E_DRAM})
    out["E_total_pJ"] = out[["E_mac_pJ", "E_sram_pJ", "E_dram_pJ"]].sum(axis=1)
    out["EDP"] = out["E_total_pJ"] * out["cycles"]
    print(out.to_string(index=False))
    t = out.sum(numeric_only=True)
    print(f"\nTOTAL ({pe}): {t.E_total_pJ/1e6:.3f} uJ, MAC share {t.E_mac_pJ/t.E_total_pJ:.1%}, "
          f"cycles {int(t.cycles)}")

if __name__ == "__main__":
    main(*sys.argv[1:])
