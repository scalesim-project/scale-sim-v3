"""Experiment sweep for Project 3: PE type x array size x dataflow x workload.

Run from the repo root (the folder with scale.py), inside the scalesim-env venv:

    python project/sweep.py --quick          # ~1 min smoke test (GEMM test, WS only)
    python project/sweep.py                  # full sweep (hours: run on onyx overnight)
    python project/sweep.py --workloads resnet18 --dataflows ws os
    python project/sweep.py --quick --sram 64 512   # on-chip buffer size experiment

SCALE-Sim cycles depend only on (array size, dataflow, PeCyclesPerMac, workload),
not on which PE does the math, so each unique simulation runs once and energy for
every PE type is computed from it with energy.py. Finished simulations are skipped,
so an interrupted sweep can simply be restarted. Results are merged into the CSV, so
separate runs (or machines, via --out) add up instead of overwriting each other.
"""
import argparse, configparser, os, shutil, subprocess, sys, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import energy  # noqa: E402  (project/energy.py)

# ---- Experiment definition ---------------------------------------------------
WORKLOADS = {  # name: (topology file, SCALE-Sim input type)
    "gemm_test": ("topologies/GEMM_mnk/test_mnk_input.csv", "gemm"),
    "resnet18":  ("topologies/conv_nets/Resnet18.csv",      "conv"),
    "vit_s":     ("topologies/GEMM_mnk/vit_s.csv",          "gemm"),
}
PE_CYCLES = {"baseline": 1, "pot": 1, "apot": 1, "mitchell": 1, "bitserial": 8}
ISO_AREA_SIDE = {"baseline": 32, "pot": 45, "apot": 35, "mitchell": 34, "bitserial": 45}  # costs.md sec. 4
FIXED_SIDE = 32
TEMPLATE_CFG = "configs/scale.cfg"
RUNS_DIR = Path("project/runs")       # simulator outputs (in .gitignore)
RESULTS_CSV = Path("project/results/all.csv")
KEY = ["workload", "dataflow", "sram_kb", "array_mode", "pe", "side"]   # one row per experiment point


def default_sram_kb():
    """On-chip buffer size (kB) in the template config; runs at this size keep their old names."""
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(TEMPLATE_CFG)
    return int(cfg["architecture_presets"]["IfmapSramSzkB"])


def plan(pes, dataflows, workloads, array_modes, srams):
    """List every (pe, array_mode, side, dataflow, workload, sram_kb) point to report."""
    points = []
    for wl in workloads:
        for df in dataflows:
            for sram in srams:
                for pe in pes:
                    for mode in array_modes:
                        side = FIXED_SIDE if mode == "fixed" else ISO_AREA_SIDE[pe]
                        points.append(dict(pe=pe, array_mode=mode, side=side, dataflow=df,
                                           workload=wl, sram_kb=sram))
    return points


def run_name(side, df, k, wl, sram):
    name = f"{wl}_{side}x{side}_{df}_k{k}"
    return name if sram == DEFAULT_SRAM else f"{name}_sram{sram}"


def write_cfg(name, side, df, k, sram):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str                      # keep SCALE-Sim's key capitalisation
    cfg.read(TEMPLATE_CFG)
    cfg["general"]["run_name"] = name
    a = cfg["architecture_presets"]
    a["ArrayHeight"], a["ArrayWidth"] = str(side), str(side)
    a["Dataflow"] = df
    a["PeCyclesPerMac"] = str(k)
    for key in ("IfmapSramSzkB", "FilterSramSzkB", "OfmapSramSzkB"):   # all three buffers
        a[key] = str(sram)
    path = RUNS_DIR / "configs" / f"{name}.cfg"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        cfg.write(f)
    return path


def simulate(side, df, k, wl, sram, keep_traces):
    """Run SCALE-Sim once for this hardware point; skip if already done."""
    name = run_name(side, df, k, wl, sram)
    out_dir = RUNS_DIR / name
    if (out_dir / "COMPUTE_REPORT.csv").exists():
        print(f"  [skip] {name} (already simulated)")
        return out_dir
    topo, itype = WORKLOADS[wl]
    cfg = write_cfg(name, side, df, k, sram)
    print(f"  [run ] {name} ...", flush=True)
    t0 = time.time()
    res = subprocess.run([sys.executable, "scale.py", "-c", str(cfg), "-t", topo,
                          "-p", str(RUNS_DIR), "-i", itype],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not (out_dir / "COMPUTE_REPORT.csv").exists():
        print(res.stdout[-2000:], res.stderr[-2000:])
        raise RuntimeError(f"SCALE-Sim failed for {name}")
    if not keep_traces:                        # traces are large; reports are all we need
        for d in out_dir.glob("layer*"):
            shutil.rmtree(d, ignore_errors=True)
    print(f"         done in {time.time() - t0:.0f} s")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pes", nargs="+", default=list(PE_CYCLES))
    ap.add_argument("--dataflows", nargs="+", default=["ws", "os", "is"])
    ap.add_argument("--workloads", nargs="+", default=["gemm_test", "resnet18", "vit_s"])
    ap.add_argument("--arrays", choices=["fixed", "iso", "both"], default="both",
                    help="fixed = all PEs on 32x32; iso = iso-area sizes from costs.md")
    ap.add_argument("--sram", nargs="+", type=int, default=None,
                    help="on-chip buffer size in kB for each of ifmap/filter/ofmap (e.g. 64 512); "
                         "default = the template config's size")
    ap.add_argument("--keep-traces", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_CSV),
                    help="results CSV; new rows are merged into it (default project/results/all.csv)")
    ap.add_argument("--quick", action="store_true", help="GEMM test, WS only, smoke test")
    args = ap.parse_args()
    if args.quick:
        args.workloads, args.dataflows = ["gemm_test"], ["ws"]
    if not Path("scale.py").exists():
        sys.exit("Run this from the repo root (the folder that contains scale.py).")

    modes = ["fixed", "iso"] if args.arrays == "both" else [args.arrays]
    srams = args.sram or [DEFAULT_SRAM]
    points = plan(args.pes, args.dataflows, args.workloads, modes, srams)
    sims = sorted({(p["side"], p["dataflow"], PE_CYCLES[p["pe"]], p["workload"], p["sram_kb"])
                   for p in points})
    print(f"{len(points)} result rows from {len(sims)} unique simulations\n")

    run_dirs = {s: simulate(*s, keep_traces=args.keep_traces) for s in sims}

    rows = []
    for p in points:
        k = PE_CYCLES[p["pe"]]
        topo, itype = WORKLOADS[p["workload"]]
        layers = energy.compute(str(run_dirs[(p["side"], p["dataflow"], k, p["workload"], p["sram_kb"])]),
                                topo, itype, p["pe"])
        t = layers.sum(numeric_only=True)
        rows.append({**p, "pe_cycles_per_mac": k, "pes": p["side"] ** 2,
                     "cycles": int(t.cycles), "macs": int(t.macs),
                     "E_mac_uJ": t.E_mac_pJ / 1e6, "E_sram_uJ": t.E_sram_pJ / 1e6,
                     "E_dram_uJ": t.E_dram_pJ / 1e6, "E_total_uJ": t.E_total_pJ / 1e6,
                     "mac_share": t.E_mac_pJ / t.E_total_pJ,
                     "EDP_uJ_cycles": t.E_total_pJ / 1e6 * t.cycles})
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():                           # merge: new rows replace same points, others are kept
        old = pd.read_csv(out)
        if "sram_kb" not in old.columns:       # rows from before the --sram option
            old["sram_kb"] = DEFAULT_SRAM
        df = (pd.concat([old, df], ignore_index=True)
                .drop_duplicates(subset=KEY, keep="last")
                .sort_values(KEY, ignore_index=True))
    df.to_csv(out, index=False)
    print(f"\nAdded {len(rows)} rows; {out} now has {len(df)} rows\n")
    print(df[["workload", "dataflow", "sram_kb", "array_mode", "pe", "side", "cycles",
              "E_total_uJ", "mac_share"]].to_string(index=False))


DEFAULT_SRAM = default_sram_kb() if Path(TEMPLATE_CFG).exists() else 64

if __name__ == "__main__":
    main()
