# Project 3

"""
Test PeCyclesPerMac (k) with os/is dataflows (ws = control) vs unmodified SCALE-Sim.`   
"""
import configparser, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT.parent / "scale-sim-upstream"
TOPO = ROOT / "topologies/GEMM_mnk/test_mnk_input.csv"
OUT = ROOT / "project/runs/dftest"


def run(repo, name, df, k=None):
    cfg = configparser.ConfigParser(); cfg.optionxform = str
    cfg.read(ROOT / "configs/scale.cfg")
    cfg["general"]["run_name"] = name
    cfg["architecture_presets"]["Dataflow"] = df
    if k is not None:
        cfg["architecture_presets"]["PeCyclesPerMac"] = str(k)

    
    path = OUT / "configs" / f"{name}.cfg"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        cfg.write(f)

    # running 
    print(f"running {name} ...", flush=True)
    subprocess.run([sys.executable, "scale.py", "-c", str(path), "-t", str(TOPO), 
                    "-p", str(OUT), "-i", "gemm"], cwd=repo, check=True, 
                    stdout=subprocess.DEVNULL)

    
    rep = pd.read_csv(OUT / name / "COMPUTE_REPORT.csv", skipinitialspace=True)
    rep.columns = rep.columns.str.strip()
    col = "Total Cycles" if "Total Cycles" in rep else [c for c in rep if c.startswith("Total Cycles")][0]
    return dict(total_cycles=int(rep["Total Cycles (incl. prefetch)"].sum()),
                compute_cycles=int(rep["Total Cycles"].sum()),
                stalls=int(rep["Stall Cycles"].sum()),
                util=float(rep["Overall Util %"].iloc[0]))

rows = []
for df in ["ws", "os", "is"]:
    upstream_result = run(UPSTREAM, f"upstream_{df}", df)
    for k in [1, 4, 8]:
        r = run(ROOT, f"patched_{df}_k{k}", df, k)
        rows.append(dict(dataflow=df, k=k,
                         upstream_total=upstream_result["total_cycles"],
                         upstream_compute=upstream_result["compute_cycles"],
                         **r))

t = pd.DataFrame(rows)
first = lambda c: t.groupby("dataflow")[c].transform("first")
t["ratio_compute"] = (t.compute_cycles / first("compute_cycles")).round(3)
t["ratio_total"] = (t.total_cycles / first("total_cycles")).round(3)
t["real_util"] = (t.util * t.k).round(2)
t["k1_matches_upstream"] = (t.k != 1) | ((t.total_cycles == t.upstream_total) & (t.compute_cycles == t.upstream_compute))
t.to_csv(ROOT / "project/results/dataflow_tests.csv", index=False)
print("\n| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns))
for r in t.itertuples(index=False):
    print("| " + " | ".join(map(str, r)) + " |")