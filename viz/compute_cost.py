"""
compute_cost.py
---------------
Policy and GP compute time per run, from run.h5.

  ergodic (mh): compute_log col 3 = P (MH) build per p_update   -> / p_update, per agent
  hedac       : compute_log col 3 = summed solves per gp_update  -> / gp_update, per agent
  mac_dt      : attrs policy_s_total / policy_steps              -> central planner, total per step

Decentralized planners report per-agent latency (agents run in parallel);
MAC-DT reports its single central planner's latency (all agents' moves).

Usage (from MAES repo root):
    python viz/compute_cost.py results/PLANNER
"""
import sys, glob
import h5py
import numpy as np
import pandas as pd

batch = sys.argv[1] if len(sys.argv) > 1 else "results/PLANNER"
rows = []
for p in sorted(glob.glob(f"{batch}/**/run.h5", recursive=True)):
    with h5py.File(p, "r") as f:
        m = str(f.attrs.get("planner", f.attrs.get("method", "unknown")))
        n = int(f.attrs["n_agents"])
        cl = (np.asarray(f["compute_log"]).reshape(-1, 4)
              if "compute_log" in f else np.zeros((0, 4)))
        gp = cl[cl[:, 2] > 0, 2]
        if "policy_s_total" in f.attrs:                  # MAC-DT: central, total per step
            steps = max(int(f.attrs["policy_steps"]), 1)
            pol_ms = 1e3 * float(f.attrs["policy_s_total"]) / steps
        else:                                            # decentralized: per agent per step
            tau = int(f.attrs["gp_update"] if m == "hedac" else f.attrs.get("p_update", 1))
            pol = cl[cl[:, 3] > 0, 3]
            pol_ms = 1e3 * pol.mean() / tau if pol.size else np.nan
        rows.append(dict(
            planner=m, n_agents=n, policy_ms_per_step=pol_ms,
            gp_ms_per_fit=1e3 * gp.mean() if gp.size else np.nan,
            wall_s=float(f.attrs.get("wall_time_s", np.nan)), run=p))

df = pd.DataFrame(rows)
if df.empty:
    sys.exit("no runs found")
print(df.drop(columns="run").round(3).to_string(index=False))
print("\nmean per (planner, N):")
print(df.groupby(["planner", "n_agents"])[["policy_ms_per_step", "gp_ms_per_fit", "wall_s"]]
        .mean().round(3).to_string())
df.to_csv(f"{batch}/compute_cost.csv", index=False)
print(f"\nwrote {batch}/compute_cost.csv")