"""
table_robust.py
---------------
Compact LaTeX table (rows: comms None / Global; columns: agents lost 0/2/4) of
  - mean ROI time-to-hit   (roi_episodes.csv, hits only)
  - mean team cover time   (cover_cycles.csv, complete cycles only)
Each run is averaged first, then runs are averaged (equal weight per run).

Needs the CSVs from export_dynamic_metrics.py with the `attrition` column.

Usage (from MAES repo root):
    python viz/export_dynamic_metrics.py results/ROBUST
    python viz/table_robust.py results/ROBUST
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BATCH_DIR = Path("results/ROBUST")
T_MIN     = 0            # only events starting at t >= T_MIN (900 = after last kill)
ROWS      = [(0.0, "None"), (15.0, "Global")]     # (comm_radius, label)
LOST      = [0, 2, 4]


def run_means(df, col):
    """per-run mean, then mean over runs; std = pooled within-map std across starts."""
    pr = (df.groupby(["comm_radius", "attrition", "filename"])[col].mean()
            .reset_index())
    pr["map"] = pr.filename.str.extract(r"dynamic_(v\d+)_", expand=False)
    pr["res"] = pr[col] - pr.groupby(["comm_radius", "attrition", "map"])[col].transform("mean")
    g = pr.groupby(["comm_radius", "attrition"])
    std = g.apply(lambda d: np.sqrt((d.res ** 2).sum() / max(len(d) - d["map"].nunique(), 1)))
    print(pr.groupby(["comm_radius", "attrition", "map"])[col].mean().unstack().round(0))
    return g[col].mean(), std, g.size()


def main(batch):
    c = pd.read_csv(batch / "cover_cycles_robust.csv")
    h = pd.read_csv(batch / "roi_episodes_robust.csv")
    keep = lambda d: d[(d.comm_delay == 0) & d.comm_radius.isin([r for r, _ in ROWS])
                       & d.attrition.isin(LOST) & (d.t_start >= T_MIN)]
    c, h = keep(c), keep(h)

    c = c[c.complete.astype(str) == "True"]
    miss = h[h.t_hit.isna() & (h.end == "moved")]                 # true misses
    hits = h[h.t_hit.notna()]                                     # censored run_end NaNs dropped

    hit_m, hit_s, hit_n = run_means(hits, "t_hit")
    cov_m, cov_s, cov_n = run_means(c, "length")

    print(f"T_MIN={T_MIN}")
    for r, lab in ROWS:
        for k in LOST:
            nm = len(miss[(miss.comm_radius == r) & (miss.attrition == k)])
            print(f"  {lab:6s} lost={k}: runs(hit)={hit_n.get((r, k), 0)} "
                  f"runs(cover)={cov_n.get((r, k), 0)} misses={nm}")

    def fmt(m, s, key):
        if key not in m.index:
            return "--"
        return rf"{m[key]:.0f}$\pm${s[key]:.0f}"
    body = []
    for r, lab in ROWS:
        cells = ([fmt(hit_m, hit_s, (r, k)) for k in LOST]
                 + [fmt(cov_m, cov_s, (r, k)) for k in LOST])
        body.append(f"{lab} & " + " & ".join(cells) + r" \\")

    n = len(LOST)
    tex = "\n".join([
        r"\begin{table}[t]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{0.95}",
        r"\caption{ROI hit time and team cover time (steps), mean $\pm$ std over 6 runs "
        r"(6 agents, 2 maps $\times$ 3 starts), vs.\ number of agents lost.}",
        r"\label{tab:robust}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l" + "c" * (2 * n) + "@{}}",
        r"\toprule",
        rf" & \multicolumn{{{n}}}{{c}}{{ROI hit time}} & \multicolumn{{{n}}}{{c}}{{Cover time}} \\",
        rf"\cmidrule(lr){{2-{n+1}}}\cmidrule(lr){{{n+2}-{2*n+1}}}",
        r"Agents lost & " + " & ".join(str(k) for k in LOST * 2) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ])
    out = batch / "table_robust.tex"
    out.write_text(tex + "\n")
    print(tex)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else BATCH_DIR)