"""Analysis and plots over results/grid.csv: Pareto frontier (gain_retained
vs cost_fraction) per topology/centrality, and best-tau-vs-topology table."""
import sys
sys.path.insert(0, ".")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("../results/grid.csv")

# Collapse seeds: average R/cost/gain per (family, centrality, policy, param)
agg = (
    df[df["family"].isin(["ER", "BA", "WS", "SBM"])]
    .groupby(["family", "centrality", "policy", "param"], dropna=False)
    .agg(R=("R", "mean"), cost_fraction=("cost_fraction", "mean"),
         gain_retained=("gain_retained", "mean"))
    .reset_index()
)

families = ["ER", "BA", "WS", "SBM"]
centralities = ["degree", "betweenness"]

fig, axes = plt.subplots(len(centralities), len(families), figsize=(18, 8), sharey=True)
for i, cfn in enumerate(centralities):
    for j, fam in enumerate(families):
        ax = axes[i][j]
        sub = agg[(agg["family"] == fam) & (agg["centrality"] == cfn)]
        for policy, marker, color in [
            ("fixed_k", "o", "tab:blue"),
            ("trigger_T1_edgeloss", "^", "tab:orange"),
            ("trigger_T2_topdrift", "s", "tab:green"),
        ]:
            pts = sub[sub["policy"] == policy].sort_values("cost_fraction")
            ax.plot(pts["cost_fraction"], pts["gain_retained"], marker=marker,
                     color=color, label=policy, linewidth=1)
        static_row = sub[sub["policy"] == "static"]
        if not static_row.empty:
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"{fam} / {cfn}", fontsize=9)
        ax.set_xlabel("cost fraction (recomputations / N)")
        if j == 0:
            ax.set_ylabel("gain retained")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("../figures/pareto_by_topology.png", dpi=150)
print("Saved ../figures/pareto_by_topology.png")

# Best tau per family/centrality for each trigger, by gain_retained at
# comparable (lowest) cost bracket
summary_rows = []
for fam in families:
    for cfn in centralities:
        sub = agg[(agg["family"] == fam) & (agg["centrality"] == cfn)]
        for policy in ["trigger_T1_edgeloss", "trigger_T2_topdrift"]:
            pts = sub[sub["policy"] == policy]
            if pts.empty:
                continue
            best = pts.loc[pts["gain_retained"].idxmax()]
            summary_rows.append({
                "family": fam, "centrality": cfn, "policy": policy,
                "best_tau": best["param"], "gain_retained": best["gain_retained"],
                "cost_fraction": best["cost_fraction"],
            })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("../results/best_tau_by_topology.csv", index=False)
print(summary_df.to_string(index=False))

# Real networks table
real = df[df["family"] == "REAL"]
if not real.empty:
    real_summary = (
        real.groupby(["network", "centrality", "policy", "param"], dropna=False)
        .agg(R=("R", "mean"), cost_fraction=("cost_fraction", "mean"),
             gain_retained=("gain_retained", "mean"))
        .reset_index()
    )
    real_summary.to_csv("../results/real_networks_summary.csv", index=False)
    print("\nSaved ../results/real_networks_summary.csv")
