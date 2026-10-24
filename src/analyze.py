"""Analysis and plots over results/grid.csv: Pareto frontier (gain_retained
vs cost_fraction) per topology/centrality including a random-recompute
control, an honest (in-range vs extrapolated) equal-budget comparison, and
best-tau-vs-topology table using a non-noise-driven selection rule."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("../results/grid.csv")
families = ["ER", "BA", "WS", "SBM"]
centralities = ["degree", "betweenness"]

synth = df[df["family"].isin(families)]

# Collapse seeds: mean AND std per (family, centrality, policy, param), so
# we can see when a result (e.g. gain_retained > 1) is noise from a small
# R_static - R_adaptive denominator rather than a real effect.
agg = (
    synth.groupby(["family", "centrality", "policy", "param"], dropna=False)
    .agg(R=("R", "mean"), R_std=("R", "std"),
         cost_fraction=("cost_fraction", "mean"),
         gain_retained=("gain_retained", "mean"),
         gain_std=("gain_retained", "std"))
    .reset_index()
)

print("=== R_static - R_adaptive denominator (small denom -> noisy gain_retained) ===")
for fam in families:
    for cfn in centralities:
        sub = agg[(agg.family == fam) & (agg.centrality == cfn)]
        rs = sub[sub.policy == "static"].R.values
        ra = sub[sub.policy == "adaptive"].R.values
        if len(rs) and len(ra):
            denom = rs[0] - ra[0]
            flag = "  <-- SMALL, treat gain_retained here with caution" if denom < 0.015 else ""
            print(f"{fam}/{cfn}: R_static={rs[0]:.4f} R_adaptive={ra[0]:.4f} denom={denom:.4f}{flag}")

# --- Pareto plots: fixed_k, T1, T2, and the random-recompute control ---
fig, axes = plt.subplots(len(centralities), len(families), figsize=(18, 8), sharey=True)
for i, cfn in enumerate(centralities):
    for j, fam in enumerate(families):
        ax = axes[i][j]
        sub = agg[(agg["family"] == fam) & (agg["centrality"] == cfn)]
        for policy, marker, color, ls in [
            ("fixed_k", "o", "tab:blue", "-"),
            ("trigger_T1_edgeloss", "^", "tab:orange", "-"),
            ("trigger_T2_topdrift", "s", "tab:green", "-"),
            ("random_control", "x", "gray", "--"),
        ]:
            pts = sub[sub["policy"] == policy].sort_values("cost_fraction")
            ax.plot(pts["cost_fraction"], pts["gain_retained"], marker=marker,
                     color=color, linestyle=ls, label=policy, linewidth=1)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"{fam} / {cfn}", fontsize=9)
        ax.set_xlabel("cost fraction (recomputations / N)")
        if j == 0:
            ax.set_ylabel("gain retained")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("../figures/pareto_by_topology.png", dpi=150)
print("\nSaved ../figures/pareto_by_topology.png")

# --- Best tau: smallest tau that reaches gain_retained >= 0.95, not argmax ---
# (argmax picks up noise - e.g. a gain_retained of 1.26 from 3 seeds on a
# network where R_static - R_adaptive is tiny - rather than the policy's
# real behavior.)
GAIN_THRESHOLD = 0.95
summary_rows = []
for fam in families:
    for cfn in centralities:
        sub = agg[(agg["family"] == fam) & (agg["centrality"] == cfn)]
        for policy in ["trigger_T1_edgeloss", "trigger_T2_topdrift"]:
            pts = sub[sub["policy"] == policy].sort_values("param")
            if pts.empty:
                continue
            reaching = pts[pts["gain_retained"] >= GAIN_THRESHOLD]
            if not reaching.empty:
                best = reaching.iloc[0]  # smallest tau reaching threshold
                note = f"first tau reaching gain>={GAIN_THRESHOLD}"
            else:
                best = pts.loc[pts["gain_retained"].idxmax()]
                note = "never reaches threshold; reporting max achieved"
            summary_rows.append({
                "family": fam, "centrality": cfn, "policy": policy,
                "best_tau": best["param"], "gain_retained": best["gain_retained"],
                "cost_fraction": best["cost_fraction"], "note": note,
            })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("../results/best_tau_by_topology.csv", index=False)
print("\n=== Best tau (first to reach gain_retained >= 0.95, not argmax) ===")
print(summary_df.to_string(index=False))

# --- Equal-budget comparison: T2 and random_control vs fixed_k, explicit
# about which comparisons are genuinely in T2's tested cost range vs
# extrapolated/clipped to its cheapest point. ---
print("\n=== Equal-budget comparison (T2 and random_control vs fixed_k) ===")
in_range_n, total_n, t2_wins_in_range = 0, 0, 0
random_beats_fixed_k = 0
for fam in families:
    for cfn in centralities:
        sub = agg[(agg.family == fam) & (agg.centrality == cfn)]
        fk = sub[sub.policy == "fixed_k"].sort_values("cost_fraction")
        t2 = sub[sub.policy == "trigger_T2_topdrift"].sort_values("cost_fraction")
        rnd = sub[sub.policy == "random_control"].sort_values("cost_fraction")
        if fk.empty or len(t2) < 2:
            continue
        for _, row in fk.iterrows():
            total_n += 1
            in_range = t2.cost_fraction.min() <= row.cost_fraction <= t2.cost_fraction.max()
            x = np.clip(row.cost_fraction, t2.cost_fraction.min(), t2.cost_fraction.max())
            t2_gain = np.interp(x, t2.cost_fraction, t2.gain_retained)
            tag = "IN-RANGE" if in_range else "extrapolated"
            if in_range:
                in_range_n += 1
                if t2_gain >= row.gain_retained:
                    t2_wins_in_range += 1
            rnd_gain = None
            if len(rnd) >= 2:
                xr = np.clip(row.cost_fraction, rnd.cost_fraction.min(), rnd.cost_fraction.max())
                rnd_gain = np.interp(xr, rnd.cost_fraction, rnd.gain_retained)
                if rnd_gain >= row.gain_retained:
                    random_beats_fixed_k += 1
            print(f"{fam}/{cfn}: cost={row.cost_fraction:.3f} fixed_k={row.gain_retained:.3f} "
                  f"T2={t2_gain:.3f} [{tag}] random={rnd_gain:.3f}" if rnd_gain is not None else
                  f"{fam}/{cfn}: cost={row.cost_fraction:.3f} fixed_k={row.gain_retained:.3f} T2={t2_gain:.3f} [{tag}]")

print(f"\n{in_range_n}/{total_n} comparisons are genuinely in T2's tested cost range "
      f"({total_n - in_range_n} are extrapolated to T2's cheapest tested point).")
print(f"Of the in-range comparisons, T2 beats fixed_k in {t2_wins_in_range}/{in_range_n}.")

# --- T2 vs random_control at matched cost: is T2 actually smarter than
# recomputing at random with the same average cost? ---
print("\n=== T2 vs random_control at matched cost (is the trigger smarter than noise?) ===")
t2_beats_random, random_total = 0, 0
for fam in families:
    for cfn in centralities:
        sub = agg[(agg.family == fam) & (agg.centrality == cfn)]
        t2 = sub[sub.policy == "trigger_T2_topdrift"].sort_values("cost_fraction")
        rnd = sub[sub.policy == "random_control"].sort_values("cost_fraction")
        if len(t2) < 2 or len(rnd) < 2:
            continue
        lo = max(t2.cost_fraction.min(), rnd.cost_fraction.min())
        hi = min(t2.cost_fraction.max(), rnd.cost_fraction.max())
        if lo > hi:
            print(f"{fam}/{cfn}: no overlapping cost range between T2 and random_control")
            continue
        test_costs = np.linspace(lo, hi, 5)
        for c in test_costs:
            t2_gain = np.interp(c, t2.cost_fraction, t2.gain_retained)
            rnd_gain = np.interp(c, rnd.cost_fraction, rnd.gain_retained)
            random_total += 1
            if t2_gain >= rnd_gain:
                t2_beats_random += 1
        print(f"{fam}/{cfn}: overlap cost range [{lo:.3f}, {hi:.3f}], "
              f"T2 beats random in {sum(np.interp(c, t2.cost_fraction, t2.gain_retained) >= np.interp(c, rnd.cost_fraction, rnd.gain_retained) for c in test_costs)}/5 sampled points")
if random_total:
    print(f"\nOverall: T2 beats random_control in {t2_beats_random}/{random_total} matched-cost samples.")

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
