"""Analysis and plots over results/grid.csv: Pareto frontier (gain_retained
vs cost_fraction) per topology/centrality including a random-recompute
control, an honest (in-range vs extrapolated) equal-budget comparison,
best-tau-vs-topology table using a non-noise-driven selection rule, and
paired per-instance significance tests on both recomputation-count and
total-wall-clock-time cost axes (T3's checks are not free, unlike T1/T2)."""
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
TRIGGERS = ["trigger_T1_edgeloss", "trigger_T2_topdrift", "trigger_T3_probedrift"]

synth = df[df["family"].isin(families)].copy()
synth["total_time"] = synth["score_time"] + synth["trigger_time"]

# Collapse instances: mean AND std per (family, centrality, policy, param).
# Each point here averages 6 instances (2 network sizes x 3 seeds each),
# NOT 3 seeds - the groupby below has no "N0" key, so it pools both sizes.
# The std lets us see when a result (e.g. gain_retained > 1) is noise from
# a small R_static - R_adaptive denominator rather than a real effect.
agg = (
    synth.groupby(["family", "centrality", "policy", "param"], dropna=False)
    .agg(R=("R", "mean"), R_std=("R", "std"),
         cost_fraction=("cost_fraction", "mean"),
         gain_retained=("gain_retained", "mean"),
         gain_std=("gain_retained", "std"),
         total_time=("total_time", "mean"))
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

# --- Pareto plots: fixed_k, T1, T2, T3, and the random-recompute control,
# on cost_fraction (recomputations/N) - the PRIMARY metric throughout this
# project. T3's checks are not free (unlike T1/T2's near-O(1) checks); see
# the total_time figure below for the transparent secondary view. Shaded
# bands show +/-1 std across the 6 instances per point. Y-axis clipped to
# [-0.2, 1.3] (see caption in the paper for why); each panel uses its own
# x-range since cost_fraction has no reason to share scale across rows. ---
Y_LO, Y_HI = -0.2, 1.3
POLICY_STYLE = [
    ("fixed_k", "o", "tab:blue", "-"),
    ("trigger_T1_edgeloss", "^", "tab:orange", "-"),
    ("trigger_T2_topdrift", "s", "tab:green", "-"),
    ("trigger_T3_probedrift", "D", "tab:red", "-"),
    ("random_control", "x", "gray", "--"),
]

fig, axes = plt.subplots(len(centralities), len(families), figsize=(18, 8), sharey=True)
for i, cfn in enumerate(centralities):
    for j, fam in enumerate(families):
        ax = axes[i][j]
        sub = agg[(agg["family"] == fam) & (agg["centrality"] == cfn)]
        for policy, marker, color, ls in POLICY_STYLE:
            pts = sub[sub["policy"] == policy].sort_values("cost_fraction")
            ax.plot(pts["cost_fraction"], pts["gain_retained"], marker=marker,
                     color=color, linestyle=ls, label=policy, linewidth=1)
            std = pts["gain_std"].fillna(0)
            ax.fill_between(pts["cost_fraction"], pts["gain_retained"] - std,
                             pts["gain_retained"] + std, color=color, alpha=0.12)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(Y_LO, Y_HI)
        ax.set_title(f"{fam} / {cfn}", fontsize=9)
        ax.set_xlabel("cost fraction (recomputations / N)")
        if j == 0:
            ax.set_ylabel("gain retained")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("../figures/pareto_by_topology.png", dpi=150)
print("\nSaved ../figures/pareto_by_topology.png")

# --- Secondary, transparency figure: the betweenness row re-plotted on
# TOTAL TIME (score_time + trigger_time) instead of recomputation count,
# since T3's checks do real work unlike T1/T2/fixed_k's near-free ones. ---
fig2, axes2 = plt.subplots(1, len(families), figsize=(18, 4.5))
for j, fam in enumerate(families):
    ax = axes2[j]
    sub = agg[(agg["family"] == fam) & (agg["centrality"] == "betweenness")]
    for policy, marker, color, ls in POLICY_STYLE:
        pts = sub[sub["policy"] == policy].sort_values("total_time")
        ax.plot(pts["total_time"], pts["gain_retained"], marker=marker,
                 color=color, linestyle=ls, label=policy, linewidth=1)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_title(f"{fam} / betweenness", fontsize=9)
    ax.set_xlabel("total time (s)")
    if j == 0:
        ax.set_ylabel("gain retained")
handles2, labels2 = axes2[0].get_legend_handles_labels()
fig2.legend(handles2, labels2, loc="upper center", ncol=5)
fig2.tight_layout(rect=[0, 0, 1, 0.88])
fig2.savefig("../figures/pareto_betweenness_by_time.png", dpi=150)
print("Saved ../figures/pareto_betweenness_by_time.png")

# --- Best tau: the CHEAPEST tau that still reaches gain_retained >= 0.95,
# not argmax (argmax picks up noise) and not "smallest tau reaching
# threshold" (smaller tau fires more often, i.e. is MORE expensive). ---
GAIN_THRESHOLD = 0.95
summary_rows = []
for fam in families:
    for cfn in centralities:
        sub = agg[(agg["family"] == fam) & (agg["centrality"] == cfn)]
        for policy in TRIGGERS:
            pts = sub[sub["policy"] == policy].sort_values("cost_fraction")
            if pts.empty:
                continue
            reaching = pts[pts["gain_retained"] >= GAIN_THRESHOLD]
            if not reaching.empty:
                best = reaching.iloc[0]  # cheapest point reaching threshold
                note = f"cheapest tau reaching gain>={GAIN_THRESHOLD}"
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
print("\n=== Best tau (cheapest to reach gain_retained >= 0.95, not argmax) ===")
print(summary_df.to_string(index=False))

# --- Equal-budget comparison: each trigger vs fixed_k, explicit about
# in-range vs extrapolated comparisons. ---
for trig in TRIGGERS:
    print(f"\n=== Equal-budget comparison ({trig} vs fixed_k) ===")
    in_range_n, total_n, wins_in_range = 0, 0, 0
    in_range_by_metric = {"degree": [0, 0], "betweenness": [0, 0]}
    for fam in families:
        for cfn in centralities:
            sub = agg[(agg.family == fam) & (agg.centrality == cfn)]
            fk = sub[sub.policy == "fixed_k"].sort_values("cost_fraction")
            tr = sub[sub.policy == trig].sort_values("cost_fraction")
            if fk.empty or len(tr) < 2:
                continue
            for _, row in fk.iterrows():
                total_n += 1
                in_range = tr.cost_fraction.min() <= row.cost_fraction <= tr.cost_fraction.max()
                x = np.clip(row.cost_fraction, tr.cost_fraction.min(), tr.cost_fraction.max())
                tr_gain = np.interp(x, tr.cost_fraction, tr.gain_retained)
                if in_range:
                    in_range_n += 1
                    in_range_by_metric[cfn][1] += 1
                    if tr_gain >= row.gain_retained:
                        wins_in_range += 1
                        in_range_by_metric[cfn][0] += 1
    print(f"{in_range_n}/{total_n} comparisons genuinely in {trig}'s tested cost range.")
    if in_range_n:
        print(f"Of those, {trig} beats fixed_k in {wins_in_range}/{in_range_n}.")
    for cfn in centralities:
        w, t = in_range_by_metric[cfn]
        if t:
            print(f"  split by metric - {cfn}: {trig} beats fixed_k in {w}/{t}")


# --- Paired per-instance comparison: since every policy runs on the SAME
# six network instances (2 sizes x 3 seeds) per family, a paired
# difference per instance is a much more powerful test than comparing
# independently-aggregated mean curves. Report mean +/- standard error
# across the paired differences. cost_col lets the same function compare
# on recomputation count OR total wall-clock time. ---
def paired_diff(policy_a, policy_b, fam, cfn, cost_col="cost_fraction", n_points=5):
    sub = synth[(synth.family == fam) & (synth.centrality == cfn)]
    instances = sub[["N0", "seed"]].drop_duplicates()
    diffs = []
    for _, inst in instances.iterrows():
        inst_df = sub[(sub.N0 == inst.N0) & (sub.seed == inst.seed)]
        a = inst_df[inst_df.policy == policy_a].sort_values(cost_col)
        b = inst_df[inst_df.policy == policy_b].sort_values(cost_col)
        if len(a) < 2 or len(b) < 2:
            continue
        lo = max(a[cost_col].min(), b[cost_col].min())
        hi = min(a[cost_col].max(), b[cost_col].max())
        if lo > hi:
            continue
        for c in np.linspace(lo, hi, n_points):
            ag = np.interp(c, a[cost_col], a.gain_retained)
            bg = np.interp(c, b[cost_col], b.gain_retained)
            diffs.append(ag - bg)
    if not diffs:
        return None, None, 0
    diffs = np.array(diffs)
    return diffs.mean(), diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0, len(diffs)


for trig in TRIGGERS:
    print(f"\n=== Paired per-instance comparison: {trig} minus random_control (cost_fraction) ===")
    for fam in families:
        for cfn in centralities:
            mean, se, n = paired_diff(trig, "random_control", fam, cfn)
            if mean is None:
                continue
            sig = "significant" if abs(mean) > 2 * se else "within ~2 SE of zero"
            print(f"{fam}/{cfn}: {trig} - random = {mean:+.3f} +/- {se:.3f} (n={n}, {sig})")

    print(f"\n=== Paired per-instance comparison: {trig} minus fixed_k (cost_fraction) ===")
    for fam in families:
        for cfn in centralities:
            mean, se, n = paired_diff(trig, "fixed_k", fam, cfn)
            if mean is None:
                continue
            sig = "significant" if abs(mean) > 2 * se else "within ~2 SE of zero"
            print(f"{fam}/{cfn}: {trig} - fixed_k = {mean:+.3f} +/- {se:.3f} (n={n}, {sig})")

# T3-specific: the same fixed_k comparison but on TOTAL TIME, since that's
# the axis where T3's non-free checks actually matter.
print("\n=== Paired per-instance comparison: trigger_T3_probedrift minus fixed_k (TOTAL TIME) ===")
for fam in families:
    for cfn in centralities:
        mean, se, n = paired_diff("trigger_T3_probedrift", "fixed_k", fam, cfn, cost_col="total_time")
        if mean is None:
            continue
        sig = "significant" if abs(mean) > 2 * se else "within ~2 SE of zero"
        print(f"{fam}/{cfn}: T3 - fixed_k (time) = {mean:+.3f} +/- {se:.3f} (n={n}, {sig})")

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
