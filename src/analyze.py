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

# --- Pareto plots: fixed_k, T1, T2, and the random-recompute control.
# Shaded bands show +/-1 std across the 6 instances per point (2 network
# sizes x 3 seeds - NOT 3 seeds; cost_fraction counts recomputations, not
# time, so there is no reason for degree and betweenness to need different
# x-ranges - each panel instead uses its own data's x-range, since forcing
# a shared range wastes space whenever one policy on one topology happens
# to run much longer than the rest (e.g. WS/degree's high-tau T2 points).
# The y-axis is clipped to [-0.2, 1.3]: a few (policy, tau) points on
# WS/degree have such high seed-to-seed variance (T1 at tau=0.3 reaches a
# mean of 1.26 with std 0.86) that an unclipped shared y-axis would
# compress every other panel into a thin strip near 0-1; the caption notes
# the clip and where it truncates a band. ---
Y_LO, Y_HI = -0.2, 1.3
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
fig.legend(handles, labels, loc="upper center", ncol=4)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("../figures/pareto_by_topology.png", dpi=150)
print("\nSaved ../figures/pareto_by_topology.png")

# --- Best tau: the CHEAPEST tau that still reaches gain_retained >= 0.95,
# not argmax (argmax picks up noise - e.g. a gain_retained of 1.26 from 3
# seeds on a network where R_static - R_adaptive is tiny) and not "smallest
# tau reaching threshold" (smaller tau fires more often, so that previous
# version of this selection picked the MOST expensive qualifying point,
# exactly backwards from "the most efficient trigger setting").
GAIN_THRESHOLD = 0.95
summary_rows = []
for fam in families:
    for cfn in centralities:
        sub = agg[(agg["family"] == fam) & (agg["centrality"] == cfn)]
        for policy in ["trigger_T1_edgeloss", "trigger_T2_topdrift"]:
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
print("\n=== Best tau (first to reach gain_retained >= 0.95, not argmax) ===")
print(summary_df.to_string(index=False))

# --- Equal-budget comparison: T2 and random_control vs fixed_k, explicit
# about which comparisons are genuinely in T2's tested cost range vs
# extrapolated/clipped to its cheapest point. ---
print("\n=== Equal-budget comparison (T2 and random_control vs fixed_k) ===")
in_range_n, total_n, t2_wins_in_range = 0, 0, 0
in_range_by_metric = {"degree": [0, 0], "betweenness": [0, 0]}  # [wins, total]
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
                in_range_by_metric[cfn][1] += 1
                if t2_gain >= row.gain_retained:
                    t2_wins_in_range += 1
                    in_range_by_metric[cfn][0] += 1
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
for cfn in centralities:
    w, t = in_range_by_metric[cfn]
    print(f"  split by metric - {cfn}: T2 beats fixed_k in {w}/{t}")

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

# --- Paired per-instance comparison: since every policy runs on the SAME
# six network instances (2 sizes x 3 seeds) per family, a paired
# T2-minus-other difference per instance is a much more powerful test than
# comparing independently-aggregated mean curves - it controls for
# instance-to-instance variance instead of treating seed spread as noise
# on each side separately. Report mean +/- standard error across the 6
# paired differences. ---
def paired_diff(policy_a, policy_b, fam, cfn, n_points=5):
    sub = synth[(synth.family == fam) & (synth.centrality == cfn)]
    instances = sub[["N0", "seed"]].drop_duplicates()
    diffs = []
    for _, inst in instances.iterrows():
        inst_df = sub[(sub.N0 == inst.N0) & (sub.seed == inst.seed)]
        a = inst_df[inst_df.policy == policy_a].sort_values("cost_fraction")
        b = inst_df[inst_df.policy == policy_b].sort_values("cost_fraction")
        if len(a) < 2 or len(b) < 2:
            continue
        lo = max(a.cost_fraction.min(), b.cost_fraction.min())
        hi = min(a.cost_fraction.max(), b.cost_fraction.max())
        if lo > hi:
            continue
        for c in np.linspace(lo, hi, n_points):
            ag = np.interp(c, a.cost_fraction, a.gain_retained)
            bg = np.interp(c, b.cost_fraction, b.gain_retained)
            diffs.append(ag - bg)
    if not diffs:
        return None, None, 0
    diffs = np.array(diffs)
    return diffs.mean(), diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0, len(diffs)


print("\n=== Paired per-instance comparison: T2 minus random_control (mean +/- SE) ===")
for fam in families:
    for cfn in centralities:
        mean, se, n = paired_diff("trigger_T2_topdrift", "random_control", fam, cfn)
        if mean is None:
            continue
        sig = "significant" if abs(mean) > 2 * se else "within ~2 SE of zero"
        print(f"{fam}/{cfn}: T2 - random = {mean:+.3f} +/- {se:.3f} (n={n} paired points, {sig})")

print("\n=== Paired per-instance comparison: T2 minus fixed_k (mean +/- SE) ===")
for fam in families:
    for cfn in centralities:
        mean, se, n = paired_diff("trigger_T2_topdrift", "fixed_k", fam, cfn)
        if mean is None:
            continue
        sig = "significant" if abs(mean) > 2 * se else "within ~2 SE of zero"
        print(f"{fam}/{cfn}: T2 - fixed_k = {mean:+.3f} +/- {se:.3f} (n={n} paired points, {sig})")

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
