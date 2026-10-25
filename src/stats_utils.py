"""Shared paired-comparison statistics, used by both analyze.py and the
Streamlit app. Every policy runs on the SAME six network instances (2
sizes x 3 seeds) per family, so a paired difference per instance is a much
more powerful test than comparing independently-aggregated mean curves.

The real sample size is 6 instances, NOT 6 instances x 5 sampled cost
points = 30: points sampled within one instance's cost-gain curve are
strongly correlated (same run, read at nearby costs), so treating them as
independent understates the standard error severely. We average the 5
sampled points WITHIN each instance first, giving exactly one number per
instance, then run a paired t-test (and the nonparametric Wilcoxon
signed-rank test as a check) across those instance-level means against
zero. With only 6 instances, report the test as supporting evidence
alongside the "k of 6 instances positive" consistency count, not as a
standalone p<0.05 claim - and note this does not correct for the many
comparisons run across families/metrics/policies.
"""
import numpy as np
from scipy import stats


def paired_instance_means(synth, policy_a, policy_b, fam, cfn, cost_col="cost_fraction", n_points=5):
    """One difference (policy_a - policy_b) per network instance, each
    itself averaged over n_points sampled costs within that instance's own
    overlapping cost range. Returns an array of per-instance means."""
    sub = synth[(synth.family == fam) & (synth.centrality == cfn)]
    instances = sub[["N0", "seed"]].drop_duplicates()
    instance_means = []
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
        diffs = []
        for c in np.linspace(lo, hi, n_points):
            ag = np.interp(c, a[cost_col], a.gain_retained)
            bg = np.interp(c, b[cost_col], b.gain_retained)
            diffs.append(ag - bg)
        instance_means.append(np.mean(diffs))
    return np.array(instance_means)


def paired_test(synth, policy_a, policy_b, fam, cfn, cost_col="cost_fraction", n_points=5):
    """t-test and Wilcoxon signed-rank test of instance-level means against
    zero. Returns (mean, t_pvalue, wilcoxon_pvalue, n_positive, n_total), or
    None if fewer than 2 instances have an overlapping cost range."""
    means = paired_instance_means(synth, policy_a, policy_b, fam, cfn, cost_col, n_points)
    if len(means) < 2:
        return None
    mean = means.mean()
    t_p = stats.ttest_1samp(means, 0.0).pvalue
    try:
        w_p = stats.wilcoxon(means).pvalue
    except ValueError:
        w_p = float("nan")  # all differences identical (e.g. all zero)
    n_pos = int((means > 0).sum())
    return mean, t_p, w_p, n_pos, len(means)
