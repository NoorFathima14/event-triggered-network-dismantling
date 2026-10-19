"""Full experiment grid: synthetic network families (ER/BA/WS/SBM) across
sizes and seeds, plus real networks, for both degree- and betweenness-based
attacks, across all policies (static/adaptive/fixed_k/trigger T1/T2).

Writes one row per (network, centrality, policy, param) to results/grid.csv.
"""
import sys
import time
sys.path.insert(0, ".")

import pandas as pd

from networks import make_er, make_ba, make_ws, make_sbm
from real_networks import load_real_networks
from experiment import run_all_policies, summarize

SYNTH_SIZES = [300, 600]
SYNTH_SEEDS = [0, 1, 2]
K_VALUES = (10, 25, 50)
TAU_VALUES = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
BETWEENNESS_APPROX_K = 60  # sampled betweenness for speed on larger graphs


def synthetic_networks():
    nets = []
    for n in SYNTH_SIZES:
        for seed in SYNTH_SEEDS:
            nets.append((f"ER_n{n}_s{seed}", "ER", n, seed, make_er(n=n, avg_deg=4, seed=seed)))
            nets.append((f"BA_n{n}_s{seed}", "BA", n, seed, make_ba(n=n, m=2, seed=seed)))
            nets.append((f"WS_n{n}_s{seed}", "WS", n, seed, make_ws(n=n, k=4, p=0.1, seed=seed)))
            nets.append((f"SBM_n{n}_s{seed}", "SBM", n, seed, make_sbm(n=n, seed=seed)))
    return nets


def main():
    rows = []
    t_start = time.time()

    all_nets = []
    for name, family, n, seed, G in synthetic_networks():
        all_nets.append((name, family, seed, G))

    real_nets = load_real_networks(include_midsize=True, midsize_cap=600)
    for name, G in real_nets.items():
        all_nets.append((name, "REAL", None, G))

    for name, family, seed, G in all_nets:
        n_nodes = G.number_of_nodes()
        print(f"[{time.time()-t_start:6.1f}s] {name} (N={n_nodes}, E={G.number_of_edges()})")
        for centrality_fn in ["degree", "betweenness"]:
            approx_k = BETWEENNESS_APPROX_K if (centrality_fn == "betweenness" and n_nodes > 300) else None
            results = run_all_policies(
                G, centrality_fn=centrality_fn,
                k_values=K_VALUES, tau_values=TAU_VALUES,
                approx_betweenness_k=approx_k,
            )
            results = summarize(results)
            for r in results:
                rows.append({
                    "network": name,
                    "family": family,
                    "seed": seed,
                    "N0": r["N0"],
                    "centrality": centrality_fn,
                    "policy": r["policy_name"],
                    "param": r["param"],
                    "R": r["R"],
                    "n_recomputations": r["n_recomputations"],
                    "cost_fraction": r["cost_fraction"],
                    "gain_retained": r["gain_retained"],
                })

    df = pd.DataFrame(rows)
    out_path = "../results/grid.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path} in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
