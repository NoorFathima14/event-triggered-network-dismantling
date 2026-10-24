"""Reproducible measurement of the naive T2 trigger's design flaw: how much
the top-m nodes by betweenness disagree with the top-m nodes by degree, on
the unattacked network, before any edges are removed. This is the number
cited in the paper to justify comparing the degree-based proxy against
itself over time rather than against the attack ranking directly.

Averaged over every synthetic network instance actually used in the main
experiment grid (run_grid.py's SYNTH_SIZES x SYNTH_SEEDS), not a single
seed, so the reported numbers are reproducible from this script alone.
"""
import sys
sys.path.insert(0, ".")

from networks import make_er, make_ba, make_ws, make_sbm
from strategies import betweenness_centrality
from run_grid import SYNTH_SIZES, SYNTH_SEEDS, BETWEENNESS_APPROX_K

TOP_M = 20

GENERATORS = {
    "ER": lambda n, seed: make_er(n=n, avg_deg=4, seed=seed),
    "BA": lambda n, seed: make_ba(n=n, m=2, seed=seed),
    "WS": lambda n, seed: make_ws(n=n, k=4, p=0.1, seed=seed),
    "SBM": lambda n, seed: make_sbm(n=n, seed=seed),
}


def initial_disagreement(G, top_m=TOP_M):
    approx_k = BETWEENNESS_APPROX_K if G.number_of_nodes() > 300 else None
    bet = betweenness_centrality(G, k=approx_k)
    bet_top = set(sorted(bet.keys(), key=lambda n: bet[n], reverse=True)[:top_m])
    deg_top = set(n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_m])
    union = bet_top | deg_top
    overlap = len(bet_top & deg_top) / len(union)
    return 1 - overlap


if __name__ == "__main__":
    for fam, gen in GENERATORS.items():
        values = []
        for n in SYNTH_SIZES:
            for seed in SYNTH_SEEDS:
                G = gen(n, seed)
                d = initial_disagreement(G)
                values.append(d)
                print(f"{fam} n={n} seed={seed}: disagreement={d:.3f}")
        avg = sum(values) / len(values)
        print(f"{fam}: mean over {len(values)} instances = {avg:.3f}\n")
