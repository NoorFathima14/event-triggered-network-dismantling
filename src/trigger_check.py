"""Quick check that trigger policies run and land between static and
adaptive in both R-measure and recomputation cost."""
import sys
sys.path.insert(0, ".")

from networks import make_ba
from experiment import run_all_policies, summarize

G = make_ba(n=300, m=2, seed=1)
results = run_all_policies(G, centrality_fn="degree",
                            k_values=(10, 25), tau_values=(0.1, 0.2))
results = summarize(results)

print(f"{'policy':24s} {'param':>6s} {'R':>8s} {'recomps':>8s} {'cost_frac':>10s} {'gain_ret':>9s}")
for r in results:
    param = "" if r["param"] is None else str(r["param"])
    gr = f"{r['gain_retained']:.3f}" if r["gain_retained"] is not None else "n/a"
    print(f"{r['policy_name']:24s} {param:>6s} {r['R']:8.4f} {r['n_recomputations']:8d} "
          f"{r['cost_fraction']:10.4f} {gr:>9s}")
