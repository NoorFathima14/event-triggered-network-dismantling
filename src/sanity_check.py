"""Reproduces the baseline expectation from Wandelt 2018: adaptive attacks
beat static ones, and betweenness-based attacks beat degree-based ones on
most networks. Run this before trusting any trigger results."""
import sys
sys.path.insert(0, ".")

from networks import make_ba, make_er, make_ws
from strategies import run_policy, r_measure

NETS = {
    "ER": make_er(n=300, avg_deg=4, seed=1),
    "BA": make_ba(n=300, m=2, seed=1),
    "WS": make_ws(n=300, k=4, p=0.1, seed=1),
}

for net_name, G in NETS.items():
    print(f"\n=== {net_name} (N={G.number_of_nodes()}, E={G.number_of_edges()}) ===")
    N0 = G.number_of_nodes()
    for cfn in ["degree", "betweenness"]:
        r_static = r_measure(run_policy(G, cfn, "static")["gcc_curve"], N0)
        r_adaptive = r_measure(run_policy(G, cfn, "adaptive")["gcc_curve"], N0)
        ok = r_adaptive <= r_static
        print(f"  {cfn:11s} R_static={r_static:.4f}  R_adaptive={r_adaptive:.4f}  "
              f"adaptive<=static: {ok}")

    r_hd_static = r_measure(run_policy(G, "degree", "static")["gcc_curve"], N0)
    r_hb_static = r_measure(run_policy(G, "betweenness", "static")["gcc_curve"], N0)
    print(f"  betweenness<=degree (static): {r_hb_static <= r_hd_static} "
          f"(R_HB={r_hb_static:.4f}, R_HD={r_hd_static:.4f})")
