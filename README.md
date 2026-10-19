# Event-Triggered Lazy Adaptive Network Dismantling

Network Science course project (M.Sc. Data Science). Takes the experimental
framework from Wandelt et al. 2018, *"A comparative analysis of approaches
to network-dismantling"* (Scientific Reports), and proposes an improvement:
**recompute the attack ranking only when a cheap signal says it has gone
stale, instead of on a fixed schedule.**

## Background

- **Static attacks** (HD, HB) rank nodes once and remove them in that
  order — cheap but weak.
- **Fully adaptive attacks** (HDA, HBA) recompute the ranking after every
  single removal — strong but expensive, especially for betweenness.
- **Fixed-k batching** is the only existing middle ground: recompute every
  k removals, whether or not the ranking has actually gone stale.

## Contribution

Two **event-triggered** recomputation policies that fire on network state
rather than a fixed step count:

- **T1 (edge-loss trigger):** recompute once the fraction of edges removed
  since the last recomputation exceeds τ.
- **T2 (top-set drift trigger):** track current node degrees (cheap to
  maintain) and recompute once the top-m nodes of the stored ranking
  diverge from the top-m nodes by current degree beyond τ.

Both are compared against static, fully adaptive, and fixed-k baselines at
**equal recomputation budget**, using:

- `R`: area under the GCC-vs-fraction-removed curve (lower = better attack)
- `gain_retained`: how much of the adaptive-over-static improvement a
  policy keeps `(R_static - R_policy) / (R_static - R_adaptive)`
- `cost_fraction`: recomputations / N

## Key finding so far

T2 (top-set drift) retains almost all of the fully-adaptive gain
(`gain_retained ≈ 1.0`) at a fraction of the recomputation cost, across
ER/BA/WS/SBM synthetic networks and several real networks, for both degree
and betweenness centrality. T1 (edge-loss) works well for degree but
underperforms for betweenness — edge count alone is a weak staleness
signal when betweenness is the attack metric.

## Repo layout

```
src/
  networks.py        synthetic network generators (ER, BA, WS, SBM)
  real_networks.py    real-world networks (networkx built-ins + SNAP edgelists)
  strategies.py        removal engine: static/adaptive/fixed_k/trigger policies, T1/T2, R-measure
  experiment.py         runs all policies on one graph, computes gain_retained/cost_fraction
  run_grid.py           full experiment grid -> results/grid.csv
  analyze.py            Pareto plots + summary tables from results/grid.csv
  sanity_check.py       reproduces "adaptive beats static" as a regression check
  trigger_check.py      quick single-network check that triggers land between static/adaptive
data/real/              downloaded SNAP edgelists (facebook_combined.txt, ca-GrQc.txt) — gitignored, re-fetch as needed
results/                 generated CSVs (gitignored)
figures/                 generated plots (gitignored)
```

## Running it

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cd src
../.venv/bin/python sanity_check.py   # regression check
../.venv/bin/python run_grid.py       # full experiment grid -> ../results/grid.csv
../.venv/bin/python analyze.py        # plots + summary tables
```

Real-network edgelists (Facebook ego-network, GrQc collaboration network)
are downloaded from SNAP (https://snap.stanford.edu/data/) into
`data/real/` on first run if not already present; small ones (karate club,
Les Misérables, Florentine families, Davis Southern women) come bundled
with networkx.

## Reference paper

Wandelt, S., Sun, X., Feng, D., Zanin, M., & Havlin, S. (2018). A
comparative analysis of approaches to network-dismantling. *Scientific
Reports*, 8, 13513. https://pmc.ncbi.nlm.nih.gov/articles/PMC6131543
