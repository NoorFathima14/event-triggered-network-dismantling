"""Streamlit dashboard for the Event-Triggered Lazy Adaptive Network
Dismantling project. Run from the project root:

    ./.venv/bin/streamlit run app.py

Reads results/grid.csv, results/best_tau_by_topology.csv,
results/real_networks_summary.csv, and figures/*.png - all produced by
src/run_grid.py and src/analyze.py. Does not re-run the experiment grid;
it only visualizes what is already there.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from stats_utils import paired_test

ROOT = os.path.dirname(__file__)
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

st.set_page_config(page_title="Event-Triggered Lazy Adaptive Dismantling", layout="wide")


@st.cache_data
def load_grid():
    df = pd.read_csv(os.path.join(RESULTS, "grid.csv"))
    df["total_time"] = df["score_time"] + df["trigger_time"]
    return df


@st.cache_data
def load_best_tau():
    return pd.read_csv(os.path.join(RESULTS, "best_tau_by_topology.csv"))


@st.cache_data
def load_real_summary():
    return pd.read_csv(os.path.join(RESULTS, "real_networks_summary.csv"))


@st.cache_data
def example_network(family, n=40, seed=0):
    if family == "ER":
        return nx.gnp_random_graph(n, 4 / (n - 1), seed=seed)
    if family == "BA":
        return nx.barabasi_albert_graph(n, 2, seed=seed)
    if family == "WS":
        return nx.watts_strogatz_graph(n, 4, 0.1, seed=seed)
    if family == "SBM":
        sizes = [n // 4] * 4
        sizes[-1] += n - sum(sizes)
        # p_in here (0.5) is much higher than the real experiment's 0.02 -
        # chosen only so the N=40 illustration looks visually clustered;
        # see the caption below the plots for why the real SBM instances
        # are far sparser than this picture.
        p_in, p_out = 0.5, 0.02
        probs = [[p_in if i == j else p_out for j in range(4)] for i in range(4)]
        return nx.stochastic_block_model(sizes, probs, seed=seed)
    raise ValueError(family)


POLICY_STYLE = [
    ("fixed_k", "o", "tab:blue", "-"),
    ("trigger_T1_edgeloss", "^", "tab:orange", "-"),
    ("trigger_T2_topdrift", "s", "tab:green", "-"),
    ("trigger_T3_probedrift", "D", "tab:red", "-"),
    ("random_control", "x", "gray", "--"),
]
POLICY_LABEL = {
    "fixed_k": "Fixed-k batch",
    "trigger_T1_edgeloss": "T1 (edge-loss)",
    "trigger_T2_topdrift": "T2 (top-set drift)",
    "trigger_T3_probedrift": "T3 (sampled-probe drift)",
    "random_control": "Random control",
}

st.title("Event-Triggered Lazy Adaptive Network Dismantling")
st.caption(
    "Does recomputing an attack ranking only when a cheap signal says it has gone "
    "stale beat a fixed recomputation schedule? Network Science course project."
)

tab_problem, tab_data, tab_method, tab_results = st.tabs(
    ["Problem", "Data & Networks", "Method", "Results"]
)

# ---------------------------------------------------------------- Problem
with tab_problem:
    st.header("The problem")
    st.markdown(
        """
Network dismantling asks: given a network, what short sequence of node
removals destroys its connectivity (giant component) fastest? It is the
attack-side mirror of network robustness, used both to find which parts of
critical infrastructure most need protecting and, directly, to design
disruption strategies (stopping disease spread, breaking up criminal
networks).

Attacks fall on a quality-vs-cost spectrum:

- **Static**: rank nodes once (e.g. by degree or betweenness centrality),
  remove them in that order. Cheap, but the ranking goes stale as the
  network changes.
- **Fully adaptive**: recompute the ranking after *every* removal. Strong,
  but expensive — betweenness recomputation costs up to O(N³) over a full
  attack.
- **Fixed-k batching**: recompute every *k* removals. The only existing
  middle ground — but it recomputes on a schedule, blind to whether the
  ranking has actually gone stale.
"""
    )
    st.subheader("Our question")
    st.markdown(
        """
Can a trigger that recomputes based on the network's **own state** —
rather than a fixed step count — recover more of the adaptive quality gain
per recomputation than fixed-k batching?

We built three such triggers (T1, T2, T3 — see **Method**) and tested
three hypotheses:

- **H1** — trigger policies retain most of the adaptive quality gain using
  far fewer recomputations than fixed-k batching.
- **H2** — the best trigger threshold (τ) depends on network topology.
- **H3** — the benefit of triggering is largest for **betweenness**
  attacks, where each recomputation is most expensive.
"""
    )
    st.info(
        "This project builds on **Wandelt et al. (2018), \"A comparative "
        "analysis of approaches to network-dismantling\"** (Scientific "
        "Reports) for its experimental framing, network families, and the "
        "R-measure robustness metric — see the **Method** tab."
    )

# ------------------------------------------------------------------ Data
with tab_data:
    st.header("Networks used")

    st.subheader("Synthetic families")
    st.markdown(
        """
Four families, at N = 300 and N = 600 nodes, 3 random seeds each
(6 instances per family/size combination):
"""
    )
    synth_table = pd.DataFrame(
        [
            ["ER", "Erdős–Rényi", "avg_deg = 4", "Homogeneous random graph"],
            ["BA", "Barabási–Albert", "m = 2", "Scale-free, hub-dominated"],
            ["WS", "Watts–Strogatz", "k = 4, p = 0.1", "Small-world, ring-lattice based"],
            ["SBM", "Stochastic Block Model", "4 blocks, p_in=0.02, p_out=0.001", "Modular / community structure"],
        ],
        columns=["Code", "Name", "Parameters", "Notes"],
    )
    st.dataframe(synth_table, hide_index=True, use_container_width=True)

    st.markdown("One small example of each family (N=40, for illustration only — experiments use N=300/600):")
    cols = st.columns(4)
    for col, family in zip(cols, ["ER", "BA", "WS", "SBM"]):
        with col:
            G = example_network(family)
            fig, ax = plt.subplots(figsize=(3, 3))
            if family == "WS":
                # spring_layout (force-directed) does not preserve the ring
                # shape WS is built from - circular_layout shows it directly.
                pos = nx.circular_layout(G)
                nx.draw(G, pos, ax=ax, node_size=30, width=0.5, node_color="tab:blue")
            elif family == "SBM":
                # Color by block so the four communities are visually
                # obvious. Note p_in is boosted to 0.5 here purely for this
                # picture (see example_network) - at the real experiment's
                # p_in=0.02, same-community pairs are mostly NOT connected
                # (an edge is a 2% roll per pair, not a guarantee), so a
                # faithful plot would look sparse and scattered, not
                # visibly clustered like this one.
                partition = G.graph["partition"]
                block_of = {}
                for b, members in enumerate(partition):
                    for node in members:
                        block_of[node] = b
                colors = [block_of[n] for n in G.nodes()]
                pos = nx.spring_layout(G, seed=0)
                nx.draw(G, pos, ax=ax, node_size=30, width=0.5,
                        node_color=colors, cmap="tab10")
            else:
                pos = nx.spring_layout(G, seed=0)
                nx.draw(G, pos, ax=ax, node_size=30, width=0.5, node_color="tab:blue")
            ax.set_title(family, fontsize=10)
            st.pyplot(fig)
            plt.close(fig)
    st.caption(
        "WS is drawn with a circular layout to show its ring-lattice origin "
        "(force-directed layout hides it). SBM is colored by community "
        "block, with p_in boosted to 0.5 (vs. the real experiment's 0.02) "
        "purely so the four communities are visibly clustered here — at "
        "the real p_in, an edge within a community is only a 2% chance per "
        "pair, not a guarantee, so the actual networks are far sparser and "
        "less visibly clustered than this picture, and are not fully "
        "connected either (avg degree < 2, near the percolation threshold, "
        "as the paper's Limitations section discusses)."
    )

    st.subheader("Real networks")
    st.markdown(
        """
Four small classic networks (exact betweenness) plus two mid-size
networks from SNAP, capped at 600 nodes (approximate betweenness, k=60,
same as the N=600 synthetic instances):
"""
    )
    real_table = pd.DataFrame(
        [
            ["karate_club", 34, "Zachary's karate club"],
            ["les_miserables", 77, "Character co-appearances, Les Misérables"],
            ["florentine_families", 15, "Marriage ties, Renaissance Florence"],
            ["davis_southern_women", 32, "Southern Women bipartite-derived network"],
            ["ca_grqc", "≤600 (sampled)", "Collaboration network, SNAP"],
            ["facebook_ego", "≤600 (sampled)", "Facebook ego-network, SNAP"],
        ],
        columns=["Network", "Nodes", "Description"],
    )
    st.dataframe(real_table, hide_index=True, use_container_width=True)
    st.caption(
        "Florentine families is excluded from the Results comparisons: both T3 "
        "and the baselines converge to identical, degenerate curves there."
    )

# ---------------------------------------------------------------- Method
with tab_method:
    st.header("Method")

    st.subheader("Robustness measure (R)")
    st.markdown("Following Wandelt et al. (2018):")
    st.latex(r"R = \frac{1}{N} \sum_{Q=1}^{N} s(Q)")
    st.markdown(
        "where *s(Q)* is the giant-component size (as a fraction of N) after "
        "removing Q nodes. **Lower R = a more effective attack.**"
    )

    st.subheader("Policy spectrum")
    st.markdown(
        """
| Policy | Recomputes when... |
| --- | --- |
| **Static** | Never (ranking computed once) |
| **Fully adaptive** | After every single removal |
| **Fixed-k batch** | Every *k* removals, k ∈ {2,3,5,10,25,50,100} |
| **T1 — edge-loss** | Fraction of edges lost since last recompute exceeds τ |
| **T2 — top-set drift** | Top-20 nodes by *current degree* differ from the top-20 by degree *at last recompute* (Jaccard overlap below 1−τ) |
| **T3 — sampled-probe drift** | Top-20 by a cheap betweenness *probe* (computed from 10 fixed source nodes) differ from that probe's own last snapshot |
| **Random control** | Fixed per-step probability *p*, independent of network state |
"""
    )
    st.markdown(
        """
**Why T2 and T3 are built the way they are:** a naive design would compare
the *stored attack ranking* directly against a *different* cheap proxy
(e.g. degree). When the attack metric is betweenness, this conflates
"the ranking has drifted" with "degree and betweenness rank nodes
differently to begin with" — measured initial disagreement of 0.23–0.82
across topologies (`src/initial_drift.py`). T2 and T3 instead compare
their own cheap signal **against its own last snapshot**, decoupled from
whichever metric ranks the actual attack.
"""
    )

    st.subheader("Evaluation metrics")
    st.markdown(
        """
- **cost_fraction** (primary) = recomputations / N
- **total_time** (secondary) = score_time + trigger_time (wall-clock
  seconds) — needed because T3's checks are *not* free (real shortest-path
  computation), unlike T1/T2/fixed-k's near-instant checks
- **gain_retained** = (R_static − R_policy) / (R_static − R_adaptive) — 1.0
  means a policy matches full adaptive quality, 0.0 means no better than
  static
"""
    )

    st.subheader("Statistical methodology")
    st.markdown(
        """
Every policy runs on the **same 6 network instances** (2 sizes × 3 seeds)
per topology, so comparisons are **paired**: for each instance, we sample
5 cost points in the two policies' overlapping range and average them —
giving exactly **one number per instance**. We then run a **paired
t-test** and a **Wilcoxon signed-rank test** on those 6 instance-level
means against zero, reported alongside the "k of 6 instances positive"
consistency count (6 instances gives limited statistical power; we do not
correct for multiple comparisons across families/metrics/policies).

An earlier pass over this data instead sampled 5 points × 6 instances = 30
values and treated them as independent — they are not, since points from
the same instance are correlated — which understated the standard error
and manufactured significance that a proper test did not support.
"""
    )

# --------------------------------------------------------------- Results
with tab_results:
    st.header("Results")

    st.subheader("Headline")
    st.success(
        "**T3 (sampled-probe drift) beats T2 for betweenness attacks in all "
        "four topologies (p ≤ 0.008, the cleanest result in this project)**, "
        "and beats fixed-k / random in most of them. The same signal shows no "
        "reliable benefit — and a large but statistically unconfirmed drop on "
        "Watts-Strogatz — for degree attacks."
    )

    df = load_grid()
    families = ["ER", "BA", "WS", "SBM"]
    centralities = ["degree", "betweenness"]
    synth = df[df.family.isin(families)].copy()

    st.subheader("Pareto comparison (primary metric: recomputation count)")
    st.image(os.path.join(FIGURES, "pareto_by_topology.png"), use_container_width=True)
    st.caption(
        "gain_retained vs cost_fraction, all 5 policies, 4 topologies × "
        "{degree, betweenness}. Shaded bands: ±1 std across 6 instances. "
        "Y-axis clipped to [-0.2, 1.3] (a few WS/degree points have very "
        "high instance-to-instance variance)."
    )

    st.subheader("Cost accounting: recomputations vs. wall-clock time")
    st.image(os.path.join(FIGURES, "pareto_betweenness_by_time.png"), use_container_width=True)
    st.caption(
        "Same betweenness panels, re-plotted on total_time. T3's margin over "
        "fixed-k is not statistically significant on this axis in any "
        "topology (all p > 0.2) — the fair summary is that T3 trades "
        "recomputations for check time roughly evenly, not a win on both axes."
    )

    st.subheader("Interactive explorer")
    c1, c2 = st.columns(2)
    sel_family = c1.selectbox("Network family", families)
    sel_cfn = c2.selectbox("Attack metric", centralities)

    sub = synth[(synth.family == sel_family) & (synth.centrality == sel_cfn)]
    agg = (
        sub.groupby(["policy", "param"], dropna=False)
        .agg(cost_fraction=("cost_fraction", "mean"), gain_retained=("gain_retained", "mean"),
             gain_std=("gain_retained", "std"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy, marker, color, ls in POLICY_STYLE:
        pts = agg[agg.policy == policy].sort_values("cost_fraction")
        if pts.empty:
            continue
        ax.plot(pts.cost_fraction, pts.gain_retained, marker=marker, color=color,
                linestyle=ls, label=POLICY_LABEL[policy], linewidth=1.5)
        std = pts.gain_std.fillna(0)
        ax.fill_between(pts.cost_fraction, pts.gain_retained - std, pts.gain_retained + std,
                         color=color, alpha=0.12)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(-0.2, 1.3)
    ax.set_xlabel("cost fraction (recomputations / N)")
    ax.set_ylabel("gain retained")
    ax.set_title(f"{sel_family} / {sel_cfn}")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Paired significance (T3 vs. each baseline)")
    rows = []
    for comparator in ["fixed_k", "random_control", "trigger_T2_topdrift"]:
        result = paired_test(synth, "trigger_T3_probedrift", comparator, sel_family, sel_cfn)
        if result is None:
            continue
        mean, t_p, w_p, n_pos, n = result
        rows.append({
            "Comparison": f"T3 vs. {POLICY_LABEL.get(comparator, comparator)}",
            "Mean diff": f"{mean:+.3f}",
            "t-test p": f"{t_p:.3f}",
            "Wilcoxon p": f"{w_p:.3f}",
            "Instances positive": f"{n_pos}/{n}",
            "p<0.05": "✓" if t_p < 0.05 else "",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.write("No overlapping-cost comparison available for this selection.")

    st.subheader("Best τ per topology (cheapest to reach gain_retained ≥ 0.95)")
    st.dataframe(load_best_tau(), hide_index=True, use_container_width=True)

    st.subheader("Real-network summary")
    st.dataframe(load_real_summary(), hide_index=True, use_container_width=True)
    st.caption(
        "Single deterministic run per network (no seeds) — margins, not a "
        "statistic. See the paper for the full breakdown against random and "
        "fixed-k."
    )
