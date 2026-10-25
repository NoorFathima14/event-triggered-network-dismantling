"""Dismantling policies: static, fully adaptive, fixed-k batch, and the
event-triggered lazy adaptive policies (T1 edge-loss, T2 top-set drift,
T3 sampled-probe drift) that are this project's contribution.
"""
import random
import time

import networkx as nx


def degree_centrality_raw(G):
    return dict(G.degree())


def betweenness_centrality(G, k=None, seed=0):
    # k enables approximate (sampled) betweenness for large graphs; as the
    # graph shrinks during dismantling, k must not exceed the node count
    if k is not None:
        k = min(k, G.number_of_nodes())
    return nx.betweenness_centrality(G, k=k, seed=seed)


def _rank(scores):
    return sorted(scores.keys(), key=lambda n: scores[n], reverse=True)


def run_policy(G, centrality_fn, policy, k=None, trigger=None, tau=None,
               top_m=20, stop_frac=0.01, approx_betweenness_k=None,
               random_p=None, rng=None, probe_k=None, probe_seed=0):
    """
    Remove nodes from G (a copy is made) under one of:
      policy='static'   -> rank once, remove in that order
      policy='adaptive' -> recompute ranking after every removal
      policy='fixed_k'  -> recompute every k removals
      policy='trigger'  -> recompute when `trigger` fires (T1, T2 or T3)
      policy='random'   -> recompute with per-step probability random_p
                           (a matched-cost control: is a real trigger any
                           smarter than recomputing at random?)

    probe_k: number of fixed probe source nodes for T3 (trigger_sampled_probe).
             Only set this (non-None) when running T3 - it is the one trigger
             whose check itself does real algorithmic work, so other policies
             must not pay for maintaining probe state they don't use.

    Returns dict with:
      gcc_curve: list of largest-component sizes (fraction of N0) after each removal
      n_recomputations: int
      removed_order: list of removed node ids
      score_time: total wall-clock seconds spent inside compute_scores
      trigger_time: total wall-clock seconds spent inside the trigger function
                    (for T3, this includes the probe betweenness computation
                    both at recomputation refresh and at every check)
      work_units: total "work" across all recomputes - sources_used * current_edges
                  for betweenness (sources_used = approx_betweenness_k or current
                  node count for exact), current node count for degree
    """
    G = G.copy()
    N0 = G.number_of_nodes()
    E0 = G.number_of_edges()
    initial_largest = max((len(c) for c in nx.connected_components(G)), default=0)
    gcc_curve = [initial_largest / N0]
    n_recomputations = 0
    removed_order = []
    score_time = 0.0
    trigger_time = 0.0
    work_units = 0.0

    def compute_scores(H):
        nonlocal score_time, work_units
        t0 = time.perf_counter()
        if centrality_fn == "degree":
            result = degree_centrality_raw(H)
            work_units += H.number_of_nodes()
        elif centrality_fn == "betweenness":
            result = betweenness_centrality(H, k=approx_betweenness_k)
            sources_used = approx_betweenness_k if approx_betweenness_k else H.number_of_nodes()
            sources_used = min(sources_used, H.number_of_nodes())
            work_units += sources_used * H.number_of_edges()
        else:
            raise ValueError(centrality_fn)
        score_time += time.perf_counter() - t0
        return result

    def degree_top(H):
        return set(n for n, _ in sorted(H.degree(), key=lambda x: x[1], reverse=True)[:top_m])

    probe_sources = None
    if probe_k is not None:
        probe_sources = random.Random(probe_seed).sample(list(G.nodes()), min(probe_k, N0))

    def refresh_probe_top(H):
        nonlocal trigger_time
        t0 = time.perf_counter()
        result = probe_betweenness_top(H, probe_sources, top_m)
        trigger_time += time.perf_counter() - t0
        return result

    def call_trigger(**kwargs):
        nonlocal trigger_time
        t0 = time.perf_counter()
        result = trigger(**kwargs)
        trigger_time += time.perf_counter() - t0
        return result

    scores = compute_scores(G)
    n_recomputations += 1
    ranking = _rank(scores)
    edges_at_last_recompute = G.number_of_edges()
    stored_degree_top = degree_top(G)
    stored_probe_top = refresh_probe_top(G) if probe_sources is not None else None

    removed_since_recompute = 0
    stop_size = stop_frac * N0

    while ranking and G.number_of_nodes() > 0:
        # pick the next alive node from the stored ranking
        node = None
        while ranking:
            cand = ranking.pop(0)
            if cand in G:
                node = cand
                break
        if node is None:
            break

        G.remove_node(node)
        removed_order.append(node)
        removed_since_recompute += 1

        largest = max((len(c) for c in nx.connected_components(G)), default=0)
        gcc_curve.append(largest / N0)

        if largest <= stop_size:
            break

        recompute = False
        if policy == "static":
            recompute = False
        elif policy == "adaptive":
            recompute = True
        elif policy == "fixed_k":
            recompute = removed_since_recompute >= k
        elif policy == "trigger":
            recompute = call_trigger(
                G=G,
                E0=edges_at_last_recompute,
                removed_since_recompute=removed_since_recompute,
                stored_degree_top=stored_degree_top,
                probe_sources=probe_sources,
                stored_probe_top=stored_probe_top,
                tau=tau,
                top_m=top_m,
            )
        elif policy == "random":
            recompute = rng.random() < random_p
        else:
            raise ValueError(policy)

        if recompute:
            scores = compute_scores(G)
            n_recomputations += 1
            ranking = _rank(scores)
            edges_at_last_recompute = G.number_of_edges()
            stored_degree_top = degree_top(G)
            stored_probe_top = refresh_probe_top(G) if probe_sources is not None else None
            removed_since_recompute = 0

    return {
        "gcc_curve": gcc_curve,
        "n_recomputations": n_recomputations,
        "removed_order": removed_order,
        "N0": N0,
        "score_time": score_time,
        "trigger_time": trigger_time,
        "work_units": work_units,
    }


def trigger_edge_loss(G, E0, tau, **kwargs):
    """T1: recompute when the fraction of edges lost since the last
    recomputation exceeds tau."""
    if E0 == 0:
        return False
    current_edges = G.number_of_edges()
    frac_lost = (E0 - current_edges) / E0
    return frac_lost > tau


def trigger_topset_drift(G, stored_degree_top, tau, top_m, **kwargs):
    """T2: recompute when the top-m nodes by degree *at the last
    recomputation* differ from the top-m nodes by *current* degree by more
    than tau (Jaccard overlap below 1 - tau).

    This compares the cheap degree-based proxy signal against itself over
    time, not against the (possibly different) attack ranking. Comparing a
    betweenness-ranked stored top-m against a degree-ranked current top-m
    would conflate "the ranking has drifted" with "degree and betweenness
    rank nodes differently to begin with" - the two top-m sets can already
    differ substantially at the moment of recomputation (fraction of
    non-overlap measured at 0.4-0.7 on our test networks), which forces the
    trigger toward near-every-step recomputation at any reasonably small
    tau regardless of whether the network has actually changed."""
    if G.number_of_nodes() == 0:
        return True
    current_degree_top = set(
        n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_m]
    )
    stored_alive = {n for n in stored_degree_top if n in G}
    union = stored_alive | current_degree_top
    if not union:
        return False
    overlap = len(stored_alive & current_degree_top) / len(union)
    return (1 - overlap) > tau


def probe_betweenness_top(H, probe_sources, top_m):
    """Betweenness restricted to shortest paths from a small, fixed set of
    source nodes to every other node (nx.betweenness_centrality_subset),
    returning the top-m nodes by this probe score. Used by T3 both to
    refresh its stored snapshot at recomputation and, by the trigger
    itself, to get the current reading to compare against that snapshot."""
    alive_sources = [s for s in probe_sources if s in H]
    if not alive_sources or H.number_of_nodes() == 0:
        return set()
    scores = nx.betweenness_centrality_subset(
        H, sources=alive_sources, targets=list(H.nodes()), normalized=True
    )
    return set(n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_m])


def trigger_sampled_probe(G, probe_sources, stored_probe_top, tau, top_m, **kwargs):
    """T3: recompute once the top-m nodes by a cheap betweenness *probe*
    (computed from a small, fixed set of source nodes, chosen once before
    any removals) differ from the same probe's top-m at the last
    recomputation by more than tau (Jaccard overlap below 1 - tau).

    Unlike T1 (edge count, structural) and T2 (a degree-based proxy,
    regardless of the attack metric), T3 is a genuine - if coarse -
    approximation of betweenness itself: it uses real shortest-path
    computations from a handful of fixed pivots, not a different quantity
    assumed to correlate with betweenness drift. Keeping the source set
    fixed across the whole attack (rather than resampling it at every
    check) makes "the probe's ranking now" and "the probe's ranking at
    the last recomputation" a like-for-like comparison. This costs real
    work per check - about (len(probe_sources) / N) of a full
    recomputation - unlike T1/T2's near-free checks, which is why this
    project's Phase 1 timing/work accounting exists: to judge whether that
    extra per-check cost is worth what it buys in recomputations saved."""
    if G.number_of_nodes() == 0 or stored_probe_top is None or not probe_sources:
        return False
    current_top = probe_betweenness_top(G, probe_sources, top_m)
    stored_alive = {n for n in stored_probe_top if n in G}
    union = stored_alive | current_top
    if not union:
        return False
    overlap = len(stored_alive & current_top) / len(union)
    return (1 - overlap) > tau


def r_measure(gcc_curve, N0):
    """Area under the GCC-fraction-removed curve (trapezoidal), normalized
    by the fixed original node count N0, not by curve length. Policies stop
    early once the GCC falls below stop_frac, so the curve is padded with
    its last (small) value out to N0 removals before integrating — using
    curve length instead would unfairly penalize policies that dismantle
    the network in fewer steps. Lower R = more effective attack."""
    curve = list(gcc_curve) + [gcc_curve[-1]] * (N0 + 1 - len(gcc_curve))
    area = 0.0
    for i in range(len(curve) - 1):
        area += (curve[i] + curve[i + 1]) / 2.0
    return area / N0
