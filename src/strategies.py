"""Dismantling policies: static, fully adaptive, fixed-k batch, and the
event-triggered lazy adaptive policies (T1 edge-loss, T2 top-set drift)
that are this project's contribution.
"""
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
               top_m=20, stop_frac=0.01, approx_betweenness_k=None):
    """
    Remove nodes from G (a copy is made) under one of:
      policy='static'   -> rank once, remove in that order
      policy='adaptive' -> recompute ranking after every removal
      policy='fixed_k'  -> recompute every k removals
      policy='trigger'  -> recompute when `trigger` fires (T1 or T2)

    Returns dict with:
      gcc_curve: list of largest-component sizes (fraction of N0) after each removal
      n_recomputations: int
      removed_order: list of removed node ids
    """
    G = G.copy()
    N0 = G.number_of_nodes()
    E0 = G.number_of_edges()
    gcc_curve = [1.0]
    n_recomputations = 0
    removed_order = []

    def compute_scores(H):
        if centrality_fn == "degree":
            return degree_centrality_raw(H)
        elif centrality_fn == "betweenness":
            return betweenness_centrality(H, k=approx_betweenness_k)
        else:
            raise ValueError(centrality_fn)

    scores = compute_scores(G)
    n_recomputations += 1
    ranking = _rank(scores)
    edges_at_last_recompute = G.number_of_edges()
    stored_top_m = set(ranking[:top_m])

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
            recompute = trigger(
                G=G,
                E0=edges_at_last_recompute,
                removed_since_recompute=removed_since_recompute,
                stored_top_m=stored_top_m,
                tau=tau,
                top_m=top_m,
            )
        else:
            raise ValueError(policy)

        if recompute:
            scores = compute_scores(G)
            n_recomputations += 1
            ranking = _rank(scores)
            edges_at_last_recompute = G.number_of_edges()
            stored_top_m = set(ranking[:top_m])
            removed_since_recompute = 0

    return {
        "gcc_curve": gcc_curve,
        "n_recomputations": n_recomputations,
        "removed_order": removed_order,
        "N0": N0,
    }


def trigger_edge_loss(G, E0, tau, **kwargs):
    """T1: recompute when the fraction of edges lost since the last
    recomputation exceeds tau."""
    if E0 == 0:
        return False
    current_edges = G.number_of_edges()
    frac_lost = (E0 - current_edges) / E0
    return frac_lost > tau


def trigger_topset_drift(G, stored_top_m, tau, top_m, **kwargs):
    """T2: recompute when the top-m nodes of the stored ranking differ from
    the top-m nodes by *current* degree (cheap to maintain) by more than tau,
    i.e. when their Jaccard overlap falls below (1 - tau)."""
    if G.number_of_nodes() == 0:
        return True
    current_degree_top = set(
        n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_m]
    )
    stored_alive = {n for n in stored_top_m if n in G}
    union = stored_alive | current_degree_top
    if not union:
        return False
    overlap = len(stored_alive & current_degree_top) / len(union)
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
