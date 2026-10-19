"""Real-world networks: small ones bundled with networkx, plus mid-size
edge-list networks downloaded from SNAP into data/real/. Each is reduced to
its largest connected component as an undirected simple graph."""
import os
import networkx as nx

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real")


def _lcc_undirected(G):
    G = nx.Graph(G)  # drop direction/multiplicity
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() == 0:
        return G
    largest_cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(largest_cc).copy())


def _load_edgelist(filename):
    path = os.path.join(DATA_DIR, filename)
    return nx.read_edgelist(path, nodetype=int)


def load_real_networks(include_midsize=True, midsize_cap=None):
    """midsize_cap: if set, subsample mid-size networks down to this many
    nodes (via a BFS ego-sample) to keep betweenness tractable."""
    loaders = {
        "karate_club": nx.karate_club_graph,
        "les_miserables": nx.les_miserables_graph,
        "florentine_families": nx.florentine_families_graph,
        "davis_southern_women": nx.davis_southern_women_graph,
    }
    out = {}
    for name, fn in loaders.items():
        out[name] = _lcc_undirected(fn())

    if include_midsize:
        midsize_files = {
            "facebook_ego": "facebook_combined.txt",
            "ca_grqc": "ca-GrQc.txt",
        }
        for name, fname in midsize_files.items():
            path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(path):
                continue
            G = _lcc_undirected(_load_edgelist(fname))
            if midsize_cap and G.number_of_nodes() > midsize_cap:
                G = _bfs_sample(G, midsize_cap, seed=0)
            out[name] = G
    return out


def _bfs_sample(G, n_target, seed=0):
    import random
    rng = random.Random(seed)
    start = rng.choice(list(G.nodes()))
    visited = [start]
    seen = {start}
    frontier = [start]
    while frontier and len(visited) < n_target:
        node = frontier.pop(0)
        neighbors = list(G.neighbors(node))
        rng.shuffle(neighbors)
        for nb in neighbors:
            if nb not in seen:
                seen.add(nb)
                visited.append(nb)
                frontier.append(nb)
                if len(visited) >= n_target:
                    break
    return _lcc_undirected(G.subgraph(visited).copy())
