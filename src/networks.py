"""Synthetic network generators used throughout the experiments."""
import networkx as nx


def make_er(n=1000, avg_deg=4, seed=0):
    p = avg_deg / (n - 1)
    return nx.gnp_random_graph(n, p, seed=seed)


def make_ba(n=1000, m=2, seed=0):
    return nx.barabasi_albert_graph(n, m, seed=seed)


def make_ws(n=1000, k=4, p=0.1, seed=0):
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def make_sbm(n=1000, n_blocks=4, p_in=0.02, p_out=0.001, seed=0):
    sizes = [n // n_blocks] * n_blocks
    sizes[-1] += n - sum(sizes)
    probs = [[p_in if i == j else p_out for j in range(n_blocks)] for i in range(n_blocks)]
    return nx.stochastic_block_model(sizes, probs, seed=seed)


GENERATORS = {
    "ER": make_er,
    "BA": make_ba,
    "WS": make_ws,
    "SBM": make_sbm,
}
