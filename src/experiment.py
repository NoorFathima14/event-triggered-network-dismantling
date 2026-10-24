"""Run all dismantling policies on a graph and collect comparable metrics."""
import random

from strategies import run_policy, trigger_edge_loss, trigger_topset_drift, r_measure

RANDOM_P_VALUES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def run_all_policies(G, centrality_fn, k_values=(10, 25, 50), tau_values=(0.05, 0.1, 0.2, 0.3),
                      top_m=20, stop_frac=0.01, approx_betweenness_k=None,
                      random_p_values=RANDOM_P_VALUES, random_seed=0):
    """Run static, adaptive, fixed-k (several k), trigger T1/T2 (several
    tau), and a random-recompute control (several per-step probabilities)
    on G. The random control answers a question the trigger results alone
    cannot: is a trigger's cost-quality tradeoff actually better than
    recomputing at random with a matched average cost, or does any
    schedule with a similar number of recomputations do about as well?
    Returns a list of result dicts, one per policy/param."""
    results = []

    common = dict(centrality_fn=centrality_fn, stop_frac=stop_frac,
                  approx_betweenness_k=approx_betweenness_k)

    for name, policy, extra in [
        ("static", "static", {}),
        ("adaptive", "adaptive", {}),
    ]:
        res = run_policy(G, policy=policy, **common, **extra)
        res.update(policy_name=name, param=None)
        results.append(res)

    for k in k_values:
        res = run_policy(G, policy="fixed_k", k=k, **common)
        res.update(policy_name=f"fixed_k", param=k)
        results.append(res)

    for tau in tau_values:
        res = run_policy(G, policy="trigger", trigger=trigger_edge_loss, tau=tau,
                          top_m=top_m, **common)
        res.update(policy_name="trigger_T1_edgeloss", param=tau)
        results.append(res)

        res = run_policy(G, policy="trigger", trigger=trigger_topset_drift, tau=tau,
                          top_m=top_m, **common)
        res.update(policy_name="trigger_T2_topdrift", param=tau)
        results.append(res)

    for p in random_p_values:
        res = run_policy(G, policy="random", random_p=p, rng=random.Random(random_seed),
                          **common)
        res.update(policy_name="random_control", param=p)
        results.append(res)

    return results


def summarize(results):
    """Attach R-measure, gain-retained and cost-fraction to each result.
    Requires 'static' and 'adaptive' entries to be present in results."""
    r_static = None
    r_adaptive = None
    for res in results:
        res["R"] = r_measure(res["gcc_curve"], res["N0"])
        if res["policy_name"] == "static":
            r_static = res["R"]
        if res["policy_name"] == "adaptive":
            r_adaptive = res["R"]

    denom = (r_static - r_adaptive) if (r_static is not None and r_adaptive is not None and r_static != r_adaptive) else None

    for res in results:
        N0 = res["N0"]
        res["cost_fraction"] = res["n_recomputations"] / N0
        if denom:
            res["gain_retained"] = (r_static - res["R"]) / denom
        else:
            res["gain_retained"] = None
    return results
