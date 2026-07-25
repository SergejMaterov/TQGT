#!/usr/bin/env python3
"""
uv_dimensional_reduction_check.py — Quantumograph UV dimensional-reduction check
==================================================================================
Tests whether the local (t-resolved) spectral dimension of the Z^4 torus shows
any reduction toward d_s~2 at short diffusion times (the UV/small-t regime),
as seen in CDT / causal-set quantum gravity. Unlike finite_size_scaling.py
(which fits a single global exponent over a diffusive window), this script
computes dim_S(t) as a local central-difference slope at each even t, with
bootstrap error bars, on a single LARGE, fixed L (so finite-size/IR effects
are separated from short-time/UV behavior by construction).

Method: identical random-walk-return-probability estimator as
finite_size_scaling.py / auto_continuum_check_v3.py (even-t sampling to avoid
bipartite oscillation on the bipartite Z^4 torus), but reporting the local
slope dim_S(t) = -2 d(log P_return)/d(log t) via central differences, rather
than a single global power-law fit.

Usage:
  python3 uv_dimensional_reduction_check.py --L 21 --Tmax 24 --nwalks 1500000
"""
import argparse
import random
import numpy as np
import networkx as nx


def odd(L):
    return L if L % 2 == 1 else L + 1


def build_Z4(L):
    L = odd(L)
    G = nx.cycle_graph(L)
    for _ in range(3):
        G = nx.cartesian_product(G, nx.cycle_graph(L))
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def local_dimS_curve(G, nwalks, Tmax, seed=42, n_starts=400, n_boot=20):
    """Returns list of (t, dim_S(t), bootstrap_err, P_return(t))."""
    rng = random.Random(seed)
    n = G.number_of_nodes()
    nbrs = {u: list(G.neighbors(u)) for u in G.nodes()}
    starts = rng.sample(list(G.nodes()), min(n, n_starts))
    per_s = max(1, nwalks // len(starts))

    returns = []
    P = np.zeros(Tmax + 1)
    cnt = np.zeros(Tmax + 1)
    for s in starts:
        for _ in range(per_s):
            cur = s
            ret = np.zeros(Tmax + 1, dtype=bool)
            P[0] += 1
            cnt[0] += 1
            for t in range(1, Tmax + 1):
                cur = rng.choice(nbrs[cur])
                cnt[t] += 1
                if cur == s:
                    P[t] += 1
                    ret[t] = True
            returns.append(ret)
    returns = np.array(returns)
    prob = P / np.maximum(1., cnt)

    t_even = np.arange(2, Tmax - 1, 2)
    results = []
    brng = random.Random(seed + 777)
    nwalk_total = returns.shape[0]
    for t in t_even:
        t1, t2 = t - 2, t + 2
        if t1 < 2 or prob[t1] <= 0 or prob[t2] <= 0:
            continue
        slope = (np.log(prob[t2]) - np.log(prob[t1])) / (np.log(t2) - np.log(t1))
        d_central = -2.0 * slope
        boots = []
        for _ in range(n_boot):
            idx = brng.choices(range(nwalk_total), k=nwalk_total)
            sub = returns[idx]
            p1 = sub[:, t1].mean()
            p2 = sub[:, t2].mean()
            if p1 > 0 and p2 > 0:
                s_boot = (np.log(p2) - np.log(p1)) / (np.log(t2) - np.log(t1))
                boots.append(-2.0 * s_boot)
        err = float(np.std(boots)) if len(boots) > 3 else float('nan')
        results.append((int(t), float(d_central), err, float(prob[t])))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--L", type=int, default=21, help="Torus side length (rounded up to odd)")
    ap.add_argument("--Tmax", type=int, default=24,
                     help="Max diffusion time (kept modest to stay in the high-statistics regime)")
    ap.add_argument("--nwalks", type=int, default=1_500_000)
    ap.add_argument("--n_starts", type=int, default=400)
    ap.add_argument("--n_boot", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    G = build_Z4(args.L)
    print(f"Z^4 torus, L={odd(args.L)} (odd), N={G.number_of_nodes()} vertices, degree=8")
    print(f"{'t':>4} {'dim_S(t)':>10} {'+/- boot':>10} {'P_return(t)':>14}")
    for t, d, err, pr in local_dimS_curve(
        G, args.nwalks, args.Tmax, seed=args.seed,
        n_starts=args.n_starts, n_boot=args.n_boot
    ):
        print(f"{t:4d} {d:10.3f} {err:10.3f} {pr:14.6e}")


if __name__ == "__main__":
    main()
