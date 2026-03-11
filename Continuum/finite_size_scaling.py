#!/usr/bin/env python3
"""
finite_size_scaling.py  —  Quantumograph d_s finite-size scaling
=================================================================
Runs Z^d lattices for L = Lmin..Lmax (odd only), computes spectral
dimension d_s via random-walk return probability P(t)~t^{-d_s/2},
extrapolates d_s(L→∞) = d_inf + a/L², saves a publication-quality plot.

Usage examples:
  python3 finite_size_scaling.py
  python3 finite_size_scaling.py --Lmax 13 --dims 2 3 4 --nwalks 30000
  python3 finite_size_scaling.py --Lmax 9  --dims 4     --nwalks 60000
  python3 finite_size_scaling.py --Lmax 11 --dims 4     --x_axis n
"""
import argparse, os, json, math, random, time
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ─── CLI ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(
    description="Finite-size scaling of spectral dimension for Z^d lattices")
ap.add_argument("--Lmin",   type=int,   default=5,
                help="Minimum L (will be rounded up to odd, default 5)")
ap.add_argument("--Lmax",   type=int,   default=13,
                help="Maximum L (will be rounded up to odd, default 13)")
ap.add_argument("--Lstep",  type=int,   default=2,
                help="Step between L values (default 2 → all odd L)")
ap.add_argument("--dims",   type=int,   nargs="+", default=[2, 3, 4],
                help="Lattice dimensions to scan (default: 2 3 4)")
ap.add_argument("--nwalks", type=int,   default=30000,
                help="Total number of random walks per (L,d) point")
ap.add_argument("--n_boot", type=int,   default=30,
                help="Bootstrap resamples for error bars")
ap.add_argument("--seed",   type=int,   default=42)
ap.add_argument("--out",    type=str,   default="fss_out",
                help="Output directory")
ap.add_argument("--x_axis", choices=["L","n"], default="L",
                help="X-axis of the main plot: ring size L or vertex count n")
ap.add_argument("--force_model", type=str, default=None,
                choices=["power","log","power4"],
                help="Override AIC: force fit model for all dims. "
                     "power=standard, log=+logcorr(d=2), power4=+L4(d=3,4)")
ap.add_argument("--min_pts_log", type=int, default=6,
                help="Min data points before log/power4 models enter AIC competition (default 6)")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)

def odd(L): return L if L % 2 == 1 else L + 1

# ─── BUILD Z^d TORUS ─────────────────────────────────────────────────────────
def build_Zd(L, d):
    L = odd(L)
    G = nx.cycle_graph(L)
    for _ in range(d - 1):
        G = nx.cartesian_product(G, nx.cycle_graph(L))
    return nx.convert_node_labels_to_integers(G, ordering="sorted")

# ─── SPECTRAL DIM VIA RANDOM WALK + BOOTSTRAP ────────────────────────────────
def linfit(x, y):
    return float(np.polyfit(x, y, 1)[0])

def ds_rw(G, nwalks, seed, n_boot):
    """
    Returns (d_s, sigma, diam, t_fit_max).
    Uses even-t return probability to avoid bipartite oscillation.
    Fits in early diffusive window [4, diam²/8] before saturation.
    Bootstrap resampling for error bars.
    """
    rng  = random.Random(seed)
    n    = G.number_of_nodes()
    nbrs = {u: list(G.neighbors(u)) for u in G.nodes()}

    # diameter via BFS
    s0      = rng.choice(list(G.nodes()))
    lengths = nx.single_source_shortest_path_length(G, s0, cutoff=2000)
    diam    = max(lengths.values(), default=4)

    # fit window: even t in [4, diam²/8]
    t_fit_max = max(10, diam * diam // 8)
    Tmax      = t_fit_max + 2

    starts = rng.sample(list(G.nodes()), min(n, 200))
    per_s  = max(1, nwalks // len(starts))

    # collect walks
    walks = []
    P     = np.zeros(Tmax + 1)
    cnt   = np.zeros(Tmax + 1)

    for s in starts:
        for _ in range(per_s):
            cur = s
            ret = np.zeros(Tmax + 1, dtype=bool)
            P[0]   += 1; cnt[0]   += 1
            for t in range(1, Tmax + 1):
                cur = rng.choice(nbrs[cur])
                cnt[t] += 1
                if cur == s:
                    P[t]   += 1
                    ret[t]  = True
            walks.append((s, ret))

    def fit_p(P_arr):
        prob   = P_arr / np.maximum(1., cnt)
        t_even = np.arange(4, t_fit_max + 1, 2)
        mask   = (prob[t_even] > 1e-9) & np.isfinite(prob[t_even])
        if mask.sum() < 4:
            return None
        lx    = np.log(t_even[mask].astype(float))
        ly    = np.log(prob[t_even[mask]])
        n_fit = max(4, int(mask.sum() * 0.6))
        return float(-2.0 * linfit(lx[:n_fit], ly[:n_fit]))

    ds_c = fit_p(P)

    # bootstrap
    brng = random.Random(seed + 31337)
    nw   = len(walks)
    boot = []
    for _ in range(n_boot):
        idx = brng.choices(range(nw), k=nw)
        Pb  = np.zeros(Tmax + 1)
        for i in idx:
            _, ret = walks[i]
            Pb[0] += 1
            for t in range(1, Tmax + 1):
                if ret[t]: Pb[t] += 1
        d = fit_p(Pb)
        if d is not None:
            boot.append(d)

    se = float(np.std(boot)) if len(boot) > 3 else None
    return ds_c, se, diam, t_fit_max

# ─── UNIVERSAL EXTRAPOLATION WITH AIC MODEL SELECTION ────────────────────────
#
# Three candidate models:
#   M1 "power":  d_s(L) = d_inf + a/L²              (standard FSS)
#   M2 "log":    d_s(L) = d_inf + a/L² + b/ln(L)    (log correction, d=2)
#   M3 "power4": d_s(L) = d_inf + a/L² + b/L⁴       (higher order, d=3)
#
# Best model chosen automatically by AIC = n·ln(RSS/n) + 2k
# No dimension hardcoding — fully universal.

def _wls(A, y, w):
    W  = np.diag(np.sqrt(w))
    Aw = W @ A; yw = W @ y
    sol, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    rss = float(np.sum(w * (y - A @ sol) ** 2))
    cov = np.linalg.pinv(A.T @ np.diag(w) @ A)
    return sol, cov, rss

def _aic(n, rss, k):
    """AICc — corrected AIC for small samples.
    AICc = AIC + 2k(k+1)/(n-k-1)
    Penalizes extra parameters more strongly when n is small,
    preventing overfitting of log/power4 models at moderate L.
    Reduces to standard AIC as n -> inf.
    """
    if rss <= 0 or n <= k + 1: return np.inf
    aic  = n * math.log(rss / n) + 2 * k
    aicc = aic + 2 * k * (k + 1) / max(1, n - k - 1)
    return aicc

def extrapolate(L_arr, ds_arr, se_arr,
                force_model=None, min_pts_log=6):
    """
    Universal FSS extrapolation with automatic AIC model selection.

    Models:
      "power"  — d_inf + a/L²           (always, needs n>=3)
      "log"    — d_inf + a/L² + b/lnL   (needs n>=min_pts_log, best for d=2)
      "power4" — d_inf + a/L² + b/L⁴   (needs n>=min_pts_log, best for d=3,4)

    AIC = n*ln(RSS/n) + 2k  selects the model automatically.
    Override: force_model="log"|"power"|"power4"
    """
    L  = np.asarray(L_arr, dtype=float)
    ds = np.asarray(ds_arr, dtype=float)
    w  = 1.0 / np.maximum(se_arr, 0.05) ** 2
    n  = len(L)
    x2   = 1.0 / L ** 2
    xlnL = 1.0 / np.log(np.maximum(L, 2.0))
    x4   = 1.0 / L ** 4
    ones = np.ones(n)
    # (design_matrix, param_names, latex_label, min_points_required)
    candidates = {
        "power":  (np.column_stack([ones, x2]),
                   ["d_inf","a"],
                   r"$d_s(\infty) + a/L^2$",
                   3),
        "log":    (np.column_stack([ones, x2, xlnL]),
                   ["d_inf","a","b"],
                   r"$d_s(\infty) + a/L^2 + b/\ln L$",
                   min_pts_log),
        "power4": (np.column_stack([ones, x2, x4]),
                   ["d_inf","a","b"],
                   r"$d_s(\infty) + a/L^2 + b/L^4$",
                   min_pts_log),
    }
    if force_model is not None:
        if force_model not in candidates:
            raise ValueError(f"Unknown model: {force_model}")
        candidates = {force_model: candidates[force_model]}

    best = None
    for name, (A, pnames, label, min_n) in candidates.items():
        if n < min_n: continue
        if n < A.shape[1] + 1: continue
        sol, cov, rss = _wls(A, ds, w)
        aic = _aic(n, rss, A.shape[1])
        # Physical constraint: log correction b/lnL should decrease d_s(L)
        # i.e. b < 0 (since P(t) ~ 1/(t*lnT) makes apparent d_s < true d_s)
        # If b > 0 the log model is overfitting — disqualify it
        if name == "log" and len(sol) > 2 and sol[2] > 0:
            continue
        if best is None or aic < best["aic"]:
            best = dict(model_name=name, model_label=label,
                        params=sol, param_names=pnames,
                        cov=cov, rss=rss, aic=aic)
    if best is None:
        raise ValueError("Not enough data points for any model")
    d_inf     = float(best["params"][0])
    d_inf_err = float(np.sqrt(best["cov"][0, 0]))
    a         = float(best["params"][1])
    fit_params = {pn: float(pv)
                  for pn, pv in zip(best["param_names"], best["params"])}
    return dict(d_inf=d_inf, d_inf_err=d_inf_err, a=a,
                model_name=best["model_name"],
                model_label=best["model_label"],
                aic=float(best["aic"]), fit_params=fit_params,
                rss=float(best["rss"]))

def predict_fit(L_dense, fit_result):
    p  = fit_result["fit_params"]
    mn = fit_result["model_name"]
    y  = p["d_inf"] + p["a"] / L_dense ** 2
    if mn == "log":    y += p["b"] / np.log(np.maximum(L_dense, 2.0))
    if mn == "power4": y += p["b"] / L_dense ** 4
    return y

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
all_data = {}   # d -> list of row dicts

for d in args.dims:
    print(f"\n{'='*62}")
    print(f"  Z^{d} lattice  (d_s target = {d})")
    print(f"{'='*62}")
    print(f"{'L':>4}  {'n':>7}  {'diam':>5}  {'t_fit':>6}  "
          f"{'d_s':>7}  {'±σ':>6}  {'time':>6}")
    print("─" * 55)

    rows   = []
    L_list = range(odd(args.Lmin), odd(args.Lmax) + 1, args.Lstep)

    for L in L_list:
        L = odd(L)
        if L > odd(args.Lmax):
            break
        t0 = time.time()
        G  = build_Zd(L, d)
        n  = G.number_of_nodes()

        ds, se, diam, tfm = ds_rw(G, nwalks=args.nwalks,
                                   seed=args.seed, n_boot=args.n_boot)
        elapsed = time.time() - t0

        ds_s = f"{ds:.4f}" if ds is not None else "  None"
        se_s = f"{se:.4f}" if se is not None else "   ?"
        print(f"L={L:2d}  n={n:7d}  d={diam:4d}  t={tfm:5d}  "
              f"ds={ds_s}  ±{se_s}  {elapsed:.1f}s")

        if ds is not None:
            rows.append(dict(L=L, n=n, ds=ds, se=se if se else 0.1,
                             diam=diam, d=d))

    all_data[d] = rows
    json.dump(rows, open(os.path.join(args.out, f"d{d}_rows.json"), "w"), indent=2)

    if len(rows) >= 3:
        Lr  = np.array([r["L"]  for r in rows], dtype=float)
        dsr = np.array([r["ds"] for r in rows], dtype=float)
        ser = np.array([r["se"] for r in rows], dtype=float)
        fit = extrapolate(Lr, dsr, ser,
                          force_model=args.force_model,
                          min_pts_log=args.min_pts_log)
        print(f"\n  Model: {fit['model_name']}  (AIC={fit['aic']:.1f})")
        print(f"  Fit:   {fit['model_label']}")
        print(f"  Params: " + "  ".join(f"{k}={v:.4f}" for k,v in fit['fit_params'].items()))
        print(f"  d_s(∞) = {fit['d_inf']:.4f} ± {fit['d_inf_err']:.4f}   (target: {d})")
        for r in rows:
            r.update(d_inf=fit["d_inf"], a=fit["a"],
                     d_inf_err=fit["d_inf_err"],
                     model_name=fit["model_name"],
                     fit_params=fit["fit_params"])
        json.dump(rows, open(os.path.join(args.out, f"d{d}_rows.json"), "w"), indent=2)

# ─── PLOT ─────────────────────────────────────────────────────────────────────
COLORS  = {2: "#2176AE", 3: "#20A39E", 4: "#EF5B5B", 5: "#9B59B6"}
MARKERS = {2: "o",       3: "s",       4: "^",       5: "D"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    "Quantumograph: spectral dimension $d_s$ vs lattice size\n"
    r"(FSS with auto model selection via AIC: power / log / power4)",
    fontsize=12, fontweight="bold")

for ax, xscale in [(ax1, "linear"), (ax2, "log")]:
    for d, rows in all_data.items():
        if not rows:
            continue
        Lr  = np.array([r["L"]  for r in rows], dtype=float)
        nr  = np.array([r["n"]  for r in rows], dtype=float)
        dsr = np.array([r["ds"] for r in rows], dtype=float)
        ser = np.array([r["se"] for r in rows], dtype=float)

        col = COLORS.get(d, "gray")
        mk  = MARKERS.get(d, "o")
        xv  = nr if args.x_axis == "n" else Lr

        # data + 2σ error bars
        ax.errorbar(xv, dsr, yerr=2*ser,
                    fmt=mk, color=col, ms=7, lw=1.8,
                    capsize=4, capthick=1.5, zorder=4,
                    label=f"$\\mathbb{{Z}}^{d}$, d={d}")

        # fit curve via universal predict_fit
        if len(rows) >= 3 and "d_inf" in rows[0]:
            d_inf  = rows[0]["d_inf"]
            fr     = dict(model_name=rows[0]["model_name"],
                          fit_params=rows[0]["fit_params"])
            L_dense = np.linspace(Lr.min() * 0.9, Lr.max() * 1.2, 300)
            fit_ds  = predict_fit(L_dense, fr)
            x_dense = L_dense ** d if args.x_axis == "n" else L_dense
            ax.plot(x_dense, fit_ds, "--", color=col, lw=1.5, alpha=0.75)
            ax.axhline(d_inf, color=col, lw=0.7, ls=":", alpha=0.4)
            mn_short = {"power":"", "log":" +log", "power4":" +L⁴"}
            ax.annotate(
                "$d_s(\\infty)=" + f"{d_inf:.2f}$" + mn_short.get(rows[0]["model_name"],""),
                xy=(x_dense[-1], d_inf),
                xytext=(5, 3), textcoords="offset points",
                fontsize=8.5, color=col, va="bottom")

        # integer reference
        ax.axhline(d, color=col, lw=0.5, ls="-.", alpha=0.2)

    if xscale == "log":
        ax.set_xscale("log")
    xlabel = ("$n = L^d$ (vertices)" if args.x_axis == "n"
              else "Ring size $L$")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Spectral dimension $d_s$", fontsize=11)
    ax.set_title(f"{'Linear' if xscale=='linear' else 'Log'} scale", fontsize=10)
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.45)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # custom legend: add dashed=fit, dotted=asymptote
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0],[0], ls="--", color="gray", lw=1.5, label="FSS fit"),
        Line2D([0],[0], ls=":",  color="gray", lw=0.8, label="$d_s(\\infty)$"),
    ]
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.9, loc="upper left")

plt.tight_layout()
plot_path = os.path.join(args.out, "ds_vs_L.png")
plt.savefig(plot_path, dpi=160, bbox_inches="tight")
print(f"\nPlot saved → {plot_path}")

# ─── CONSOLE SUMMARY ─────────────────────────────────────────────────────────
print(f"\n{'='*66}")
print(f"{'d':>4}  {'L range':>10}  {'d_s(∞)':>9}  {'±σ':>7}  "
      f"{'target':>7}  {'|Δ|':>6}  status")
print("─" * 66)
for d, rows in all_data.items():
    if not rows or "d_inf" not in rows[0]:
        print(f"  d={d}  — insufficient data"); continue
    d_inf = rows[0]["d_inf"]; err = rows[0]["d_inf_err"]
    Lmin_r = min(r["L"] for r in rows)
    Lmax_r = max(r["L"] for r in rows)
    delta  = abs(d_inf - d)
    status = "✓  consistent" if delta < 3 * err else "?  check"
    model  = rows[0].get("model_name","?")
    print(f"  d={d}  L={Lmin_r}..{Lmax_r}  "
          f"d_s(∞)={d_inf:.4f}  ±{err:.4f}  "
          f"target={d}  |Δ|={delta:.4f}  model={model}  {status}")
print(f"{'='*66}")
print(f"\nAll files in: {args.out}/")