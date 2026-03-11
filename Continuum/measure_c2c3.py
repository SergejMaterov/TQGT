#!/usr/bin/env python3
"""
measure_c2c3.py  —  Quantumograph v14
========================================
Measures the error-bound coefficients C2 and C3 from Section 6:

    ||R(ell)|| <= C1*(a/ell)^2  +  C2/z_bar  +  C3*eps_spec

C1 is already provided by finite_size_scaling.py as |a|/4.
This script handles C2 and C3 only.

======================================================================
C2  —  finite-coordination coefficient
----------------------------------------------------------------------
Physics: a larger z_bar means the discrete Laplacian better
approximates the continuum one, so d_s(inf; z_bar) -> 4 faster.

Construction: Z^4 torus with EDGE DIAGONALS (preserves Z^4 symmetry).
  hop=1  (NN only):      z_bar =  8   (standard Z^4)
  hop=1 + 2D diagonals:  z_bar = 32   (+4 diags per each of C(4,2)=6 planes)
  hop=1+2D+3D diagonals: z_bar = 64   (+8 diags per each of C(4,3)=4 triples)

All three are Z^4 tori with different approximating Laplacians;
d_s(inf) -> 4 for all three as L -> inf.

FSS extrapolation for each z_bar:
  d_s(L; z_bar) = d_s(inf; z_bar) + a(z_bar)/L^2

Then:  4 - d_s(inf; z_bar) ~ C2 / z_bar
C2 = slope of (4 - d_s_inf) vs 1/z_bar.

======================================================================
C3  —  spectral-leakage coefficient
----------------------------------------------------------------------
Physics: the field-theory operator uses all Laplacian eigenmodes; if
the spectrum is truncated to the first k modes, the error is
eps_spec(k) = 1 - sum(lambda_i, i<k) / sum(lambda_i, all).

Graph: Z^3 torus L=9  (n=729, dense spectrum, d_s^hk well-defined).
Method: heat-kernel  K(t) = sum exp(-lambda_i * t)  at full and
        truncated spectra. Fit: |Delta d_s| / d_s_full ~ C3 * eps_spec.

======================================================================
Usage (fast, ~8 min):
  python3 measure_c2c3.py --out c2c3_out

More accurate (~25 min):
  python3 measure_c2c3.py --out c2c3_out --Lvals 5,7,9,11,13 \\
      --nwalks 80000 --nseeds 7 --n_boot 500

With C1 loaded from finite_size_scaling output:
  python3 measure_c2c3.py --fss_json fss_final/d4_rows.json --out c2c3_out
"""

import os, json, argparse, random, time
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── utilities ───────────────────────────────────────────────────────────────

def ensure(p):
    os.makedirs(p, exist_ok=True); return p

def jdump(obj, path):
    with open(path, 'w') as f: json.dump(obj, f, indent=2, default=float)

def ols(x, y):
    """Ordinary least squares: returns (slope, intercept)."""
    A = np.column_stack([x, np.ones_like(x)])
    c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(c[0]), float(c[1])

def bootstrap_ols(x, y, n_boot=300):
    """
    Bootstrap 95% CI on both slope and intercept.
    Returns (s_med, s_lo, s_hi, i_med, i_lo, i_hi).
    """
    rng = np.random.default_rng(0)
    sl, ic = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(idx)) < 2: continue
        try:
            s, i = ols(x[idx], y[idx])
            sl.append(s); ic.append(i)
        except Exception:
            pass
    if len(sl) < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    return (float(np.median(sl)),
            float(np.percentile(sl, 2.5)),
            float(np.percentile(sl, 97.5)),
            float(np.median(ic)),
            float(np.percentile(ic, 2.5)),
            float(np.percentile(ic, 97.5)))

# ─── graph constructors ──────────────────────────────────────────────────────

def make_z4_nn(L):
    """
    Z_L^4 torus, nearest-neighbour (hop=1) edges only.
    z_bar = 8 = 2 * 4 (two neighbours per axis).
    L is forced odd to avoid bipartite oscillation in RW.
    """
    if L % 2 == 0: L += 1
    r = nx.cycle_graph(L)
    G = nx.cartesian_product(
            nx.cartesian_product(
                nx.cartesian_product(r, r), r), r)
    return nx.convert_node_labels_to_integers(G, ordering='sorted')

def make_z4_with_diagonals(L, include_2d=True, include_3d=False):
    """
    Z_L^4 torus + edge diagonals, preserving full Z^4 symmetry.

    2D diagonals: for each pair of axes (i,j) from C(4,2)=6 pairs,
                  add +-e_i +- e_j  ->  4 vectors * 6 pairs = 24 extra edges/node
                  z_bar = 8 + 24 = 32

    3D diagonals: for each triple of axes (i,j,k) from C(4,3)=4 triples,
                  add +-e_i +- e_j +- e_k  ->  8 vectors * 4 triples = 32 extra
                  z_bar = 32 + 32 = 64  (both 2D and 3D)
    """
    if L % 2 == 0: L += 1
    n = L**4

    def idx(c):
        return c[0] + L*(c[1] + L*(c[2] + L*c[3]))

    def coords(i):
        c = [0]*4
        c[0] = i % L; i //= L
        c[1] = i % L; i //= L
        c[2] = i % L; c[3] = i // L
        return c

    G = make_z4_nn(L)

    if include_2d:
        axis_pairs = [(a, b) for a in range(4) for b in range(a+1, 4)]
        for u in range(n):
            c = coords(u)
            for a, b in axis_pairs:
                for sa in (+1, -1):
                    for sb in (+1, -1):
                        c2 = c[:]
                        c2[a] = (c2[a] + sa) % L
                        c2[b] = (c2[b] + sb) % L
                        v = idx(c2)
                        if u != v: G.add_edge(u, v)

    if include_3d:
        axis_triples = [(a, b, c_)
                        for a in range(4)
                        for b in range(a+1, 4)
                        for c_ in range(b+1, 4)]
        for u in range(n):
            c = coords(u)
            for a, b, c_ in axis_triples:
                for sa in (+1, -1):
                    for sb in (+1, -1):
                        for sc in (+1, -1):
                            c2 = c[:]
                            c2[a]  = (c2[a]  + sa) % L
                            c2[b]  = (c2[b]  + sb) % L
                            c2[c_] = (c2[c_] + sc) % L
                            v = idx(c2)
                            if u != v: G.add_edge(u, v)
    return G

def make_z3(L):
    """Z_L^3 torus, L forced odd."""
    if L % 2 == 0: L += 1
    r = nx.cycle_graph(L)
    G = nx.cartesian_product(nx.cartesian_product(r, r), r)
    return nx.convert_node_labels_to_integers(G, ordering='sorted')

def mean_deg(G):
    return float(np.mean([d for _, d in G.degree()]))

# ─── Laplacian ───────────────────────────────────────────────────────────────

def laplacian(G):
    n = G.number_of_nodes()
    row, col, dat = [], [], []
    for u, v in G.edges():
        row += [u, v]; col += [v, u]; dat += [1., 1.]
    A = sp.csr_matrix((dat, (row, col)), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel()
    return (sp.diags(deg) - A).tocsr(), deg

# ─── Random-walk spectral dimension (from auto_continuum_check_v3.py) ─────────

def ds_rw(G, nwalks=40000, seed=None):
    """
    Estimates d_s via P(return | t) ~ t^{-d_s/2}.

    Key fixes (identical to auto_continuum_check_v3.py):
      - even t only  ->  avoids bipartite oscillation
      - fit window [4, diam^2/8]  ->  early diffusive regime
      - fit on the first half of available even-t points
    """
    rng  = random.Random(seed)
    n    = G.number_of_nodes()
    nbrs = {u: list(G.neighbors(u)) for u in G.nodes()}

    sample = random.sample(list(G.nodes()), min(8, n))
    diam   = max(
        (max(nx.single_source_shortest_path_length(G, s, cutoff=500).values(),
             default=1) for s in sample),
        default=4)
    t_fit_max = max(12, diam * diam // 8)
    Tmax      = t_fit_max + 4
    starts    = random.sample(list(G.nodes()), min(n, 120))
    per_s     = max(1, nwalks // len(starts))

    P   = np.zeros(Tmax + 1)
    cnt = np.zeros(Tmax + 1)
    for s in starts:
        for _ in range(per_s):
            cur = s; P[0] += 1; cnt[0] += 1
            for t in range(1, Tmax + 1):
                nb = nbrs.get(cur, [])
                if not nb: break
                cur = rng.choice(nb); cnt[t] += 1
                if cur == s: P[t] += 1

    prob   = P / np.maximum(1., cnt)
    t_even = np.arange(4, t_fit_max + 1, 2)
    mask   = (prob[t_even] > 1e-9) & np.isfinite(prob[t_even])
    if mask.sum() < 5: return None
    lx = np.log(t_even[mask].astype(float))
    ly = np.log(prob[t_even[mask]])
    n_fit = max(4, mask.sum() // 2)
    s_fit, _ = ols(lx[:n_fit], ly[:n_fit])
    return float(-2.0 * s_fit)

def ds_rw_mean(G, nwalks, nseeds):
    """Average d_s over nseeds independent RW runs."""
    vals = [ds_rw(G, nwalks=nwalks, seed=s) for s in range(nseeds)]
    vals = [v for v in vals if v is not None]
    if not vals: return None, None
    return float(np.mean(vals)), float(np.std(vals) if len(vals) > 1 else 0.0)

# ─── Heat-kernel spectral dimension (for C3) ──────────────────────────────────

def ds_hk(pos, wf=0.4):
    """
    K(t) = sum exp(-lambda_i * t) ~ C * t^{-d_s/2}.
    Fit window: [1/lambda_max, wf/lambda_gap].
    wf=0.4 works well for Z^3 L>=9.
    """
    if len(pos) < 8: return None
    t_lo = 1.0 / pos[-1]; t_hi = wf / pos[0]
    if t_hi < 3.0 * t_lo: return None
    ts = np.logspace(np.log10(t_lo), np.log10(t_hi), 80)
    K  = np.array([np.sum(np.exp(-pos * t)) for t in ts])
    m  = (K > 0) & np.isfinite(K)
    if m.sum() < 10: return None
    lx = np.log(ts[m]); ly = np.log(K[m])
    i0, i1 = int(len(lx)*0.15), int(len(lx)*0.85)
    if i1 <= i0 + 3: return None
    s, _ = ols(lx[i0:i1], ly[i0:i1])
    return float(-2.0 * s)

def eps_spec(pos, k):
    """Spectral leakage: fraction of Laplacian weight above mode k."""
    return 1.0 - float(np.sum(pos[:k])) / float(np.sum(pos))

# ─── MODULE C2 ───────────────────────────────────────────────────────────────

def run_C2(Lvals, nwalks, nseeds, n_boot):
    """
    Run FSS for each z_bar in {8, 32, 64}:
        d_s(L; z_bar) = d_s(inf; z_bar) + a(z_bar) / L^2

    Then fit:
        4 - d_s(inf; z_bar) = C2 / z_bar

    This gives C2 = slope of (4 - d_s_inf) vs 1/z_bar.
    """
    print('\n====  MODULE C2: d_s(inf; z_bar) at varying z_bar  ============')

    configs = [
        ('Z4 NN  (z_bar= 8)', lambda L: make_z4_nn(L),                        'NN'),
        ('Z4 +2D (z_bar=32)', lambda L: make_z4_with_diagonals(L, True, False),'2D'),
        ('Z4 +3D (z_bar=64)', lambda L: make_z4_with_diagonals(L, True, True), '3D'),
    ]

    results = []
    for label, builder, tag in configs:
        print(f'\n  -- {label} --')
        G_probe = builder(min(Lvals))
        zm = mean_deg(G_probe)
        del G_probe

        xs, ys, errs = [], [], []
        for L in Lvals:
            if L % 2 == 0: L += 1
            t0 = time.time()
            G  = builder(L)
            n  = G.number_of_nodes()
            d_m, d_s = ds_rw_mean(G, nwalks, nseeds)
            if d_m is None:
                print(f'    L={L}: d_s=None — skipped'); continue
            xs.append(1.0 / float(L)**2)
            ys.append(d_m)
            errs.append(d_s or 0.0)
            print(f'    L={L:3d}  n={n:8d}  z_bar={zm:.0f}  '
                  f'd_s={d_m:.4f}+-{d_s:.4f}  ({time.time()-t0:.1f}s)')

        if len(xs) < 3:
            print('    Not enough points (need >=3)'); continue
        xs = np.array(xs); ys = np.array(ys)
        C1z, ds_inf_ols = ols(xs, ys)
        s_m, s_lo, s_hi, i_m, i_lo, i_hi = bootstrap_ols(xs, ys, n_boot)
        print(f'    d_s(inf; z_bar={zm:.0f}) = {i_m:.4f}  [{i_lo:.4f}, {i_hi:.4f}]')
        print(f'    FSS slope a(z_bar)      = {s_m:.4f}  [{s_lo:.4f}, {s_hi:.4f}]')
        results.append({
            'label': label, 'tag': tag, 'zm': zm,
            'inv_z': round(1.0/zm, 6),
            'ds_inf': i_m, 'ds_inf_lo': i_lo, 'ds_inf_hi': i_hi,
            'a_fss': s_m, 'a_fss_lo': s_lo, 'a_fss_hi': s_hi,
            'xs': xs.tolist(), 'ys': ys.tolist(), 'errs': errs,
        })

    if len(results) < 2:
        print('\n  Not enough configurations for C2 fit'); return {}

    inv_z = np.array([r['inv_z']         for r in results])
    resid = np.array([4.0 - r['ds_inf']  for r in results])

    print(f'\n  Fitting  (4 - d_s_inf) = C2 * (1/z_bar):')
    for r, rv in zip(results, resid):
        print(f'    z_bar={r["zm"]:.0f}  d_s(inf)={r["ds_inf"]:.4f}  '
              f'4-d_s(inf)={rv:+.4f}  1/z_bar={r["inv_z"]:.5f}')

    C2_ols, _ = ols(inv_z, resid)
    C2_m, C2_lo, C2_hi, _, _, _ = bootstrap_ols(inv_z, resid, n_boot)
    print(f'\n  C2 (OLS)    = {C2_ols:.4f}')
    print(f'  C2 (median) = {C2_m:.4f}   95% CI [{C2_lo:.4f}, {C2_hi:.4f}]')

    return {
        'C2': C2_m, 'C2_lo': C2_lo, 'C2_hi': C2_hi, 'C2_ols': C2_ols,
        'configs': results,
        'inv_z': inv_z.tolist(), 'resid_4_minus_ds': resid.tolist(),
    }

# ─── MODULE C3 ───────────────────────────────────────────────────────────────

def run_C3(n_boot):
    """
    Graph: Z^3 torus L=9 (n=729, full dense spectrum).
    Truncate to first k positive eigenvalues; compute:
        eps_spec(k) = 1 - sum(lambda_i, i<k) / sum(lambda_i)
        norm_err(k) = |d_s_hk(k) - d_s_hk_full| / d_s_hk_full
    Fit: norm_err = C3 * eps_spec.

    Note: d_s_hk(full) < 3 at L=9 due to finite-size effects;
    we use only the *relative* change, not the absolute value.
    """
    print('\n====  MODULE C3: spectral leakage (Z^3 torus L=9)  ============')
    L = 9
    G = make_z3(L); n = G.number_of_nodes()
    print(f'  Graph: Z^3 L={L}, n={n}. Computing dense spectrum...')
    t0 = time.time()
    L_mat, _ = laplacian(G)
    pos = np.sort(np.linalg.eigvalsh(L_mat.toarray()))
    pos = pos[pos > 1e-10]
    n_pos = len(pos)
    d_s_full = ds_hk(pos)
    if d_s_full is None or d_s_full < 0.5:
        print('  ds_hk(full spectrum) undefined — try larger L'); return {}
    print(f'  n_pos={n_pos}, d_s_hk(full)={d_s_full:.4f}  ({time.time()-t0:.2f}s)')
    print(f'  (d_s_hk < 3 at this L due to finite-size; only relative change matters)')

    k_fracs = np.array([0.02, 0.05, 0.08, 0.12, 0.18, 0.25,
                        0.35, 0.47, 0.60, 0.75, 0.90])
    k_vals  = np.unique(np.maximum(10, (k_fracs * n_pos).astype(int)))

    xs, ys, rows = [], [], []
    for k in k_vals:
        eps   = eps_spec(pos, k)
        d_s_t = ds_hk(pos[:k])
        if d_s_t is None: continue
        ne    = abs(d_s_t - d_s_full) / abs(d_s_full)
        xs.append(eps); ys.append(ne)
        rows.append({'k': int(k), 'frac': round(k/n_pos, 3),
                     'eps': round(eps, 5), 'd_s_t': round(d_s_t, 4),
                     'norm_err': round(ne, 6)})
        print(f'  k={k:4d}/{n_pos}  eps_spec={eps:.4f}  '
              f'd_s_hk={d_s_t:.4f}  |delta_ds|/ds={ne:.5f}')

    if len(xs) < 5:
        print('  Not enough points'); return {}
    xs = np.array(xs); ys = np.array(ys)
    m  = xs > 0.01
    if m.sum() < 3: m = np.ones(len(xs), dtype=bool)
    C3_ols, _ = ols(xs[m], ys[m])
    C3_m, C3_lo, C3_hi, _, _, _ = bootstrap_ols(xs[m], ys[m], n_boot)
    print(f'\n  C3 (OLS)    = {C3_ols:.4f}')
    print(f'  C3 (median) = {C3_m:.4f}   95% CI [{C3_lo:.4f}, {C3_hi:.4f}]')
    return {
        'C3': C3_m, 'C3_lo': C3_lo, 'C3_hi': C3_hi, 'C3_ols': C3_ols,
        'd_s_full_hk': d_s_full,
        'rows': rows, 'xs': xs.tolist(), 'ys': ys.tolist(),
    }

# ─── load C1 from FSS output ─────────────────────────────────────────────────

def load_C1_from_fss(path):
    """
    Reads d4_rows.json produced by finite_size_scaling.py.
    C1 = |a| / 4  (normalisation: ||R|| = Delta_d_s / 4).
    CI estimate: sigma_a ~ sigma_d_inf * L_min^2 (from linear model).
    """
    if path is None or not os.path.exists(path):
        return None
    data   = json.load(open(path))
    a      = data[0]['a']
    d_inf  = data[0]['d_inf']
    d_inf_err = data[0]['d_inf_err']
    C1     = abs(a) / 4.0
    L_min  = min(r['L'] for r in data)
    C1_err = d_inf_err * L_min**2 / 4.0
    return {
        'C1': C1, 'C1_err': C1_err,
        'C1_lo': max(0.0, C1 - 2*C1_err),
        'C1_hi': C1 + 2*C1_err,
        'a_fss': a, 'd_inf': d_inf, 'd_inf_err': d_inf_err,
        'source': path, 'L_min': L_min,
        'n_points': len(data),
    }

# ─── plots ───────────────────────────────────────────────────────────────────

def make_plots(r_C1, r_C2, r_C3, outdir):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.40, hspace=0.48)
    fig.suptitle(
        'Quantumograph v14 — coefficients $C_2$ and $C_3$\n'
        r'($C_1$ from finite\_size\_scaling.py)',
        fontsize=12, fontweight='bold')

    # C2: FSS curves at each z_bar
    ax = fig.add_subplot(gs[0, :2])
    colors_z = {'NN': 'steelblue', '2D': 'darkorange', '3D': 'forestgreen'}
    if r_C2 and r_C2.get('configs'):
        for cfg in r_C2['configs']:
            c   = colors_z.get(cfg['tag'], 'gray')
            xs  = np.array(cfg['xs']); ys = np.array(cfg['ys'])
            ers = np.array(cfg['errs'])
            ax.errorbar(xs, ys, yerr=ers, fmt='o', color=c, ms=6, capsize=3,
                        label=f"{cfg['label']}  $d_s(\\infty)$={cfg['ds_inf']:.3f}",
                        zorder=5)
            x_f = np.linspace(0, max(xs)*1.05, 200)
            ax.plot(x_f, cfg['a_fss']*x_f + cfg['ds_inf'], '--', color=c,
                    lw=1.5, alpha=0.7)
            ax.scatter([0], [cfg['ds_inf']], color=c, s=100, zorder=8, marker='*')
    ax.axhline(4.0, color='black', lw=1, ls=':', label='theoretical $d_s=4$')
    ax.set_xlabel('$1/L^2$', fontsize=10)
    ax.set_ylabel('$d_s$ (RW)', fontsize=10)
    ax.set_title('FSS at different $\\bar{z}$:\n'
                 '$d_s(L;\\bar{z}) = d_s(\\infty;\\bar{z}) + a(\\bar{z})/L^2$',
                 fontsize=9)
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
    ax.set_xlim(left=-0.001)

    # C2: final fit  (4 - d_s_inf) vs 1/z_bar
    ax2 = fig.add_subplot(gs[1, 0])
    if r_C2 and r_C2.get('configs'):
        inv_z  = np.array(r_C2['inv_z'])
        resid  = np.array(r_C2['resid_4_minus_ds'])
        errs_r = np.array([abs(c['ds_inf_hi']-c['ds_inf_lo'])/2
                           for c in r_C2['configs']])
        ax2.errorbar(inv_z, resid, yerr=errs_r, fmt='o', c='navy',
                     ms=8, capsize=4, zorder=5)
        x_f = np.linspace(0, max(inv_z)*1.1, 200)
        ax2.plot(x_f, r_C2['C2_ols']*x_f, 'r-', lw=2,
                 label=f"$C_2$={r_C2['C2']:.3f}")
        for cfg in r_C2['configs']:
            ax2.annotate(f"$\\bar{{z}}$={cfg['zm']:.0f}",
                         (cfg['inv_z'], 4.0 - cfg['ds_inf']),
                         xytext=(5, 3), textcoords='offset points', fontsize=8)
    c2v = r_C2.get('C2'); c2l = r_C2.get('C2_lo'); c2h = r_C2.get('C2_hi')
    ax2.set_xlabel('$1/\\bar{z}$', fontsize=10)
    ax2.set_ylabel('$4 - d_s(\\infty;\\bar{z})$', fontsize=10)
    ax2.set_title(
        f'$C_2$ = {c2v:.3f}  [{c2l:.3f}, {c2h:.3f}]' if c2v else '$C_2$',
        fontsize=9)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # C3: eps_spec vs Delta d_s
    ax3 = fig.add_subplot(gs[1, 1])
    if r_C3 and r_C3.get('xs'):
        xs3 = np.array(r_C3['xs']); ys3 = np.array(r_C3['ys'])
        ax3.scatter(xs3, ys3, c='forestgreen', s=60, zorder=5)
        x_f = np.linspace(0, max(xs3)*1.05, 200)
        ax3.plot(x_f, r_C3['C3_ols']*x_f, 'r-', lw=2,
                 label=f"$C_3$={r_C3['C3']:.3f}")
    c3v = r_C3.get('C3'); c3l = r_C3.get('C3_lo'); c3h = r_C3.get('C3_hi')
    ax3.set_xlabel(
        r'$\varepsilon_{\rm spec} = 1 - \sum_{i<k}\lambda_i / \sum\lambda_i$',
        fontsize=9)
    ax3.set_ylabel('$|\\Delta d_s^{\\rm hk}| / d_s^{\\rm hk,full}$', fontsize=9)
    ax3.set_title(
        f'$C_3$ = {c3v:.3f}  [{c3l:.3f}, {c3h:.3f}]' if c3v else '$C_3$',
        fontsize=9)
    if r_C3 and r_C3.get('C3'): ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # Summary panel
    ax4 = fig.add_subplot(gs[1, 2]); ax4.axis('off')

    def sig(lo, hi):
        return 'significant' if (lo > 0 or hi < 0) else 'CI includes 0'

    lines = [
        ('||R(ell)|| <= C1*(a/ell)^2 + C2/z + C3*eps', True, 'navy'),
        ('', False, 'black'),
    ]
    if r_C1:
        lines += [
            (f'C1 = {r_C1["C1"]:.3f} +/- {r_C1["C1_err"]:.3f}', False, '#27ae60'),
            (f'  from FSS ({r_C1["n_points"]} points, L_min={r_C1["L_min"]})',
             False, '#555'),
        ]
    if r_C2 and r_C2.get('C2'):
        st = sig(r_C2['C2_lo'], r_C2['C2_hi'])
        col = '#27ae60' if 'significant' in st else '#e67e22'
        lines += [
            (f'C2 = {r_C2["C2"]:.3f}  [{r_C2["C2_lo"]:.3f},{r_C2["C2_hi"]:.3f}]',
             False, col),
            (f'  {st}', False, '#555'),
        ]
    if r_C3 and r_C3.get('C3'):
        st3 = sig(r_C3['C3_lo'], r_C3['C3_hi'])
        col3 = '#27ae60' if 'significant' in st3 else '#e67e22'
        lines += [
            (f'C3 = {r_C3["C3"]:.3f}  [{r_C3["C3_lo"]:.3f},{r_C3["C3_hi"]:.3f}]',
             False, col3),
            (f'  {st3}', False, '#555'),
        ]

    if r_C1 and r_C2 and r_C3 and r_C1.get('C1') and r_C2.get('C2') and r_C3.get('C3'):
        C1 = r_C1['C1']; C2 = r_C2['C2']; C3 = r_C3['C3']
        bnd = C1/49 + C2/8 + C3*0.05
        lines += [
            ('', False, 'black'),
            ('At ell=7a, z_bar=8, eps=0.05:', True, 'black'),
            (f'||R(7a)|| <= {bnd:.4f}  ({bnd*100:.1f}%)', False, 'navy'),
        ]

    y = 0.97
    for item in lines:
        if item == '':
            y -= 0.04; continue
        line, bold, color = item
        ax4.text(0.03, y, line, transform=ax4.transAxes,
                 fontsize=9, va='top', color=color,
                 fontweight='bold' if bold else 'normal')
        y -= 0.09

    p = os.path.join(outdir, 'measure_c2c3.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Figure saved: {p}')

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            'Quantumograph v14: measure C2 and C3.\n'
            'C1 is already given by finite_size_scaling.py as |a|/4.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out',      default='c2c3_out',
                    help='Output directory')
    ap.add_argument('--fss_json', default=None,
                    help='d4_rows.json from finite_size_scaling.py (to display C1)')
    ap.add_argument('--Lvals',    default='5,7,9',
                    help='Comma-separated L values (forced odd). Default: 5,7,9')
    ap.add_argument('--nwalks',   type=int, default=60000,
                    help='RW steps per run')
    ap.add_argument('--nseeds',   type=int, default=5,
                    help='Independent RW runs to average over')
    ap.add_argument('--n_boot',   type=int, default=400,
                    help='Bootstrap resamples for CI')
    ap.add_argument('--skip_C2',  action='store_true')
    ap.add_argument('--skip_C3',  action='store_true')
    args = ap.parse_args()
    ensure(args.out)

    Lvals = sorted(set(int(x) if int(x) % 2 == 1 else int(x) + 1
                       for x in args.Lvals.split(',')))

    print('=' * 65)
    print('  Quantumograph v14 — measure_c2c3.py')
    print(f'  Lvals={Lvals}, nwalks={args.nwalks}, nseeds={args.nseeds}')
    print('=' * 65)
    print()
    print('  C1 is read from finite_size_scaling.py output: C1 = |a|/4')
    print('  This script measures C2 (z_bar variation) and C3 (spectral leakage).')

    r_C1 = load_C1_from_fss(args.fss_json)
    if r_C1:
        print(f'\n  C1 from {args.fss_json}:')
        print(f'    a_fss    = {r_C1["a_fss"]:.4f}')
        print(f'    d_s(inf) = {r_C1["d_inf"]:.4f} +/- {r_C1["d_inf_err"]:.4f}')
        print(f'    C1 = |a|/4 = {r_C1["C1"]:.4f} +/- {r_C1["C1_err"]:.4f}')

    r_C2 = {}
    if not args.skip_C2:
        r_C2 = run_C2(Lvals, args.nwalks, args.nseeds, args.n_boot)
        jdump(r_C2, os.path.join(args.out, 'C2.json'))

    r_C3 = {}
    if not args.skip_C3:
        r_C3 = run_C3(args.n_boot)
        jdump(r_C3, os.path.join(args.out, 'C3.json'))

    make_plots(r_C1, r_C2, r_C3, args.out)

    print('\n' + '=' * 65)
    print('  SUMMARY')
    print('=' * 65)
    if r_C1:
        print(f'\n  C1 = {r_C1["C1"]:.4f} +/- {r_C1["C1_err"]:.4f}'
              f'  (from FSS, {r_C1["n_points"]} points)')
    for name, res in [('C2', r_C2), ('C3', r_C3)]:
        v = res.get(name)
        if v:
            lo = res[name+'_lo']; hi = res[name+'_hi']
            sig = 'significant' if (lo > 0 or hi < 0) else 'CI includes 0'
            print(f'  {name} = {v:.4f}   95% CI [{lo:.4f}, {hi:.4f}]  — {sig}')
        else:
            print(f'  {name} — not computed')

    if r_C1 and r_C2.get('C2') and r_C3.get('C3'):
        C1 = r_C1['C1']; C2 = r_C2['C2']; C3 = r_C3['C3']
        bnd = C1/49 + C2/8 + C3*0.05
        print(f'\n  Bound at ell=7a, z_bar=8, eps_spec=0.05:')
        print(f'  ||R(7a)|| <= {C1:.3f}/49 + {C2:.3f}/8 + {C3:.3f}*0.05')
        print(f'            = {bnd:.4f}  ({bnd*100:.2f}%)')

    summary = {
        'C1_from_fss': r_C1,
        'C2': r_C2.get('C2'), 'C2_lo': r_C2.get('C2_lo'), 'C2_hi': r_C2.get('C2_hi'),
        'C3': r_C3.get('C3'), 'C3_lo': r_C3.get('C3_lo'), 'C3_hi': r_C3.get('C3_hi'),
        'notes': {
            'C1': 'C1 = |a_fss|/4 from finite_size_scaling.py d4_rows.json',
            'C2': ('FSS extrapolation at z_bar=8,32,64 (NN / 2D-diag / 3D-diag); '
                   'C2 = slope of (4 - d_s_inf) vs 1/z_bar'),
            'C3': ('spectral truncation on Z^3 L=9 (n=729, dense spectrum); '
                   'C3 = slope of |delta_d_s_hk| / d_s_hk_full vs eps_spec'),
        }
    }
    jdump(summary, os.path.join(args.out, 'summary_c2c3.json'))
    print(f'\n  All outputs saved to {args.out}/')
    print('=' * 65)


if __name__ == '__main__':
    main()
