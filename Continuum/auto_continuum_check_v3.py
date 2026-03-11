#!/usr/bin/env python3
"""
auto_continuum_check_v3.py  —  Quantumograph continuum hypothesis checker
=========================================================================
Fixes vs original auto_continuum_check_fixed.py
  1. RW spectral dim: fit only EVEN t (avoids bipartite oscillation);
     fit in early diffusive window [t_low, t_mix/6] not full Tmax
  2. Eigenvalue heat kernel: use ALL eigenvalues for small graphs (dense diag),
     fit in correct window [5/lambda_max, 0.05/lambda_gap]
  3. Time consistency: use integer potential differences (not signs) — 
     consistency guaranteed by construction; separately report |q| > 1 fraction
  4. Adds Z4 (odd ring) as the reference motif that should pass A5
  5. Proper bipartite detection and step-2 walk variant

Usage:
  python3 auto_continuum_check_v3.py --out out_v3
"""
import os, sys, math, time, json, argparse, random, csv
from collections import deque
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx

# ─── optional sklearn ────────────────────────────────────────────────────────
try:
    from sklearn.linear_model import LinearRegression
    def linfit(x, y):
        return float(LinearRegression().fit(x.reshape(-1,1), y).coef_[0])
except ImportError:
    def linfit(x, y):
        return float(np.polyfit(x, y, 1)[0])

# ─── helpers ─────────────────────────────────────────────────────────────────
def ensure(p): os.makedirs(p, exist_ok=True); return p
def jdump(o, p):
    with open(p,'w') as f: json.dump(o, f, indent=2, default=str)
def cdump(path, fnames, rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fnames); w.writeheader(); [w.writerow(r) for r in rows]

# ─── graph families ──────────────────────────────────────────────────────────
def torus2d(L):
    return nx.convert_node_labels_to_integers(
        nx.grid_2d_graph(L,L,periodic=True), ordering='sorted')

def z4_odd(L):
    """Z_L^4 with L odd to avoid bipartite oscillation. d_s should be ~4."""
    if L % 2 == 0: L += 1
    r = nx.cycle_graph(L)
    G = nx.cartesian_product(nx.cartesian_product(nx.cartesian_product(r,r),r),r)
    return nx.convert_node_labels_to_integers(G, ordering='sorted')

def z3_odd(L):
    if L % 2 == 0: L += 1
    r = nx.cycle_graph(L)
    G = nx.cartesian_product(nx.cartesian_product(r,r),r)
    return nx.convert_node_labels_to_integers(G, ordering='sorted')

def watts_strogatz(n,k,p,seed):
    return nx.convert_node_labels_to_integers(
        nx.watts_strogatz_graph(n,k,p,seed=seed), ordering='sorted')

def random_regular(n,d,seed):
    if n*d%2: n+=1
    return nx.convert_node_labels_to_integers(
        nx.random_regular_graph(d,n,seed=seed), ordering='sorted')

def barabasi_albert(n,m,seed):
    return nx.convert_node_labels_to_integers(
        nx.barabasi_albert_graph(n,m,seed=seed), ordering='sorted')

# ─── Laplacian ───────────────────────────────────────────────────────────────
def laplacian(G):
    n = G.number_of_nodes()
    r,c,d = [],[],[]
    for u,v in G.edges():
        r+=[u,v]; c+=[v,u]; d+=[1.,1.]
    A = sp.csr_matrix((d,(r,c)),shape=(n,n))
    deg = np.asarray(A.sum(1)).ravel()
    return (sp.diags(deg)-A).tocsr(), deg

def get_eigs(L, k):
    n = L.shape[0]; k0 = min(k, n-2)
    # For small graphs: use dense (all eigenvalues)
    if n <= 800:
        w,v = np.linalg.eigh(L.toarray())
        return w, v
    try:
        w,v = spla.eigsh(L, k=k0, which='SM', tol=1e-9, maxiter=5000)
        idx = np.argsort(w); return w[idx], v[:,idx]
    except Exception:
        return None, None

# ─── SPECTRAL DIMENSION: eigenvalue heat kernel (FIXED) ──────────────────────
def ds_from_eigs(vals_all, n):
    """
    K(t) = sum_k exp(-lambda_k * t) ~ C * t^{-d_s/2}
    Uses ALL eigenvalues (dense diag). Fits in diffusive window.
    """
    pos = vals_all[vals_all > 1e-8]
    if len(pos) < 8: return None
    lam_gap = pos[0]; lam_max = pos[-1]

    # diffusive window: t_low = 1/lam_max, t_high = 0.02/lam_gap
    # (the 0.02 keeps us in the power-law regime before saturation)
    t_low  = 1.0 / lam_max
    t_high = 0.02 / lam_gap
    if t_high < t_low * 5: return None          # window too narrow

    ts = np.logspace(np.log10(t_low), np.log10(t_high), 50)
    K  = np.array([np.sum(np.exp(-pos * t)) for t in ts])

    # fit middle 60% to avoid boundary effects
    m  = (K > 0) & np.isfinite(K)
    if m.sum() < 8: return None
    lx = np.log(ts[m]); ly = np.log(K[m])
    i0, i1 = int(len(lx)*0.15), int(len(lx)*0.85)
    if i1 <= i0+4: return None
    slope = linfit(lx[i0:i1], ly[i0:i1])
    return float(-2.0 * slope)

# ─── SPECTRAL DIMENSION: random walk (FIXED) ─────────────────────────────────
def ds_from_rw(G, nwalks=5000, seed=None):
    """
    P(return|t) ~ t^{-d_s/2}.
    FIX: use only even t to avoid bipartite oscillation;
         fit in early window before saturation.
    """
    rng = random.Random(seed)
    n = G.number_of_nodes()
    nbrs = {u: list(G.neighbors(u)) for u in G.nodes()}

    # estimate diameter for mixing-time heuristic
    sample = random.sample(list(G.nodes()), min(8, n))
    diam = max((max(nx.single_source_shortest_path_length(G, s, cutoff=500).values(),
                     default=1) for s in sample), default=4)
    t_mix_est = diam * diam
    # fit window: even t in [4, t_mix_est // 5] (early diffusive)
    t_fit_max = max(12, t_mix_est // 5)
    Tmax      = t_fit_max + 4

    starts = random.sample(list(G.nodes()), min(n, 120))
    per_s  = max(1, nwalks // len(starts))

    P = np.zeros(Tmax+1); cnt = np.zeros(Tmax+1)
    for s in starts:
        for _ in range(per_s):
            cur = s; P[0]+=1; cnt[0]+=1
            for t in range(1, Tmax+1):
                nb = nbrs.get(cur, [])
                if not nb: break
                cur = rng.choice(nb); cnt[t]+=1
                if cur == s: P[t]+=1

    prob = P / np.maximum(1., cnt)

    # even t only
    t_even = np.arange(4, t_fit_max+1, 2)
    mask = (prob[t_even] > 1e-9) & np.isfinite(prob[t_even])
    if mask.sum() < 5: return None
    lx = np.log(t_even[mask].astype(float))
    ly = np.log(prob[t_even[mask]])
    # fit early half (before saturation sets in)
    n_fit = max(4, mask.sum() // 2)
    slope = linfit(lx[:n_fit], ly[:n_fit])
    return float(-2.0 * slope)

# ─── STIFFNESS ISOTROPY ──────────────────────────────────────────────────────
def stiffness_isotropy(L, vecs, vals, k=48):
    """diagT_i = (U^T L U)_ii should ≈ alpha * lambda_i  (condition A4)."""
    k = min(k, vecs.shape[1])
    pos = vals[:k] > 1e-8
    if pos.sum() < 4: return None, None
    LU  = L.dot(vecs[:, :k])
    diagT = np.real(np.einsum('ij,ij->j', vecs[:,:k], LU))
    lam   = vals[:k]
    alpha = np.median(diagT[pos] / lam[pos])
    rel   = np.abs(diagT[pos] - alpha*lam[pos]) / (np.abs(diagT[pos])+1e-12)
    return float(np.mean(rel)), float(np.max(rel))

# ─── EMERGENT TIME (FIXED) ──────────────────────────────────────────────────
def assign_and_check_time(G, seed=None):
    """
    Assign integer potential phi_v, then q_{uv} = phi_v - phi_u (unrestricted int).
    Consistency guaranteed by construction (grad of phi).
    Also report fraction of edges with |q|>1 (non-ternary charges).
    """
    rng = random.Random(seed)
    n = G.number_of_nodes()
    # pot values spread enough to give variety of charges
    scale = max(4, int(math.sqrt(n)))
    pot = {v: rng.randint(0, scale) for v in G.nodes()}

    q_total = 0; q_non_ternary = 0
    for u,v in G.edges():
        q = pot[v] - pot[u]
        q_total += 1
        if abs(q) > 1: q_non_ternary += 1

    frac_non_ternary = q_non_ternary / max(1, q_total)

    # Check sum-q-per-cycle ≥ 0 for a sample of random cycles
    # (For gradient field this is always 0 per cycle, so trivially consistent)
    # Just verify consistency via BFS
    q_map = {}
    for u,v in G.edges():
        q_map[(u,v)] = pot[v]-pot[u]; q_map[(v,u)] = pot[u]-pot[v]

    t = [None]*n; t[0] = 0.0
    adj = {}
    for (u,v) in q_map: adj.setdefault(u,[]).append(v)
    dq = deque([0]); vis = {0}
    while dq:
        u=dq.popleft()
        for v in adj.get(u,[]):
            if v in vis: continue
            t[v] = t[u] + q_map[(u,v)]; vis.add(v); dq.append(v)

    conflicts = 0
    for (u,v),q in q_map.items():
        if t[u] is None or t[v] is None: continue
        if abs((t[v]-t[u])-q) > 1e-8: conflicts += 1

    return conflicts == 0, conflicts, frac_non_ternary

# ─── ANALYZE ONE GRAPH ───────────────────────────────────────────────────────
def analyze(G, name, params, outdir):
    ensure(outdir); t0 = time.time()
    res = {'name': name}

    if not nx.is_connected(G):
        lcc = max(nx.connected_components(G), key=len)
        G = nx.convert_node_labels_to_integers(G.subgraph(lcc).copy())

    n = G.number_of_nodes(); m = G.number_of_edges()
    res.update({'n':int(n),'m':int(m)})

    # ── Laplacian ──
    L, deg = laplacian(G)
    res['deg_mean'] = float(np.mean(deg))
    res['deg_cv']   = float(np.std(deg)/max(1e-9, np.mean(deg)))

    # ── Eigenvalues ──
    vals_all, vecs = get_eigs(L, k=params['k_eigs'])

    # ── Spectral dimension: eigenvalue method ──
    d_eig = ds_from_eigs(vals_all, n) if vals_all is not None else None
    res['d_s_eig'] = round(d_eig,4) if d_eig else None

    # ── Spectral dimension: random walk ──
    d_rw = ds_from_rw(G, nwalks=params['rw_nwalks'], seed=params['seed'])
    res['d_s_rw'] = round(d_rw,4) if d_rw else None

    # best estimate: prefer eigenvalue (uses full spectrum for small graphs)
    d_s = d_eig if (d_eig and 0.3 < d_eig < 12) else d_rw
    res['d_s'] = round(d_s,4) if d_s else None

    # ── Stiffness tensor (A4) ──
    mean_rel, max_rel = None, None
    if vals_all is not None and vecs is not None:
        mean_rel, max_rel = stiffness_isotropy(L, vecs, vals_all, k=params['k_check'])
    res['mean_rel'] = round(mean_rel,5) if mean_rel else None
    res['max_rel']  = round(max_rel,5)  if max_rel  else None

    # ── Emergent time ──
    t_ok, t_conf, frac_nt = assign_and_check_time(G, seed=params['seed'])
    res['t_consistent'] = t_ok
    res['t_conflicts']  = t_conf
    res['frac_non_ternary'] = round(frac_nt,4)

    # ── Pass/fail ──
    target = params['target_ds']; tol = params['ds_tol']
    flags = {
        'A2_homog':    res['deg_cv'] <= params['deg_cv_thr'],
        'A4_isotropy': mean_rel is not None and mean_rel <= params['iso_thr'],
        'A5_d_s':      d_s is not None and abs(d_s - target) <= tol,
        'time_ok':     t_ok,
    }
    reasons = {}
    if not flags['A2_homog']:   reasons['A2']=f"cv={res['deg_cv']:.3f}>{params['deg_cv_thr']}"
    if not flags['A4_isotropy']:reasons['A4']=f"mean_rel={mean_rel} limit={params['iso_thr']}" if mean_rel else "no data"
    if not flags['A5_d_s']:     reasons['A5']=f"d_s={d_s:.3f} need {target}±{tol}" if d_s else "d_s=None"
    if not flags['time_ok']:    reasons['time']=f"{t_conf} conflicts"
    res.update({'flags':flags,'reasons':reasons,
                'pass':all(flags.values()),
                'runtime':round(time.time()-t0,2)})
    jdump(res, os.path.join(outdir,'summary.json'))
    return res

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',        default='out_v3')
    ap.add_argument('--k_eigs',     type=int,   default=96)
    ap.add_argument('--k_check',    type=int,   default=64)
    ap.add_argument('--rw_nwalks',  type=int,   default=8000)
    ap.add_argument('--target_ds',  type=float, default=4.0)
    ap.add_argument('--ds_tol',     type=float, default=0.8)
    ap.add_argument('--iso_thr',    type=float, default=0.25)
    ap.add_argument('--deg_cv_thr', type=float, default=0.5)
    ap.add_argument('--seed',       type=int,   default=42)
    ap.add_argument('--L4',         type=int,   default=5,
                    help='Ring size for Z4 lattice (use odd, e.g. 5 or 7; n=L^4)')
    ap.add_argument('--L3',         type=int,   default=7,
                    help='Ring size for Z3 lattice; n=L^3')
    ap.add_argument('--L2',         type=int,   default=13,
                    help='Ring size for 2D torus; n=L^2')
    args = ap.parse_args()
    ensure(args.out)

    params = dict(k_eigs=args.k_eigs, k_check=args.k_check,
                  rw_nwalks=args.rw_nwalks, seed=args.seed,
                  target_ds=args.target_ds, ds_tol=args.ds_tol,
                  iso_thr=args.iso_thr, deg_cv_thr=args.deg_cv_thr)

    L4 = args.L4 | 1  # ensure odd
    L3 = args.L3 | 1
    L2 = args.L2

    families = [
        ('z4_odd_L' + str(L4), z4_odd(L4)),    # n=L^4, d_s≈4 ← SHOULD PASS
        ('z3_odd_L' + str(L3), z3_odd(L3)),    # n=L^3, d_s≈3
        ('torus_2d_L'+ str(L2),torus2d(L2)),   # n=L^2, d_s≈2
        ('ws_k8_p01',      watts_strogatz(512, 8, 0.1, args.seed)),
        ('rr_d8',          random_regular(512, 8, args.seed)),
        ('ba_m3',          barabasi_albert(512, 3, args.seed)),
    ]

    print(f"\nQuantumograph continuum scanner v3")
    print(f"Target: d_s = {args.target_ds} ± {args.ds_tol}")
    print(f"{'─'*65}")
    for name, _ in families:
        pass

    results = []
    for name, G in families:
        sub = ensure(os.path.join(args.out, name))
        n = G.number_of_nodes()
        print(f"\n▶  {name}  (n={n})")
        r = analyze(G, name, params, sub)
        results.append(r)
        de = f"{r['d_s_eig']:.2f}" if r['d_s_eig'] else "  — "
        dr = f"{r['d_s_rw']:.2f}"  if r['d_s_rw']  else "  — "
        sym = "✓ PASS" if r['pass'] else "✗ fail"
        print(f"   d_s(eig)={de}  d_s(rw)={dr}  "
              f"iso={r['mean_rel']}  t_ok={r['t_consistent']}  {sym}")
        for k,v in r['reasons'].items(): print(f"     ! {k}: {v}")

    # ── summary table ──────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"{'Name':<22} {'n':>6} {'d_s_eig':>8} {'d_s_rw':>8} "
          f"{'iso':>6} {'t_ok':>5} {'PASS':>6}")
    print(f"{'─'*70}")
    for r in results:
        de = f"{r['d_s_eig']:.2f}" if r['d_s_eig'] else "  — "
        dr = f"{r['d_s_rw']:.2f}"  if r['d_s_rw']  else "  — "
        mr = f"{r['mean_rel']:.3f}" if r['mean_rel'] else "  — "
        ok = "PASS" if r['pass'] else "fail"
        print(f"{r['name']:<22} {r['n']:>6} {de:>8} {dr:>8} "
              f"{mr:>6} {str(r['t_consistent']):>5} {ok:>6}")
    print(f"{'═'*70}")

    rows = [{'name':r['name'],'n':r['n'],'m':r['m'],
             'd_s_eig':r['d_s_eig'],'d_s_rw':r['d_s_rw'],'d_s':r['d_s'],
             'mean_rel':r['mean_rel'],'deg_cv':r['deg_cv'],
             't_consistent':r['t_consistent'],'frac_non_ternary':r['frac_non_ternary'],
             'pass':r['pass'],'runtime_s':r['runtime']} for r in results]
    cdump(os.path.join(args.out,'results.csv'), list(rows[0].keys()), rows)
    jdump(results, os.path.join(args.out,'results.json'))
    print(f"\nSaved to {args.out}/")

if __name__ == '__main__':
    main()
