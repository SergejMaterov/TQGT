# Colab-ready: full demo + auto-report for Scheme A (classical weights) & Scheme B (edge-qubits)
# WARNING: Scheme B uses full statevector 2^E - keep E small (<=12 recommended).

# 0) Mount Drive and imports
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, json, time, zipfile
import numpy as np
import scipy.linalg as sla
from scipy.integrate import odeint
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1) Parameters & output dirs - edit these if desired
BASE_DIR_DRIVE = "/content/drive/MyDrive/Quantumograph_matched/demo_full"
os.makedirs(BASE_DIR_DRIVE, exist_ok=True)
OUTDIR = BASE_DIR_DRIVE
ZIP_PATH = os.path.join(OUTDIR, "demo_full_report.zip")

print("Output ->", OUTDIR)

# ---------- Utilities ----------
def laplacian_from_W(W):
    W = 0.5*(W + W.T)
    deg = np.sum(W, axis=1)
    return np.diag(deg) - W

def Evac_from_eigvals(eigvals, t0):
    ev = np.maximum(eigvals, 0.0)
    omega = np.sqrt(ev)
    return 0.5 * np.mean(omega * np.exp(-(omega**2) * t0))

def compute_ipr_from_vecs(vecs, kmax=10):
    # vecs shape (N, m); return IPR for first kmax modes
    kmax = min(kmax, vecs.shape[1])
    ipr = np.sum(np.abs(vecs[:, :kmax])**4, axis=0)
    return ipr

# ---------- Scheme A implementation (enhanced) ----------
def run_scheme_A(graph=None, params=None, tmax=6.0, dt=0.02, t0_eval=1e-2, save_prefix="schemeA"):
    if graph is None:
        graph = nx.grid_2d_graph(2,4)
        graph = nx.convert_node_labels_to_integers(graph)
    N = graph.number_of_nodes()
    edges = list(graph.edges())
    E = len(edges)
    edge_to_idx = {e:i for i,e in enumerate(edges)}

    # default params
    if params is None:
        params = {}
    kappa = float(params.get('kappa', 2.0))
    gamma = float(params.get('gamma', 0.3))
    alpha = float(params.get('alpha', 0.5))
    m_w = float(params.get('m_w', 1.0))
    w0 = float(params.get('w0', 1.0))
    phi0_amp = float(params.get('phi0_amp', 0.05))
    seed = int(params.get('seed', 0))

    rng = np.random.RandomState(seed)

    # initial conds
    phi = rng.randn(N) * phi0_amp
    pi = np.zeros(N)
    w = np.ones(E) * w0 + 0.01 * rng.randn(E)
    pw = np.zeros(E)

    nsteps = int(np.ceil(tmax/dt))
    times = np.linspace(0, tmax, nsteps+1)

    # storage
    Evac_list = []
    gap_list = []
    Lnormdiff = []
    eig_snapshots = []
    eigvecs_snapshots = []
    ipr_snapshots = []
    energies_field = []
    energies_graph = []
    energies_total = []

    L_prev = None

    for step, t in enumerate(times):
        # build W and L
        W = np.zeros((N,N))
        for (i,j), idx in edge_to_idx.items():
            val = float(max(0.0, w[idx]))
            W[i,j] = W[j,i] = val
        L = laplacian_from_W(W)
        vals, vecs = sla.eigh(L)
        vals_sorted = np.real(vals)
        eig_snapshots.append(vals_sorted)
        eigvecs_snapshots.append(vecs)
        # Evac and gap
        Ev = Evac_from_eigvals(vals_sorted, t0_eval)
        Evac_list.append(Ev)
        gap = vals_sorted[1] if len(vals_sorted)>1 else vals_sorted[0]
        gap_list.append(gap)
        # smoothness
        if L_prev is None:
            Lnormdiff.append(0.0)
        else:
            Lnormdiff.append(np.linalg.norm(L - L_prev, ord='fro'))
        L_prev = L.copy()
        # energies
        E_field_kin = 0.5 * np.sum(pi**2)
        E_field_pot = 0.5 * phi.dot(L.dot(phi))
        E_field = E_field_kin + E_field_pot
        E_w_kin = 0.5 * np.sum(pw**2 / m_w)
        E_w_pot = 0.5 * kappa * np.sum((w - w0)**2)
        E_graph = E_w_kin + E_w_pot
        E_total = E_field + E_graph
        energies_field.append(E_field); energies_graph.append(E_graph); energies_total.append(E_total)
        # IPR
        ipr = compute_ipr_from_vecs(vecs, kmax=10)
        ipr_snapshots.append(ipr)

        # evolution: leapfrog-like update
        # field update
        pi = pi - dt * (L.dot(phi))
        phi = phi + dt * pi

        # grad^2 driving for weights
        grad2 = np.zeros(E)
        for (i,j), idx in edge_to_idx.items():
            grad2[idx] = (phi[i] - phi[j])**2

        # weight dynamics (classical oscillator)
        pw_dot = -kappa*(w - w0) - gamma * pw - alpha * grad2
        pw = pw + dt * pw_dot
        w = w + dt * (pw / m_w)

    # Save results
    out = {}
    out['times'] = times.tolist()
    out['Evac'] = np.array(Evac_list).tolist()
    out['gap'] = np.array(gap_list).tolist()
    out['Lnormdiff'] = np.array(Lnormdiff).tolist()
    out['energies_field'] = np.array(energies_field).tolist()
    out['energies_graph'] = np.array(energies_graph).tolist()
    out['energies_total'] = np.array(energies_total).tolist()
    out['ipr_first10'] = np.array(ipr_snapshots).tolist()
    # dump to files
    prefix = os.path.join(OUTDIR, save_prefix)
    np.savetxt(prefix + "_Evac.csv", np.c_[times, np.array(Evac_list)], header="t,Evac")
    np.savetxt(prefix + "_gap.csv", np.c_[times, np.array(gap_list)], header="t,gap")
    np.savetxt(prefix + "_Lnormdiff.csv", np.c_[times, np.array(Lnormdiff)], header="t,||DeltaL||F")
    np.savetxt(prefix + "_energies.csv", np.c_[times, np.array(energies_field), np.array(energies_graph), np.array(energies_total)], header="t,E_field,E_graph,E_total")
    # eigen snapshots (save only first 50)
    maxsave = min(50, len(eig_snapshots))
    for i in range(maxsave):
        np.savetxt(prefix + f"_eigs_{i:03d}.csv", eig_snapshots[i], header="eig")
    # ipr
    np.savetxt(prefix + "_ipr_first10.csv", np.array(ipr_snapshots), header="ipr_mode0,ipr_mode1,...")
    # plots
    fig, axes = plt.subplots(2,3, figsize=(15,8))
    ax = axes.flatten()
    ax[0].plot(times, Evac_list, '-o', ms=3); ax[0].set_title("Evac vs time"); ax[0].set_xlabel("t")
    ax[1].plot(times, gap_list, '-o', ms=3); ax[1].set_title("spectral gap"); ax[1].set_xlabel("t")
    ax[2].plot(times, Lnormdiff, '-o', ms=3); ax[2].set_title("||Delta L||_F"); ax[2].set_xlabel("t")
    ax[3].plot(times, energies_field, label='E_field'); ax[3].plot(times, energies_graph, label='E_graph'); ax[3].plot(times, energies_total, label='E_total'); ax[3].legend(); ax[3].set_title("energies")
    # quick IPR plot (mean of first 3 modes)
    ipr_arr = np.array(ipr_snapshots)
    ax[4].plot(times, np.mean(ipr_arr[:, :3], axis=1)); ax[4].set_title("mean IPR (first 3 modes)")
    # DOS heatmap (stack eigenvalues)
    # build matrix for heatmap: rows=time, cols=bins
    all_eigs = np.array(eig_snapshots)
    vmin = np.nanmin(all_eigs[all_eigs>0]) if np.any(all_eigs>0) else 1e-6
    vmax = np.nanmax(all_eigs)
    # sample up to 50 times for heatmap
    idxs = np.linspace(0, all_eigs.shape[0]-1, maxsave).astype(int)
    mat = all_eigs[idxs, :]
    ax[5].imshow(mat, aspect='auto', interpolation='nearest', cmap='viridis'); ax[5].set_title("eigs snapshot (rows=time)")
    plt.tight_layout()
    plt.savefig(prefix + "_summary.png", dpi=200)
    plt.close()

    # json summary
    with open(prefix + "_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    return out

# ---------- Scheme B implementation (enhanced) ----------
def run_scheme_B(graph=None, n_steps=60, theta=0.25, t0_eval=1e-2, save_prefix="schemeB"):
    if graph is None:
        graph = nx.path_graph(6)
    N = graph.number_of_nodes()
    edges = list(graph.edges())
    E = len(edges)
    if E > 14:
        raise RuntimeError("Too many edges for statevector demo (E>14). Use smaller graph.")

    # Pauli matrices
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    id2 = np.eye(2, dtype=complex)

    # Rx operator
    Rx_single = sla.expm(-1j * theta/2.0 * sx)

    # helper: op on full space
    def op_on_qubit(single_op, k):
        op = 1
        for q in range(E):
            if q == k:
                op = np.kron(op, single_op)
            else:
                op = np.kron(op, id2)
        return op

    Rx_ops = [op_on_qubit(Rx_single, k) for k in range(E)]
    Sz_ops = [op_on_qubit(sz, k) for k in range(E)]

    # initial product |0...0>
    dim = 2**E
    psi = np.zeros(dim, dtype=complex); psi[0] = 1.0

    edge_to_idx = {e:i for i,e in enumerate(edges)}

    Evac_list = []
    gap_list = []
    Lnormdiff = []
    Aexp_snapshots = []
    eig_snapshots = []
    ipr_snapshots = []
    E_graph_expect_list = []

    L_prev = None

    for step in range(n_steps):
        # apply single-qubit rotations (could be parallel)
        for k in range(E):
            psi = Rx_ops[k].dot(psi)
        # compute expected adjacency
        Aexp = np.zeros((N,N), dtype=float)
        sz_expect = np.zeros(E)
        for (i,j), idx in edge_to_idx.items():
            sz_exp = np.vdot(psi, Sz_ops[idx].dot(psi)).real
            sz_expect[idx] = sz_exp
            Aexp[i,j] = Aexp[j,i] = 0.5*(1.0 + sz_exp)
        Aexp_snapshots.append(Aexp.copy())
        Lexp = laplacian_from_W(Aexp)
        vals, vecs = sla.eigh(Lexp)
        vals_sorted = np.real(vals)
        eig_snapshots.append(vals_sorted)
        Ev = Evac_from_eigvals(vals_sorted, t0_eval)
        Evac_list.append(Ev)
        gap = vals_sorted[1] if len(vals_sorted)>1 else vals_sorted[0]
        gap_list.append(gap)
        if L_prev is None:
            Lnormdiff.append(0.0)
        else:
            Lnormdiff.append(np.linalg.norm(Lexp - L_prev, ord='fro'))
        L_prev = Lexp.copy()
        # pseudo graph energy: eps * sum <A>
        eps = 1.0
        E_graph_expect = eps * 0.5 * np.sum(1.0 + sz_expect)
        E_graph_expect_list.append(E_graph_expect)
        # ipr
        ipr = compute_ipr_from_vecs(vecs, kmax=10)
        ipr_snapshots.append(ipr)

    # save
    prefix = os.path.join(OUTDIR, save_prefix)
    np.savetxt(prefix + "_Evac.csv", np.c_[np.arange(len(Evac_list)), np.array(Evac_list)], header="step,Evac")
    np.savetxt(prefix + "_gap.csv", np.c_[np.arange(len(gap_list)), np.array(gap_list)], header="step,gap")
    np.savetxt(prefix + "_Lnormdiff.csv", np.c_[np.arange(len(Lnormdiff)), np.array(Lnormdiff)], header="step,||DeltaL||F")
    np.savetxt(prefix + "_Egraph_expect.csv", np.c_[np.arange(len(E_graph_expect_list)), np.array(E_graph_expect_list)], header="step,E_graph_expect")
    # eig snapshots (first 50)
    maxsave = min(50, len(eig_snapshots))
    for i in range(maxsave):
        np.savetxt(prefix + f"_eigs_{i:03d}.csv", eig_snapshots[i], header="eig")
    np.savetxt(prefix + "_ipr_first10.csv", np.array(ipr_snapshots), header="ipr_mode0,...")

    # plots
    fig, axes = plt.subplots(2,3, figsize=(15,8))
    ax = axes.flatten()
    times = np.arange(len(Evac_list))
    ax[0].plot(times, Evac_list, '-o', ms=4); ax[0].set_title("Evac vs step")
    ax[1].plot(times, gap_list, '-o', ms=4); ax[1].set_title("expected spectral gap")
    ax[2].plot(times, Lnormdiff, '-o', ms=4); ax[2].set_title("||Delta <L>||_F")
    ax[3].plot(times, E_graph_expect_list); ax[3].set_title("E_graph_expect (pseudo)")
    # ipr mean
    ax[4].plot(times, np.mean(np.array(ipr_snapshots)[:, :3], axis=1)); ax[4].set_title("mean IPR (first 3)")
    # image of A expectation for first snapshots
    # take up to 9 snapshots and tile them
    nshow = min(9, len(Aexp_snapshots))
    A_stack = Aexp_snapshots[:nshow]
    # show as average
    avgA = np.mean(np.array(A_stack), axis=0)
    ax[5].imshow(avgA, cmap='viridis'); ax[5].set_title("avg <A> (first few steps)")
    plt.tight_layout()
    plt.savefig(prefix + "_summary.png", dpi=200)
    plt.close()

    out = {
        'Evac': Evac_list, 'gap': gap_list, 'Lnorm': Lnormdiff, 'E_graph_expect': E_graph_expect_list
    }
    with open(prefix + "_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    return out

# ---------- parameter scans & run ----------
summary = {'schemeA':{}, 'schemeB':{}}
t0_eval = 1e-2

# Scheme A: baseline + scan over alpha
graphA = nx.grid_2d_graph(2,4)
graphA = nx.convert_node_labels_to_integers(graphA)
alphas = [0.1, 0.3, 0.6]
for alpha in alphas:
    params = {'kappa':2.0, 'gamma':0.3, 'alpha':alpha, 'm_w':1.0, 'w0':1.0, 'phi0_amp':0.05, 'seed':0}
    pref = f"schemeA_alpha_{alpha:.2f}"
    print("Running Scheme A alpha=", alpha)
    outA = run_scheme_A(graph=graphA, params=params, tmax=6.0, dt=0.02, t0_eval=t0_eval, save_prefix=pref)
    summary['schemeA'][f'alpha_{alpha:.2f}'] = {'Evac_mean': float(np.mean(outA['Evac'])), 'Evac_std': float(np.std(outA['Evac']))}

# Scheme B: baseline + scan over theta (be conservative with E)
graphB = nx.path_graph(6)
graphB = nx.convert_node_labels_to_integers(graphB)
thetas = [0.05, 0.15, 0.30]
for theta in thetas:
    print("Running Scheme B theta=", theta)
    outB = run_scheme_B(graph=graphB, n_steps=60, theta=theta, t0_eval=t0_eval, save_prefix=f"schemeB_theta_{theta:.2f}")
    summary['schemeB'][f'theta_{theta:.2f}'] = {'Evac_mean': float(np.mean(outB['Evac'])), 'Evac_std': float(np.std(outB['Evac']))}

# Save summary
with open(os.path.join(OUTDIR, "demo_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Zip results
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUTDIR):
        for fn in files:
            if fn.endswith(".zip"): continue
            full = os.path.join(root, fn)
            arcname = os.path.relpath(full, OUTDIR)
            zf.write(full, arcname)
print("All done. Results saved and zipped at:", ZIP_PATH)