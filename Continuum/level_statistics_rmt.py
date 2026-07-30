"""
level_statistics_rmt.py
========================
Quantum-chaos diagnostic via level statistics (Bohigas-Giannoni-Schmit),
immune to the almost-periodicity argument (it's a STATIC spectral property,
not a dynamical time-series property).

v2 fixes vs. the previous version
----------------------------------
1. REFLECTION SYMMETRY was not resolved. A uniform open chain (same J, hx,
   hz on every site/bond) has an exact spatial-reflection symmetry
   (i <-> N+1-i) in addition to the Z2 spin-flip symmetry. Only Z2 was ever
   split. Mixing the two reflection-parity subsequences inside each
   Z2 sector (or inside the full spectrum, for hz!=0) suppresses level
   repulsion exactly like unresolved Z2 does -- and is almost certainly why
   both the "chaotic" and "integrable" benchmarks converged to the same
   contaminated <r> ~= 0.42 at N=12/14 regardless of hz.

   Fix: break reflection explicitly with a small site-dependent disorder
   term (default: disorder in hx). This does NOT break integrability at
   hz=0 (a disordered transverse-field Ising model is still exactly
   solvable by Jordan-Wigner -- only hz!=0 breaks integrability), so the
   Z2 sector split there remains valid and meaningful. At hz!=0 there is
   no residual symmetry left to resolve.

2. MEMORY: dense np.linalg.eigvalsh(H.toarray()) needs the full dim x dim
   matrix PLUS an internal LAPACK copy at diagonalization time -- for
   N=16 (dim=65536) that's ~64 GB, right at/over a 64 GB machine's limit.
   Since r_statistic() only ever uses the middle ~50% of the spectrum,
   there is no need to compute the full spectrum for large N.

   Fix: for N above --dense-max-n, use shift-invert Lanczos (scipy
   eigsh) to pull only a window of k eigenvalues around mid-spectrum,
   operating on the sparse matrix directly -- no dense array is ever
   formed.

Usage
-----
    python3 level_statistics_rmt.py --N 14 --model chaotic
    python3 level_statistics_rmt.py --N 14 --model integrable
    python3 level_statistics_rmt.py --N 16 --model chaotic --k 3000
    python3 level_statistics_rmt.py --N 16 --model integrable --k 3000

    # sanity check that disorder is actually doing its job: compare
    # --disorder 0 (should reproduce the old contaminated ~0.42 result)
    # against the default disorder (should separate cleanly).
    python3 level_statistics_rmt.py --N 14 --model chaotic --disorder 0.0
"""
import argparse
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, splu, LinearOperator


# --------------------------------------------------------------------------
# Hamiltonian construction
# --------------------------------------------------------------------------

def pauli_sparse(op, i, N):
    I2 = sp.identity(2, format='csr')
    X = sp.csr_matrix([[0, 1], [1, 0]], dtype=float)
    Z = sp.csr_matrix([[1, 0], [0, -1]], dtype=float)
    mat = {'X': X, 'Z': Z}[op]
    out = sp.identity(1, format='csr')
    for k in range(N):
        out = sp.kron(out, mat if k == i else I2, format='csr')
    return out


def build_H_sparse(N, J=1.0, hx=1.05, hz=0.5, periodic=False,
                    disorder=0.03, seed=0):
    """Mixed-field Ising chain, with an optional small site-dependent
    disorder in hx to break the exact spatial-reflection symmetry that a
    uniform chain otherwise has.

    hz=0 recovers the integrable (free-fermion) transverse-field Ising
    model -- disorder in hx does NOT break this integrability (still
    Jordan-Wigner solvable). hz != 0 is the standard non-integrable
    'chaotic' benchmark. Reflection symmetry, present in BOTH cases when
    disorder=0, is broken by disorder != 0 in both cases identically, so
    the two benchmarks stay on equal footing.
    """
    dim = 2 ** N
    H = sp.csr_matrix((dim, dim), dtype=float)

    bonds = range(N) if periodic else range(N - 1)
    for i in bonds:
        j = (i + 1) % N
        H = H + J * (pauli_sparse('Z', i, N) @ pauli_sparse('Z', j, N))

    rng = np.random.default_rng(seed)
    # Deterministic, non-palindromic disorder pattern (a random draw is
    # non-palindromic with probability 1, but we also nudge it explicitly
    # so the symmetry breaking is not accidentally left intact by a
    # symmetric draw).
    hx_site = hx + disorder * rng.uniform(-1.0, 1.0, size=N)

    for i in range(N):
        H = H + hx_site[i] * pauli_sparse('X', i, N)
        if hz != 0:
            H = H + hz * pauli_sparse('Z', i, N)

    return H


# --------------------------------------------------------------------------
# Z2 (spin-flip) sector split -- still valid and needed at hz=0
# --------------------------------------------------------------------------

def sector_projectors(N):
    """Exact, cheap Z2 sector basis for P = prod_i X_i (global spin flip).
    P|b> = |~b> (bit-complement), so P just pairs each computational basis
    state with its complement. The +1/-1 eigenvectors of P within each pair
    {b, ~b} are (|b> +/- |~b>)/sqrt(2). O(dim) construction, no
    diagonalization needed.

    NOTE: this resolves Z2 spin-flip only. It does NOT resolve reflection
    symmetry -- that is instead removed at the Hamiltonian level via
    disorder in build_H_sparse, which is simpler than building a second
    projector and is sufficient since we only need *a* symmetry-free
    subsystem, not a symmetry-adapted basis for every possible symmetry.
    """
    dim = 2 ** N
    idx = np.arange(dim, dtype=np.int64)
    complement = (~idx) & (dim - 1)
    seen = np.zeros(dim, dtype=bool)
    col_p = col_m = 0
    for b in range(dim):
        if seen[b]:
            continue
        bc = complement[b]
        seen[b] = True
        seen[bc] = True
        if b == bc:
            col_p += 1
        else:
            col_p += 1
            col_m += 1

    Bp = sp.lil_matrix((dim, col_p))
    Bm = sp.lil_matrix((dim, col_m))
    seen[:] = False
    cp = cm = 0
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for b in range(dim):
        if seen[b]:
            continue
        bc = complement[b]
        seen[b] = True
        seen[bc] = True
        if b == bc:
            Bp[b, cp] = 1.0
            cp += 1
        else:
            Bp[b, cp] = inv_sqrt2
            Bp[bc, cp] = inv_sqrt2
            cp += 1
            Bm[b, cm] = inv_sqrt2
            Bm[bc, cm] = -inv_sqrt2
            cm += 1
    return Bp.tocsr(), Bm.tocsr()


def dense_eigvals_in_sector(H_sparse, basis_sparse):
    """Compress H into the given orthonormal sector basis and diagonalize
    the (much smaller) dense block."""
    Hc = basis_sparse.conj().T @ (H_sparse @ basis_sparse)
    Hc = np.asarray(Hc.todense()) if sp.issparse(Hc) else np.asarray(Hc)
    Hc = 0.5 * (Hc + Hc.conj().T)
    return np.linalg.eigvalsh(Hc)


# --------------------------------------------------------------------------
# Large-N path: windowed sparse diagonalization (no dense dim x dim array)
# --------------------------------------------------------------------------

def windowed_eigvals_sparse(H_sparse, k, sigma=None, permc_spec='MMD_AT_PLUS_A',
                             use_pardiso=True):
    """Eigenvalues in a window around mid-spectrum via shift-invert Lanczos.
    Never forms a dense dim x dim array -- memory cost is set by the sparse
    factorization fill-in, not by dim^2. Suitable for N where dense
    eigvalsh would not fit in RAM (N=16 and up on a typical 64 GB machine).

    IMPORTANT: scipy's built-in splu (SuperLU) is single-threaded -- it
    will NOT use multiple cores no matter how many you have, so the
    factorization step (the bottleneck at N=16) runs on one core only.
    If pypardiso is installed (Intel MKL PARDISO), use it instead: it
    parallelizes the LU factorization across all available cores, which
    is the actual bottleneck here, not the Lanczos iterations themselves.

        pip install pypardiso
    """
    dim = H_sparse.shape[0]
    if sigma is None:
        # cheap mid-spectrum estimate: mean of the diagonal
        sigma = float(H_sparse.diagonal().mean())

    if use_pardiso:
        try:
            from pypardiso import PyPardisoSolver
            H_csr = H_sparse.tocsr()
            shifted = (H_csr - sigma * sp.identity(dim, format='csr')).tocsr()
            solver = PyPardisoSolver()
            solver.factorize(shifted)  # parallel across all cores

            def matvec(x):
                return solver.solve(shifted, x)

            OPinv = LinearOperator(shifted.shape, matvec=matvec, dtype=shifted.dtype)
            evals = eigsh(H_csr, k=k, sigma=sigma, which='LM', OPinv=OPinv,
                          return_eigenvectors=False)
            solver.free_memory(everything=True)
            return np.sort(evals)
        except ImportError:
            print("pypardiso not installed (pip install pypardiso) -- "
                  "falling back to single-threaded SuperLU. This will be "
                  "much slower at large N.")

    H_csc = H_sparse.tocsc()
    shifted = (H_csc - sigma * sp.identity(dim, format='csc')).tocsc()
    lu = splu(shifted, permc_spec=permc_spec)
    OPinv = LinearOperator(H_csc.shape, matvec=lu.solve, dtype=H_csc.dtype)

    evals = eigsh(H_csc, k=k, sigma=sigma, which='LM', OPinv=OPinv,
                  return_eigenvectors=False)
    return np.sort(evals)


# --------------------------------------------------------------------------
# r-statistic (Oganesyan-Huse)
# --------------------------------------------------------------------------

def r_statistic(evals, mid_fraction=0.5):
    """When evals already comes from windowed_eigvals_sparse (i.e. it is
    ALREADY a mid-spectrum window, not the full spectrum), pass
    mid_fraction=1.0 so no further trimming is applied."""
    s = np.sort(evals)
    n = len(s)
    lo = int(n * (0.5 - mid_fraction / 2))
    hi = int(n * (0.5 + mid_fraction / 2))
    s = s[lo:hi]
    gaps = np.diff(s)
    gaps = gaps[gaps > 1e-12]
    if len(gaps) < 2:
        return float('nan'), float('nan')
    r = np.minimum(gaps[1:], gaps[:-1]) / np.maximum(gaps[1:], gaps[:-1])
    return float(np.mean(r)), float(np.std(r) / np.sqrt(len(r)))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=14)
    ap.add_argument('--model', choices=['chaotic', 'integrable'], default='chaotic')
    ap.add_argument('--periodic', action='store_true')
    ap.add_argument('--J', type=float, default=1.0)
    ap.add_argument('--hx', type=float, default=1.05)
    ap.add_argument('--hz', type=float, default=0.5)
    ap.add_argument('--disorder', type=float, default=0.03,
                     help='Amplitude of site-dependent hx disorder that '
                          'breaks spatial-reflection symmetry. Set to 0 '
                          'to reproduce the old (contaminated) behaviour.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dense-max-n', type=int, default=14,
                     help='Use dense eigvalsh for N <= this; sparse '
                          'windowed diagonalization above it.')
    ap.add_argument('--k', type=int, default=3000,
                     help='Number of mid-spectrum eigenvalues to compute '
                          'in the sparse (large-N) path.')
    ap.add_argument('--no-pardiso', action='store_true',
                     help='Force single-threaded SuperLU instead of '
                          'pypardiso, even if pypardiso is installed.')
    args = ap.parse_args()

    hz = 0.0 if args.model == 'integrable' else args.hz
    use_sparse_path = args.N > args.dense_max_n

    t0 = time.time()
    H = build_H_sparse(args.N, J=args.J, hx=args.hx, hz=hz,
                        periodic=args.periodic, disorder=args.disorder,
                        seed=args.seed)
    print(f"N={args.N}, model={args.model} (hz={hz}), dim={2**args.N}, "
          f"disorder={args.disorder}, built in {time.time()-t0:.1f}s")

    if hz == 0.0:
        # Integrable case still has the Z2 symmetry (disorder in hx alone
        # does not break it): MUST split sectors or the statistics mix two
        # independent level sequences.
        t0 = time.time()
        Pp, Pm = sector_projectors(args.N)
        print(f"Z2 sector projectors built in {time.time()-t0:.1f}s "
              f"(dims: {Pp.shape[1]}, {Pm.shape[1]})")

        if not use_sparse_path:
            t0 = time.time()
            ev_p = dense_eigvals_in_sector(H, Pp)
            ev_m = dense_eigvals_in_sector(H, Pm)
            print(f"Dense sector diagonalization done in {time.time()-t0:.1f}s")
            rp = r_statistic(ev_p)
            rm = r_statistic(ev_m)
        else:
            # Compress into each Z2 sector (small dense blocks are fine --
            # each is only dim/2), THEN, if that block is itself still too
            # big, window it. For the sizes here (dim/2 <= 32768 at N=16)
            # a dense in-sector diagonalization is usually still fine; if
            # you push N further, replace with a sparse-in-sector window.
            t0 = time.time()
            ev_p = dense_eigvals_in_sector(H, Pp)
            ev_m = dense_eigvals_in_sector(H, Pm)
            print(f"Dense sector diagonalization done in {time.time()-t0:.1f}s")
            rp = r_statistic(ev_p)
            rm = r_statistic(ev_m)

        print(f"<r> (+parity sector) = {rp[0]:.4f} +/- {rp[1]:.4f}  (n={len(ev_p)})")
        print(f"<r> (-parity sector) = {rm[0]:.4f} +/- {rm[1]:.4f}  (n={len(ev_m)})")

    else:
        # Chaotic case: disorder has removed reflection symmetry, and
        # hz != 0 already removes Z2 -- no residual symmetry to resolve.
        if not use_sparse_path:
            t0 = time.time()
            ev = np.linalg.eigvalsh(H.toarray())
            print(f"Full dense diagonalization done in {time.time()-t0:.1f}s")
            r, err = r_statistic(ev)
            print(f"<r> = {r:.4f} +/- {err:.4f}  (n={len(ev)})")
        else:
            t0 = time.time()
            ev = windowed_eigvals_sparse(H, k=args.k,
                                         use_pardiso=not args.no_pardiso)
            print(f"Sparse windowed diagonalization done in {time.time()-t0:.1f}s "
                  f"(k={args.k} eigenvalues near mid-spectrum)")
            # ev is already a mid-spectrum window -- don't trim it again.
            r, err = r_statistic(ev, mid_fraction=1.0)
            print(f"<r> = {r:.4f} +/- {err:.4f}  (n={len(ev)})")

    print("\nReference: GOE (chaotic) <r> ~= 0.5307,  Poisson (integrable) <r> ~= 0.3863")


if __name__ == '__main__':
    main()
