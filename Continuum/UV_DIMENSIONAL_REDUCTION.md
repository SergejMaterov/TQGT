# UV Dimensional Reduction Check — Negative Result

**Status: checked, not observed. Do not re-propose "d_s → 2 in the UV" as a
Quantumograph prediction without first revisiting this note.**

## Question

CDT, causal sets, and asymptotic safety all report a spectral-dimension flow
from d_s≈4 at large scales down to d_s≈2 near the discreteness scale. Since
`finite_size_scaling.py` already establishes d_s→4.0000 for L≥5 via a single
global power-law fit over an early diffusive window, it is natural to ask
whether the *same* random-walk data, examined at short diffusion times on a
single large graph, shows a similar UV flow.

**It does not — see below.**

## Why this needed a separate check

`finite_size_scaling.py` extracts one global exponent from a single power-law
fit over the window `[4, diam²/8]`. That number is dominated by the shared IR
regime and does not by itself tell you whether d_s(t) *varies* across the
window (e.g. rises from 2 at small t to 4 at large t) or is flat throughout.
Answering that requires the **local** (t-resolved) slope, computed on a graph
large enough that finite-size effects don't contaminate the small-t region —
otherwise a genuine UV signal and an ordinary finite-size artifact look the
same.

## Method

Same estimator as `finite_size_scaling.py` / `auto_continuum_check_v3.py`
(random-walk return probability, even-t sampling to avoid the Z^4 torus's
bipartite oscillation), but instead of one global fit, compute the local
central-difference slope at each even t:

```
dim_S(t) = -2 · [ln P_return(t+2) - ln P_return(t-2)] / [ln(t+2) - ln(t-2)]
```

with bootstrap error bars from resampling individual walks. Run on a single
large, fixed L (so the IR/finite-size scale is far from the t-window probed)
rather than scanning L, which is what isolates short-time behavior from
finite-size behavior.

Script: `uv_dimensional_reduction_check.py` (this folder).

```bash
python3 uv_dimensional_reduction_check.py --L 21 --Tmax 24 --nwalks 1500000
```

## Result

Z⁴ torus, L=21 (odd), N=194,481 vertices, degree 8. 1.5M random walks, 400
start points, 20 bootstrap resamples per point.

| t  | dim_S(t) | ± bootstrap | P_return(t) |
|----|----------|-------------|-------------|
| 4  | 3.378    | 0.013       | 4.10×10⁻²   |
| 6  | 3.716    | 0.018       | 1.96×10⁻²   |
| 8  | 3.846    | 0.044       | 1.13×10⁻²   |
| 10 | 3.815    | 0.064       | 7.33×10⁻³   |
| 12 | 3.764    | 0.090       | 5.21×10⁻³   |
| 14 | 3.957    | 0.110       | 3.89×10⁻³   |
| 16 | 4.068    | 0.213       | 2.95×10⁻³   |
| 18 | 3.952    | 0.190       | 2.33×10⁻³   |
| 20 | 3.837    | 0.206       | 1.90×10⁻³   |
| 22 | 3.779    | 0.444       | 1.59×10⁻³   |

dim_S(t) rises monotonically from ≈3.4 at t=4 to a plateau at ≈4 by t≈8–10,
and stays consistent with 4 (within bootstrap error) out to t=22, where
statistics start to run out. **No dip toward d_s≈2, or toward any value
below 4, appears anywhere in the reliable window.** The slight suppression
at t=4 is the ordinary short-step lattice artifact of a cubic graph (walks
haven't yet explored enough directions to look isotropic) and disappears by
t≈8 — it is not a trend toward 2.

A first pass with `Tmax=80` was also run and showed wild swings (dim_S
ranging from −6 to +18 for t≳40). This is **not** a physical signal: at
those t, P_return(t) has decayed to ~10⁻⁴, leaving only tens of raw return
events out of 200k walks, so the central-difference derivative is dominated
by shot noise. Anyone re-running this check should restrict the fit window
to where P_return(t) is still ≳10⁻³ (roughly t≲25 for these parameters) or
scale `nwalks` up accordingly — large-t noise should not be mistaken for a
UV/IR crossover in either direction.

## Interpretation

The Z⁴ torus used throughout Quantumograph v14 is a **fixed, non-dynamical
lattice** — Q1–Q4 put quantum structure in the *state* on vertices/edges, not
in a superposition over graph topologies. UV dimensional reduction in CDT and
causal sets is a consequence of summing over (or superposing) geometries /
causal structure, not of discreteness per se. A fixed classical Z^4 lattice
with an ordinary nearest-neighbour random walk has no such mechanism, and the
numerics confirm there isn't one hiding in this construction either.

**This is not a flaw in the theory** — it simply means "d_s→2 in the UV" is
not a prediction of the current (v14) graph construction, and should not be
listed as one. If a UV dimensional-reduction prediction is wanted, it would
have to come from a genuinely different mechanism (e.g. quantum superposition
of edge configurations, a dynamical/fluctuating graph rather than a fixed
one) that does not yet exist in any document in this repository — introducing
it would be new physics, not a corollary of what's already here.

## Reproducibility

- Script: `uv_dimensional_reduction_check.py`
- Dependencies: `numpy`, `networkx` (same as `finite_size_scaling.py`)
- Runtime: ~2–5 min for the table above on a single core; scale `--nwalks`
  down for a faster, noisier check, or up for tighter error bars.
