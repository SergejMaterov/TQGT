# Continuum Simulations

**Purpose —** Deriving a continuum from a graph. Numerical validation of continuum-like behavior on discrete graph motifs and extraction of spectral-scaling coefficients from random-walk diffusion. The primary motif studied here is the 4D torus (Z^4).

---


**Step 1 — verify check that the torus satisfies all the axioms**

```bash
python3 auto_continuum_check_v3.py --out out_v3 --L4 5
```

---

**Step 2 — precise d_s and C1**

```bash
python3 finite_size_scaling.py --dims 4 --Lmin 5 --Lmax 39 --Lstep 4 --nwalks 60000 --n_boot 50 --out fss_final
```

C1 = |a| / 4  (from d4_rows.json, field "a")

---

**Step 3 — C2 and C3**

```bash
python3 measure_c2c3.py --fss_json fss_final/d4_rows.json --Lvals 5,7,9 --nwalks 60000 --nseeds 5 --out c2c3_out
```
C2 currently overshoots because diagonal edges at small L (L=5) nearly saturate the graph and the FSS fit extrapolates above 4. The C2 number is statistically significant but physically the 3-point fit is fragile — C1 and C3 are the reliable ones.

---

**Step 4 — UV dimensional-reduction check (negative result, checked)**

```bash
python3 uv_dimensional_reduction_check.py --L 21 --Tmax 24 --nwalks 1500000
```

Tests whether the *local* (t-resolved) spectral dimension dips toward d_s≈2
at short diffusion times, as in CDT / causal sets. **It does not**: dim_S(t)
rises from ≈3.4 at t=4 to a plateau at ≈4 by t≈8–10 and stays there within
bootstrap error out to t≈22 (beyond which return-probability statistics run
out and the estimator becomes noise-dominated — not a physical signal).

This is expected, not a bug: the Z^4 torus here is a fixed, non-dynamical
lattice (Q1–Q4 put quantum structure in the *state*, not in a superposition
of graph topologies), and UV dimensional reduction elsewhere in the
literature comes from summing/superposing geometries, not from discreteness
alone. **"d_s→2 in the UV" is not a prediction of the current (v14) graph
construction — do not re-list it as one without first revisiting
[`UV_DIMENSIONAL_REDUCTION.md`](UV_DIMENSIONAL_REDUCTION.md).**
