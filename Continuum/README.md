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
