# Continuum Simulations

**Purpose —** Deriving a continuum from a graph. Numerical validation of continuum-like behavior on discrete graph motifs and extraction of spectral-scaling coefficients from random-walk diffusion. The primary motif studied here is the 4D torus (Z^4).

---


**Step 1 — verify the torus passes all axioms**

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

**Step 3**



