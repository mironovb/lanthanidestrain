# Presentation plan — Vogiatzis meeting

## Novelty check (web, August 2026) — what the contraction result is against prior art

**Already known, and must be cited rather than claimed:**

| prior work | what it established |
|---|---|
| Bursch, Hansen, Grimme, *Inorg. Chem.* 2017 | GFN1-xTB usable for lanthanoid geometries: 80 complexes, per-complex heavy-atom RMSD 0.65 Å vs CSD X-ray |
| Zhang et al., *J. Comput. Chem.* **2026** (Ln-xTB) | GFN2-xTB lanthanide **bond lengths are inaccurate in absolute terms**: MAD 0.134 Å vs all-electron DFT; their new Ln-xTB reaches 0.037 Å |
| g-xTB preprint (ChemRxiv 2025) | benchmarks ~32k relative **energies** (incl. actinides); f-shell Mulliken populations used as fitting targets. **No lanthanide geometry validation** |
| GFN2 parameter file / docs | the Ce→Lu linear interpolation is the authors' documented design, not our discovery |

**Genuinely new in our result (the claim to make):**
1. **The observable**: per-ligand contraction *slope* c_L = d⟨M–donor⟩/d r_Shannon at **fixed ligand environment** (anchor-substitution series). Prior work reports per-complex error statistics, which cannot see a *correlated trend error* — bonds can be individually acceptable while the series derivative is 2.5× too shallow. Nobody has measured the derivative.
2. **Validated against experiment** (Shannon 1976 radii), not against DFT.
3. The **2.47× systematic under-response**, plus its noise structure (23 % of GFN2's non-linear response shared across ligands vs 96 % for g-xTB).
4. **First lanthanide geometry validation of g-xTB at all** — its own preprint only benchmarks energies.
5. The **mechanism chain**: linear-in-Z parameters → rank-1 metal response → measured ML encoder interchangeability (effective rank 1.05/8). No one has connected these.
6. Scale and relevance: 71 extraction-relevant ligands, 2,130 optimisations.

**Obligations before any submission:** read the Ln-xTB full text (paywalled; closest prior art, appeared this year) and run our benchmark on Ln-xTB itself — it is the natural third method and the pre-registered falsifying test for the f-in-valence explanation.

---

## Narrative arc (three acts)

- **Act 1 — the ML result (slides 2–11).** Open on Kostas's own question; answer it precisely on one slide; then earn that answer: data provenance → pipeline → EDA → protocol → models → July-vs-now → the best system and its honest boundary.
- **Act 2 — the surprise (slides 12–15).** Why doesn't 3D do more? Because the Hamiltonian is rank-1 in the metal. The contraction benchmark (figure), the novelty slide (web check), and the decisive negative: fixing the chemistry does not fix the score.
- **Act 3 — what actually works and what's next (slides 16–19).** The objective is the lever (transfer figure); the ceiling; methodology; summary + four discussion items.

Principles: one message per slide, stated in the title; every number from a real artefact; caveats on the slide that makes the claim, never later; speaker notes carry the depth so slides stay sparse.

## Slide-by-slide

| # | title (the message) | body | figure | speaker notes carry |
|---|---|---|---|---|
| 1 | Title | — | — | 30-second framing |
| 2 | Your question — the answer up front | "constantly more accurate?" → no as single model / yes the best system needs it | — | the July numbers he read and what changed |
| 3 | Why adjacent lanthanides are the hard case | 0.013 Å; industrial extraction; the metric | — | why R²=0 baseline is already non-trivial |
| 4 | Data: SAFE database | 48,138 rows · 31 elements · 181 DOIs → 4,746/162/14 | — | curation, replicate structure, who built it |
| 5 | From SMILES to simplicial complex | 5-step pipeline table | — | Architector choices, CN 8–9, force check |
| 6 | The data at a glance | composition + targets | f1, f2 | Pm gap; heavy left tail of log D |
| 7 | Why leave-extractants-out is the hard split | PCA reading | f3 | winsorisation note; 19 %/8 % var |
| 8 | Evaluation protocol | LOEO 5×3, 16–32 seeds, OOF; pair construction | — | replicate averaging; why per-seed sd 0.0285 matters |
| 9 | Three models + pair-fitted stack | table | — | why row-fitted stacking fails (79 % to the wrong model) |
| 10 | Results: July → today | both gains are objective changes | f4 | the loss chain on CatBoost |
| 11 | Best system and its honest boundary | 0.313 vs 0.291; weight 0.46; strict key; exploratory | — | why contribution shrank (mechanism) |
| 12 | Why 3D contributes so little | GFN2 rank-1 in the metal; encoder interchangeability | — | parameter file numbers |
| 13 | The contraction benchmark | GFN2 0.41 vs g-xTB 1.08 vs Shannon 1.00 | f5 | construction (anchor substitution), solvent replication |
| 14 | **Is this new? (web check)** | prior-art table + the 6 new elements | — | Ln-xTB obligation; g-xTB energies-only |
| 15 | Fixing the chemistry does not fix the score | 4 nulls table + shrinkage artefact story | — | the 237 % rescaling test |
| 16 | What works: align the loss with the contrast | transfer across encoders | f6 | C15/C17 design, power |
| 17 | Headroom: ceiling ≈ +0.53 | noise cancellation on differencing | f7 | why the naive ceiling is impossible |
| 18 | Methodology | power / scale-free / silent no-ops / selection | — | the three killed positives, one line each |
| 19 | Summary + discussion | 3 claims + 4 discussion items | — | suggested order for the call |
