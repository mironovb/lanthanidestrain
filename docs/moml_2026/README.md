# MoML 2026 short paper

`main.tex` + `reference.bib` + `figures/` follow the MoML template structure
(LoG 2022 style). The style file `log_2022.sty` is **not** in this directory:
download it from the MoML submission page and place it next to `main.tex`.

Build:

    pdflatex main && bibtex main && pdflatex main && pdflatex main

`preview_standin_style.pdf` was compiled with a minimal stand-in for the style
(plain `article`, 1-inch margins) to check that the source is complete; page
count and layout will differ under the real style.

Open items before submission:

- author block: affiliation of B. Mironov; second author only with his agreement (email placeholder);
- `\cite{safe}`: supply the citation for the SAFE solvent-extraction database (bib entry is a placeholder);
- acknowledgements (computing, funding, data providers);
- check the page limit; if it is four pages of main text, drop Table 2 into the appendix and compress Section 5.

Numbers come from: `automl/reports/SCIENTIFIC_FINDINGS.md` (I14, I15, I17–I21),
`automl/reports/decision_quality.json`, `automl/artifacts/gxtb_series/lnxtb_benchmark.json`,
and the R² anatomy computed on 2026-09-08 (bootstrap, permutation, variance decomposition).
Figures are copies of `docs/talk_acs/{T2_architecture,T3_block,T9_ranking,contraction}.png`.
