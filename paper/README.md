# Paper artifacts and CPU reproduction

The result maps use the preprint numbering for five figures and nineteen tables. Figures 1–5 and Tables 1–16 are represented by the earlier reproduction package’s display assets. The three Appendix L table sources match the preprint. [Paper result map](RESULT_MAP.md) · [Machine-readable map](result_map.json)

From the release root, the existing numerical analyses are:

```sh
python analysis/abc/recompute_windows.py --output build/abc_windows
python analysis/abc/recompute_headline.py --output build/abc_headline
python analysis/wide_d/recompute.py --output build/wide_d
python secondary/recompute.py --output build/secondary
python paper/verification_metrics/reproduce.py --output build/verification_metrics
```

These commands consume archived scores and perform CPU statistics. They do not load weights or run models. The last command reproduces baseline, decision flips, normalized T, and wide-D supplementary metrics with the archived 50,000-draw bootstrap, then regenerates the three metric tables. It copies inputs into the output folder before running and checks all nine result CSVs against the included references (absolute tolerance `1e-12`). Repeating all metric stages takes several minutes. For a short run, add `--stages baseline tables`; the table stage uses the included remaining result CSVs.

The Appendix B.1 check recomputes the token-window contrasts after excluding two PRMBench categories. The Appendix L commands reproduce Tables 17–18, the sensitivity analyses in the placebo text, and Table 19, respectively.

```sh
python paper/category_exclusion/recompute_category_exclusion.py
OPENBLAS_NUM_THREADS=1 python paper/appendix_l/placebo_surface/reproduce.py --output build/appendix_l/placebo_surface.csv
OPENBLAS_NUM_THREADS=1 python paper/appendix_l/placebo_surface/reproduce_sensitivities.py --output-dir build/appendix_l
python paper/appendix_l/routing/reproduce.py --output build/appendix_l/routing
```

The category check uses the original donor assignments. The placebo and surface-gap analyses use the saved fixed-receiver margins and TF-IDF scores. Table 19 starts from per-receiver router measurements. [Category exclusion](category_exclusion/README.md) · [Placebo and surface gap](appendix_l/placebo_surface/README.md) · [Routing diagnostics](appendix_l/routing/README.md)

`verification_metrics/extracted/` stores text-free candidate-score projections for all nine window and wide-D runs. Its manifests describe the historical extraction. `references/windows/` retains only the native pair-level T values needed for cross-checking; native preliminary confidence intervals are not included. The retained reference values are not rerun predictions. The three generated metric tables reproduce the reviewed numerical cells; their automated typography predates the manuscript's final table layout.

The primary margin is `M = mean_logprob(INCORRECT) − mean_logprob(CORRECT)`; `M > 0` predicts INCORRECT, including zero in CORRECT. Mean and total candidate log probabilities remain separate analyses. The baseline is this study's paired assay, not an official benchmark score. Normalization is `R = sum(T_i) / sum(g_i)` on the same eligible pairs, with `2 T_i = g_i − g_after_i`; it is not a mediation share, probability, or mean of per-pair ratios. Undefined conditional flip rates remain undefined. Wide-D baselines are compared within their own run when forward scores differ from the window run. See each analysis script's scientific definitions.

To regenerate the four numerical figures:

```sh
python -m pip install -r paper/figure_source/requirements.txt
python paper/figure_source/reproduce.py --output build/figures
```

Rendering also needs Poppler (`pdftotext`) and Times New Roman available to Matplotlib. Fonts are not bundled. The retained plotting functions, data, palette, marker roles, axes, and final label positions come from the actual final figure source. The script checks text tokens and page sizes against the reviewed PDFs, plus the appendix layout and 144 window-profile intervals. On the preparation machine (Matplotlib 3.10.8), all four regenerated figures rendered pixel-identically to the reviewed PDFs at 144 dpi. Font availability and renderer versions can affect other environments.

The earlier Figure 1 schematic is editable in `figure_source/overview.svg`. Its source builder uses ReportLab to measure system Arial and Inkscape to export PDF:

```sh
ROUTEVERIFY_FIGURE_OUTPUT=build/figures python paper/figure_source/overview.py
python paper/figure_source/fix_gradient.py \
  build/figures/abc_overview_20260908.pdf \
  build/figures/abc_overview_fixed.pdf
```

`INKSCAPE` can name the Inkscape executable; `ROUTEVERIFY_FONT_DIR` can name a directory containing `Arial.ttf` and `Arial Bold.ttf`. The defaults use the local executable and macOS system fonts. The final command applies the recorded vector-gradient encoding repair; it does not change the drawing. During preparation of the earlier reproduction package, the regenerated schematic matched its archived Figure 1 PDF pixel-for-pixel at 144 dpi. Existing `figures/` PDFs can be inspected without any plotting dependencies.

Full model-forward experiments require pinned weights, upstream datasets, compatible GPU runtimes, and reconstructed run locations; see [execution](../execution/README.md) and [secondary analyses](../secondary/README.md). No end-to-end GPU rerun is claimed. Post hoc comparisons, representation access, and exploratory ProcessBench strata retain their separate scopes.
