# Paper result map

The map follows the preprint numbering for five figures and nineteen tables. Paths are relative to `paper/`. Run the CPU commands from the release root. The [execution guide](../execution/README.md) covers model-forward replication.

| Paper item | Result | Released source and renderer |
|---|---|---|
| Figure 1 | Intervention schematic | `../execution/` |
| Figure 2 | Interface, token coverage, and direct gap contrasts | `../analysis/abc/unified_windows_v1.json` |
| Figure 3 | Full-step all-router fixed-receiver donor correctness | `../analysis/wide_d/wide_d_nine_model_v1.json`, `../analysis/wide_d/secondary_contrasts.json`, `figure_source/inputs/abc/c/cross_checkpoint_contrasts_v1.json` |
| Figure 4 | Nine checkpoints, four windows and four interfaces | `../analysis/abc/unified_windows_v1.json` |
| Figure 5 | Exploratory ProcessBench receiver-state strata | `../secondary/processbench/analysis/pair_effects.csv`, `../secondary/processbench/analysis/receiver_state_effects.csv` |
| Table 1 | Last-token local fixed-receiver D | `../analysis/abc/headline_d_bundle_v1.csv`, `../analysis/abc/headline_inputs/` |
| Table 2 | Layer and intervention coverage | `../execution/`, `../secondary/probe/configs/` |
| Table 3 | Eight-token window eligibility | `../analysis/abc/unified_windows_v1.json`, `../analysis/abc/inputs/` |
| Table 4 | All three local interfaces and receiver roles | `../analysis/abc/headline_d_bundle_v1.csv` |
| Table 5 | Full-step all-router controls and paired differences | `../analysis/abc/unified_windows_v1.json` |
| Table 6 | All-layer last-token MoE-output controls | `../analysis/abc/headline_d_bundle_v1.csv`, `../analysis/abc/headline_inputs/` |
| Table 7 | Unpatched assay performance | `verification_metrics/baseline/baseline_panels_v1.csv`, `verification_metrics/extracted/windows/` |
| Table 8 | Verdict transitions and controls | `verification_metrics/flips/flip_panels_v1.csv`, `verification_metrics/extracted/windows/` |
| Table 9 | Normalized transfer and baseline gap | `verification_metrics/normalized_t/normalized_t_panels_v1.csv`, `verification_metrics/extracted/windows/` |
| Table 10 | Primary wide D | `../analysis/wide_d/wide_d_nine_model_v1.json`, `../analysis/wide_d/` |
| Table 11 | Total-logprob sensitivity | `../analysis/wide_d/wide_d_nine_model_v1.json` |
| Table 12 | Post hoc direct wide D contrasts | `../analysis/wide_d/secondary_contrasts.json` |
| Table 13 | Representation-access AUROC | `../secondary/probe/canonical_six/`, `../secondary/probe/configs/` |
| Table 14 | Text baseline and router-minus-text | `../secondary/probe/text_baseline/` |
| Table 15 | Natural-error transfer | `../secondary/processbench/runs/`, `../secondary/processbench/analysis/effect_distribution.csv` |
| Table 16 | Exploratory receiver-state effects | `../secondary/processbench/analysis/receiver_state_effects.csv`, `../secondary/processbench/analysis/receiver_state_contrasts.csv` |
| Table 17 | Same-label placebo at the last-token residual interface | [Table source](tables/appendix_l_placebo.tex) · [CPU reproduction](appendix_l/placebo_surface/README.md) |
| Table 18 | Residual D and the TF-IDF donor gap | [Table source](tables/appendix_l_surface_gap.tex) · [CPU reproduction](appendix_l/placebo_surface/README.md) |
| Table 19 | Routing changes and perturbation-size comparisons | [Table source](tables/appendix_l_routing.tex) · [CPU reproduction](appendix_l/routing/README.md) |

[The machine-readable map](result_map.json) lists inputs and commands for each item. Figures 1–5 and Tables 1–16 use display assets from the earlier reproduction package. The Table 17–19 TeX files match the preprint source.

Statistical scope: mean-logprob is primary; total-logprob is a separate sensitivity. M = log P(INCORRECT) − log P(CORRECT), using candidate-wise mean scores in the primary analysis. T summarizes paired donor-aligned changes. D compares erroneous and correct external donors on the same receiver; roles average within pair. Neither T−D nor D/T is a mediation decomposition. Bootstrap intervals are pointwise and do not correct for multiplicity. Donor assignments and checkpoints remain fixed.

Appendix G/H supplementary textual results and remaining activation/native-group sensitivity limits are documented in `../secondary/README.md`; adapter and source snapshot provenance is in `../execution/README.md`.

Appendix B.1 reports the category-exclusion sensitivity of the token-window contrasts. Appendix L.1 separately tests the sensitivity of the same-label placebo to donor length, total log probability, category exclusion and within-run centering. Appendix L.3 reports the ranges of routing changes and KL ratios computed from the Table 19 measurements. [B.1 reproduction](category_exclusion/README.md) · [L.1 reproduction](appendix_l/placebo_surface/README.md) · [L.3 reproduction](appendix_l/routing/README.md)
