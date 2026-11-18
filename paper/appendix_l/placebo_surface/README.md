# Same-label placebo and donor surface gap

Run from the repository root with Python 3.11 and the packages in `requirements-cpu.txt`.

```sh
OPENBLAS_NUM_THREADS=1 python paper/appendix_l/placebo_surface/reproduce.py --output build/appendix_l/placebo_surface.csv
```

The command reconstructs all estimates and 95% intervals in Tables 17 and 18 from numerical observations already included in this repository. It checks all 120 values against the full-precision reference in `reference.csv` with an absolute tolerance of 1e-12. Omit `--output` to print the CSV.

## Inputs

- `analysis/abc/headline_inputs/<model>/D.json.gz` supplies the receiver and donor identities, source components and four fixed-receiver margins for each pair.
- `paper/verification_metrics/extracted/windows/<model>.csv.gz` supplies the alternate external donor and patched margin from the matched last-token residual intervention.
- `secondary/probe/text_baseline/oof_predictions.jsonl` supplies the out-of-fold TF-IDF error score. The five canonical copies share the same scores and boundary steps. The script uses the Gemma4 copy and checks its boundary steps against the eight models' fixed-receiver records.

The model forwards and the TF-IDF predictions are frozen inputs. This script starts at the saved scores. It needs no GPU, model weights, dataset text, or network access.

## Estimands

For Table 17, the fixed-receiver contrast subtracts the margin under a correct donor from the margin under its erroneous counterpart. The placebo compares two different external donors with the same label and subtracts the margin under the lower-scoring donor from the margin under the higher-scoring donor. Both estimands average the two receiver roles within each receiver pair. The three columns use the same 195 eligible pairs in 166 source components. The third column takes the difference within pairs before constructing the interval.

The three pass conditions require a positive lower bound for real D, a placebo interval that contains zero or whose upper bound is below one quarter of real D's point estimate, and a positive lower bound for the paired difference. Six models pass. The two exceptions are Qwen3.6 and OLMoE.

For Table 18, the donor gap is the erroneous donor's TF-IDF score minus the correct donor's score. The conditional mean uses the 80 receiver pairs in 78 source components with a nonpositive gap. The zero-gap intercept comes from an ordinary least-squares fit to all 290 pairs in 235 components. Each fitted intercept uses the pair multiplicities from its own bootstrap draw.

All intervals use 50,000 PCG64 resamples of sorted receiver source components, seed 0, and the 2.5th and 97.5th percentiles. Qwen3 is omitted from Table 17 because its two runs targeted different layers and from Table 18 because its archived TF-IDF boundary steps were misaligned. The primary CSV covers the two tables. Run the companion command below to reproduce the sensitivity analyses.

## Same-label placebo sensitivity checks

```sh
OPENBLAS_NUM_THREADS=1 python paper/appendix_l/placebo_surface/reproduce_sensitivities.py --output-dir build/appendix_l
```

`placebo_sensitivities.csv` reports the real contrast, oriented placebo, their paired difference, 95% intervals, pair and component counts, bootstrap seed and pass decision for each model. Each sensitivity applies Table 17's three pass conditions. The command checks all estimates, interval endpoints, sample counts and decisions against the frozen references at an absolute numerical tolerance of 1e-12.

| Change from Table 17 | Pairs | Components | Seed | Models passing |
| --- | ---: | ---: | ---: | ---: |
| Orient by boundary-step character count | 194 | 165 | 0 | 6/8 |
| Orient by boundary-step word count | 183 | 160 | 0 | 5/8 |
| Use total-log-probability margins, with TF-IDF orientation | 195 | 166 | 0 | 5/8 |
| Exclude circular and redundancy pairs, with TF-IDF orientation | 194 | 166 | 20260908 | 6/8 |
| Center patched margins within each run for Nano and Lightning, with TF-IDF orientation | 195 | 166 | 0 | 2/2 |

Count orientations subtract the shorter donor's margin from the longer donor's and drop a receiver pair when either role ties. Category exclusion removes six circular and three redundancy receiver pairs from the full 290 before applying placebo eligibility. In the centered analysis, each patched margin is differenced from the unpatched margin of the same receiver in the same run. `identical_donor_drift.csv` also measures the raw and centered cross-run differences on the 190 receiver traces (95 pairs) assigned the same donor in both runs.

Two small inputs supplement the public margins and out-of-fold scores. `boundary_counts.csv` records the evaluated boundary-step index, whitespace-separated word count and Unicode character count for the 410 donor traces used in the count comparisons. The counts were calculated on the frozen boundary steps with Python's `len(step.split())` and `len(step)` after matching the archived sample identity and content hash. `fixed_run_baselines.csv` records the unpatched mean margin for both receiver roles in all 290 fixed-run pairs for Nano and Lightning. These values come from the two corresponding fixed-receiver runs. The matched-window baseline values and both kinds of patched margins come from the existing public run records. These projections contain numerical observations and identifiers only.
