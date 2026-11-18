# Routing manipulation and donor perturbation size (Appendix L, Table 19)

Run from the repository root with Python 3.11 or later. The calculation uses the standard library and the bundled receiver measurements.

```sh
python paper/appendix_l/routing/reproduce.py --output build/appendix_l/routing
```

`table19.csv` reports the published values for all nine models. The script checks all 45 numbers, including both endpoints of each range, against `table19_reference.csv` at the paper's three-decimal precision. `condition_means.csv` contains the 180 underlying means of each metric and their receiver counts. Omit `--output` to print the table.

## Measurements and coverage

Each row of [observations.csv.gz](observations.csv.gz) records a single receiver trace under one intervention and donor condition. The key is the model, run, window, arm, condition, pair ID and receiver role. Data comprise 80,668 window-run observations and 10,440 observations from each of the single-layer and all-router fixed-receiver runs. Pair IDs refer to the same frozen PRMBench selection used elsewhere in the release.

At each patched router site and scoring forward, `topk_changed` measures the fraction of patched positions whose selected expert set changes. Changes in expert weights alone do not change this statistic. `router_kl` measures KL(receiver ‖ replacement) in nats after softmax over all expert logits. The recorded diagnostic averages over patched positions. The release projection then averages the diagnostic over both candidate-scoring forwards and all patched sites of a receiver trace. Both averages retain equal weights at their respective level. The script takes the arithmetic mean of these projected observations within each condition. The metric definitions are in [the archived implementation](../../../execution/campaign_D/src/routeverify/instrumentation/patching.py).

- The first result column uses the changed-set fraction for `windows / last_token / router_logits / paired`. Each model contributes 580 receiver traces.
- For the range, the script computes mean matched KL divided by mean paired KL separately for each of four windows and two routing arms. It reports the lowest and highest of those eight ratios per model. Each treatment uses its own eligible receiver set (519–580 traces). Windows spanning multiple tokens may include a shared prefix in the paired treatment.
- The final two columns use fixed-receiver runs. `donor_E` and `donor_C` denote erroneous and correct external donors. Their ratio of mean KL is computed at the single-layer last token and at all routed layers across the full step. Each condition includes 580 receiver traces. The corresponding single-layer changed-set means range from 0.447 to 1.000 across models and donor roles, as reported in the text.

The KL values describe within-model perturbations. Native dispatch can use a different gate function, and the receiver at a later all-router site already reflects upstream patches. [Source archive identities](source_archives.csv) record the original byte sizes and SHA-256 hashes of the nine window-row logs, nine fixed-receiver prediction logs and nine full-step archives from which the diagnostic records were extracted. This CPU reproduction begins with the saved per-receiver numerical measurements.
