# Boundary features for CPU probe refitting

This bundle contains the captured boundary vectors needed for the eight-model
canonical interface-access panel: 580 rows / 290 pairs / 235 source components
per model, 23 model-anchor combinations and six feature families. The eight
compressed NPZ files total **107,292,676 bytes**; the original numerical payload
is 139,306,720 bytes. Each individual NPZ is smaller than 16 MB.

The five captured arrays at each selected layer are `post_moe_residual`,
`router_input`, `router_logits`, `router_topk_ids` and `router_topk_weights`.
Float16 values and int64 expert IDs retain their original stored dtype and bits.
The loader expands them into the original six float32 feature matrices, including
expert-ID membership, expert-ID-aligned weights and the secondary legacy
rank-only weights. It uses the archived feature-construction code and does not
renormalize routing weights. No model weights or original problem/solution text
are included.

## Refit

From the release root, use Python 3.11 and the CPU requirements:

```bash
python -m pip install -r requirements-cpu.txt
python secondary/probe/refit.py --output build/probe-refit
```

This fits 690 classifiers (138 cells × five saved folds), writes new OOF scores
and a summary table, recomputes the original 1,000-draw per-cell intervals, and
compares the new OOF scores and AUROCs with the recorded results. Numerical
libraries use at most two threads. No model forwards or GPU are involved.

For the shorter fit-and-score comparison without repeating interval calculations:

```bash
python secondary/probe/refit.py --fit-only --output build/probe-refit-scores
```

An individual model can be selected with, for example,
`--models gpt_oss_120b`. Every selected anchor and feature family is still fitted.
The output directory must be new or empty. Invalid identity, feature-digest or
convergence checks exit with an error. Completed fits separately report whether
they match saved scores, including every difference and the unchanged strict
comparison tolerances in `validation.json`. Add `--strict-reference-match` to
also return an error when the saved-score comparison differs.

## Row identity and validation

`index.json` describes every NPZ member, its exact dtype, shape and byte digest,
the six reconstructed X digests, and the corresponding saved prediction file.
Each model's `*.rows.jsonl` is in the NPZ row order (sorted `sample_id`). It retains
sample, pair and component identities, clean/corrupt labels, selected boundary
metadata and the five-fold assignments. Identity and label fields are separate
from the NPZ and never enter X. The canonical grouping and original fold manifest
are in `../canonical_six/source_component_folds.json`; this remains a post hoc
source-component analysis rather than a prospective freeze.

During packaging, all 138 reconstructed feature matrices were compared exactly
with the original archive extractor. Every NPZ round trip retained the same
values and dtype. The 18 GPT-OSS 120B feature digests also matched its recorded
execution provenance. `validation.json` records these checks and the actual CPU
refit comparison against saved predictions. All 126 cells of the seven models
other than GLM reproduced the saved scores exactly in the packaging runtime.
GLM's 12 cells completed but do not all reproduce its saved predictions exactly;
the largest AUROC difference was 0.0003092 and the largest individual
decision-score difference was 0.5851. The original GLM fit used Linux x86_64 and
this refit used macOS arm64 with matching recorded numerical-package versions.
Platform-dependent numerical behavior is a plausible explanation, not an
experimentally established cause. These differences are retained in the
validation report and the original predictions are unchanged.

For GLM L24, the main comparisons retain their conclusions, but the refit does
not reproduce every number at the paper's displayed precision. At four decimals,
expert-membership AUROC changes from 0.7010 to 0.7009 and its 95% interval from
[0.6644, 0.7397] to [0.6644, 0.7396]. The membership AUROC changes by 0.0000832 because
16 positive-negative comparisons involve exact ties whose ordering changes at
score differences no larger than 1.35e-13. The aligned-weight-minus-membership
contrast's lower interval endpoint changes from 0.02234 to 0.02255, or from
0.022 to 0.023 at three decimals. The paper and original archive results remain
unchanged. The separate, larger score differences in
continuous GLM features have not been uniquely attributed. Full values are in
`glm-refit-differences.json`.

## Coverage boundaries

The bundle covers the main 235-component interface panel for Gemma 4,
Nemotron-3 Nano, Nemotron-3.5 Lightning, Qwen3.6-35B-A3B, GPT-OSS-20B,
Qwen3-30B-A3B, GLM-4.7-Flash and GPT-OSS-120B. GLM has two recorded anchors;
the other seven models have three. Qwen3 here is the 580-row canonical subset
at L0/L24/L47; its historical boundary protocol is unchanged. OLMoE has no
archived probe activations and remains unavailable.

The separate historical Qwen routing/text comparison uses 1,160 rows and is
outside this bundle. Text baselines require the pinned upstream PRMBench input
and their own CPU feature construction. This command also does not refit the
separate 238-component sensitivity. Their saved predictions and statistical
recomputation remain in the repository. This bundle does not claim to reproduce
GPU activation capture or every secondary analysis from scratch.
