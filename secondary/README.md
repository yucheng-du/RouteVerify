# Representation probes and ProcessBench

These analyses are separate from the nine-checkpoint intervention campaigns. All supplied predictions are from completed runs, not smoke runs. No model weights or original problem/solution text are included. In the public GPT-OSS behavior records, `input_provenance.rendered_prompt_text` has been removed; recorded input hashes, sample identities and candidate scores remain unchanged.

## CPU reproduction from saved scores

From the release root, in an environment with NumPy:

```bash
python secondary/recompute.py --output build/secondary
```

The script recomputes 246 interface-probe cells (including the 238-group sensitivity), their recorded direct/adjacent contrasts, the matched text baseline and router-minus-text differences, and the separate historical Qwen3 routing/text contrast. It uses saved OOF labels/groups and checks row/component counts, within-group fold consistency, and score statistics. For ProcessBench it starts from all three models' atomic candidate scores, verifies behavior and intervention coverage, checks identities and native routing dispatch, and recomputes the 50,000-draw directional, receiver-state and source-component analyses. The output records validation results and computed values. It performs no model forwards or classifier fits.

Probe intervals condition on fitted OOF predictions. The six-model panel uses a **post hoc** 290-pair, 580-row, 235-source-component mapping and five folds. Positive class means the error-bearing step. The bootstrap has 1,000 draws, seed 20260829. GLM and GPT-OSS 120B retain separate analysis records. Qwen3 L24 is distinct from its L36 intervention anchor; OLMoE has no archived probe activations and remains unavailable. The separate historical Qwen3 matched surface comparison uses 580 pairs, 384 components, 2,000 draws, seed 20260808; it is not the six-model text-baseline panel.

ProcessBench uses 64 same-generator, cross-problem pairs, 128 unique traces and 61 shared-problem components. All boundary indices and selection IDs are retained. The margin is **mean log P(INCORRECT) − mean log P(CORRECT)**. `clean_to_corrupt` inserts the matched-correct donor into a first-error receiver; negative changes are expected. `corrupt_to_clean` reverses that direction; positive changes are expected. The ordinary pair bootstrap and the 61-component sensitivity remain separate. Receiver-state grouping is exploratory and uses the baseline margin that also enters the effect, so it is subject to baseline conditioning.

| Paper evidence | Saved input and implementation |
|---|---|
| Appendix G interface-access table | `probe/canonical_six/`, `probe/configs/`, `probe/source/` |
| Appendix G GLM L24 and GPT-OSS 120B L18 | `probe/glm/`, `probe/gpt120/` |
| Appendix G text-baseline table | `probe/text_baseline/`; original fit implementation `probe/source/scripts/text_baseline.py` |
| Historical Qwen3 matched surface comparison | `probe/qwen3_historical/` |
| Appendix H ProcessBench transfer table | `processbench/runs/*/transfer/predictions.jsonl`; `processbench/analysis/effect_distribution.csv` |
| Appendix H receiver-state figure/table | `processbench/analysis/pair_effects.csv`, `receiver_state_effects.csv`, `receiver_state_contrasts.csv` |
| Appendix H construction and component sensitivity | `processbench/selection.json`, `analysis/construction_pairs.csv`, `analysis/problem_component_sensitivity.csv` |

## Regenerate ProcessBench inputs, then optionally run GPU forwards

The original three-model runtime implementation is in `processbench/source/`, with its dependency lock and fixed model/tokenizer revisions. Obtain only `gsm8k.json` from [Qwen/ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench/tree/3bdcd5371ed567559a78f559c01c13a6deee7604), revision `3bdcd5371ed567559a78f559c01c13a6deee7604`. Its expected SHA-256 is `fb1c59dfd1e83e1c5a48dbe598db27e4bc0b8bc524a09445f2d4b69178f61349`. Follow the upstream dataset license. The text-free selection manifest records all trace IDs and content hashes.

```bash
cd secondary/processbench/source
uv sync --frozen --all-extras
uv run --frozen python scripts/prepare_processbench_transfer.py --source-root /path/to/processbench-snapshot
uv run --frozen python scripts/run_processbench_natural_transfer.py --dry-run
```

The preparation command is CPU only. It regenerates the frozen 64 pairs; the original processed-data SHA-256 is `b1d7f21aff6d5a17d1494b5e8bcc2a4bf43d0e2f7d6689921ccb368d4e735c94`. This regeneration and the three-model dry run were executed successfully during packaging, without downloading weights or accessing CUDA.

A full model rerun needs supported GPU hardware, separately downloaded pinned model weights and kernels, and the model's upstream access terms. For example, after setup:

```bash
ALLOW_GPU_RUN=1 ALLOW_GPT_OSS_PROCESSBENCH_TRANSFER_FULL=1 \
uv run --frozen --all-extras python scripts/run_processbench_natural_transfer.py \
  --model-profile gpt_oss --run-role full --run-root /path/to/new-output
```

Use `gemma4` with `ALLOW_GEMMA4_PROCESSBENCH_TRANSFER_FULL=1`, or `nemotron3_nano` with `ALLOW_NEMOTRON3_NANO_PROCESSBENCH_TRANSFER_FULL=1`, for the other profiles. These commands run model forwards; they were **not** executed during release preparation. The original runtime's authorization environment switches are local execution guards, not evidence that GPU reproduction has been completed. The runtime's stored transfer metrics use its original smaller bootstrap; the paper's 50,000-draw results are the separate `analysis/` tables and the CPU command above.

## CPU probe fitting from bundled features

`probe/features/` contains the selected native boundary vectors for all eight models in the canonical interface panel: 580 rows per model, 23 model-anchor combinations and six feature families per anchor. The eight compressed NPZ files total 107,292,676 bytes and preserve 139,306,720 bytes of original float16 values/int64 expert IDs without quantization or rescaling. The index records row order, identities, layers, feature shapes and hashes. The public entry point reconstructs the six float32 feature matrices and uses the archived numerical implementation with the saved 235-source-component, five-fold split. StandardScaler is fitted on each training fold only.

From the release root, after installing `requirements-cpu.txt`:

```sh
python secondary/probe/refit.py --output build/probe-refit
```

The output directory must be new or empty. The command performs 690 classifier fits across 138 cells, compares fitted OOF scores and AUROCs with saved predictions, and computes each cell's original 1,000-draw source-component interval. `--fit-only` retains every classifier fit and score/AUROC comparison while omitting interval output. `--models gpt_oss_120b` selects one model for a shorter run. Direct contrasts and the other secondary analyses remain available through `secondary/recompute.py`. No model weights, original dataset text, GPU or safetensors dependency is needed for this NPZ-based refit.

Packaging validation refitted all 138 cells (690 fold fits): the 126 cells from seven models reproduce their saved OOF scores exactly. GLM has declared differences on this macOS arm64 host relative to the saved Linux x86_64 run, despite matching row/fold identities, recorded estimator settings and features reconstructed from the archive. Its largest AUROC difference is 0.0003092; individual score differences can be larger (maximum 0.5851). GLM L24's main comparisons retain their conclusions, but the refit does not reproduce every number at the paper's displayed precision: expert-membership AUROC changes from 0.7010 to 0.7009 and its 95% interval from [0.6644, 0.7397] to [0.6644, 0.7396] at four decimals. The aligned-weight-minus-membership interval's lower endpoint changes from 0.022 to 0.023 at three decimals. The exact cause has not been isolated; the paper and original archive results remain unchanged. See `probe/features/validation.json` and `probe/features/glm-refit-differences.json`. The command reports execution completion separately from reference matching; use `--strict-reference-match` to return a nonzero exit code when the original comparison tolerances fail.

The all-model packaging run used `--fit-only`; the new command's default 1,000-draw branch was additionally executed for GPT-OSS 120B in a clean environment installed only from `requirements-cpu.txt`. Its 18 AUROCs, mean-fold AUROCs and confidence intervals exactly match the saved table (`../validation/probe-default-command.json`).

`probe/source/` retains the original analysis implementations and `probe/recapture_metadata/` records capture boundaries, model topology and input hashes. Original recapture configurations refer to archive-relative locations; they are provenance records for constructing a new activation archive. The portable refit above reads the included feature bundle instead. The boundary text baseline still requires the pinned upstream PRMBench data; `probe/text_baseline/input_row_manifest.csv` records input text hashes, rows and folds.

Twenty-one of 22 recorded analysis source hashes match the selected pre-anonymization sources. The GPT-OSS 120B orchestration script currently available differs from its recorded execution hash; its seven shared numerical modules match. The discrepancy is explicit in `probe/source_hash_checks.json`; the candidate wrapper is not represented as a recovered byte-identical execution snapshot. Original archive hashes identify pre-release sources; the release inventory identifies the packaged files after redaction.

The retained GPT-OSS 120B wrapper has also been executed on the original stored activations and fixed folds: all 18 feature-matrix hashes, 10,440 freshly fitted OOF scores, 18 cell statistics/intervals and 30 direct contrasts/intervals exactly match the original records. See `../validation/gpt120-wrapper-refit.json`. This verifies the retained wrapper's behavior on those inputs; the missing original source bytes and the reason for their difference remain unknown.

The narrower 238-group recapture-native sensitivity is preserved separately in `probe/native238/` and is also recomputed from saved scores. Its saved result files were recovered from the historical source revision, without including version history in this release. Every middle-anchor table cell changes by at most 0.031 relative to the 235-component canonical table. The portable refit does not cover that sensitivity, the separate historical 1,160-row Qwen comparison, or text baselines. Full activation archives are omitted beyond the selected vectors above. No end-to-end GPU reproduction is claimed.
