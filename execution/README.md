# Executed intervention code

These snapshots implement the nine-checkpoint interventions. They come from completed campaign archives, with per-model overlays for campaign A. They are independent Python packages: use the source tree for the model and experiment below. The original development checkout is not substituted for the executed code.

| Source tree | Model aliases | Experiments |
| --- | --- | --- |
| `campaign_A/qwen36_moe` | `qwen36_moe` | Local fixed-receiver D, all-layer MoE-output bundle, token windows |
| `campaign_A/qwen3_moe` | `qwen3_moe` | Same three experiment families |
| `campaign_A/glm4_moe_lite` | `glm4_moe_lite` | Same families; original common-core entry also retained |
| `campaign_B` | `nemotron35_lightning`, `nemotron3_nano`, `gemma4_moe`, `olmoe` | Same three experiment families |
| `campaign_C` | `gpt_oss_20b`, `gpt_oss_120b` | Same three experiment families |
| `campaign_D` | All nine aliases | Full-step all-router fixed-receiver D |

`MODEL_REVISIONS.md` lists exact model/tokenizer revisions and coverage. Configs retain architecture-specific intervention boundaries, native routing, layer indices, candidate scoring, control definitions and seeds. The source registry also contains unused adapters inherited from the execution snapshot; these are not additional evaluated checkpoints.

`source-verification.json` records final-window source identities for all nine checkpoints. `campaign_D/execution-source-verification.json` records completed full-step D runner and adapter identities, including both retained Gemma invocations. These are **original archived identities**, checked before public path sanitization; public-file SHA-256 values are in the release-level inventory.

## CPU analysis and validation

The release-level `analysis/` directories recompute reported statistics from archived outputs. They do not need these GPU environments. A/B/C's native `window_interventions.py` also writes a preliminary pair bootstrap: use the release-level common synthesis for the paper's final 50,000-draw **source-component** intervals and direct contrasts. Never substitute native preliminary intervals for final paper estimates.

Packaging validation performed no checkpoint forwards and downloaded no weights. It checked Python syntax, all nine donor dry-run configurations, all frozen content hashes through the data loaders, 290 discovery pairs/235 source components, and manifest self-hashes. The fixed upstream PRMBench JSONL was downloaded and rebuilt into the exact archived 580-row discovery file; its SHA-256 is `dd8ae3ae2fc6669748e3d9ab7dd12775c2143e68ea8de82b39a922f5b822964a`. See `validation.json`. GPU execution and a fresh complete end-to-end rerun have not been validated for this public candidate.

## Prepare a GPU environment and the fixed inputs

Use Python 3.11 and Linux/CUDA. Each tree contains `pyproject.toml` and its archived `uv.lock`; the recorded versions include Transformers 5.12.1 and kernels 0.12.3. D also includes separate recorded core/FLA environment freezes. These historical Linux GPU environments have not been reinstalled by this packaging task. Hardware-compatible Torch/CUDA, Triton and FLA kernels are required; GPT-OSS requires the pinned native MXFP4 kernel revision in its model YAML. Model weights and third-party runtime packages retain their upstream terms.

The following example uses OLMoE. From the release root:

```bash
cd execution/campaign_B
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
export PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export MODEL=olmoe
mkdir -p inputs/raw
curl --fail --location \
  https://huggingface.co/datasets/hitsmy/PRMBench_Preview/resolve/59abf58d51100c6befccf45fe8114d995fcc3b8e/prmbench_preview.jsonl \
  --output inputs/raw/prmbench_preview.jsonl
python scripts/prepare_window_discovery_20260907.py \
  --source inputs/raw/prmbench_preview.jsonl \
  --output inputs/discovery.jsonl \
  --selected-manifest docs/results/gpt_oss_full_20260809/full_selection_manifest.json
```

The preparation program rejects a raw SHA mismatch and validates every discovery sample against the frozen selection. It produces exactly 290 pairs/580 examples; it does not select a new split. The declared dataset revision is `5cc7683d0ae5797f84d7aeac0607966f277c39e1`; the archived resolved data revision is `59abf58d51100c6befccf45fe8114d995fcc3b8e`. Required raw SHA-256: `3efe79ff688751369da6161635c247318611032071ee46cde6a20a95125bc966`. Dataset text is acquired separately; `inputs/discovery-identities.jsonl` retains IDs, content hashes, grouping, role and label fields. Both fixed dataset cards declare Apache-2.0; exact evidence and links are in `inputs/dataset-licenses.json`.

Each model YAML pins model and tokenizer revisions. The model loaders fetch those revisions through the Hugging Face client when missing. Obtain gated-model access under the upstream terms before running. To stage tokenizer files without weights, run within the chosen source tree:

```bash
python - <<'PY'
import os, yaml
from huggingface_hub import snapshot_download
c = yaml.safe_load(open(f"configs/models/{os.environ['MODEL']}.yaml"))
snapshot_download(c['model_id'], revision=c['tokenizer_revision'],
                  allow_patterns=['*.json', '*.jinja', 'merges.txt', 'vocab.json'])
PY
```

The commands below load full model weights and require a suitable GPU. Use fresh output directories; a new public rerun has its own configuration/source hashes and must generate its own qualifying smoke report.

## Local D and all-layer output bundle (GPU)

Run in the corresponding A/B/C tree, with `MODEL` set to that tree's alias:

```bash
python -m routeverify.donor_contrast \
  --config "configs/experiments/donor_bundle_20260907/$MODEL.yaml" --dry-run
python -m routeverify.donor_contrast \
  --config "configs/experiments/donor_bundle_20260907/$MODEL.yaml" \
  --pairs-jsonl inputs/discovery.jsonl --mode both --limit 2 \
  --output-dir "runs/$MODEL/donor_smoke" --n-resamples 50000
python -m routeverify.donor_contrast \
  --config "configs/experiments/donor_bundle_20260907/$MODEL.yaml" \
  --pairs-jsonl inputs/discovery.jsonl --mode both \
  --output-dir "runs/$MODEL/donor_full" --n-resamples 50000
```

Inspect the saved identity and native-dispatch diagnostics before starting a full run. The archived `records/<alias>/matched_donor_manifest.json` supplies the published fixed external donor assignment. The runner deterministically reconstructs that assignment from the frozen discovery pool and seed; compare the edges in a new run with this record. The two donor roles share the same receiver in each D contrast. A smoke result is a technical check, not paper evidence.

## Token-window interventions (GPU)

```bash
python -m routeverify.window_interventions \
  --config "configs/windows/${MODEL}_core_20260907.yaml" \
  --pairs inputs/discovery.jsonl --mode smoke --output "runs/$MODEL/window_smoke"
python -m routeverify.window_interventions \
  --config "configs/windows/${MODEL}_core_20260907.yaml" \
  --pairs inputs/discovery.jsonl --mode full --output "runs/$MODEL/window_full" \
  --smoke-report "runs/$MODEL/window_smoke/execution_summary.json"
```

Both commands include the recorded last-token, suffix, divergence and full-step windows, with original eligibility masks and controls. Use the same exact model/tokenizer/config/data for smoke and full. Optional `--tokenizer-path` accepts a local snapshot of the pinned tokenizer. The original campaign A orchestration helpers additionally expect their historical asset-manifest layout; the module commands above are the portable entry points.

## Full-step all-router D (GPU)

Use `campaign_D`, recreate its environment/input file as above, and select any alias from the table:

```bash
python -m routeverify.window_donor_contrast \
  --config "configs/experiments/window_donor_role_20260908/$MODEL.yaml" \
  --pairs inputs/discovery.jsonl --mode smoke \
  --execution-version v1 --output "runs/$MODEL/wide_d_smoke"
python -m routeverify.window_donor_contrast \
  --config "configs/experiments/window_donor_role_20260908/$MODEL.yaml" \
  --pairs inputs/discovery.jsonl --mode full \
  --execution-version v1 --output "runs/$MODEL/wide_d_full" \
  --smoke-report "runs/$MODEL/wide_d_smoke/execution_summary.json"
```

The fixed `matching.json` has 290 receivers and 204 distinct external donor pairs. All receiver-step tokens at all routed sites are covered, with four donor-role cells per receiver pair. The native analysis uses 50,000 source-component draws. The separately labeled post hoc secondary contrasts remain in the release-level wide-D analysis; they are not a new preregistered experiment.

Across these experiments, the margin is mean log probability(INCORRECT) minus mean log probability(CORRECT). Positive fixed-receiver D favors INCORRECT more under an erroneous donor. T and D have different estimands and are not a mediation decomposition. The original eligibility, paired masks, component bootstrap units and pointwise interval interpretation must be retained when adapting commands.
