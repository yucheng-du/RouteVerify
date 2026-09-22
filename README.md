# RouteVerify — anonymous reproduction materials

Code and non-weight artifacts for the nine-checkpoint intervention study, with separate representation-probe and ProcessBench analyses, supplied under the [review-use permission](LICENSES.md). These materials include no original project history or model weights.

## Start here: CPU reproduction

Python 3.11 is the common supported version. Create an environment and install the CPU dependencies:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python scripts/verify_release.py
```

The default check supports a normal Git clone: it ignores only the top-level
`.git` metadata. Before distributing a standalone release directory, use
`python scripts/verify_release.py --release-package`; this mode rejects all Git
metadata. Both modes reject nested `.git` entries and nonempty stored
`rendered_prompt_text` fields. Run the regression checks with
`python -B -m unittest discover -s tests -v`.

Recompute from the included archived numerical observations, without loading any model:

```sh
OPENBLAS_NUM_THREADS=1 python analysis/abc/recompute_windows.py
OPENBLAS_NUM_THREADS=1 python analysis/abc/recompute_headline.py
OPENBLAS_NUM_THREADS=1 python analysis/wide_d/recompute.py
python secondary/recompute.py
python paper/verification_metrics/reproduce.py --output build/verification_metrics
```

The first three commands retain the reported 50,000 source-component bootstrap draws and write to `build/`. They cover all 1,170 window panels / 161,480 pair-level T values, 126 local-D / all-layer-output panels, and the nine-model full-step all-router D analysis. See [secondary analyses](secondary/README.md), [paper assets and result map](paper/README.md), and [analysis details](analysis/README.md) for their executable commands and exact scope.

Refit the eight-model canonical interface probes from the included compact boundary features, without model weights, dataset text or GPU execution:

```sh
python secondary/probe/refit.py --output build/probe-refit
```

Use a new or empty output directory. This command fits 138 cells with five folds each and recomputes their 1,000-draw intervals. Add `--fit-only` to fit and compare OOF scores/AUROCs without repeating intervals. The compressed feature arrays occupy about 107 MB and preserve the original capture precision. The separate 238-group sensitivity, historical Qwen3 comparison and text baselines retain their saved-score reproduction paths; they are not refitted by this command. Packaging checks found exact OOF matches for seven models and small AUROC differences for GLM, which are reported separately from execution success; see the [refit validation and limits](secondary/README.md#cpu-probe-fitting-from-bundled-features). Use `--strict-reference-match` when a nonzero exit is required for those differences.

## Full experiments require separate GPU execution

[Execution sources](execution/README.md) preserves the actual campaign snapshots and model revisions. The A checkpoints use different overlays; B, C and D retain their own sources. Download model weights and datasets separately from the recorded upstream revisions, reconstruct the frozen input manifests, then use the corresponding campaign environment and command. GPU dependencies are in those snapshots, not `requirements-cpu.txt`.

This candidate has been validated by CPU reanalysis and lightweight source/input checks. **A complete GPU rerun has not been performed for this release.** Archived scores reproduce the reported analysis; this does not establish bitwise determinism of new GPU forwards.

## Interpretation and provenance

- Margin is mean log P(INCORRECT) minus mean log P(CORRECT). Positive is more INCORRECT-favoring. Total log-probability is separately labelled sensitivity analysis.
- T averages the two sign-aligned receiver changes; D switches correct/error roles of one fixed external donor while holding the receiver fixed, then averages receiver roles within pair. D/T and T−D are not causal decompositions.
- The intervention study uses 290 ordered pairs / 235 source components. Missing windows are excluded on the actual eligible-pair intersection. Bootstrap draws sample source components and retain pair weighting; donor assignments are fixed. Intervals are pointwise, without multiplicity adjustment or population-of-models inference.
- Full-step all-router post hoc contrasts, representation probes, and the 64-pair ProcessBench analysis remain separate. Consult the result map for layer/token coverage, sample counts and analysis versions.

[MANIFEST.json](MANIFEST.json) identifies **these public copies**. [Original artifact identities](docs/original-artifact-identities.json) preserve the three hashes quoted in the paper separately. Paths and selected metadata have been anonymized; public hashes must not be substituted for the original archive hashes.

[Release scope and remaining decisions](docs/release-scope.md) records omissions and limitations. [Licensing](LICENSES.md) distinguishes project sources from upstream datasets, model code and model weights. No model weights, raw dataset problem text, private audit/chat records, credentials or private repository addresses are included.
