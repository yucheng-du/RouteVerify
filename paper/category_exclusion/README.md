# PRMBench category exclusion

This analysis reproduces the appendix paragraph “Category-exclusion sensitivity.” It removes six discovery pairs labeled `circular` and three labeled `redundency` (the dataset spelling). The full 290-pair discovery sample provides the primary estimates.

From the release root, run:

```sh
python -m pip install -r paper/category_exclusion/requirements.txt
python paper/category_exclusion/recompute_category_exclusion.py
```

The first command can be skipped in an environment prepared with the release's `requirements-cpu.txt`. The script uses three local inputs and checks their SHA256 hashes against `input-manifest.json`. It writes the complete estimates and a verification report to `build/category_exclusion/`. The reference estimates in this directory were computed with Python 3.14.6 and NumPy 2.4.4. The separate `verification.json` records the comparison using the release environment.

## Contrasts and uncertainty

For each contrast, intersect the eligible receiver pairs across conditions, then remove pairs from the two excluded categories. The predictions and donor assignments come from the original runs. Full-step and last-token contrasts retain 281 pairs in 230 source components. The eight-token windows have their own eligible sets.

- Compare single-layer residual transfer with router-logit transfer and with MoE-output transfer in each of four windows, for nine checkpoints (72 contrasts).
- Compare the residual-minus-all-router gap at divergence-8 with the gap at suffix-8 for each checkpoint (nine contrasts).
- Compare full-step paired all-router transfer with its norm-random control for each checkpoint (nine contrasts). Repeat the OLMoE contrast on the full discovery set and check it against the primary estimate in `analysis/abc/unified_windows_v1.csv`.

The bootstrap samples eligible receiver source components with replacement. Each draw divides the sum of pair effects by the number of sampled pairs. The script uses 50,000 PCG64 draws with seed 20260908 and reports marginal 95% percentile intervals. Contrasts with identical eligible pairs share draws. It checks the eligible pairs and components, sample counts, and all 91 means and intervals against the frozen `results.json`, with an absolute numerical tolerance of `1e-12`.

After exclusion, all 72 interface intervals remain positive and all nine position-gap intervals remain negative. For OLMoE, the full-step paired-minus-norm-random effect goes from 0.0466 [0.0003, 0.0938] to 0.0450 [−0.0013, 0.0928]. The analysis measures the sensitivity of these estimates to the category labels. The underlying mathematical correctness of the retained responses was not independently assessed.
