# CPU analysis inputs

The numerical release uses compact projections to avoid copying large activation archives or dataset text. The private source-to-public mapping is retained outside this repository.

| Entry point | Included inputs | Verification performed |
|---|---|---|
| `python analysis/abc/recompute_windows.py` | Every original row's pair/component, window, arm, treatment, receiver role, margin and baseline margin; native pair estimands and summary estimates, compressed with gzip | Reconstructs all 161,480 T values; recomputes all 1,170 panels with 50,000 PCG64 draws, seed 20260908 |
| `python analysis/abc/recompute_headline.py` | Native per-pair four-cell D scores and all-layer-output effects/contrasts for all nine checkpoints | Recomputes all 126 headline panels; original seeds (A/C 0, B 20260908), 50,000 source-component draws |
| `python analysis/wide_d/recompute.py` | Nine compact four-cell score projections with complete stored input-identity projections, fixed donor IDs and candidate lengths | Rederives D from cells, reproduces 54 primary/sensitivity endpoints and 10 post hoc direct contrasts, verifies stored cross-model identities and Gemma unequal candidate weighting |

All commands accept `--output` and write to `build/` by default. Repeated CPU computation does not access protected validation data, change selection, or run a model. The original library bootstrap implementation is retained in `analysis/lib/`. Portable adaptations change input/output paths and read compressed projections; they do not change the statistical estimands.

The wide-D input projections store complete-input hashes and target token IDs, not the original full token arrays. The original extraction verified each selected atomic member against accepted archive manifests. This release checks equality of those stored projections and rederives numerical effects; it does not claim to rehash original arrays that are not shipped. Similarly, bundle statistics start from native per-pair effects, not every original activation or forward. Execution sources provide the route to new runs. Regenerating omitted inputs requires upstream data, weights, runtime compatibility and reconstruction of archived input settings; exact byte identity is not established.

Archived JSON `source`, `sha256`, `summary_source`, `archive_sha256_recorded`, and similar provenance fields refer to historical source identities under anonymous `archive://` labels. Only the root public manifest provides hashes for current public files.
