# GPU benchmark batches 11–15 (archived)

Plans, run scripts and evidence from the batch 11–15 GPU/inference benchmark campaigns
(H100, L4, CoreML/ANE, TensorRT, TFLite, vLLM speculative decoding, KV-cache KPIs).
Kept for reference; nothing here is imported by the application.

| Folder | Contents |
| --- | --- |
| `docs/` | Batch plans and play-by-plays (`BATCH13`–`BATCH15`) |
| `scripts/` | The runners used for batches 11–14 |
| `evidence/` | JSON results written by those runs |

**New batch outputs go in `evidence/`, not the repo root.** Root-level `*.json`, `logs/`,
`models/`, `backups/` and `shared/` are gitignored so generated artifacts stop reappearing in
`git status`. Large local artifacts (model weights, pre-run backups, logs) live outside the
repository, in `~/Projects/xcelsior-local/`.
