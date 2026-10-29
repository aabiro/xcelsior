<!-- residency-guard: documents-removal — mirrored registry snapshot -->
<!-- harness-delta-stamp: 2026-07-06T06:45:00Z -->
# AI/ML Master Index by Repo

<!-- pxl-audit-closure: 2026-07-06 §4 row18 ProcessFrame/load_model use_int8; Mac CPU verify + Linux GPU gating -->

**Status:** Active
**Created:** 2026-07-04  
**Location:** `~/Projects/pxl-registry/docs/` (Mac `aaryn@100.64.0.3`)

This document sorts every actionable item from all `AI_ML_*` planning docs and companion inventory docs **by repository**, ranked **most impactful → least** within each repo. Checkbox text is copied **word-for-word** where the source used checkboxes; implementation-phase, acceptance, risk, and audit items are normalized into explicit checklist rows without changing the technical intent. Each item cites its source.

**Tower excluded** throughout (see `pixelspark/PXL_ENGINE_AND_TOWER.md`).

---

## Source document registry

| # | File | Purpose |
|---|------|---------|
| S1 | [`AI_ML_UPGRADE_IMPLEMENTATION_CHECKLIST.md`](./AI_ML_UPGRADE_IMPLEMENTATION_CHECKLIST.md) | Original per-repo checkbox checklist (111 lines) |
| S2 | [`AI_ML_UPGRADE_PLAN.md`](./AI_ML_UPGRADE_PLAN.md) | Portfolio audit and gap analysis (~70 KB) |
| S3 | [`AI_ML_TOP7_IMPLEMENTATION_PLAN.md`](./AI_ML_TOP7_IMPLEMENTATION_PLAN.md) | Ranked top-7 checklist execution (tower excluded) |
| S4 | [`AI_ML_MEGA_INITIATIVES_PLAN.md`](./AI_ML_MEGA_INITIATIVES_PLAN.md) | M1 registry, M2 job fabric, M3 inference mesh |
| S5 | [`AI_ML_ULTRA_INITIATIVES_AND_PORTFOLIO_LIFEHACKS_PLAN.md`](./AI_ML_ULTRA_INITIATIVES_AND_PORTFOLIO_LIFEHACKS_PLAN.md) | U1–U3 above M1 + L1–L6 life-hacks |
| S6 | [`AI_ML_FRONTIER_NEXT7_PLAN.md`](./AI_ML_FRONTIER_NEXT7_PLAN.md) | F1–F7 frontier tech (SeedVR2, SAM3, Nunchaku, etc.) |
| S7 | [`AI_ML_MODEL_UPGRADES_AND_DOWNLOADS.md`](./AI_ML_MODEL_UPGRADES_AND_DOWNLOADS.md) | Model swap table + download URLs |
| S8 | [`MISSING_WEIGHTS.md`](./MISSING_WEIGHTS.md) | Stub/tiny model artifacts that need real weights and hydration |
| S9 | [`PIPELINE_ROUTER_INVENTORY.md`](./PIPELINE_ROUTER_INVENTORY.md) | Pipeline/router reconciliation inventory and normalization actions |

### Cross-document impact hierarchy (portfolio-wide)

Higher layers subsume or gate lower layers:

```
U1 Outcome Spine > U2 Agent Plane > U3 Eval Factory
    ↓ feeds
M1 Model Registry > M2 Job Fabric > M3 Inference Mesh
    ↓ absorbs
Top-7 quality gates (#1–#7)
    ↓ extended by
Frontier F1–F7 + Model swaps
    ↓ parallel
Life-hacks L1–L6 (non–top-8 repos)
```

---

## 1. `pxl-registry`

*Central authority for schemas, codegen, quality, prediction, and ultra-layer packages.*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] `outcome_event_v1.schema.json` | S5 U1.0 |
| 2 | - [ ] OTel GenAI tracer wrapper `pxl_outcomes.trace()` | S5 U1.0 |
| 3 | - [ ] JSON Schema \| `model_v1.schema.json` with required fields above | S4 M1.0 |
| 4 | - [ ] Validator \| `pxl-registry validate` — fails CI on schema drift | S4 M1.0 |
| 5 | - [ ] `pxl_eval run --bundle production_serverless --surface all` \| Single entrypoint | S4 M3.0 |
| 6 | - [ ] Zero hand-maintained `MODELS` dicts in any repo | S4 M1 DoD |
| 7 | - [ ] `pxl-registry promote` is the only path to production | S4 M1 DoD |
| 8 | - [ ] `gen_handler_registry.py` \| Emit `generated_models.py` imported by `handler.py` | S4 M1.1 |
| 9 | - [ ] `gen_regression_scope.py` \| Single source for regression_test MODELS | S4 M1.1 |
| 10 | - [ ] `gen_cpp_registry.py` \| Emit `generated_model_registry.inc` included by `model_registry.cpp` | S4 M1.2 |
| 11 | - [ ] `gen_tflite_allowlist.py` \| JSON for pixelspark Gradle asset filter | S4 M1.3 |
| 12 | - [ ] Android TFLite allowlist generated from registry | S4 M1 DoD |
| 13 | - [ ] 10-model pilot \| Migrate top traffic models from `MANIFEST.md` + handler | S4 M1.0 |
| 14 | - [ ] Promotion CLI \| `pxl-registry promote <id> --to staging --eval-run <id>` | S4 M1.0 |
| 15 | - [ ] Create `pxl-registry/quality/golden_manifest.yaml` | S3 Phase 0.1 |
| 16 | - [ ] Shared package `pxl-registry/quality/pxl_metrics/` | S3 Phase 2.1 |
| 17 | - [ ] Single `pxl_metrics` API used by both repos | S3 Phase 2.4 |
| 18 | - [ ] `pxl-registry/schema/job_v1.schema.json` | S4 M2 |
| 19 | - [ ] `pxl-registry/predict/` (or standalone): `forecast(df, horizon, covariates) → quantiles` (Chronos-2), `tabular(train_df, test_df) → proba/quantiles` (TabPFN v2), `conformalize(preds, calib) → intervals` (MAPIE/crepes). CPU-first; 2060 optional. | S6 F6.0 |
| 20 | - [ ] Pick canonical schema in `pxl-registry/schema/creator_event_v1.json` | S5 L2.1 |
| 21 | - [ ] `mcp_aggregate_server` \| Single SSE endpoint; tool namespaces | S5 U2.0 |
| 22 | - [ ] `policy_engine.py` \| YAML rules evaluated before tool invoke | S5 U2.1 |
| 23 | - [ ] Failure → JSON case file in `shelf/datasets/inbox/` within 5 minutes. | S5 U3.0 acceptance |
| 24 | - [ ] **U1:** ≥95% GPU jobs emit outcome events; promotions require outcome rollup | S5 DoD |
| 25 | - [ ] **U2:** Single MCP gateway; ≥40% LLM cost cut on cached surfaces; zero unguarded deploys | S5 DoD |
| 26 | - [ ] **U3:** ≥90% failures become replay cases; repeat incident rate ↓50% | S5 DoD |
| 27 | - [ ] DuckDB or PEL Postgres `outcome_rollups` \| model_id × day × outcome_class counts | S5 U1.1 |
| 28 | - [ ] `routing_policy/outcome_weights.yaml` \| Boost models with better outcome rollups | S5 U1.2 |
| 29 | - [ ] Register in pxl-registry (`tools/import_model_store.py --introspect`) with license + quality-baseline fields per M1, and run the golden-set eval before promotion (M3/TOP7 gates). | S7 closing |
| 30 | - [ ] Add `registry/pipelines/*.yaml` entries for mobile preset families so PixelSpark, iOS, Playground, backend, and AI Data Factory stop maintaining separate routing documents. | S9 normalization |
| 31 | - [ ] Import `serverless_bundle/handler.py` route keys as cloud pipeline or endpoint metadata, preserving aliases and making an explicit logical choice for route-vs-pipeline modeling. | S9 normalization |
| 32 | - [ ] Pick one TFLite routing matrix source and mark the other copies as historical or generated; make the real source obvious. | S9 normalization |
| 33 | - [ ] Decide whether `pxl-engine` is modeled as runtime capability metadata, reusable pipeline metadata, or both, and encode that decision in registry schema/docs. | S9 normalization |
| 34 | - [ ] Add an audit script comparing `pixelspark-ios/ModelRoutingRules.json`, ASUS `pixelspark/TFLITE_ROUTING_MATRIX.md`, ASUS `serverless_bundle/handler.py`, and `pxl-registry/registry/models/*.yaml`. | S9 normalization |
| 35 | - [ ] Add `bundles/production_serverless.yaml` and `bundles/production_android.yaml` so production scope is declarative and codegen targets are bundle-driven. | S4 M1 architecture |
| 36 | - [ ] Add `pxl-registry diff` and `pxl-registry explain` alongside validate/promote so reviewers can see model and route changes before promotion. | S4 M1 architecture |
| 37 | - [ ] Enforce immutable stable model IDs, promotion states, per-surface export maps, and hardware compatibility matrices as required schema fields. | S4 M1 requirements |
| 38 | - [ ] Add pre-commit hook: regenerate codegen on YAML change and fail if generated files are stale. | S4 M1.1 |
| 39 | - [ ] Promotion state machine: `experiment → staging → production → quarantine → retired`, with required tests, eval artifact, and human ack at each boundary. | S4 M1.4 |
| 40 | - [ ] Each promotion writes `promotions/<id>/<timestamp>.json` audit log with actor, state transition, eval run, and reason. | S4 M1.4 |
| 41 | - [ ] Add `pxl-registry/pricing/rate_card.yaml` with CAD per GPU-second by tier for M2 cost estimates. | S4 M2 billing |
| 42 | - [ ] Add `routing/mesh_policy.yaml` with ASUS local and cloud-burst tiers, health rules, queue-depth triggers, and model compatibility triggers. | S4 M3 routing |
| 43 | - [ ] Implement `quality/pxl_eval/` as the orchestrator over all M3 eval surfaces, extending `pxl_metrics` instead of creating another one-off script. | S4 M3 eval matrix |
| 44 | - [ ] Emit `eval_reports/<git_sha>.json` plus markdown PR scorecards for every deploy candidate. | S4 M3.1 |
| 45 | - [ ] Add MLflow experiment `pxl-eval/production` with one run per deploy candidate. | S4 M3.1 |
| 46 | - [ ] Store M1 promotion artifact IDs for M3 eval passes and U3 SHELF sign-off; block production when either is missing. | S4 M1.4 + S5 U3 |
| 47 | - [ ] Add SQL views for refund rate, alert CTR, and trade P&L attribution over outcome events. | S5 U1.1 |
| 48 | - [ ] Add Grafana or PostHog dashboard: "Models by business outcome". | S5 U1.1 |
| 49 | - [ ] Add 5% shadow traffic support for challenger models and log explore/exploit decisions. | S5 U1.2 |
| 50 | - [ ] Router explains why model B was chosen, including outcome-score deltas against model A. | S5 U1.2 acceptance |
| 51 | - [ ] Add `pxl-agent-plane/rings/` docs for Ring 0 read-only, Ring 1 write, and Ring 2 deploy permissions. | S5 U2 architecture |
| 52 | - [ ] Add deterministic policy rules requiring `m3.eval_pass`, Ring 2, and `human_ack_id` before deploy tools run. | S5 U2 policy |
| 53 | - [ ] Add semantic cache with similarity threshold around 0.92, cache-hit/cost-saved metrics, and local sentence-transformer embeddings. | S5 U2.2 |
| 54 | - [ ] Add `shelf/ingest`, `shelf/synthesize`, `shelf/label`, `shelf/judge`, and `shelf/act` modules exactly matching the SHELF architecture. | S5 U3 architecture |
| 55 | - [ ] Add `shelf replay <case_id>` to reproduce synthesized failures deterministically. | S5 U3.1 |
| 56 | - [ ] Add active-learning queue that prioritizes judge confidence 0.4–0.6 and tracks Brier score against human labels. | S5 U3.2 |
| 57 | - [ ] Auto-quarantine a model after 3 failures with same `model_id` in 24h, then notify Slack/Telegram and open a test-insertion PR through U2. | S5 U3.3 |
| 58 | - [ ] Hydrate every stub entry in `MISSING_WEIGHTS.md` with real weights, then run `tools/import_model_store.py --introspect` or the hydration pass to fill sha256 and size. | S8 |
| 59 | - [ ] Add registry records for DDColor, SEA-RAFT, Nomos8kDAT, Parakeet, Kokoro, Chronos-2, Qwen3-8B, FlashVSR, SeedVR2, FLUX-schnell, SANA-1.5, Wan2.2, SAM3, and Gemma 3n as downloads land. | S7 download order |

**Repo impact note:** These changes make `pxl-registry` the executable contract for the whole portfolio: models, routes, pipelines, prices, evals, outcomes, promotions, and generated clients all originate here. The expected result is less drift between Android/iOS/server/serverless/C++ surfaces, safer model promotion, faster rollback, and evidence-based routing decisions tied to quality, cost, and business outcomes instead of manually maintained Markdown and handler dictionaries.

---

## 2. `serverless_bundle`

*Production GPU inference hot path. Path: `/mnt/storage/models/onnx/production/serverless_bundle` (ASUS).*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] `deploy.sh` impossible to run without eval pass | S4 M3 DoD |
| 2 | - [ ] Add a PSNR/SSIM/LPIPS quality regression gate to the existing latency/VRAM benchmark path. | S1 §1 |
| 3 | - [ ] Deploy blocked on regression failure | S3 Phase 1.4 |
| 4 | - [ ] All `MANIFEST.md` production models have baseline entries with PSNR/SSIM/LPIPS | S3 Phase 1.4 |
| 5 | - [ ] Middleware validates `X-PXL-Dispatch-Token` (HMAC, short TTL, ties to job_id) | S4 M2.2 |
| 6 | - [ ] Reject unsigned requests in production compose (dev mode optional bypass) | S4 M2.2 |
| 7 | - [ ] serverless handler rejects unsigned dispatch in prod | S4 M2 DoD |
| 8 | - [ ] Shrink `handler.py` \| Replace inline `MODELS = {` with `from generated_models import MODELS` | S4 M1.1 |
| 9 | - [ ] Emit from serverless `handler.py` on job complete | S5 U1.0 |
| 10 | - [ ] Wire `pyiqa` metrics into CI: LPIPS, DISTS, MUSIQ, MANIQA, and NIQE. | S1 §1 |
| 11 | - [ ] Wire VMAF into CI for video routes. | S1 §1 |
| 12 | - [ ] CI fails on LPIPS regression > 0.01 vs baseline | S3 Phase 2.4 |
| 13 | - [ ] `regression_results.json` includes perceptual metrics | S3 Phase 1.4 |
| 14 | - [ ] Synthetic weight perturbation fails CI on ≥1 model | S3 Phase 1.4 |
| 15 | - [ ] Add an INT8 calibration-quality gate: INT8 only ships per model when it passes against FP16/FP32. | S1 §2 (pxl-engine) + S3 #6 |
| 16 | - [ ] ≥5 high-traffic SR models pass INT8 gate on sm75 | S3 Phase 6.4 |
| 17 | - [ ] Handler falls back to FP16 when INT8 not approved | S3 Phase 6.4 |
| 18 | - [ ] Throughput ≥1.5× on approved models | S3 Phase 6.4 |
| 19 | - [ ] Operationalize INT8/FP8 instead of shipping everything FP16. | S1 §1 |
| 20 | - [ ] New handler route `video_restore_seedvr2` in `serverless_bundle` (`handler.py` + `add_diffusion_generate.py` pattern): chunked frames → tiled one-step restore → temporal-window stitch. | S6 F1.1 |
| 21 | - [ ] Register in pxl-registry as `restore.seedvr2_3b` with `quality_baselines` from F1.0 so M3's eval gate can police it. | S6 F1.1 |
| 22 | - [ ] RealBasicVSR / RVRT / VRT → **FlashVSR v1.1** (video super-resolution) | S7 Part A #1 |
| 23 | - [ ] `serverless_bundle/segmentation/sam3_service.py`: batch endpoint `segment(concepts[], image\|clip) → RLE masks`; cache masks keyed by content-hash + concept. | S6 F2.1 |
| 24 | - [ ] Add `image_edit_kontext` route: prompt+image → Kontext INT4 edit. Port top-3 revenue LoRAs through Nunchaku's no-requant LoRA path; visual-diff against BF16 (SSIM ≥ 0.95 gate on golden prompts). | S6 F3.1 |
| 25 | - [ ] Move ONNX/TRT artifacts out of the Docker image into object storage (Novita-local or R2/S3), manifest-addressed (M1 registry paths if live). | S6 F4.0 |
| 26 | - [ ] Lazy per-route loading in `handler.py`: load-on-first-use + LRU eviction (6 models hot max); measure cold start before/after. Target: image ≤8GB, first-request penalty <10s, steady-state unchanged. | S6 F4.0 |
| 27 | - [ ] SD-Turbo / SDXL-Turbo / LCM / Hyper-SD → **FLUX.1-schnell** + **SANA-1.5** (fast text-to-image) | S7 Part A #3 |
| 28 | - [ ] GFPGAN v1.4 / CodeFormer → **OSDFace** (face restoration) | S7 Part A #4 |
| 29 | - [ ] DeOldify → **DDColor** (colorization) | S7 Part A #6 |
| 30 | - [ ] Add a `torch.compile` or `stable-fast` path for diffusion models. | S1 §1 |
| 31 | - [ ] Standardize on ORT-TRT EP with embedded engine cache. | S1 §1 |
| 32 | - [ ] Add an engine compatibility manifest keyed by GPU arch and TensorRT version. | S1 §1 |
| 33 | - [ ] Archive `handler.py.bak*` \| Move to `archive/` outside git; add `.gitignore` rule | S4 M3.0 |
| 34 | - [ ] Rollback demotes model in <5 minutes globally | S4 M3 DoD |
| 35 | - [ ] **F1** SeedVR2 default for one production video-restore route; loser engines archived in registry. | S6 DoD |
| 36 | - [ ] **F3** INT4 FLUX route serving with SSIM≥0.95 parity gate; measured ≥2.5× throughput. | S6 DoD |
| 37 | - [ ] **F4** Cold start p50 −70%; one preempt→resume demo on xcelsior. | S6 DoD |
| 38 | - [ ] Add `quality_gate.py` with `QualityScore`, `score_vs_reference(output, reference)`, and `check_gates(score, thresholds, baseline=None)`. | S3 Phase 1.1 |
| 39 | - [ ] Support reference-based PSNR/SSIM/LPIPS gates for upscalers, frozen-baseline LPIPS drift caps for face restore, and no-reference NIQE/MUSIQ baseline deltas as warnings. | S3 Phase 1.1 |
| 40 | - [ ] Add CLI flags: `python3 regression_test.py --gpu --quality-gate --category upscale_4x` and `--update-baseline --include-lpips`. | S3 Phase 1.2 |
| 41 | - [ ] Add `.github/workflows/quality-regression.yml` for self-hosted ASUS GPU PR matrix and nightly full model set. | S3 Phase 1.3 |
| 42 | - [ ] GPU smoke test: 1 model and 1 image fail fast before full golden pass. | S3 Phase 0.2 |
| 43 | - [ ] Upload MLflow/artifact report from the quality CI job. | S3 Phase 0.2 |
| 44 | - [ ] Pin quality dependencies with `uv lock` or equivalent; include `pyiqa`, `ffmpeg`, and `libvmaf` on ASUS/serverless. | S3 Phase 0.2-0.3 |
| 45 | - [ ] Add nightly `pxl_metrics.video` VMAF command for slow video routes rather than blocking every PR on full video. | S3 Phase 2.2 |
| 46 | - [ ] Log `job_id` to structured JSON logs from handler middleware. | S4 M2.2 |
| 47 | - [ ] Raw handler URL cannot be called without dispatch token in prod. | S4 M2.2 acceptance |
| 48 | - [ ] Wire `deploy.sh` with `set -e` so `pxl_eval` failure prevents `docker push`. | S4 M3.0 |
| 49 | - [ ] Deploy candidate blocked on LPIPS > 0.01 regression. | S4 M3.1 acceptance |
| 50 | - [ ] Tag immutable image digest `pixelspark-serverless@sha256:...` in each job record. | S4 M3.4 |
| 51 | - [ ] Use sm75-only TRT matrix until tower returns; do not produce sm86 builds as production targets while tower is excluded. | S4 M3.4 |
| 52 | - [ ] Add a dynamic batching layer for SR/restoration models so concurrent frame jobs share GPU batches. | S2 serverless_bundle |
| 53 | - [ ] Split or modularize the monolithic `handler.py` hot path enough that route definitions, model loading, quality gates, auth middleware, and diffusion helpers are separately testable. | S2 serverless_bundle |
| 54 | - [ ] Add `seedvr2` conda env; pull `SeedVR2-3B`; run SeedVR2 vs RVRT vs RealBasicVSR on A40 using the existing golden/degraded clips. | S6 F1.0 |
| 55 | - [ ] Score SeedVR2 eval with pyiqa/VMAF and record fps, VRAM, and per-clip cost before serving cutover. | S6 F1.0 |
| 56 | - [ ] Torch-only SeedVR2 path first; pin `torch.compile` on and measure before attempting ONNX. | S6 F1.1 |
| 57 | - [ ] Demote redundant RVRT/VRT/RealBasicVSR per-degradation engines to `archived` after SeedVR2 wins ≥80% of A/B comparisons. | S6 F1.2 |
| 58 | - [ ] Add FlashVSR v1.1 as the throughput/streaming VSR primary and keep SeedVR2-3B as the heavy-degradation quality tier until golden eval picks per-route winners. | S7 Part A #1 |
| 59 | - [ ] Replace i2vgen-xl routes with Wan2.2-TI2V-5B / I2V-A14B candidates after A40 validation and license check. | S7 Part A #2 |
| 60 | - [x] Request SAM3/SAM3.1 gated HF access and verify commercial terms before creator-platform deployment. | S6 F2.0 + risks |
| 61 | - [ ] Bench SAM3 noun phrases `{face, skin, hair, text, watermark, tattoo, background}` on 50 frames per product domain and record precision plus ms/frame. | S6 F2.0 |
| 62 | - [ ] Wire SAM3 masks into region-aware enhancement: face restore inside face masks only, conservative sharpening on skin, LANCZOS/text-safe handling for text regions. | S6 F2.1 |
| 63 | - [ ] Cache SAM3 masks keyed by content hash and concept. | S6 F2.1 |
| 64 | - [ ] Install and validate Nunchaku in a fresh A40 serverless env with FLUX.1-schnell INT4, then repeat on RTX 2060 to settle Turing kernel support. | S6 F3.0 |
| 65 | - [ ] Register INT4 diffusion entries with `precision: int4-svdquant` and BF16 parity baselines so M3 gates every INT4 route. | S6 F3.1 |
| 66 | - [ ] Rebuild the diffusion layer of the Novita image with Nunchaku kernels and measure container size delta. | S6 F3.2 |
| 67 | - [ ] Move model payloads out of the Docker image; target image size ≤8GB, first-request penalty <10s, steady-state unchanged. | S6 F4.0 |
| 68 | - [ ] Adapt Run:ai Model Streamer for safetensors diffusion weights. | S6 F4.1 |
| 69 | - [ ] For TRT `.engine` files, implement chunked parallel S3 reads into pinned host memory before deserialize; benchmark 8/16/32 workers and pick the knee. | S6 F4.1 |
| 70 | - [ ] Document Novita cold-start before/after in `DEPLOY.md`. | S6 F4 acceptance |
| 71 | - [ ] Replace Real-ESRGAN/RealPLKSR/DAT routes with 4xNomos8kSCHAT-L, 4xNomos8kDAT, or 4xSPANkendata candidates where golden eval and GPU fit approve them. | S7 Part B |
| 72 | - [ ] Keep CodeFormer as commercial-safe fallback until OSDFace license is cleared for production. | S7 Part A #4 |
| 73 | - [ ] Evaluate BioIR/AdaIR for NAFNet replacement, but keep academic/research license routes out of commercial production until cleared. | S7 Part B |
| 74 | - [ ] Evaluate PowerPaint v2 as commercial-safe LaMa replacement; keep FLUX.1-Fill-dev gated behind a BFL license because `-dev` weights are non-commercial for serving. | S7 Part B |
| 75 | - [ ] Evaluate GIMM-VFI as RIFE upgrade while retaining RIFE for cheap 2× interpolation. | S7 Part B |
| 76 | - [ ] Evaluate Arc2Face plus expression adapter for identity-consistent face generation, while keeping StyleGAN3 for pure unconditional synthesis. | S7 Part B |
| 77 | - [ ] Add Parakeet TDT 0.6B v3 route when transcription/search/captioning is requested. | S7 honorable + Part B |
| 78 | - [ ] Add Kokoro-82M and Chatterbox/F5-TTS candidates for CPU/all-GPU TTS routes when voice generation is requested. | S7 Part B |
| 79 | - [ ] Recheck the license watchlist before any commercial route: OSDFace, BioIR/AdaIR, FADING, FLUX dev variants, insightface buffalo, HunyuanVideo, and LTX. | S7 watchlist |
| 80 | - [ ] Fill `b2_colorize_dh_colorizer` with DDColor and hydrate RVRT/VRT/BasicVSR++/DOVE/EDSR/EDVR/NAFNet/RAFT/VSR-DUF stubs only when real weights exist, then introspect hashes. | S8 |

**Repo impact note:** These changes turn `serverless_bundle` from a manually patched GPU endpoint into a measurable, gated production runtime. Quality gates, signed dispatch, model streaming, route-specific baselines, and modern replacement models should reduce bad deploys, lower cold-start and GPU cost, raise throughput, and make every production inference traceable to a job, model version, eval report, and rollback path.

---

## 3. `pixelenhance-labs`

*SaaS video enhancement platform. Path: `/mnt/storage/projects/pixelenhance-labs` (ASUS).*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] Every production GPU request has `job_id` + `user_id` | S4 M2 DoD |
| 2 | - [ ] Alembic migration: `jobs`, `job_events`, `wallet_reservations` | S4 M2.0 |
| 3 | - [ ] `POST /v1/jobs`, `GET /v1/jobs/{id}` | S4 M2.0 |
| 4 | - [ ] Job Router (new: pixelenhance-labs/backend/app/jobs/router.py) | S4 M2 |
| 5 | - [ ] Finish the QA harness already planned around VMAF, PSNR, SSIM, and golden-set replay. | S1 §4 |
| 6 | - [ ] Golden regression unblocked in orchestrator | S3 Phase 7.4 |
| 7 | - [ ] Add a temporal-consistency CI metric: warp error or temporal LPIPS. | S1 §4 |
| 8 | - [ ] Temporal metric catches injected flicker | S3 Phase 7.4 |
| 9 | - [ ] QA report attached to deploy workflow | S3 Phase 7.4 |
| 10 | - [ ] PR CI < 15 min on ASUS | S3 Phase 7.4 |
| 11 | - [ ] pixelenhance-labs: add "One-Step Restore (beta)" pipeline option routing to the new endpoint; A/B against the RVRT route on real jobs. | S6 F1.2 |
| 12 | - [ ] Wire `pipeline_core/qa/runner.py` → `pxl_metrics`. Extend `qa/gating.py` for image + video jobs. Add job to `quality-gates.yml`. | S3 Phase 2.3 |
| 13 | - [ ] VMAF runs on ≥3 golden video pairs | S3 Phase 2.4 |
| 14 | - [ ] Scorecard artifact uploaded per run | S3 Phase 2.4 |
| 15 | - [ ] Wire DVC, `pipeline_rl_lab`, QA scores, and PostHog user choices into one reward loop. | S1 §4 |
| 16 | - [ ] Add off-policy evaluation before shipping learned routing policies: OBP or d3rlpy. | S1 §4 |
| 17 | - [ ] Add a no-reference quality auto-gate that PostHog-logs every enhancement's quality delta. | S1 §4 |
| 18 | - [ ] Consolidate the multiple MCP servers into one server with tool namespaces. | S1 §4 |
| 19 | - [ ] Compile chaiNNer `.chn` graphs into the `pipeline_core` operator DAG for headless GPU execution. | S1 §4 |
| 20 | - [ ] Operationalize FP8/Transformer Engine and the INT8 calibration phases already planned. | S1 §4 |
| 21 | - [ ] `cli/model_catalog.py` \| Load from registry HTTP or git submodule, not Mac volume paths | S4 M1.3 |
| 22 | - [ ] Move committed `actionlint`, `.vsix`, `branding.png`, `servers.html`, and the stray `=` file out of the repo after explicit removal approval. | S1 §4 |
| 23 | - [ ] OpenAPI published for `POST /v1/jobs` and `GET /v1/jobs/{id}`; Android client stub generated from it. | S4 M2.0 |
| 24 | - [ ] Create job without executing GPU; ledger row must exist. | S4 M2.0 acceptance |
| 25 | - [ ] Job Router selects `tier=local` if ASUS mirror is healthy, else `cloud_burst`, and dispatches to serverless handler, Celery GPU, or VTT worker. | S4 M2 dispatch |
| 26 | - [ ] Add wallet reserve/capture/release path: reserve estimated cost on dispatch, capture actual on success, release on failure. | S4 M2 dispatch |
| 27 | - [ ] Attach M3 eval sample when a job is marked as a golden/eval job. | S4 M2 dispatch |
| 28 | - [ ] Rate limits per `subscription_tier` from `UNIFIED_AUTH.md`. | S4 M2.4 |
| 29 | - [ ] Free tier cannot exceed daily GPU budget; studio tier can. | S4 M2.4 acceptance |
| 30 | - [ ] Wire `pixelenhance` Celery golden regression as a `pxl_eval --surface celery` surface. | S4 M3.5 |
| 31 | - [ ] One `pxl_eval` command runs serverless, Celery, mobile, and VTT surfaces with a unified scorecard. | S4 M3.5 acceptance |
| 32 | - [ ] `pxl qa regression --suite golden_v1` via `qa/runner.py`, `qa/baselines.py`, and `qa/gating.py`. | S3 Phase 7.1 |
| 33 | - [ ] Add `qa/temporal.py`: PR tier frame-diff MAD smoke; nightly full temporal-LPIPS via optical-flow warp. | S3 Phase 7.2 |
| 34 | - [ ] CI tiers: PR = 3 image + 1 video clip; nightly = full golden + VMAF; weekly cloud 4090 = 10 video sequences temporal-LPIPS. | S3 Phase 7.3 |
| 35 | - [ ] Wire PostHog user choices + QA deltas + DVC data versions into `pipeline_rl_lab` rewards. | S2 pixelenhance-labs |
| 36 | - [ ] Add DVC-versioned dataset of `(input features → best pipeline)` for learned routing. | S2 pixelenhance-labs |
| 37 | - [ ] Add OBP or d3rlpy off-policy evaluation before any RL route selector ships. | S2 pixelenhance-labs |
| 38 | - [ ] Treat chaiNNer `.chn` graphs as authoring format and compile them to optimized `pipeline_core` operator DAGs for headless GPU execution. | S2 pixelenhance-labs |
| 39 | - [ ] Collapse business/devops/gpu-strategist/model-advisor/posthog/technical MCP servers into one namespace-based MCP service. | S2 pixelenhance-labs |
| 40 | - [ ] Remove committed `dump.rdb` after explicit removal approval, inspect it for queue/session/cache data, and rotate any secrets/PII found. | S2 pixelenhance-labs |
| 41 | - [ ] Remove committed `.env.cli`, `.env.sentry`, and `.env.grafana` after explicit removal approval; rotate every value that appeared in them and add safe examples only. | S2 pixelenhance-labs |
| 42 | - [ ] Choose one Python dependency manager and resolve drift between Poetry, `requirements.txt`, and bun-managed frontend deps. | S2 pixelenhance-labs |
| 43 | - [ ] Add `uv`/lockfile discipline for Python runtime and CI so QA/RL dependencies do not drift. | S2 cross-cutting |
| 44 | - [ ] Run FlashVSR v1.1 against the same golden video sets as SeedVR2 and select throughput vs heavy-degradation winners per route. | S7 Part A #1 |
| 45 | - [ ] Add Parakeet TDT transcription/captioning as a creator/video workflow only after a real transcription feature request. | S6 HM + S7 Part B |

**Repo impact note:** These changes make `pixelenhance-labs` the product owner of job accounting, user-visible enhancement quality, and learned routing. The expected outcome is fewer untracked GPU costs, faster QA feedback, safer RL policy rollout, cleaner repo operations, and a production feedback loop where real user choices and objective quality metrics continuously improve pipeline selection.

---

## 4. `pxl-engine`

*Fast C++ inference core. Path: `/mnt/storage/projects/pxl-engine` (ASUS).*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [x] Add CPU-FP32 vs C++ TRT/FP16 golden-image tests with PSNR/SSIM thresholds; INT8 gated structurally + via `tools/int8_gate.py` subprocess (live TRT INT8 blocked on sm75 6GB — see rows 17–18). | S1 §2 |
| 2 | - [x] INT8 calibration gate (`tools/int8_gate.py` → `int8_manifest.json`); engine refuses unapproved INT8. No models `int8_approved: true` on this host (ORT TRT INT8 subprocess SIGSEGV exit -11 on sm75 6GB). | S1 §2 |
| 3 | - [x] Fix **production blockers** from `pxl_engine_improvements.md` before trusting mesh: CUDA graph incomplete capture \| Disable graphs until inference in capture region OR complete capture | S4 M3.2 |
| 4 | - [x] Cache eviction race \| `shared_ptr` lifetime guard on session | S4 M3.2 |
| 5 | - [x] Multi-input ORT retry \| Exponential backoff on OOM | S4 M3.2 |
| 6 | - [x] TRT profile flags \| Read `trt.sm75.fp16.approved` before building engines | S4 M1.2 |
| 7 | - [x] Unblock SwinIR, RealPLKSR, and RGT via pxl-engine ORT TRT EP + CUDA EP fallback (`ort_accel.py`, `engine.cpp`). torch-tensorrt/`torch.compile` is serverless_bundle — out of scope. | S1 §2 |
| 8 | - [x] Generate a per-model speed-vs-quality frontier report: latency, VRAM, PSNR, SSIM, and LPIPS for FP32/FP16/INT8 per GPU. | S1 §2 |
| 9 | - [x] Standardize CoreML export and add an ANE-targeted verification path for iOS-bound models. | S1 §2 |
| 10 | - [x] Add a golden-image correctness test for tiled inference so output quality is tested, not only runtime success. | S1 §2 |
| 11 | - [x] Add a tile-seam regression test for the overlapped tiling path. | S1 §2 |
| 12 | - [x] Regenerate the ONNX test fixture after approval so the full suite runs on the repo's ONNX Runtime. | S1 §2 |
| 13 | - [x] Add CMake hook to regenerate `generated_model_registry.inc` at configure time if registry hash changed. | S4 M1.2 |
| 14 | - [x] C++ keys match YAML IDs 1:1. | S4 M1.2 acceptance |
| 15 | - [x] `pxl_tests` pass after registry sync and generated include migration. | S4 M1.2 acceptance |
| 16 | - [x] 24h stress test: 1000 concurrent requests, zero wrong-output incidents. | S4 M3.2 acceptance |
| 17 | - [x] `tests/test_quantization_golden.cpp`: FP32 CPU vs FP16 TRT parity (family-tuned gates) + INT8 structural tests (manifest/registry/calib). Subprocess-only parity probe via `int8_gate.py` (sm75 6GB: all `int8_approved: false`, ORT TRT eight-bit subprocess SIGSEGV exit -11; no in-process engine parity). | S3 Phase 6.1 |
| 18 | - [x] `tools/int8_gate.py` emits `int8_manifest.json`; C++ `Int8Manifest` + `Engine::Enhance`/`ProcessFrame`/`load_model` and pybind `use_int8` serve INT8 only when `int8_approved: true` for the active GPU SM (`PXL_INT8_MANIFEST`). No models approved on sm75 6GB (SIGSEGV). Handler wiring is out of scope (see plan Non-goals). | S3 Phase 6.2 |
| 19 | - [x] Keep sm75 as the only approved INT8 target until tower returns. | S3 Phase 6.2 |
| 20 | - [x] Publish per-model speed-vs-quality JSON artifact that downstream repos can use to choose precision per GPU. | S2 pxl-engine |
| 21 | - [ ] Evaluate ONNX Runtime 1.23 standalone TensorRT EP plugin at the next ORT bump because in-tree TRT EP is deprecated. | S6 HM |
| 22 | - [ ] Evaluate Helion for future custom GPU kernels instead of hand-writing CUDA when the next custom op is needed. | S6 HM |
| 23 | - [x] Use ORT TRT EP with unsupported subgraphs falling back to CUDA EP as the default partial-acceleration escape hatch for SwinIR/RealPLKSR/RGT. | S2 pxl-engine |
| 24 | - [x] Standardize model export verification across server TRT and device ANE so mobile-bound CoreML packages share the same parity expectations. | S2 pxl-engine |

**Repo impact note:** These changes make `pxl-engine` the measured, registry-driven acceleration layer instead of a fast but isolated C++ runtime. The expected outcome is safer INT8 adoption, fewer silent precision or tiling regressions, better acceleration for modern SR architectures, and a shared speed/quality frontier that serverless, mobile, and marketplace routing can consume directly.

---

## 5. `pixelspark` + `pixelspark-ios` + `pxl-playground`

*Mobile enhancement clients. Path: `~/Projects/pixelspark` (Mac).*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] `ServerlessEnhanceService.kt` → create job first, pass `job_id` to serverless | S4 M2.1 |
| 2 | - [ ] Remove direct Novita URL calls from client | S4 M2.1 |
| 3 | - [ ] Feature flag `USE_JOB_FABRIC=true` default on in prod flavor | S4 M2.1 |
| 4 | - [ ] Every prod enhancement has `job_id` in PEL database. | S4 M2.1 acceptance |
| 5 | - [ ] Android **all flavors** → `api.pixelenhancelabs.ai` (verify `build.gradle.kts`) | S4 M2 auth |
| 6 | - [ ] Single auth issuer (PEL) for all mobile clients | S4 M2 DoD |
| 7 | - [ ] Android prod calls to `:8009` \| 0 (telemetry) | S4 M2 metrics |
| 8 | - [ ] Replace the triplicated Markdown/Kotlin/Swift routing logic with declarative JSON routing config consumed by all three clients. | S1 §5 |
| 9 | - [ ] pixelspark-ios + pxl-playground: `@Generable` structs for {album_title, edit_suggestion[], tags[]}; sessions with tool-calling into the existing enhancement pipeline ("make this pop" → parameterized filter calls). Ship behind iOS-26 availability check. | S6 F7.0 |
| 10 | - [ ] One shipped on-device feature per mobile app (App Store/Play build), $0 marginal inference cost. | S6 F7 acceptance |
| 11 | - [ ] Add Core ML / ANE delegate selection on iOS. | S1 §5 |
| 12 | - [ ] Prototype vision-input prompts (image → tags/captions/edit plan) and SpotlightSearchTool over the app's media index | S6 F7.1 |
| 13 | - [ ] pixelspark Android: ML Kit GenAI Prompt API (Gemini Nano) where available; **Gemma 3n E2B via Google AI Edge / LiteRT-LM** as the portable fallback; same JSON contracts as iOS (`@Generable` schema ↔ XGrammar Structural Tags — one schema file, two enforcers). | S6 F7.2 |
| 14 | - [ ] Add optical-flow-warped temporal smoothing for video enhancement. | S1 §5 |
| 15 | - [ ] Add a tiny on-device classifier to pick the pipeline from frame content. | S1 §5 |
| 16 | - [ ] Add an on-device no-reference quality metric: MUSIQ-lite, MANIQA-lite, or BRISQUE. | S1 §5 |
| 17 | - [ ] Use that quality metric to skip no-op enhancement and auto-fallback when a model degrades a frame. | S1 §5 |
| 18 | - [ ] Add NNAPI/TFLite delegate auto-selection with measured fallback. | S1 §5 |
| 19 | - [ ] Store per-device delegate benchmark results and route based on the measured result. | S1 §5 |
| 20 | - [ ] Document `:8009` Spring stack as **deprecated** in `pixelspark/README.md` | S4 M2 auth |
| 21 | - [ ] Spring auth endpoints return `410 Gone` + migration link | S4 M2 auth |
| 22 | - [ ] **F7** One on-device feature shipped per mobile app; zero malformed tool calls in ara-code benchmark. | S6 DoD |
| 23 | - [ ] Android client stub generated from the M2 OpenAPI job schema and used before direct serverless dispatch. | S4 M2.0 |
| 24 | - [ ] Remove Spring Security JWT issuance after 90d telemetry shows zero `:8009` traffic. | S4 M2.5 |
| 25 | - [ ] Archive `pixelspark/backend` auth to `legacy/` or remove after migration window, after explicit approval. | S4 M2.5 |
| 26 | - [ ] Weekly Android TFLite eval job via Firebase Test Lab or local device farm. | S4 M3.5 |
| 27 | - [ ] Device eval gates: timeout, output size, and BRISQUE delta; quarantine model on fail. | S4 M3 eval matrix |
| 28 | - [ ] Convert mobile preset families into generated registry pipeline YAML, then consume the same declarative routes in Kotlin, Swift, and Playground. | S9 normalization |
| 29 | - [ ] Reconcile `pixelspark-ios/ModelRoutingRules.json`, ASUS `pixelspark/TFLITE_ROUTING_MATRIX.md`, and `pxl-playground` adapters against generated registry output. | S9 normalization |
| 30 | - [ ] CoreML manifest stub for pixelspark-ios when ANE path resumes. | S4 M1.3 |
| 31 | - [ ] iOS 26 baseline: `@Generable` contracts for album title, edit suggestions, and tags, with tool calls into existing enhancement pipeline. | S6 F7.0 |
| 32 | - [ ] iOS 27 beta features stay flagged until GA; no creator-platform/NSFW assumptions move onto Apple FM. | S6 F7 risks |
| 33 | - [ ] Android Gemma 3n fallback is mandatory when Gemini Nano / ML Kit GenAI is unavailable. | S6 F7.2 + risks |
| 34 | - [ ] Maintain one JSON schema file enforced by Swift `@Generable`/XGrammar and Android XGrammar Structural Tags. | S6 F7.2 |
| 35 | - [ ] Benchmark every model/delegate combination at install time and store per-device results for routing decisions. | S2 mobile trio |
| 36 | - [ ] Remove committed PixelSpark `.mp4` test outputs and `.venv/` after explicit removal approval; add `.gitignore` rules for regenerated artifacts. | S2 hygiene |
| 37 | - [ ] Remove pxl-playground `logs.rtf`, `nohup.out`, and `.ara_*` session files after explicit removal approval; add `.gitignore` rules. | S2 hygiene |
| 38 | - [ ] Add no-op enhancement skip telemetry so U1 can measure skip-enhance rate and session return for Android. | S5 U1.3 |
| 39 | - [ ] Evaluate Transformers.js/WebGPU only for dashboard or client-side preview features when a try-before-you-buy preview epic exists. | S6 HM |

**Repo impact note:** These changes make the mobile apps consume the same registry and routing contract as server-side inference while adding private, zero-marginal-cost on-device intelligence. The expected outcomes are less Kotlin/Swift/Markdown drift, better battery and latency through measured delegate routing, fewer bad mobile model rollouts, and richer offline features that improve retention without increasing cloud inference spend.

---

## 6. `phantom-trades-mvp`

*Trading validation. Path: `~/Projects/phantom-trades-mvp` (Mac).*

> **2026-07-04 SUPERSEDED-IN-PART:** A full production-readiness audit measured the live paper record at **93 trades, 24.7% win rate, profit factor 0.425, net −$28.45** (fee-blind fills) and validation Brier 0.2483 vs 0.2480 base-rate baseline (zero model skill). The "79% win rate" README claim traces to a synthetic-data grid search (2,592 combos). The items below remain valid but are now sequenced inside a deeper 7-phase rebuild — see **`AI_ML_PHANTOM_TRADES_AUDIT.md`** (evidence) and **`AI_ML_PHANTOM_TRADES_ULTIMATE_PLAN.md`** (execution). Headline deltas beyond the original list: horizon shift 5min→4h/1d (cost-to-edge math), triple-barrier + meta-labeling replaces raw direction labels, LightGBM/TabPFN ensemble replaces LSTM as champion, statistical deployment gates (DSR>0.95, PBO<0.10, SPA p<0.05, ≥200 OOS trades), freqtrade migration for execution (post-only entries, stoploss-on-exchange, reconciliation, dead-man's switch), and a $500→$5k capital ramp gated on live implementation-shortfall <0.2%/trade.

| Rank | Item | Source |
|------|------|--------|
| 0 | - [ ] **Execute `AI_ML_PHANTOM_TRADES_ULTIMATE_PLAN.md` Phases 0–5** (safety freeze → Kraken-native data spine → GBDT+meta-label core → stat-gated validation harness → freqtrade execution hardening → 90-day gated paper + capital ramp). Items below fold into it. | Audit 2026-07-04 |
| 1 | - [ ] Replace the fragile 80/20 split with a rigorous walk-forward harness. | S1 §3 |
| 2 | - [ ] No production path uses single 80/20 split | S3 Phase 3.4 |
| 3 | - [ ] Gate models on `vectorbt`/`quantstats` walk-forward P&L metrics: Sharpe, Sortino, max drawdown, and profit factor. | S1 §3 |
| 4 | - [ ] Promotion gate blocks bad configs | S3 Phase 3.4 |
| 5 | - [ ] Add Purged K-Fold validation. | S1 §3 |
| 6 | - [ ] Purged K-Fold in any HPO script | S3 Phase 3.4 |
| 7 | - [ ] Walk-forward report per `backtest_*.py` entrypoint | S3 Phase 3.4 |
| 8 | - [ ] Chronos-2 zero-shot BTC return quantiles as *features + bounds* into the existing signal stack (never as sole alpha); crepes online-calibrated intervals gate position size (reject trades whose stop sits inside the 80% interval). Run martingale exchangeability test first; if it fails on BTC returns, restrict conformal use to volatility (it usually passes there). | S6 F6.2 |
| 9 | - [ ] Validate through the walk-forward harness from TOP7 Phase 3 — no new validation code. | S6 F6.2 |
| 10 | - [ ] Add a regime filter before Kelly sizing. | S1 §3 |
| 11 | - [ ] Add a LightGBM/GBDT baseline for tabular OHLCV. | S1 §3 |
| 12 | - [ ] Add `pos_weight` for class imbalance. | S1 §3 |
| 13 | - [ ] phantom-trades \| OOS Sharpe, live paper P&L | S5 U1.3 surface |
| 14 | - [ ] phantom-trades paper trade breach \| daily export | S5 U3.0 |
| 15 | - [x] Fix the Kelly inconsistency: position size and stop/target logic must use one reward-risk source of truth. | S1 §3 |
| 16 | - [x] Use per-trade Kelly from the model's `prob`, not a static historical win rate. | S1 §3 |
| 17 | - [x] Add probability calibration with isotonic or Platt scaling and a reliability diagram. | S1 §3 |
| 18 | - [x] Use actual `pnl_pct` distribution for expectancy instead of reducing trades to a boolean `won`. | S1 §3 |
| 19 | - [x] Add a drawdown governor that cuts Kelly fraction as live drawdown approaches a limit. | S1 §3 |
| 20 | - [ ] Audit target creation and sequence alignment for look-ahead/normalization leakage, especially `rolling(...).max().shift(...)`, `dropna`, and `targets[i-1]`. | S2 phantom-trades |
| 21 | - [ ] Stop promoting models on `final_accuracy >= 0.60`; promotion must use P&L metrics from walk-forward backtests. | S2 phantom-trades |
| 22 | - [ ] Add `validation/walk_forward.py` with `WalkForwardConfig(train_days=180, test_days=30, step_days=30, purge_gap_days=5, min_folds=6)`. | S3 Phase 3.1 |
| 23 | - [ ] Add `validation/promotion_gate.py` with thresholds: median OOS Sharpe ≥0.8, Sortino ≥1.0, max drawdown ≤20%, profit factor ≥1.3, profitable folds ≥60%. | S3 Phase 3.3 |
| 24 | - [ ] Integrate the promotion gate into `acceptance_checker.py`. | S3 Phase 3.3 |
| 25 | - [ ] Use `vectorbt.Portfolio.from_signals()` and `quantstats` fold reports as the canonical walk-forward reporting path. | S3 Phase 3.1 |
| 26 | - [ ] Replace root sprawl of `backtest_*.py`, `retrain_v*.py`, and `strategy_v*.py` with one parameterized `backtest.py`, a `strategies/` package, and gitignored results. | S2 phantom-trades |
| 27 | - [ ] Turn `BACKTEST_REGISTRY.md` into a real backtest registry rather than a prose list. | S2 phantom-trades |
| 28 | - [ ] Add focal loss or `pos_weight` to correct rare positive targets before comparing LSTM accuracy. | S2 phantom-trades |
| 29 | - [ ] Add Temporal Convolutional Network or PatchTST candidate after the LightGBM baseline, then compare through the same walk-forward gate. | S2 phantom-trades |
| 30 | - [ ] Run martingale exchangeability test before conformal intervals; if BTC returns fail, restrict conformal use to volatility intervals. | S6 F6.2 |
| 31 | - [ ] Reject trades whose stop sits inside the online-calibrated 80% conformal interval. | S6 F6.2 |
| 32 | - [ ] Empirical interval coverage must be within ±3% of nominal 80/90% before F6 intervals are considered production-ready. | S6 F6 acceptance |
| 33 | - [ ] Track phantom-trades max drawdown reduction in walk-forward after interval-gated sizing. | S6 F6 acceptance |
| 34 | - [ ] Apply the M2 wallet/reserve pattern to paper→live gating before any real-money bridge. | S4 M2 unlocks |
| 35 | - [ ] Standardize dependencies with `uv` plus a lockfile instead of unpinned `>=` requirements. | S2 cross-cutting |

**Repo impact note:** These changes turn `phantom-trades-mvp` from a model-accuracy project into a trading-system validation project. The expected outcome is lower leakage risk, promotion decisions tied to out-of-sample P&L rather than classifier accuracy, safer Kelly sizing through calibrated probabilities and intervals, and a cleaner repo structure where the live strategy path is unambiguous.

---

## 7. `deal-ghost`

*Deal scoring. Path: `~/Projects/deal-ghost` (Mac).*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] Ground scoring in price history using trailing 30-day median and z-score for the exact query/listing class. | S1 §6 |
| 2 | - [ ] Every scored deal logs `price_z`, `median_30d`, `n_observations` | S3 Phase 4.4 |
| 3 | - [ ] Alerts include "X% below 30d median" | S3 Phase 4.4 |
| 4 | - [ ] Backtest shows fewer false-positive alerts | S3 Phase 4.4 |
| 5 | - [ ] Wrap the existing scorer's price predictor with `SplitConformalRegressor` on `price_observations` history (that table ships in TOP7 Phase 4 — reuse). Alert only when listed price < lower 90% bound. Measure alert precision vs current heuristic for 2 weeks. | S6 F6.1 |
| 6 | - [ ] Emit from deal-ghost on alert send + user feedback hook | S5 U1.0 |
| 7 | - [ ] deal-ghost "bad deal" user report \| webhook | S5 U3.0 |
| 8 | - [ ] Cache deal-ghost `score_deal` prompts \| near-duplicate listings | S5 U2.2 |
| 9 | - [ ] deal-ghost scoring \| `DealScorer(dspy.Module)` with MIPRO/GEPA optimizer | S5 U2.3 |
| 10 | - [ ] Add CLIP-based image/model mismatch checks for GPU scam detection. | S1 §6 |
| 11 | - [ ] Add reverse-image or stock-photo detection for reused listing images. | S1 §6 |
| 12 | - [ ] Add title-embedding/vector dedup across marketplaces. | S1 §6 |
| 13 | - [ ] Add proxy rotation, per-domain adaptive backoff, and captcha handling. | S1 §6 |
| 14 | - [ ] Port anti-bot hardening from `shadowscraper`. | S1 §6 |
| 15 | - [ ] **F6** Four surfaces consuming calibrated intervals with verified coverage; deal-ghost precision +15%. | S6 DoD |
| 16 | - [ ] Add append-only `price_observations` table keyed by `query_hash = sha256(normalized_query + listing_class)`. | S3 Phase 4.1 |
| 17 | - [ ] Add `scoring/price_stats.py` with robust z-score via median + MAD over trailing 30 days. | S3 Phase 4.2 |
| 18 | - [ ] Augment `score_deal()` with `final_score = 0.6 * llm_score + 0.4 * price_score(z)`. | S3 Phase 4.3 |
| 19 | - [ ] Alert only when `z <= -1.0` and `llm_legitimacy >= 70` for the first price-history rollout. | S3 Phase 4.3 |
| 20 | - [ ] Alert only when listed price is below the lower 90% conformal bound during the F6 pilot. | S6 F6.1 |
| 21 | - [ ] Measure alert precision against the current heuristic for 2 weeks before replacing the old rule. | S6 F6.1 |
| 22 | - [ ] Empirical conformal coverage must be within ±3% of nominal 80/90% on deal price history. | S6 F6 acceptance |
| 23 | - [ ] Reuse TOP7 `price_observations` table for conformal price intervals; do not create a second history store. | S6 F6.1 |
| 24 | - [ ] Add Playwright fingerprint rotation, adaptive backoff state, and proxy pool interface from `pxl-scrape-core` once extracted. | S5 L3 + S1 §6 |
| 25 | - [ ] Add vector/title embedding dedup so cross-marketplace duplicates collapse before Telegram alerts. | S2 deal-ghost |
| 26 | - [ ] Add semantic cache for near-duplicate `score_deal` prompts and report cache hit rate/cost saved. | S5 U2.2 |
| 27 | - [ ] Recompile `DealScorer(dspy.Module)` only against a golden set from U3 and block the prompt version if judge score drops. | S5 U2.3 |
| 28 | - [ ] Add bad-deal user reports to SHELF as replayable false-positive cases. | S5 U3.0 |

**Repo impact note:** These changes make `deal-ghost` score deals against observed market reality instead of trusting a single LLM pass. The expected outcome is higher alert precision, fewer duplicate or scammy notifications, more resilient marketplace scraping, and a measurable feedback loop where bad alerts become regression cases and better scoring prompts can be safely compiled.

---

## 8. `ara-code`

*Local coding agent. Path: `~/Projects/ara-code` (Mac).*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [x] Add tree-sitter/AST grounding. | S1 §7 |
| 2 | - [x] Build a repo map or symbol graph so the agent navigates by structure instead of full text. | S1 §7 |
| 3 | - [x] Index 10K-file repo in <60s cold, <2s warm | S3 Phase 5.5 |
| 4 | - [x] "Where is score_deal defined?" resolves without grep | S3 Phase 5.5 |
| 5 | - [x] Navigation-heavy prompts use ≥30% fewer tokens | S3 Phase 5.5 |
| 6 | - [x] (new capability) → **Qwen3-Coder** (local coding LLM for ara-code) | S7 Part A #5 |
| 7 | - [x] ara-code: route local models through `mlx_lm.server` (mac) / vLLM+XGrammar2 (asus 2060, Qwen3-4B/8B); all tool calls behind XGrammar Structural Tags → measure malformed-call rate (expect 0). nexus answer schemas likewise. | S6 F7.3 |
| 8 | - [x] ara-code malformed-tool-call rate = 0 across a 200-task local-model benchmark run. | S6 F7 acceptance |
| 9 | - [x] Add semantic code indexing over the codebase. | S1 §7 |
| 10 | - [x] Add LSP integration for go-to-definition, find-references, and diagnostics. | S1 §7 |
| 11 | - [x] Add a scored regression suite for model or prompt changes. | S1 §7 |
| 12 | - [x] Add LLM observability with Langfuse or OpenLLMetry. | S1 §7 |
| 13 | - [x] ara-code \| task success, human edit distance | S5 U1.3 surface |
| 14 | - [x] ara-code \| agent trace regression | S5 U3.4 |
| 15 | - [x] Rotate credentials from the committed `.env` and remove `.env`/`security.log` from git after explicit removal approval. | S1 §7 |
| 16 | - [x] `git init` and rotate credentials before first push if the repo is not already safely initialized. | S3 Phase 5.1 |
| 17 | - [x] Add `ara_code/indexing/tree_sitter_index.py` for Python, TypeScript, Go, and Rust. | S3 Phase 5.2 |
| 18 | - [x] Output Aider-style `RepoMap` JSON capped at 8K tokens. | S3 Phase 5.2 |
| 19 | - [x] Add `symbol_graph.py` backed by SQLite at `~/.ara/index/{repo_hash}.db`, incremental on mtime. | S3 Phase 5.3 |
| 20 | - [x] Add agent tools: `search_symbols`, `get_definition`, and `get_references`. | S3 Phase 5.4 |
| 21 | - [x] Inject `<repo_map>` before tool loops instead of forcing full-text navigation. | S3 Phase 5.4 |
| 22 | - [x] Restore `api_agent.py` before indexing if missing modules still block the repo-map work. | S3 risk register |
| 23 | - [x] Add Firejail/Docker/gVisor isolation for shell/code execution; resource limits alone are not enough for an agent that runs commands. | S2 ara-code |
| 24 | - [x] Split the 1MB `ara.py` and large `config.py` into testable modules so the agent can maintain itself without loading the world. | S2 ara-code |
| 25 | - [x] Add a 50+ case safety/eval harness for agent routing, tool use, and code changes before prompt/model updates ship. | S2 ara-code + S5 U2 |
| 26 | - [x] Route local models through `mlx_lm.server` on Mac and vLLM+XGrammar2 on ASUS; enforce all tool calls with Structural Tags. | S6 F7.3 |
| 27 | - [x] Add Qwen3-Coder 8B GGUF as the first local coding model target; measure against the scored regression suite before making it default. | S7 Part A #5 |
| 28 | - [x] Add OpenLLMetry/OTel export for agent traces so U1 can link task success and human edit distance to prompt/model versions. | S1 §7 + S5 U1 |
| 29 | - [x] Add semantic cache for repo-map queries and report hit rate/cost saved. | S5 U2.2 |
| 30 | - [x] Migrate intent routing to a compiled DSPy classifier only after U3 golden cases exist. | S5 U2.3 |
| 31 | - [x] Agent-implemented PR flow: opens PR adding one model YAML, human merges, no direct main push. | S5 U2.4 acceptance |

**Repo impact note:** These changes make `ara-code` a structure-aware, sandboxed coding agent rather than a large prompt-and-grep monolith. The expected outcome is fewer malformed tool calls, lower token use on navigation-heavy work, safer command execution, measurable prompt/model regressions, and enough observability for agent failures to become reproducible SHELF test cases.

---

## 9. `video-time-travel`

*Temporal face re-aging. Path: ASUS `video-time-travel` + Mac mirror.*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] Replace standalone `vtt_jobs.db` writes with PEL job API client | S4 M2.3 |
| 2 | - [ ] Job type `vtt_reage` in router | S4 M2.3 |
| 3 | - [ ] VTT jobs visible in PEL dashboard job history. | S4 M2.3 acceptance |
| 4 | - [ ] video-time-travel: use SeedVR2 as the post-re-aging cleanup pass (replaces chained denoise→SR), cutting its two-model tail to one. | S6 F1.2 |
| 5 | - [ ] GMFlow → **SEA-RAFT** (optical flow — video-time-travel temporal warping) | S7 Part A #7 |
| 6 | - [ ] Add a temporal-consistency metric to CI for video enhancement output. | S1 §8 |
| 7 | - [ ] video-time-travel \| export rate, re-run rate | S5 U1.3 surface |
| 8 | - [ ] video-time-travel: replace insightface-only face localization with SAM 3 face+hair masks for cleaner re-aging boundaries. | S6 F2.2 |
| 9 | - [ ] Add a lightweight temporal VAE or video diffusion backbone direction for the FADING/GMFlow flow. | S1 §8 |
| 10 | - [ ] Address unpinned dependency drift with a lockfile. | S1 §8 |
| 11 | - [ ] Preprocessing on cheap tier, diffusion on burst tier through the M2 router. | S4 M2.3 |
| 12 | - [ ] Add VTT temporal metric as `pxl_eval --surface vtt`, reusing temporal LPIPS from Top-7. | S4 M3.5 |
| 13 | - [ ] Add VTT failures to SHELF when a paper/eval breach or user re-run indicates a bad output. | S5 U3.4 |
| 14 | - [ ] Use SeedVR2 cleanup only after measuring re-aging identity/temporal quality against the existing denoise→SR chain. | S6 F1.2 |
| 15 | - [ ] Evaluate FlashVSR as the streaming/throughput VSR option and SeedVR2 as heavy-degradation cleanup; choose per route from golden eval. | S7 Part A #1 |
| 16 | - [ ] Use SEA-RAFT in temporal-consistency metric and warping stages before considering WAFT successor. | S7 Part A #7 |
| 17 | - [ ] Keep FADING as the best released face-age manipulation replacement unless FLUX-Kontext licensing and identity results justify a route. | S7 Part B |
| 18 | - [ ] Watch Causal Forcing++ / minWM only when a VTT streaming-preview feature is actually scoped. | S6 HM |
| 19 | - [ ] Standardize dependencies with `uv` or another lockfile mechanism so VTT is reproducible on Mac, ASUS, and burst workers. | S1 §8 + S2 cross-cutting |

**Repo impact note:** These changes connect `video-time-travel` to the shared job ledger, eval mesh, and modern temporal-restoration stack. The expected outcome is visible job history in PEL, fewer flicker or boundary regressions, cleaner face/hair masks for re-aging, reproducible deployments, and cheaper routing where preprocessing and diffusion stages run on the right tier.

---

## 10. `xcelsior`

*GPU marketplace / infra. Path: `/mnt/storage/projects/xcelsior` (ASUS).*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] Token endpoint GA with published price/SLO; ≥30% of served tokens KV-cache hits within a month of launch. | S6 F5 acceptance
| 2 | - [ ] Stand up vLLM + LMCache on 2 mesh hosts (Qwen3-8B or similar Apache model); enable EAGLE-3 (`--speculative-algorithm EAGLE3`) and validate acceptance rate ≥0.75 on real traffic samples before keeping it on. | S6 F5.0
| 3 | - [x] Meter via existing `metering.py`: per-token ledger with cached-vs-computed token split (cached tokens priced lower — the marketing headline writes itself). | S6 F5.0
| 4 | - [x] xcelsior: add `checkpoint_class: gpu-criu` capability flag to host agent + scheduler so preempted jobs on capable hosts resume instead of restart; surface "resumable" as a job attribute in the API. | S6 F4.2
| 5 | - [ ] One demonstrated preempt→migrate→resume of a running job between two xcelsior hosts (or same host restart) with no output diff. | S6 F4 acceptance
| 6 | - [x] Add Mooncake store (or LMCache remote tier) so prefix reuse survives host churn; KV-aware routing in the scheduler: route requests to the host holding their prefix (session affinity by prompt-prefix hash first; Dynamo router when >4 hosts). | S6 F5.1
| 7 | - [x] Publish `/v1/usage` SLO metrics (TTFT p95, tokens/s) per endpoint — sell against them. | S6 F5.1
| 8 | - [x] SCIP application or partner LOI submitted; attestation schema merged. | S6 F5 acceptance
| 9 | - [x] Write the SCIP alignment one-pager + application scan (Canadian ownership ✓, residency ✓, 18-month ops plan = mesh + partner DC). | S6 F5.2
| 10 | - [x] Design `attestation` fields in host admission schema now (TEE evidence JWT, NRAS verify) so H100 hosts slot in without API breaks; recruit 1 H100 host partner; price "attested-sovereign" tier at 2–3× commodity. | S6 F5.2
| 11 | - [x] stellar-subs: 14-day GPU-demand forecast with weekday covariates feeding the capacity planner; alert on interval breach (anomaly = actual outside 99% band — no extra anomaly model needed). | S6 F6.3
| 12 | - [x] MCP tool `should_i_run_this` pattern adapted for PEL jobs (internal admin / power users) | S4 M2.4
| 13 | - [x] **F5** Token SKU with published SLOs + cached-token pricing; SCIP submission/LOI out; attestation schema merged. | S6 DoD
| 14 | - [x] Ship embeddings serving through TEI or vLLM embeddings preset; stop returning `501 capability_not_available` on `/embeddings`. | S2 xcelsior
| 15 | - [x] Enable vLLM automatic prefix caching where safe and expose cached-token pricing in the product/metering layer. | S2 xcelsior
| 16 | - [x] Enable multi-LoRA serving for compatible bases so many adapters can share one GPU. | S2 xcelsior
| 17 | - [x] Enable speculative decoding after acceptance-rate validation; keep only when measured throughput improves under real traffic. | S2 xcelsior
| 18 | - [x] Replace 2-point predictive scaling extrapolation with EWMA or regression over the existing 20-sample queue-depth history. | S2 xcelsior
| 19 | - [x] Add pre-warmed pools keyed to predicted demand and model-weight cache availability; publish cold-start p50/p95 as SLO metrics. | S2 xcelsior
| 20 | - [x] Add OpenAI-style async Batch API for non-urgent embeddings, evals, and bulk inference at a discount. | S2 xcelsior
| 21 | - [x] Add LLM gateway semantic cache in front of OpenAI-compatible proxy and meter near-duplicate cache savings. | S2 xcelsior
| 22 | - [x] Add request-level chaos/fault-injection test that kills workers mid-inference and verifies requeue, idempotency, and no double billing. | S2 xcelsior
| 23 | - [x] Add idempotent metering ledger keyed by `(job_id, attempt)` so retries pay provider/customer exactly once. | S2 xcelsior
| 24 | - [x] Add hedged requests for long-tail latency: duplicate to a second worker after p95, take the winner, cancel loser, and bill once. | S2 xcelsior
| 25 | - [x] Add MCP/assistant safety eval suite for spend, provisioning, workload classification, tenant isolation, and data leakage. | S2 xcelsior
| 26 | - [x] Split or modularize `worker_agent.py` and `ai_assistant.py`; committed backup files must not be part of the active tree after explicit cleanup approval. | S2 xcelsior
| 27 | - [x] Add LLM observability with OpenLLMetry/Langfuse-style token, latency, cost, and tool traces. | S2 cross-cutting
| 28 | - [x] Add `Toto 2.0` as a monitored fallback if Chronos-2 underfits ops telemetry forecasting. | S6 HM
| 29 | - [x] Keep Dynamo optional until >4 LLM hosts; use LMCache alone at small scale. | S6 F5 risks
| 30 | - [ ] Launch with internal anchor workloads (pixelenhance-labs embed/caption, phantom-trades-mvp + ara-code chat patterns) to solve traffic cold start for token SKU. | S6 F5 risks
| 31 | - [x] Pin CUDA/driver requirements for CRIUgpu and test snapshot/restore on RTX 2060 before broader host-agent rollout. | S6 F4 risks

*Excluded from S1 checklist scope (Xcelsior business/product): token pricing, serverless metering, spot GPU pricing, free-credit UI, unit-economics dashboards, marketplace landing copy, embeddings serving, prefix cache, multi-LoRA, chaos/metering ledger — see S1 scope boundary.*

*xcelsior closure (engineering_partial, synced 2026-07-06T15:24:15Z): [x] rows: 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31. [ ] rows: 1, 2, 5, 30. row 1: engineering_partial: openai_proxy_served_traffic — proxy→live vLLM KPI 83.33% in harness; 1-month production soak ≥30% pending; row 2: engineering_partial: live_eagle_validated_gated_off — Qwen3-4B-AWQ+EAGLE accept ~0.53 on live HTTP; kept off (<0.75); Qwen3-8B 2-host mesh pending (RTX 2060 6GB); row 5: engineering_partial: live_criu_process_demo — same-host process CRIU ok; cross-host docker serverless migrate pending; row 30: engineering_partial: anchor_workloads_proxy_ledger — 2 local + 2 Mac-SSH builders with ledger rows; sustained production anchor traffic pending. Source: row-evidence.json. Billing gates unchanged. Engineering shipped (unchecked): row 1=openai_proxy_served_traffic; row 2=live_eagle_validated_gated_off; row 5=live_criu_process_demo; row 30=anchor_workloads_proxy_ledger.*

**Repo impact note:** These changes move `xcelsior` from GPU-hour marketplace plumbing toward a margin-focused inference platform with token SKUs, embeddings, KV reuse, reliable retries, and measurable SLOs.

---

## 11. `vaultwarden-secrets`

*Life-hack L1. Path: `~/Projects/vaultwarden-secrets` (Mac).*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] **L1:** `vw-deploy --all` works; no `.env` in git | S5 DoD |
| 2 | - [ ] L1.1 \| Naming convention doc + lint (`PROJECT_SERVICE_KEY`) | S5 L1 |
| 3 | - [ ] L1.2 \| `deploy.sh` all projects from Mac (ASUS rsync + env inject) | S5 L1 |
| 4 | - [ ] L1.3 \| 90-day rotation workflow + Bitwarden history | S5 L1 |
| 5 | - [ ] L1.4 \| Pre-commit hook: block `.env` commit | S5 L1 |
| 6 | - [ ] Add `secrets manifest.yaml` in each repo with keys only, never values. | S5 L1 |
| 7 | - [ ] `vw-deploy --project x` pulls from Vaultwarden and generates `.env` locally. | S5 L1 |
| 8 | - [ ] Optional age-encrypted CI artifact path for projects that need sealed env material in automation. | S5 L1 |
| 9 | - [ ] Add tagging/naming lint so keys follow `PROJECT_SERVICE_KEY` consistently across 449 imported secrets. | S5 L1 |

**Repo impact note:** These changes make `vaultwarden-secrets` the portfolio-wide control plane for environment drift and secret rotation. The expected outcome is safer deploys, fewer committed `.env` incidents, faster ASUS/Mac environment setup, and a reliable credential substrate for U2 agents and every project that needs generated env files.

---

## 12. `aarynfans-mvp`

*Creator platform + L2/L5. Path: `~/Projects/aarynfans-mvp` (Mac).*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] Add churn prediction from subscriber engagement features. | S1 §9 |
| 2 | - [ ] L2.2 \| Instrument both backends (Node emitters) | S5 L2 |
| 3 | - [ ] L2.3 \| Weekly churn risk dashboard + Telegram digest | S5 L2 |
| 4 | - [ ] L2.4 \| AI caption workflow (DSPy module) for stale creators | S5 L2 |
| 5 | - [ ] **L2:** Shared creator events; churn dashboard live | S5 DoD |
| 6 | - [ ] Wire platform-optimized AI captions/hooks per channel as a first-class creator workflow. | S1 §9 |
| 7 | - [ ] L5.1 \| Webhook normalizer + idempotent ledger table | S5 L5 |
| 8 | - [ ] L5.2 \| Cron attestation + HMAC-signed JSON | S5 L5 |
| 9 | - [ ] L5.3 \| `/trust/payments` static page | S5 L5 |
| 10 | - [ ] **L5:** BTCPay reconciliation automated; proof page public | S5 DoD |
| 11 | - [ ] Decide which creator-subscription repo is canonical before `stellar-subs`, `aarynfans-mvp`, and `content-multiplier` diverge further. | S1 §9 |
| 12 | - [ ] Rotate every credential that may have been present before the committed BFG report. | S1 §9 |
| 13 | - [ ] Remove the committed `..bfg-report` from `aarynfans-mvp` after explicit removal approval. | S1 §9 |
| 14 | - [ ] aarynfans-mvp \| subscription conversion (L2 feeds here) | S5 U1.3 surface |
| 15 | - [ ] Emit unified creator events: `content_published`, `subscription_started`, and `churn_risk` with `creator_id`, platform, `mrr_cad`, and `engagement_score`. | S5 L2 |
| 16 | - [ ] Train churn model on shared features: post frequency, DM response time, revenue slope, and engagement changes. | S5 L2 |
| 17 | - [ ] Route platform-optimized caption and hook generation through the U2 gateway with semantic cache. | S5 L2 |
| 18 | - [ ] Track caption/hook A/B outcomes so U1 can attribute conversion to prompt/model versions. | S5 L2 + U1 |
| 19 | - [ ] Use deterministic scripts, not LLMs, for money reconciliation; reserve LLMs only for customer-facing payment support drafts. | S5 L5 |
| 20 | - [ ] BTCPay webhook normalizer emits the same `payment_event` bus used by L2. | S5 L5 |
| 21 | - [ ] Embed public proof page once Axiom nightly reserve attestation is signed and stable. | S5 L5 |
| 22 | - [ ] Add Parakeet transcript/caption workflow when creator transcription becomes a real feature request. | S6 HM + S7 Part B |
| 23 | - [ ] Add Kokoro/Chatterbox/F5-TTS only after creator voice or narration product scope exists and licensing is cleared. | S7 Part B |
| 24 | - [ ] Keep canonical repo decision tied to concrete ownership: shared schema, billing owner, media pipeline owner, and caption workflow owner. | S1 §9 + S5 L2 |

**Repo impact note:** These changes make `aarynfans-mvp` part of a shared creator revenue system instead of a standalone subscription app. The expected outcomes are better churn prediction, measurable caption/hook lift, automated BTCPay reconciliation, safer post-BFG credential posture, and a proof/trust surface that can improve creator and subscriber confidence.

---

## 13. `stellar-subs`

*Creator platform + L2/F2. Path: `~/Projects/stellar-subs` (Mac).*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] L2.2 \| Instrument both backends (Node emitters) | S5 L2 |
| 2 | - [ ] stellar-subs: churn sets + LTV intervals in creator dashboard | S6 F6.3 |
| 3 | - [ ] stellar-subs: "auto-mask region by keyword" moderation tool (SAM 3.1 video tracking for clips). | S6 F2.2 |
| 4 | - [ ] Wire platform-optimized AI captions/hooks per channel as a first-class creator workflow. | S1 §9 |
| 5 | - [ ] Decide which creator-subscription repo is canonical before `stellar-subs`, `aarynfans-mvp`, and `content-multiplier` diverge further. | S1 §9 |
| 6 | - [ ] Remove committed `.DS_Store` from `stellar-subs` after explicit removal approval. | S1 §9 |
| 7 | - [ ] L2.3 \| Weekly churn risk dashboard + Telegram digest | S5 L2 |
| 8 | - [ ] L2.4 \| AI caption workflow (DSPy module) for stale creators | S5 L2 |
| 9 | - [ ] Emit unified creator events with `creator_id`, platform, `mrr_cad`, and `engagement_score`. | S5 L2 |
| 10 | - [ ] Train churn/LTV intervals on shared features across `stellar-subs` and `aarynfans-mvp`, not separate local definitions. | S5 L2 + S6 F6.3 |
| 11 | - [ ] Verify F6 churn/LTV interval empirical coverage within ±3% of nominal before dashboard decisions depend on it. | S6 F6 acceptance |
| 12 | - [ ] Route caption/hooks through U2 gateway semantic cache and pin prompt versions when DSPy recompiles. | S5 U2.3 + L2 |
| 13 | - [ ] Add SAM3 keyword masking demo on a real clip and clear gated license terms before production moderation use. | S6 F2 acceptance + risks |
| 14 | - [ ] Add Parakeet transcript/caption workflow only after a creator transcription feature request. | S6 HM + S7 Part B |

**Repo impact note:** These changes make `stellar-subs` share revenue intelligence, churn prediction, and creator AI workflows with the broader creator stack while adding SAM3 moderation capabilities. The expected outcome is clearer canonical ownership, better retention interventions, safer clip masking, lower LLM caption costs, and fewer analytics/schema divergences from `aarynfans-mvp`.

---

## 14. `shadowscraper`

*Scrape core extraction (L3). Path: Mac.*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] L3.1 \| Extract `core/backoff.py`, `core/proxy.py` from shadowscraper | S5 L3 |
| 2 | - [ ] Extract **`pxl-scrape-core`** Python package (no AI): Adaptive backoff + per-domain rate state; Proxy pool interface; Playwright fingerprint rotation; Honeypot / trap URL filter | S5 L3 |
| 3 | - [ ] L3.3 \| Integration test against httpbin + one real domain | S5 L3 |
| 4 | - [ ] Publish `pxl-scrape-core` to private PyPI or install via git submodule so consumers do not import the full 1.1GB monolith. | S5 L3 |
| 5 | - [ ] Split large scrape entrypoints such as `cli.py` into package modules before extraction so the core SDK can be tested independently. | S2 hygiene |
| 6 | - [ ] Keep SDK non-AI and deterministic; model-based scoring stays in consumers such as deal-ghost. | S5 L3 |
| 7 | - [ ] Add per-domain adaptive rate-state persistence so retry/backoff decisions survive process restarts. | S5 L3 |

**Repo impact note:** These changes turn `shadowscraper` into reusable scraping infrastructure instead of a monolith with trapped anti-bot logic. The expected outcome is more reliable Reddit and marketplace scraping, lower duplicated anti-bot work, cleaner tests around backoff/proxy behavior, and a package boundary that consumers can adopt without pulling unrelated code.

---

## 15. `reddit-mods`

*L3 consumer. Path: Mac.*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] **L3:** `pxl-scrape-core` imported by reddit-mods | S5 DoD |
| 2 | - [ ] L3.2 \| `pip install -e pxl-scrape-core` in reddit-mods | S5 L3 |
| 3 | - [ ] Replace local retry/rate-limit code with shared adaptive backoff and per-domain rate state from `pxl-scrape-core`. | S5 L3 |
| 4 | - [ ] Add consumer integration test that proves reddit-mods still handles rate limits after the SDK swap. | S5 L3 |

**Repo impact note:** These changes make `reddit-mods` a first consumer of the extracted scrape reliability SDK. The expected outcome is fewer rate-limit breakages, less duplicated scraper code, and a practical validation path proving `pxl-scrape-core` works outside its source repo.

---

## 16. `nexus`

*Knowledge bus (L4) + F7. Path: Mac.*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] **L4:** `nexus.search` MCP tool in U2 | S5 DoD |
| 2 | - [ ] L4.1 \| `git init` nexus; seed 20 portfolio notes | S5 L4 |
| 3 | - [ ] L4.2 \| MCP tool in U2 gateway | S5 L4 |
| 4 | - [ ] nexus: donate notes as `CSSearchableItem` + natural-language Q&A session. Gate on beta; ship at 27 GA (Sept 2026). | S6 F7.1 |
| 5 | - [ ] L4.3 \| Weekly `nexus audit` CLI reports stale notes | S5 L4 |
| 6 | - [ ] `nexus index` → SQLite FTS + optional embeddings | S5 L4 |
| 7 | - [ ] MCP tool `nexus.search(query)` → top-k notes injected into agent context | S5 L4 |
| 8 | - [ ] Create sync folder `~/Notes/portfolio/` with templates per project. | S5 L4 |
| 9 | - [ ] Generate `llms.txt` per project from README and PLAN.md where available, using the xcelsior docs pattern. | S5 L4 |
| 10 | - [ ] Nexus MCP runs through the U2 gateway as read-only context by default. | S5 L4 + U2 |
| 11 | - [ ] Enforce nexus answer schemas with XGrammar Structural Tags for local-agent Q&A. | S6 F7.3 |
| 12 | - [ ] Prototype natural-language Q&A over Core Spotlight only on iOS/macOS beta track and hold shipping until iOS 27 GA. | S6 F7.1 + risks |
| 13 | - [x] Feed stale-note and failed-answer cases into SHELF agent trace regression. | S5 U3.4 |

**Repo impact note:** These changes make `nexus` the portfolio memory layer for governed agents and local Q&A. The expected outcome is less project-state hallucination, faster onboarding across repos, read-only context injection for U2 tools, and a clear path to private Apple/MLX-powered note search without leaking notes to cloud models.

---

## 17. `axiom-bitcoin`

*Payments closure (L5). Path: Mac.*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] L5.1 \| Webhook normalizer + idempotent ledger table | S5 L5 |
| 2 | - [ ] L5.2 \| Cron attestation + HMAC-signed JSON | S5 L5 |
| 3 | - [ ] L5.3 \| `/trust/payments` static page | S5 L5 |
| 4 | - [ ] BTCPay webhook → unified `payment_event` (same bus as L2) | S5 L5 |
| 5 | - [ ] Nightly `verify-reserves.sh` → signed JSON attestation | S5 L5 |
| 6 | - [ ] Public **proof page** embed on aarynfans + optional aabiro.github.io | S5 L5 |
| 7 | - [ ] Payment reconciliation scripts must be deterministic; no LLM in ledger math, balance verification, or reserve attestation. | S5 L5 |
| 8 | - [ ] LLM support drafts may use U2, but only after payment_event and ledger state are already deterministic. | S5 L5 |
| 9 | - [ ] HMAC-signed JSON attestation includes timestamp, wallet/provider source, ledger totals, and verification status. | S5 L5 |
| 10 | - [ ] Add idempotency keys to BTCPay webhook processing so duplicate callbacks do not double-count revenue. | S5 L5 |

**Repo impact note:** These changes make `axiom-bitcoin` the deterministic trust and reconciliation layer for creator payments. The expected outcome is fewer manual BTCPay checks, auditable reserve proofs, safer duplicate webhook handling, and a public proof surface that reinforces the same sovereignty/trust story as xcelsior.

---

## 18. `aabiro.github.io`

*Agent-ready portfolio (L6). Path: Mac.*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] **L6:** `llms.txt` live on portfolio site | S5 DoD |
| 2 | - [ ] L6.1 \| `llms.txt` + agent.json in repo root | S5 L6 |
| 3 | - [ ] L6.2 \| JSON-LD injection in Flutter web build | S5 L6 |
| 4 | - [ ] L6.3 \| GitHub Action resume build | S5 L6 |
| 5 | - [ ] `llms.txt` + `/.well-known/agent.json` describing projects, stack, contact policy | S5 L6 |
| 6 | - [ ] GitHub Action: sync resume PDF from `RESUME.md` + contribution stats quarterly | S5 L6 |
| 7 | - [ ] Schema.org `Person` + `CreativeWork` JSON-LD in `index.html` | S5 L6 |
| 8 | - [ ] Add structured project metadata to `agent.json`: project name, repo URL, stack, status, allowed contact paths, and summary. | S5 L6 |
| 9 | - [ ] Add optional DSPy-generated cover-letter snippet tool for personal use, running locally through U2. | S5 L6 |
| 10 | - [ ] Embed Axiom proof page only after signed reserve attestation is stable and public-safe. | S5 L5 |
| 11 | - [ ] Ensure `llms.txt` and JSON-LD are generated in the Flutter web build, not hand-edited drift files. | S5 L6 |

**Repo impact note:** These changes make `aabiro.github.io` readable by humans, recruiters, and agents. The expected outcome is better structured discovery of the portfolio, less stale resume/project metadata, a visible trust surface for payment proofs when ready, and a public example of the agent-ready documentation standards used across the rest of the stack.

---

## 19. `stash-viewer`

*F2 honorable mention consumer. Path: Mac.*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] stash-viewer (mac): offline concept search — index masks for {person count, setting objects} into its SQLite; pair with F7's local embeddings for hybrid search. | S6 F2.2 |
| 2 | - [ ] **Parakeet TDT 0.6B v3** (NVIDIA, CC-BY-4.0) \| #1 Open-ASR leaderboard — transcript search for stash-viewer + free captioning for creator platforms \| Trigger: First transcription feature request | S6 HM |
| 3 | - [ ] **ColQwen3-embed-4B / SigLIP 2** \| 90.6 ViDoRe, Apache 2.0 — visual/doc retrieval for stash-viewer semantic search (pairs with F2 masks) \| Trigger: stash-viewer search epic | S6 HM |
| 4 | - [ ] Add mask/person/object indexes to SQLite with content-hash keys so concept search can reuse SAM3 service outputs or offline batches. | S6 F2.2 |
| 5 | - [ ] Pair F2 masks with local embeddings from F7 rather than requiring cloud search for private media. | S6 F2.2 + F7 |
| 6 | - [ ] Use `pxl-scrape-core` only for metadata enrichment if external pages are scraped; keep local library browsing independent. | S5 L3 |
| 7 | - [ ] Evaluate Qwen3-VL or WD EVA02 tagger only when a creator/tagging epic is scoped and licensing/content policy is cleared. | S6 HM + S7 HM |
| 8 | - [ ] Add Parakeet transcript search only after a first real transcription feature request, then index timestamps with media IDs. | S6 HM |

**Repo impact note:** These changes make `stash-viewer` a private, local-first semantic media browser rather than a filename-only viewer. The expected outcome is faster offline search by concepts, transcript, and visual embeddings while avoiding premature cloud dependencies or unscoped tagging models.

---

## 20. `ai-data-factory`

*Experiment and utility-pipeline feed-in. Path: ASUS `/mnt/storage/projects/ai-data-factory`.*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] Nightly bench in `ai-data-factory` → opens PR to M1 registry with `experiment` rows | S4 M3.6 |
| 2 | - [ ] Experiment manifests attach to registry IDs | S4 M1 unlocks |
| 3 | - [ ] Reconcile ASUS `ai-data-factory/docs/TFLITE_ROUTING_MATRIX.md` with ASUS `pixelspark/TFLITE_ROUTING_MATRIX.md`, app code, and registry models. | S9 gaps |
| 4 | - [ ] Mark imported/vendor research repos under `ai-data-factory/repos/*` as references only unless explicitly promoted into product pipelines. | S9 strict definition |
| 5 | - [ ] Represent `tools/scripts/tflite_inference_pipeline.py` as utility pipeline metadata or a registry helper capability, not a product route. | S9 inventory |
| 6 | - [ ] Represent `tools/scripts/unified_image_pipeline.py` capabilities: aspect-ratio preservation, tiling, region extraction/blending, face tracking, and video frame processing around supplied model functions. | S9 inventory |
| 7 | - [ ] Attach `organized_experiments/manifest.json` entries to registry model IDs before promotion. | S9 inventory + S4 M1 |
| 8 | - [ ] Use `ai-data-factory` for FLUX LoRA retrain batches if SDXL legacy LoRAs do not port to the F3 FLUX/Nunchaku route. | S6 F3 risks |
| 9 | - [ ] Add audit coverage so ai-data-factory utility matrices cannot silently diverge from `pxl-registry/registry/pipelines/*.yaml`. | S9 normalization |

**Repo impact note:** These changes make `ai-data-factory` the controlled experiment feeder for the registry rather than another source of routing truth. The expected outcome is cleaner promotion of successful experiments, less duplicated TFLite matrix drift, and a safe place to run model/LoRA retrains without confusing reference implementations with production pipelines.

---

## Appendix A — Mega-initiative cross-repo gates (reference)

These apply across repos; primary owner in parentheses.

| Rank | Item | Owner | Source |
|------|------|-------|--------|
| 1 | - [ ] Unified scorecard covers server + mobile + video surfaces | M3 / pxl-registry | S4 M3 DoD |
| 2 | - [ ] ASUS local tier serves ≥40% of eligible jobs | M3 / pixelenhance-labs | S4 M3 DoD |
| 3 | - [ ] Jobs without `user_id` \| 0 in production | M2 / pixelenhance-labs | S4 M2 metrics |
| 4 | - [ ] Serverless GPU $ unlinked to user \| 0 | M2 | S4 M2 metrics |
| 5 | - [ ] Deploys without eval pass \| 0 | M3 / serverless_bundle | S4 M3 metrics |

---

## Appendix B — Model download queue (ASUS staging)

From S7 suggested download order — register each in pxl-registry after pull:

```bash
# tier 1 — immediate pipeline swaps (small, license-clean)
hf download piddnad/DDColor-models --local-dir ddcolor
hf download MemorySlices/Tartan-C-T-TSKH-spring540x960-M --local-dir sea-raft
hf download Phips/4xNomos8kDAT --local-dir nomos8k-dat
hf download nvidia/parakeet-tdt-0.6b-v3 --local-dir parakeet-v3
hf download hexgrad/Kokoro-82M --local-dir kokoro
hf download amazon/chronos-2 --local-dir chronos2
hf download unsloth/Qwen3-8B-GGUF --include "*Q4_K_M*" --local-dir qwen3-8b-gguf

# tier 2 — A40-class engines (large; pull on the Novita pod or overnight)
hf download JunhaoZhuang/FlashVSR-v1.1 --local-dir flashvsr
hf download ByteDance-Seed/SeedVR2-3B --local-dir seedvr2-3b
hf download black-forest-labs/FLUX.1-schnell --local-dir flux-schnell
hf download Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers --local-dir sana15
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir wan22-ti2v5b

# tier 3 — gated (accept terms on HF first)
hf download facebook/sam3 --local-dir sam3
hf download google/gemma-3n-E4B-it --local-dir gemma3n-e4b
```

---

## Appendix C — Completed verification (do not re-open)

- [x] `pxl-engine` on Asus: `cmake --build build --target pxl_tests -j2`
- [x] `pxl-engine` on Asus: `./build/pxl_tests`
- [x] Add a golden-image correctness test for tiled inference so output quality is tested, not only runtime success.
- [x] Add a tile-seam regression test for the overlapped tiling path.
- [x] Regenerate the ONNX test fixture after approval so the full suite runs on the repo's ONNX Runtime.
- [x] Fix the Kelly inconsistency: position size and stop/target logic must use one reward-risk source of truth.
- [x] Use per-trade Kelly from the model's `prob`, not a static historical win rate.
- [x] Add probability calibration with isotonic or Platt scaling and a reliability diagram.
- [x] Use actual `pnl_pct` distribution for expectancy instead of reducing trades to a boolean `won`.
- [x] Add a drawdown governor that cuts Kelly fraction as live drawdown approaches a limit.

---

**Bottom line:** Use this file as the single per-repo execution index. Start at rank 1 in each repo you are working on; cite the source doc (S1–S7) when opening PRs. Update ranks only when a source doc changes — do not invent items here.
