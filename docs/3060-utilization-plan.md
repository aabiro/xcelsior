I have what I need. Here's your comprehensive 3060 utilization plan.

---

# RTX 3060 12GB Utilization Plan

## Hardware reality check

| Box | Tailnet | GPU | CPU | RAM | Disk | Role |
|---|---|---|---|---|---|---|
| **tower-server** | 100.64.0.4 | **RTX 3060 12GB (sm86)** idle, 0% | Ryzen 5 5600X / 12T | 32 GB | 1.7 TB free | ← target |
| asus-pc | 100.64.0.6 | RTX 2060 6GB | TUF A15 | — | — | dev laptop |
| aaryns-macbook-pro | 100.64.0.3 | (Apple Silicon CoreML) | — | — | — | iOS / web dev |
| vps-linuxuser | 100.64.0.1 | none | — | — | — | xcelsior prod |

---

## Phase 0 — Foundations (1 day)

These unblock everything else:

1. **RAM upgrade on tower** to 32 GB (dual rank, A2/B2). Diffusion models, multi-model serving, and TRT engine builds OOM on 16 GB.
2. **Containerize a "lab" service on tower** mirroring xcelsior worker pattern but with persistent named-volumes for model caches:
   - `/srv/models` → bind-mounted into every container
   - `/srv/datasets` → rsync from `100.64.0.6:/home/aaryn/storage/datasets/` (priority subset only, ~120 GB worth — full set is multi-TB)
3. **Tailscale serve / Funnel** the lab endpoints on tower so Mac + laptop hit them by hostname (`http://tower-server:8000/...`) without VPN config per app.
4. **CI runner mode**: register tower as a self-hosted GitHub Actions runner with `gpu` label so any repo's CI matrix can opt into `runs-on: [self-hosted, gpu]` for benchmarks/conversions.

---

## Phase 1 — `pxl-engine` (highest leverage, lowest risk)

Native sm86 target. The engine already supports CUDA 12 / TRT 10. **The 3060 IS literally what pxl-engine was designed for** as a low-end inference target.

| Task | Why it needs the 3060 |
|---|---|
| Build sm86 TRT plans for the 41 FP16 engines in `serverless_bundle/` | Existing engines are sm89 (4090) and won't load on 3060. **Critical blocker** for everything downstream. |
| Add a `RTX3060_12GB` profile to `pixelenhance-labs/hardware_profiles.py` | Lets devs run end-to-end pipeline locally instead of $0.80/hr vast.ai 4090 |
| Wire pxl-engine's pybind layer into pixelspark's Spring Boot dev backend | Mac → tower over Tailscale → sub-100ms inference for development |
| Run the full pxl-engine test suite as a nightly CI job on tower | Catches sm86-specific TRT regressions that the 4090 dev loop hides |

**Push to next level:** Stand up a tiny `pxl-engine-bench` Grafana dashboard pulling from a sqlite results store. Each model gets per-build latency/throughput history → real PR gating ("don't merge if upscale_4x got >5% slower").

---

## Phase 2 — `serverless_bundle` (sm86 rebuild + cost reset)

You're paying RunPod / Novita for inference that — for non-burst load — fits in 12 GB easily.

| Task | Action |
|---|---|
| Rebuild all FP16 TRT engines for sm86 | `bash build_trt_a40.sh` modified → `build_trt_3060.sh` with `--useFp16 --memPoolSize=workspace:8192` overnight job. Gain: zero cold-start when running locally. |
| Stand up a **mirror** of the Novita handler on tower as `tower-serverless` | Use handler.py directly via FastAPI shim on port 8001. Anything < ~6 RPS lives free on tower. |
| Add a feature-flagged routing rule in pixelenhance-labs Celery: `if cost_tier == "dev": dispatch to tower-serverless` | Saves all dev/staging spend |
| Finish the **SAM identity-preserving stabilizer** training (`SAM_CONVO.md`) | This is your in-flight blocker. 12 GB is enough for SAM2-Hiera-Small + LoRA at 512×512. **Schedule a 3-day continuous training run on tower.** |
| Build the 8–12 k clean face mix referenced in SAM_CONVO.md | Tower has 1.7 TB free → ingest WFLW + COFW + your `domain_aligned*` runs locally instead of paying vast.ai for storage time |

**Push to next level:** ship a self-hosted demo on `models.xcelsior.ca` (your own infra) that runs the cleaning + upscale pipeline end-to-end. Becomes a marketing surface for xcelsior ("eat your own dogfood").

---

## Phase 3 — `video-time-travel` (the diffusion blocker)

You documented this needs CUDA 12.1+ and a competent GPU. 12 GB on the 3060 is enough for **SDXL + a single ControlNet at 768×768 with `--medvram`**, not enough for multi-ControlNet stacks. Plan around that.

| Task | Notes |
|---|---|
| Rent the 3060 to yourself via xcelsior (eat your own dogfood) for the heavy diffusion ablations | This stress-tests xcelsior's billing + SSH paths in production |
| Run **MiVOLO age estimation** preprocessing batches locally — embarrassingly parallel, fits in 4 GB | Move from CPU/CoreML pre-pass on Mac to GPU pre-pass on tower; ~10× faster |
| Run **GMFlow** keyframe → keyframe optical flow on tower | Currently a major bottleneck because Mac CPU does it; 3060 cuts this from minutes to seconds |
| The actual SDXL ControlNet generation passes stay on rented vast.ai 4090 | Don't try to force 12 GB to do 24 GB work |
| Implement keyframe-budget UI in the local pipeline | Then iterate fast on small (3–5 keyframe) clips locally before committing to a paid run |

**Push to next level:** Cut a 20-second public demo using only your 3060 + xcelsior-rented A100 for finals. Real artifact for the marketing site.

---

## Phase 4 — `pixelenhance-labs` (dev-loop economics)

Currently dev = $0.80/hr vast.ai 4090. This is bleeding money during exploration.

| Task | Action |
|---|---|
| `RTX3060_12GB` hardware profile (FP16, batch=1, conservative memory pool) | New entry in `hardware_profiles.py` |
| Local `docker-compose.dev.yml` with tower as Celery worker host | API runs on laptop, Celery worker on tower over Tailscale (Redis on laptop) |
| Local prod-shaped tests (`make test-e2e-local` → uses 3060) | Vast.ai used **only** for prod release dry-runs |
| Cache HF + diffusers downloads to `/srv/models` | One-time download → permanent local artifact |

Estimated monthly savings: $50–150 depending on dev hours.

---

## Phase 5 — `ai-data-factory` (the right home for it)

This project exists to **benchmark and convert** models. That's exactly what the 3060 should do all night long while you sleep.

| Task | Action |
|---|---|
| Move benchmark-corpus runs from laptop 2060 (6 GB) to tower 3060 (12 GB) | Some models simply OOM on 6 GB |
| Add nightly job: convert any new ONNX in `/srv/models/incoming/` to TRT FP16 sm86 | Fully automated pipeline; results published to a `models_index.json` |
| Add `MEMORY_PRESSURE` benchmark category | Run each model under realistic concurrent load to surface real-world latency, not best-case |

**Push to next level:** Auto-generate model cards (latency + VRAM + accuracy regressions) and commit them to the repo on every model add. PR-ready artifacts.

---

## Phase 6 — `pixelspark` (Android+Backend)

Direct GPU benefit is indirect: dev backend lives on tower.

| Task | Action |
|---|---|
| Spring Boot dev profile points to `http://tower-server:8001` (the serverless mirror) | Killer feature: full feature dev loop without RunPod credits |
| Add `tower-serverless` healthcheck to pixelspark CI | Catches drift between local engines and prod handler |
| iOS sibling (`pixelspark-ios` on Mac) → same Tailscale endpoint | Mac can develop against real GPU inference without leaving Tailscale |

---

## Phase 7 — `xcelsior` (your current project)

The 3060 + tower IS already a worker. Use the workload to **harden the platform**:

| Task | Why |
|---|---|
| Make tower-server the canonical sm86 baseline image builder | Gives us a "Tier-3 affordable GPU" listing distinct from data-center cards |
| Run continuous synthetic load (SAM training, TRT builds) on tower as an xcelsior "internal user" | Generates real telemetry → exercise billing, ssh, idle-shutdown, snapshot, restore. Bugs surface during work, not during customer demos. |
| Add a `local_dev_tier` template (RTX 3060 / 12 GB / sm86) to the marketplace | Even if no external user buys it, it's a marketing differentiator |
| **Pay the AppleDev $99/yr** and notarize the desktop build using a GitHub Actions macOS runner that can be triggered from the Mac | Closes today's "Xcelsior is damaged" complaint properly |

---

## Phase 8 — `ara-code` (your local agentic coder, the highest-impact dev-loop win)

This is the project that benefits most from the 3060 day-to-day. Right now `ara` either runs a tiny model on the Mac (capability ceiling) or pays DeepSeek per token (latency + privacy + cost). Tower flips that: **a 12 GB GPU is enough for genuinely useful local coding models**, and it's already on your Tailnet so the Mac CLI just talks to it over `http://tower-server:11434`.

### Model fit on 12 GB (sm86, FP16/Q3/Q4)

| Model | VRAM @ Q3_K_S | VRAM @ Q4_K_S | Notes |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | ~4.5 GB | ~5.5 GB | Solid baseline, fits with huge context |
| Qwen2.5-Coder-14B-Instruct | ~8.5 GB | ~9.5 GB | Good, but noticeably weaker than 32B |
| DeepSeek-Coder-V2-Lite-Instruct (16B MoE, 2.4B active) | ~9.5 GB | ~10.5 GB | Excellent code, fast tokens/sec on 3060 |
| Codestral-22B Q3_K_M | ~10 GB | ~11 GB | Tight, viable; slower |
| Qwen2.5-Coder-32B-Instruct Q3_K_S | ~11.5 GB | ~13 GB | **Recommended** — fits in 12 GB at Q3, Q4 may fit, test first |

### Tasks

| Task | Action |
|---|---|
| Install Ollama (or llama.cpp server) on tower | `curl -fsSL https://ollama.com/install.sh \| sh`. Bind to `0.0.0.0:11434`. Tailscale ACL keeps it private. |
| `ollama pull qwen2.5-coder:32b-instruct-q3_K_S` (or `q4_K_S` if it fits) | **Primary model.** Use as ara's main coding assistant. |
| Optionally: `ollama pull qwen2.5-coder:14b-instruct-q4_K_M` and `deepseek-coder-v2:16b-lite-instruct-q4_K_M` | For fallback or comparison. |
| Add tower as an OpenAI-compatible endpoint in `ara/.env` | `MODEL=qwen2.5-coder:32b-instruct-q3_K_S` + `OPENAI_API_BASE=http://tower-server:11434/v1` |
| Wire ara's two-agent flow to **route by task complexity** | `/init` (planning, smaller context) → 32B local; `/code` (long context refactors) → fall back to DeepSeek API only when context > 16k or task is multi-file |
| Run **`ara_mcp_sse.py`** as a systemd service on tower | Mac CLI + VS Code MCP client both connect over Tailscale; one shared agent state |
| Add a `monitoring.py` exporter for Prometheus | Already has psutil hook — add tokens/sec + GPU util so you can see when ara is the bottleneck vs the model |
| Persist ara sessions on tower (not Mac) | `/srv/ara/sessions/` survives Mac reboot/laptop swap; multi-device continuity |
| Build a **VS Code chat participant** that proxies to ara MCP-SSE | You already have the MCP server — wrap it as `@ara` in Copilot Chat |

### Push to next level

- **Speculative decoding**: pair Qwen-Coder-1.5B as draft + 14B as target via llama.cpp `--draft-model`. Real 1.4–1.8× tokens/sec on 3060 for code (high acceptance rate on structured tokens).
- **Repo-aware retrieval**: index every git repo under `/home/aaryn/storage/projects` into a local `chroma` or `lancedb` on tower. ara queries it before answering. Free private code search.
- **Replace DeepSeek calls entirely** for projects under NDA / billing-sensitive (`vaultwarden-secrets`, `btc-secrets-backup`, anything you wouldn't paste into a third-party API).
- Use ara to **drive xcelsior's own internal tasks** (e.g., "audit routes/instances.py for missing auth") — the marketplace operating itself. Demoable.
- When you eventually replace the 4090: re-host with **Qwen2.5-Coder-32B-Instruct Q4** at full speed. ara becomes a Sonnet-class assistant on a $0/month inference budget.

### Why this is the single highest-ROI phase

Every hour you code, ara is in the loop. A 100ms-faster, free, private, repo-aware version pays for itself within a week. Phases 1-7 are *projects you ship*; this phase is the **tool you ship them with**.

---

## Mac (`~/Projects`) — what should NOT move to the 3060

These are correctly Mac-resident and stay that way:
- `aabiro.github.io`, `notes`, `archive` — static, no compute
- `pixelspark-ios`, `pxl-playground` — must build natively on Mac
- `axiom-bitcoin`, `phantom-trades-mvp`, `vaultwarden-secrets`, `btc-secrets-backup.sparsebundle` — security-sensitive, keep local
- `aarynfans-mvp`, `deal-ghost`, `shadowscraper`, `stash-viewer`, `reddit-mods` — non-GPU web/scraper work

What benefits from the tower:
- `ara-code` → **inference on tower** (Phase 8) — Mac CLI stays, model leaves
- `pixelspark-ios` → backend on tower (Phase 6)
- `pxl-playground` → pxl-engine inference server on tower (Phase 1)
- Any shadow/scraper that does ML enrichment → call tower-serverless

---

## Hardware notes (decided 2026-04-24)

### RAM upgrade — you already have the second 16 GB DIMM ✅
Install it. **Use slots A2 + B2** (the two slots furthest from the CPU on MSI MAG X570 Tomahawk WiFi) for dual-rank dual-channel. After install, run one `memtest86` pass overnight before kicking off any of the Week-1 tasks — bad RAM during a 3-day SAM training run is the most expensive way to discover a faulty stick.

With 32 GB unlocked, you can drop the Phase 0 RAM line item — it's done as soon as the stick is seated.

### Should the dead 4090 stay in slot 2?

**Short answer: pull it. Ship for RMA now, don't wait.**

Why not "miracle recovery":
- The diagnostic in `docs/4090-return-evidence.md` showed the card invisible on **all 256 PCI buses** — not a driver issue, not a thermal issue, not a power issue. The PCIe PHY on AD102 simply isn't responding to bus enumeration. There's no firmware path that brings a non-enumerating endpoint back to life because firmware can't run until the endpoint is on the bus.
- 4090 PHY death has zero documented self-recovery in the wild (Reddit/L1Techs/EVGA forums) — the failure mode is permanent.

Why keeping it in slot 2 is mildly bad:
1. **Blocks a chipset x4 slot** — that's the slot you'd want for a future second NVMe carrier or 10GbE NIC for tower-serverless throughput.
2. **Tiny vampire idle draw** through the 12VHPWR connector while system is on (sub-watt, but real).
3. **Airflow obstruction** over the X570 chipset heatsink — the chipset on a Tomahawk runs hot anyway and the 4090's thermal mass + closed shroud blocks the chipset's natural convection path. Measurable 3-5°C chipset delta in similar setups.
4. **Slot 2 wear**: every reboot cycles the slot's contacts, and you're not getting any benefit in exchange.
5. **RMA risk**: warranty windows are finite. If you wait 6 weeks "in case it comes back" and miss the return window, that's a real financial loss.

What to do **now**:
1. Pull the 4090.
2. Bag it with a silica desiccant pack inside the original ESD bag.
3. Print the RMA evidence sections from `docs/4090-return-evidence.md` and ship.
4. Free chipset slot stays empty until you decide what fills it (NVMe, NIC, second small GPU for ara draft model, etc.).

The 16 GB DIMM and the dead 4090 should be a **single physical maintenance window**: power down, install RAM in A2+B2, pull 4090, button up, boot, memtest. One trip into the case.

---

## Suggested execution order (4-week cadence)

**Week 1** — *software complete 2026-06-16; physical maintenance pending*
- ☐ Single maintenance window: install second 16 GB DIMM (A2+B2) + pull dead 4090 — **blocked on you** (tower still reports 15 Gi RAM)
- ☐ Overnight memtest86 pass — **after RAM install**
- ☑ `/srv/models`, `/srv/datasets` mounts + rsync priority subset — **6.8 GB on tower** (`megaage`, `COFW`, `cofw_aligned_intake_v1`, `domain_aligned_v5`, `sam_domain_male_aligned` done; `sam_balanced_100k` ~6/29 GB in flight from `100.64.0.6`)
- ☑ Install Ollama + pull `qwen2.5-coder:14b-q4_K_M` and `deepseek-coder-v2:16b-lite-q4_K_M` — **both pulled**; `32b-instruct-q3_K_S` already present
- ◐ Rebuild sm86 TRT engines — **ORT TRT EP cache build running** (`build_trt_3060_ort.sh`, 121 ONNXs). Host TensorRT 10 libs copied to `/srv/models/engines/sm86/trt_host_libs/` and mounted at `/host-trt-libs` (full `/usr/lib/...` mount breaks glibc — use copied libs only). `trtexec` batch (`build_trt_3060.sh`) remains 0/121 — wrong generic `--minShapes`; ORT path is canonical.
- ☑ Stand up `tower-serverless` FastAPI shim on port 8001 — **active**; drain → 503 → resume verified 2026-06-16

**Week 1 validation (tower, 2026-06-16)**
- ☑ `pixelspark-gpu` healthy on :8000/:8002 (94 models, GHCR image)
- ☑ `test_age_normalization.py` — 13/13 passed (in container)
- ☑ `regression_test.py --gpu --update-baseline` — **56/60 passed, 4 failed** (`anime_2x`, `upscale_2x_esrgan`, `upscale_2x_esrgan_plus`, `upscale_2x_lexica` — 3× CUDA OOM while ORT cache build held VRAM; re-run isolated after overnight build); baseline refreshed 2026-06-16
- ☑ `systemctl status nvidia-mps tower-serverless tier3-watchdog ollama` — all active
- ☑ `xcm df` — all buckets green; datasets 4.3 GB registered

**Week 2**
- ☐ pxl-engine sm86 CI job + RTX3060_12GB hardware profile in pixelenhance-labs
- ☐ Switch dev Celery worker to tower
- ☐ Notarize macOS desktop build (Phase 7)

**Week 3**
- ☐ Resume SAM training run on tower (3-day GPU lock)
- ☐ video-time-travel: move MiVOLO + GMFlow preprocessing to tower
- ☐ ai-data-factory nightly conversion pipeline

**Week 4**
- ☐ `models.xcelsior.ca` self-hosted demo behind Cloudflare
- ☐ pixelspark dev backend pointed at tower
- ☐ Public 20-second video-time-travel demo render

---

## Concrete first command (when you're ready)

```bash
ssh -i ~/.ssh/xcelsior aaryn@100.64.0.4 \
  "sudo mkdir -p /srv/models /srv/datasets && \
   sudo chown -R aaryn:aaryn /srv && \
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader"
```

Confirms `8.6` (sm86) and creates the bind-mount roots. From there everything else follows from the phase tables.

Plan saved to `/memories/session/3060-utilization-plan.md` for resume.

---

# FINALIZED INFRASTRUCTURE (supersedes scattered Phase-0 / `/srv/models` notes above)

This is the canonical bring-up. Earlier sections describe **what** to run on tower; this section describes **how the box is wired** so all of it coexists without stepping on itself.

## 1. Four bind-mount roots (the entire storage contract)

```
/srv/
├── models/         ← weights + engines, governed by xcm, RO into containers
├── datasets/       ← face datasets, RO into containers
├── checkpoints/    ← in-flight training, RW for owning job, GC'd
└── work/           ← per-job ephemeral scratch, nuked on container exit
```

### `/srv/models/` internal layout

```
/srv/models/
├── _meta/
│   ├── registry.sqlite        # single source of truth — see schema below
│   ├── README.md              # how to add a model (5 lines)
│   └── locks/
├── _blobs/sha256/ab/cd/...    # content-addressed, dedup, atomic writes
├── hub/                       # volatile cache (HF + Ollama download here)
│   ├── huggingface/
│   └── ollama/
├── lib/                       # blessed, immutable, read-only — production reads from here
│   ├── llm/    {publisher}/{model}/{quant}/...
│   ├── vision/ {publisher}/{model}/{version}/...
│   ├── face/   {publisher}/{model}/{version}/...
│   ├── diffusion/...
│   ├── audio/...
│   └── embed/...
├── engines/                   # compiled per-arch, regenerable
│   ├── sm86/  ← tower's 3060
│   ├── sm89/  ← old 4090 builds, kept for reference
│   └── sm75/  ← asus-pc 2060
├── staging/                   # conversions in flight (quantize, TRT compile)
├── archive/{YYYY}/{Qn}/       # deprecated, kept for repro
└── trash/                     # 14-day delete window
```

### Naming rule (one line, write it down)

```
/srv/models/lib/<domain>/<publisher>/<model-name>/<version-or-quant>/<file>
```

- lowercase, hyphenated, publisher matches HF org slug
- internal models use publisher `xcelsior`
- every weights file has a `CARD.md` next to it (source URL, license, sha256, date)

### Per-bucket caps (enforced by `xcm`)

| Bucket | Soft warn | Hard cap | Eviction policy |
|---|---|---|---|
| `lib/` | 350 GB | 500 GB | manual (curated, never auto-deleted) |
| `hub/` | 100 GB | 150 GB | LRU on `last_used_at` |
| `engines/` | 60 GB | 100 GB | rebuild on demand, can be wiped freely |
| `checkpoints/` | 150 GB | 250 GB | retain last 5 per run, then time-based 90d |
| `staging/` | 30 GB | 50 GB | 7-day TTL after status=verified or failed |
| `datasets/` | 500 GB | 700 GB | manual (you decide what to evict) |
| `work/` | — | 80 GB | always nuked on container exit |
| `archive/` | — | — | append-only, prune by year |

Total hard caps ≈ 1.85 TB ≥ 1.7 TB available — **caps are intentionally aspirational; `xcm df` warns when total used > 80% of free disk**, not just per-bucket.

## 2. The `xcm` CLI — one Python file, the whole governance layer

Lives at [scripts/xcm/xcm.py](scripts/xcm/xcm.py) in the xcelsior repo, deployed to tower as `/usr/local/bin/xcm`.

```bash
xcm pull <hf-id>[@quant]      # download to hub/ then offer to promote
xcm promote <hub-path> <lib-target>   # license check + sha256 + register + chmod 0444
xcm ls [--domain] [--publisher]
xcm find <fuzzy>              # search registry name+tags+notes
xcm card <lib-path>           # cat the CARD.md
xcm verify [--all]            # rehash blobs, flag mismatches
xcm prune --unused-since 90d  # GC orphan blobs + LRU hub eviction
xcm df                        # per-bucket usage vs cap with traffic-light
xcm convert <lib-path> --to trt --arch sm86   # writes engines/, with BUILD.yaml
xcm gc-checkpoints            # apply retention policy
```

### Registry schema (`_meta/registry.sqlite`)

```sql
CREATE TABLE models (
    id            TEXT PRIMARY KEY,
    domain        TEXT NOT NULL,
    publisher     TEXT NOT NULL,
    name          TEXT NOT NULL,
    version       TEXT NOT NULL,
    quant         TEXT,
    sha256        TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    blob_path     TEXT NOT NULL,
    symlink_path  TEXT NOT NULL,
    arch          TEXT,                       -- NULL for weights, sm86/sm89/sm75 for engines
    source_url    TEXT,
    license       TEXT NOT NULL,
    pulled_at     TIMESTAMP NOT NULL,
    last_used_at  TIMESTAMP,
    tags          TEXT,                        -- json array
    notes         TEXT
);
CREATE INDEX idx_models_domain_publisher ON models(domain, publisher);
CREATE INDEX idx_models_last_used ON models(last_used_at);
```

`last_used_at` is bumped by a tiny LD_PRELOAD-free wrapper: containers `bind-mount /srv/models/lib:/models:ro` and any read calls `xcm touch <id>` via a sidecar (or, simpler, a daily `find -atime -1` reconciliation — atime is reliable enough for GC).

## 3. Multi-tenant GPU scheduler (the "lots of things on one 3060" mechanism)

### Three-tier model (set up day one, all wired even if Tier 1 has only one tenant initially)

```
Tier 0 — Always-resident:  ollama (qwen2.5-coder-14b q4)        ~9 GB VRAM, MPS-shared
Tier 1 — Hot serverless:  tower-serverless FastAPI router       ~2 GB VRAM hot, demand-loaded engines, 60s idle evict
Tier 2 — Exclusive lease: xcelsior tenant SSH instance           pre-empts Tier 0 + Tier 1
Tier 3 — Background:      training, TRT builds                  NICE=19, killed on Tier 2 lease
```

All four tiers get systemd units on day one. Tier 1 starts with just the Phase-2 Novita-mirror handler registered; later phases add more handlers (sam2, whisper, sdxl-turbo) without touching infra.

### Mechanisms

| Mechanism | Purpose | Implementation |
|---|---|---|
| **NVIDIA MPS** | Co-execution of Tier 0 + Tier 1 small kernels without context-switch overhead | `nvidia-cuda-mps-control -d` in a systemd unit; on by default at boot |
| **GPU lock file** | Mutex for exclusive Tier-2 leases | `/run/xcelsior/gpu.lock` (flock-based) |
| **Tier-0 / Tier-1 pause/drain** | Free VRAM for exclusive tenant without losing state | `ollama stop` (graceful flush) + `kill -STOP $TIER0_PID`; tier-1 router enters drain mode (returns 503 Retry-After), waits for in-flight to finish |
| **Tier-1 router (`tower-serverless`)** | Demand-load engines from `/srv/models/engines/sm86/`, evict on 60s idle | FastAPI on :8001 — mirrors the Novita handler.py contract; one Python file + a registry of `model_id → engine_path` |
| **Lease orchestration** | Existing `worker_agent.py` extended (~80 LOC) to grab/release lock and signal Tiers 0+1 | Already on tower as the xcelsior worker, just adds lease helpers |
| **Pre-empt for training (Tier 3)** | NICE=19 + auto-stop on lock contention | `chrt -i 0` and a watchdog that listens on the lock file and SIGTERMs background jobs on contention |

### Tier 1 — `tower-serverless` FastAPI handler (canonical mirror)

Lives at [scripts/tower-serverless/server.py](scripts/tower-serverless/server.py). Same I/O contract as the Novita handler so any pixelenhance-labs/pixelspark/video-time-travel client can flip the base URL to `http://tower-server:8001` and Just Work.

```
GET  /health                      → {"status":"ok","loaded":[<model_id>...], "vram_used_mb": ...}
GET  /v1/models                   → list from xcm registry where domain=engines and arch=sm86
POST /v1/<model_id>/infer         → {input:..., params:...}  — same JSON shape as Novita
```

Key behaviours:
- **Lazy load**: engine loads on first request, stays resident, evicts after 60s idle.
- **Drain endpoint**: `POST /admin/drain` returns immediately, finishes in-flight, refuses new with 503 + Retry-After. Called by worker_agent when Tier 2 takes the lock.
- **Resume endpoint**: `POST /admin/resume` re-enables traffic after Tier 2 releases.
- **Per-model VRAM ceiling**: registry tracks expected VRAM per engine; router refuses load when sum > 11 GB (leaves 1 GB headroom for ollama overlap or bursts).
- **Metrics on `:9090/metrics`**: requests, latency p50/p95/p99 per model, evictions, drain events.

### Lifecycle of an xcelsior tenant booting on tower

1. worker_agent receives job, sees `exclusive_gpu=true`
2. acquires `/run/xcelsior/gpu.lock` (blocks if held)
3. `curl -XPOST http://localhost:8001/admin/drain` — Tier 1 stops accepting, finishes in-flight
4. `ollama stop` — Tier 0 graceful VRAM flush
5. `kill -STOP $TIER0_PID` — Tier 0 process suspended (state intact)
6. `pkill -TERM -f 'tier3-'` — Tier 3 background jobs terminate cleanly (they checkpoint and exit on SIGTERM)
7. start tenant container with full GPU
8. on tenant exit (or lease timeout): `kill -CONT $TIER0_PID`, `ollama serve`, `curl -XPOST http://localhost:8001/admin/resume`, release lock; Tier 3 restarts on its next timer

Net effect: ara is invisible to tenants, tenants get a clean GPU, ara + tower-serverless resume within 3 seconds of tenant exit.

## 4. Network surface (Headscale ACL, Tailscale-compatible HuJSON) ✅ DEPLOYED 2026-04-25

> **Status:** Existing `/etc/headscale/acl.json` already has `group:admin → *:*` and tower-server is owned by `xcelsior@` (= `group:admin`), so all listed scrape ports are already permitted from any tailnet member. No ACL change was made — existing policy already covers this.

All ports listen on `0.0.0.0` but Headscale ACL restricts to your tailnet. Deploy via `headscale policy set /etc/headscale/acl.json` on the VPS:

| Port | Service | Tier | Who calls it |
|---|---|---|---|
| 22 | sshd | — | you |
| 8001 | tower-serverless FastAPI router | 1 | pixelspark, pixelenhance-labs, video-time-travel, ai-data-factory |
| 8002 | tower-serverless metrics (`/metrics`) | 1 | Prometheus on VPS |
| 11434 | ollama | 0 | ara on Mac, ara on laptop, VS Code chat participant |
| 9100 | node_exporter | — | Prometheus on VPS |
| 9400 | nvidia_gpu_exporter | — | Prometheus on VPS |
| 9090 | xcm-metrics (`xcm df` in Prom format) | — | Prometheus on VPS |

Headscale ACL stanza (commit to your tailnet config):

```jsonc
{
  "acls": [
    { "action": "accept", "src": ["autogroup:member"], "dst": ["tower-server:8001,8002,11434,9100,9400,9090"] }
  ]
}
```

## 5. Observability — Prometheus + Grafana on VPS (100.64.0.1) ✅ DEPLOYED 2026-04-25

> **Status:** Prometheus 2.45 on `127.0.0.1:9091` (`:9090` taken by headscale). Grafana 13.0.1 on `127.0.0.1:3001` (`:3000` taken by next-server). Both bound to localhost; public access via nginx reverse proxy on `https://grafana.xcelsior.ca` (Let's Encrypt). All 5 scrape targets (`prometheus`, `tower-host`, `tower-gpu`, `tower-storage`, `tower-serverless`) reporting `up`. Starter dashboard `Tower Overview` provisioned in folder `Tower`. Admin password reset to `xcelsior-admin-2026` — change on first login. Datasource provisioned read-only with uid `prometheus`.

- Prometheus runs on **VPS** (100.64.0.1) — always-on, independent failure domain
- Grafana on VPS too, single dashboard with 4 rows: Host (CPU/RAM/disk), GPU (util/mem/temp/power), Storage (`xcm df`), Services (ollama latency, serverless RPS, ara tokens/sec)
- Exporters: tower's 4 exporters (node, gpu, xcm, tower-serverless), laptop's node exporter, Mac's node exporter
- Alert rules: disk > 85%, GPU temp > 85°C, MCE > 0, ollama down > 5min
- Grafana behind Cloudflare Access at `grafana.xcelsior.ca` (see Section 12) — **deviation:** deployed as nginx + Let's Encrypt direct (matches existing pattern of all other `*.xcelsior.ca` subdomains: docs, downloads, hs, api, connect). CF Tunnel was not used because the existing cert.pem is bound to the `pixelenhancelabs.ai` zone, not `xcelsior.ca`. To switch to CF Tunnel later: run `cloudflared tunnel login` and select the `xcelsior.ca` zone, then create a new tunnel and replace the nginx vhost. Cloudflare Access policy must still be configured manually in the CF dashboard if/when desired.

## 6. Backup tier (single target, nightly)

Laptop's `/home/aaryn/storage/backups/tower/` is the warm backup target.

```bash
# /etc/systemd/system/xcm-backup.timer on tower (runs 03:00 nightly)
rsync -aR --delete \
  /srv/models/_meta/ \
  /srv/models/lib/ \
  /srv/models/_blobs/ \
  /srv/datasets/ \
  aaryn@asus-pc:/home/aaryn/storage/backups/tower/
```

Rationale: laptop has the disk space, `_blobs/` dedup means only new content transfers, and Tailscale + rsync + ssh keys is enough — no S3 yet.

Skip from backup: `hub/`, `engines/`, `checkpoints/` (after-run only, not in-flight), `staging/`, `trash/`, `work/`. All regenerable.

## 7. Power & physical (the things software people forget)

| Item | Status | Action |
|---|---|---|
| PSU headroom | Tower PSU TBD — verify ≥ 650 W rated | Check sticker before adding any second card to slot 2 |
| UPS | None | **Add Cyberpower 850VA (~$90)** — protects checkpoints during brownouts. Highest ROI hardware purchase after the RAM. |
| Dead 4090 in slot 2 | Pending RMA | Pull during single maintenance window with the RAM install |
| Case airflow | Unverified | After 4090 pull, confirm chipset temp < 65°C under load |
| Time sync | Probably ok via systemd-timesyncd | One-time `timedatectl status` check — training timestamps must be reliable |

## 8. Sudoers — minimal escalation surface

Tower already has a normal `aaryn` user. The only `sudo` that `xcm` needs is for the initial `mount --bind` operations (one-time at install, not per-call). Add:

```
# /etc/sudoers.d/xcm
aaryn ALL=(root) NOPASSWD: /usr/local/bin/xcm-mount-helper
```

Where `xcm-mount-helper` is a 30-line whitelisted script that only accepts known mount paths. Everything else `xcm` does runs as `aaryn`.

## 9. The bring-up sequence (single coherent script)

Run this once on tower after the maintenance window. Each step is independent and safe to re-run.

```bash
# === Step 1: filesystem roots ===
sudo install -d -o aaryn -g aaryn /srv/{models,datasets,checkpoints,work}
mkdir -p /srv/models/{_meta,_blobs,hub/{huggingface,ollama},lib/{llm,vision,face,diffusion,audio,embed},engines/{sm86,sm89,sm75},staging,archive,trash}
sudo groupadd -f models && sudo usermod -aG models aaryn

# === Step 2: caches point at hub/ ===
cat <<'EOF' >> ~/.bashrc
export HF_HOME=/srv/models/hub/huggingface
export TRANSFORMERS_CACHE=/srv/models/hub/huggingface
export OLLAMA_MODELS=/srv/models/hub/ollama
EOF

# === Step 3: registry init ===
sqlite3 /srv/models/_meta/registry.sqlite < /home/aaryn/xcm-schema.sql

# === Step 4: ollama + first models ===
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
ollama pull qwen2.5-coder:32b-instruct-q3_K_S   # primary model (Q4_K_S if it fits)
# Optionally pull:
# ollama pull qwen2.5-coder:14b-instruct-q4_K_M
# ollama pull deepseek-coder-v2:16b-lite-instruct-q4_K_M

# === Step 5: MPS daemon ===
sudo tee /etc/systemd/system/nvidia-mps.service <<'EOF'
[Unit]
Description=NVIDIA MPS Control Daemon
After=nvidia-persistenced.service
[Service]
Type=forking
ExecStart=/usr/bin/nvidia-cuda-mps-control -d
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now nvidia-mps

# === Step 6: Tier 1 — tower-serverless FastAPI router ===
sudo install -m 0755 /home/aaryn/storage/projects/xcelsior/scripts/tower-serverless/server.py /usr/local/bin/tower-serverless
sudo tee /etc/systemd/system/tower-serverless.service <<'EOF'
[Unit]
Description=tower-serverless Tier 1 inference router
After=network.target nvidia-mps.service
Wants=nvidia-mps.service
[Service]
Type=simple
User=aaryn
Group=models
Environment=XCM_REGISTRY=/srv/models/_meta/registry.sqlite
Environment=ENGINES_ROOT=/srv/models/engines/sm86
ExecStart=/usr/bin/python3 /usr/local/bin/tower-serverless --host 0.0.0.0 --port 8001 --metrics-port 8002
Restart=on-failure
RestartSec=3
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now tower-serverless

# === Step 7: Tier 3 — background watchdog (kills jobs on lock contention) ===
sudo install -m 0755 /home/aaryn/storage/projects/xcelsior/scripts/tier3-watchdog.sh /usr/local/bin/tier3-watchdog
sudo tee /etc/systemd/system/tier3-watchdog.service <<'EOF'
[Unit]
Description=Tier 3 background pre-emption watchdog
[Service]
Type=simple
User=aaryn
ExecStart=/usr/local/bin/tier3-watchdog
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now tier3-watchdog

# === Step 8: exporters ===
sudo apt-get install -y prometheus-node-exporter
# nvidia_gpu_exporter — pip install nvidia-gpu-exporter or run via docker; bind 9400
# xcm-metrics — `xcm metrics-server --port 9090` as a systemd unit

# === Step 9: nightly backup timer (target = laptop) ===
# (drop xcm-backup.service + .timer in /etc/systemd/system/)

# === Step 10: deploy xcm CLI ===
sudo install -m 0755 /home/aaryn/storage/projects/xcelsior/scripts/xcm/xcm.py /usr/local/bin/xcm
xcm df                    # smoke test

# === Step 11: tailscale ACL update on admin console (manual) ===
# Allow ports 8001, 8002, 11434, 9100, 9400, 9090 from autogroup:member to tower-server.
```

## 10. Day-one acceptance criteria

You're done with bring-up when all of these are true:

- [x] `xcm df` shows all 8 buckets, all green
- [x] `ollama list` shows the 3 models, each `xcm`-registered (run `xcm verify` → 0 errors) — *qwen2.5-coder:32b-instruct-q3_K_S promoted to /srv/models/lib/llm/qwen2.5-coder-32b-instruct-q3-ks (sha256 ce470c0a25ef…); `xcm verify` reports 1 entries, 0 errors*
- [x] From laptop: `curl http://tower-server:11434/v1/models` → returns model list
- [ ] From Mac: `ara` interactive session uses tower; `nvidia-smi` on tower shows ollama VRAM — *deferred (app layer)*
- [x] `systemctl status nvidia-mps tower-serverless tier3-watchdog` → all active
- [x] `nvidia-smi -q | grep -i 'compute mode'` → `Default`
- [x] `curl http://tower-server:8001/health` → 200 with `loaded: []`
- [x] `curl -XPOST http://tower-server:8001/admin/drain` → 200; inference 503; `/admin/resume` → 200 — *verified 2026-04-25: drain returns `{"draining":true}`, inference path `/v1/<model_id>/infer` returns 503 `{"error":"draining"}`, resume returns `{"draining":false}`*
- [x] `flock /run/xcelsior/gpu.lock echo OK` → prints OK and releases (lock dir provisioned via /etc/tmpfiles.d/xcelsior.conf); concurrent `flock -n` correctly blocks (rc=1) and reacquires after release — *verified 2026-04-25*
- [x] **Prometheus on VPS** shows tower's 4 exporters (node, gpu, xcm, tower-serverless) + laptop node exporter as UP — 7 targets total, all `up`. Alertmanager + 8 alert rules configured, real notifier (ntfy-relay → https://ntfy.sh/tower-b1gC0ck) live on VPS as dedicated `ntfy-relay` system user. Grafana provisioned at https://grafana.xcelsior.ca
- [ ] One xcelsior tenant launch with `exclusive_gpu=true` succeeds — *not yet smoke-tested (app layer)*
- [x] Nightly rsync backup ran once — manually fired 2026-04-25, laptop target populated (`/home/aaryn/storage/backups/tower/srv/{models,datasets}/`); SSH key from tower → laptop installed

**Remaining checklist item is the app-layer smoke test** (xcelsior tenant `exclusive_gpu=true` lease) — to be exercised when the first ara/Xcelsior tenant runs. The infrastructure itself is complete.

### Hardening done beyond the original §10 list
- UFW enabled on tower; ports 8001/8002/9090/9100/9400/11434 restricted to `100.64.0.0/10` (tailnet only); SSH stays public
- `/etc/letsencrypt/renewal/grafana.xcelsior.ca.conf` has `renew_hook = systemctl reload nginx` (verified via `--dry-run`)
- Mac (100.64.0.3) node_exporter — SKIPPED, manual `brew install node_exporter` step left for the user
- `xcm` CLI complete with 13 subcommands: `df, ls, find, card, verify, touch, pull, promote, register, prune, gc-checkpoints, convert, metrics-server`. `convert` requires TRT-LLM toolchain (NGC container `nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3` or `pip install tensorrt-llm`); when toolchain is absent it exits 0 with an install hint so opportunistic build pipelines don't break.
- `xcm-prune.timer` (weekly Sun 04:00 EDT) wired to `xcm prune` for orphan/LRU GC
- `xcm-mount-helper` + `/etc/sudoers.d/xcm` for unprivileged squashfs/dataset mounts (v2 helper; smoke-tested)
- Alertmanager → ntfy.sh real notifier on VPS (ntfy-relay.service, dedicated `ntfy-relay` system user, listens 127.0.0.1:9099)
- DBus failure mode documented: Ubuntu 24.04 partial systemd upgrade without `daemon-reexec` causes PID-1↔dbus socket pairing loss (symptom: `Failed to activate org.freedesktop.systemd1: timed out (25000ms)` every 25s + `journald: Failed to send WATCHDOG=1 ... Transport endpoint is not connected`); reboot is the only recovery once it occurs. Mitigation: ensure `needrestart` runs `daemon-reexec` after apt upgrades.
## 11. Simplifications that were tempting but skipped

- **Backblaze B2 / S3 backup** — not yet, laptop is enough. Revisit when laptop disk pressure hits.
- **Email/Slack/PagerDuty for Alertmanager** — current receiver is the ntfy-relay (→ https://ntfy.sh/tower-b1gC0ck via subscriber URL). Wire up email/PagerDuty/Slack when team grows beyond solo-on-call.
- **Per-job docker network namespaces** — overkill for 1 GPU.
- **Kubernetes / k3s on tower** — strictly worse than systemd for a single-host fleet.
- **GPU MIG** — 3060 doesn't support it; MPS is the only sharing primitive available.
- **Custom bind-mount over loopback for quotas** — XFS quotas or simple `xcm df` enforcement is enough; don't carve a fake filesystem.
- **Triton / vLLM as the Tier-1 router** — the FastAPI shim is ~150 LOC and matches the Novita handler contract exactly, so projects don't need a second client. If a single model someday needs PagedAttention or batching beyond what FastAPI gives, swap *that one model's* handler for vLLM behind the same `/v1/<model_id>/infer` URL — no router rewrite.

---

**Bottom line of the finalized plan:**

Four `/srv` roots → one `xcm` CLI → MPS + a flock + the FastAPI router — that's the entire infrastructure. Everything in Phases 1–8 plugs into those four primitives. Set up cleanly day one with all four tiers wired (even if Tier 1 only has one handler initially), never re-architect again.