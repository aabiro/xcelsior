# Xcelsior

> **Sovereign GPU compute for Canada. Ever upward.**

Xcelsior is a distributed GPU scheduling platform with Canadian data sovereignty, 4-layer container security, and CAD-native billing. It routes compute jobs to admitted GPU hosts over a private Headscale mesh, enforces PIPEDA/Quebec Law 25 compliance, and manages the full lifecycle from host registration through billing reconciliation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

---

## Table of Contents

- [Architecture](#architecture)
- [Modules](#modules)
- [Host GPU Registration Flow](#host-gpu-registration-flow)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration (.env)](#configuration)
- [Database & Migrations](#database--migrations)
- [Deployment](#deployment)
  - [Local Development](#local-development)
  - [Test Environment](#test-environment)
  - [Production (xcelsior.ca)](#production-xcelsiorca)
- [Worker Agent](#worker-agent)
- [Dashboard](#dashboard)
- [CLI](#cli)
- [API Endpoints](#api-endpoints)
- [Security Model](#security-model)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Architecture

```
                         ┌──────────────────────────────────────────────────────┐
                         │              xcelsior.ca  (Cloudflare → VPS)        │
                         │  Nginx → uvicorn → api.py (FastAPI, ~50 endpoints)  │
                         └───────────────┬──────────────────────────────────────┘
                                         │
               ┌─────────────────────────┼─────────────────────────┐
               │                         │                         │
    ┌──────────▼──────────┐   ┌──────────▼──────────┐   ┌─────────▼─────────┐
    │   PostgreSQL 16     │   │   Headscale Mesh     │   │   Dashboard       │
    │  (core: hosts/jobs) │   │   (private overlay)  │   │  (6-tab SPA)      │
    │  + 4 SQLite DBs     │   │   100.x.y.z ↔ hosts  │   │  SSE live stream  │
    └─────────────────────┘   └──────────┬───────────┘   └───────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
   ┌──────────▼──────────┐   ┌──────────▼──────────┐   ┌──────────▼──────────┐
   │  Worker Agent (GPU)  │   │  Worker Agent (GPU)  │   │  Worker Agent (GPU)  │
   │  admit_node() → ✓    │   │  admit_node() → ✓    │   │  admit_node() → ✓    │
   │  telemetry @ 5s      │   │  telemetry @ 5s      │   │  telemetry @ 5s      │
   │  gVisor / runc        │   │  gVisor / runc        │   │  gVisor / runc        │
   │  egress firewall      │   │  egress firewall      │   │  egress firewall      │
   └───────────────────────┘   └───────────────────────┘   └───────────────────────┘
```

---

## Modules

| File | Purpose | Lines |
|------|---------|-------|
| `api.py` | FastAPI app, ~50 endpoints, SSE stream, dashboard | ~1960 |
| `scheduler.py` | Job queue, host registry, allocation, spot pricing, preemption | ~2850 |
| `worker_agent.py` | Pull-based GPU agent, telemetry, mining detection, Docker execution | ~1500 |
| `security.py` | 4-layer defense: version gating, least-privilege Docker, egress rules, gVisor/Kata | ~510 |
| `billing.py` | CAD pricing, 13-province GST/HST, wallets, escrow, invoices, CAF export | ~1035 |
| `events.py` | Append-only event store, tamper-evident hashing, lease state machine | ~636 |
| `verification.py` | GPU fingerprint verification, re-verification scheduling | ~577 |
| `jurisdiction.py` | Trust tiers (Community→Regulated), residency traces, CLOUD Act analysis | ~400 |
| `artifacts.py` | Two-tier B2/R2 artifact storage, residency-aware routing, presigned URLs | ~500 |
| `reputation.py` | Multi-factor scoring, 7-day grace decay, Bronze→Platinum tiers | ~675 |
| `privacy.py` | PIPEDA/Quebec Law 25, data retention, consent, PIA triggers | ~597 |
| `db.py` | SQLite ↔ PostgreSQL dual-write, connection pooling, LISTEN/NOTIFY | ~760 |
| `inference.py` | Optional local ML inference (torch/transformers) | ~200 |
| `cli.py` | Full CLI for jobs, hosts, billing, marketplace | ~300 |

---

## Host GPU Registration Flow

Hosts go through a strict admission-gated registration process:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HOST REGISTRATION & ADMISSION FLOW                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. PUT /host  ─────────────────────────────────────────────────────    │
│     • Receives: host_id, ip, gpu_model, vram, cost, country, province  │
│     • Sets: status = "pending", admitted = False                        │
│     • Host CANNOT receive work yet                                      │
│                                                                         │
│  2. Worker agent starts → gathers local versions ──────────────────    │
│     • CUDA driver, runc, nvidia-ctk, Docker versions                    │
│     • GPU model, available runtimes (gVisor/Kata/runc)                  │
│                                                                         │
│  3. POST /agent/versions  ─────────────────────────────────────────    │
│     • Server calls admit_node() from security.py:                       │
│       Layer 1: check_node_versions()                                    │
│         - runc ≥ 1.1.14, nvidia_ctk ≥ 1.17.8, docker ≥ 24.0.0        │
│         - nvidia_driver ≥ 535.0                                         │
│       Layer 2: recommend_runtime()                                      │
│         - Check GPU against GVISOR_SUPPORTED_GPUS set                   │
│         - Recommend runsc (gVisor) or fallback to runc                  │
│     • If ALL checks pass:                                               │
│       → admitted = True, status = "active"                              │
│       → Host enters the allocation pool                                 │
│     • If ANY check fails:                                               │
│       → admitted = False, status = "pending"                            │
│       → Rejection reasons logged, host excluded from scheduling         │
│                                                                         │
│  4. Scheduler allocate() — 3-step filtering  ─────────────────────    │
│     Step 1: VRAM capacity check                                         │
│     Step 2: Admission gate (admitted == True only)                      │
│     Step 3: Isolation tier enforcement                                  │
│       - sovereign/regulated/secure → prefer gVisor/Kata runtime         │
│       - community → any runtime (runc OK)                               │
│                                                                         │
│  5. Ongoing: telemetry_loop() pushes every 5s ────────────────────    │
│     • gpu_utilization, temperature, memory, power_draw, ecc_errors      │
│     • Mining detection (sustained >95% util + <5MB PCIe TX)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key rules:**
- A host with no version report stays `pending` indefinitely — zero work assigned
- Version minimums are enforced server-side (not trusting the client)
- The allocator double-checks `admitted=True` before every job assignment
- Sovereignty/regulated tiers require hardened runtime (gVisor preferred)
- Mining detection can trigger immediate job termination + reputation penalty

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/aabiro/xcelsior.git
cd xcelsior
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Start API (dev mode)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# In another terminal — start worker agent
export XCELSIOR_HOST_ID=my-gpu-01
export XCELSIOR_SCHEDULER_URL=http://localhost:8000
python worker_agent.py

# Submit a job
python cli.py run my-model 8.0

# Visit dashboard
open http://localhost:8000/dashboard
```

---

## Installation

### Prerequisites

| Component | Scheduler Node | GPU Worker Node |
|-----------|---------------|-----------------|
| OS | Ubuntu 22.04+ / Debian 12+ | Ubuntu 22.04+ / Debian 12+ |
| Python | 3.12+ | 3.12+ |
| PostgreSQL | 16+ (for production) | Not required |
| Docker | Optional | 24.0+ required |
| NVIDIA Driver | Not required | 535.0+ required |
| NVIDIA CTK | Not required | 1.17.8+ required |
| runc | Not required | 1.1.14+ required |
| gVisor (runsc) | Not required | Recommended for sovereign tier |

### Install

```bash
git clone https://github.com/aabiro/xcelsior.git
cd xcelsior
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — at minimum set XCELSIOR_API_TOKEN
```

---

## Configuration

All configuration is via environment variables. See `.env.example` for the complete reference.

### Critical Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `XCELSIOR_API_TOKEN` | **Yes** (prod) | `""` | Bearer token for all API auth |
| `XCELSIOR_ENV` | No | `dev` | `dev`, `test`, or `production` |
| `XCELSIOR_DB_BACKEND` | No | `sqlite` | `sqlite`, `postgres`, or `dual` |
| `XCELSIOR_DB_PATH` | No | `./xcelsior.db` | SQLite database path |
| `XCELSIOR_POSTGRES_DSN` | Prod | see `.env.example` | PostgreSQL connection string |

### Worker Agent Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `XCELSIOR_HOST_ID` | **Yes** | — | Unique host identifier |
| `XCELSIOR_SCHEDULER_URL` | **Yes** | — | API server URL |
| `XCELSIOR_COST_PER_HOUR` | No | `0.50` | Host hourly rate (CAD) |
| `XCELSIOR_PREFER_GVISOR` | No | `true` | Use gVisor runtime when available |
| `XCELSIOR_HEARTBEAT_INTERVAL` | No | `10` | Heartbeat frequency (seconds) |
| `XCELSIOR_POLL_INTERVAL` | No | `5` | Work poll frequency (seconds) |
| `XCELSIOR_TELEMETRY_INTERVAL` | No | `5` | GPU telemetry push frequency |

### Networking / Headscale

| Variable | Default | Purpose |
|----------|---------|---------|
| `XCELSIOR_TAILSCALE_ENABLED` | `false` | Enable Headscale mesh enrollment |
| `XCELSIOR_TAILSCALE_AUTHKEY` | `""` | Pre-auth key for mesh join |
| `XCELSIOR_HEADSCALE_URL` | `""` | Headscale coordination server URL |

### Canadian Billing

| Variable | Default | Purpose |
|----------|---------|---------|
| `XCELSIOR_CAD_USD_RATE` | `0.73` | CAD/USD exchange rate |
| `XCELSIOR_PLATFORM_CUT` | `0.20` | Marketplace platform fee (20%) |
| `XCELSIOR_CANADA_ONLY` | `false` | Restrict to Canadian hosts only |

### Artifact Storage (B2/R2)

Variables prefixed with `XCELSIOR_STORAGE_` (primary) and `XCELSIOR_CACHE_` (edge cache):

| Suffix | Default | Purpose |
|--------|---------|---------|
| `_BACKEND` | `local` | `s3` for B2/R2/S3 |
| `_ENDPOINT_URL` | `""` | S3-compatible endpoint |
| `_BUCKET` | `""` | Bucket name |
| `_REGION` | `""` | Region |
| `_ACCESS_KEY_ID` | `""` | Access key |
| `_SECRET_ACCESS_KEY` | `""` | Secret key |
| `_RESIDENCY` | `CA` | Data residency constraint |

See `.env.example` for the complete list of 50+ variables.

---

## Database & Migrations

### Database Architecture

Xcelsior uses **PostgreSQL 16** for core state (hosts, jobs) with a **dual-write** mode that mirrors to SQLite for offline resilience. Auxiliary modules use dedicated SQLite databases:

| Database | Backend | Tables | Created By |
|----------|---------|--------|------------|
| Core (hosts, jobs, state) | Postgres + SQLite | `state`, `jobs`, `hosts` | `db.py` |
| Events & leases | SQLite | `events`, `leases` | `events.py` |
| Verification | SQLite | `host_verifications`, `verification_history`, `job_failure_log` | `verification.py` |
| Billing | SQLite | `usage_meters`, `invoices`, `payout_ledger`, `wallets`, `wallet_transactions` | `billing.py` |
| Reputation | SQLite | `reputation_scores`, `reputation_events` | `reputation.py` |
| Privacy | SQLite | `retention_records`, `consent_records`, `privacy_configs` | `privacy.py` |
| Postgres-only | Postgres | `spot_prices`, `node_versions`, `benchmarks` | Alembic migrations |

**Total: 18 tables across 5 SQLite databases + PostgreSQL.**

### Running Migrations

```bash
# Ensure PostgreSQL is running and reachable at XCELSIOR_POSTGRES_DSN
pg_isready -d "$XCELSIOR_POSTGRES_DSN"

# Run Alembic migrations
alembic upgrade head

# Check current revision
alembic current

# Create a new migration
alembic revision --autogenerate -m "description"
```

### Current Migrations

| Revision | Description |
|----------|-------------|
| `001` | Initial schema: `state`, `jobs`, `hosts` tables with JSONB + GIN indexes |
| `002` | Spot pricing (`spot_prices`), node security (`node_versions`), benchmarks |

### Auto-initialization

All SQLite tables are created automatically on first access — no migration needed. PostgreSQL core tables are also auto-created by `db.py` if they don't exist, but Alembic is preferred for production.

---

## Deployment

### Local Development

```bash
# .env
XCELSIOR_ENV=dev
XCELSIOR_DB_BACKEND=sqlite

# Start
uvicorn api:app --reload --port 8000
```

No token required in dev mode. SQLite only. Hot reload enabled.

### Test Environment

For integration testing with PostgreSQL:

```bash
# Start a PostgreSQL 16 instance reachable at localhost:5432
# (local service, Docker container, or CI service container)

# .env
XCELSIOR_ENV=test
XCELSIOR_DB_BACKEND=dual
XCELSIOR_POSTGRES_DSN=postgresql://xcelsior:xcelsior@localhost:5432/xcelsior
XCELSIOR_DUAL_READ_FROM=sqlite

# Run migrations
alembic upgrade head

# Start API
uvicorn api:app --port 8000

# Run test suite
python -m pytest tests/test_scheduler.py tests/test_api.py tests/test_integration.py tests/test_worker_agent.py -v
```

Use `scripts/init-test-db.sh` as a `docker-entrypoint-initdb.d` hook if you spin up a disposable Postgres container for tests.

### Production (xcelsior.ca)

**Infrastructure:**
- **VPS**: 149.28.121.61 (Ubuntu 24.04) — runs API, PostgreSQL, Headscale
- **Domains**: xcelsior.ca, www.xcelsior.ca, hs.xcelsior.ca
- **Headscale**: v0.28.0 at `hs.xcelsior.ca` — private mesh for GPU workers
- **SSL**: Let's Encrypt terminates on the VPS; keep `hs.xcelsior.ca` DNS-only and proxy the public app endpoints separately if desired

**Deploy with Docker Compose:**

```bash
# On VPS
ssh linuxuser@149.28.121.61
cd /opt/xcelsior

# Set production secrets in .env
XCELSIOR_ENV=production
XCELSIOR_API_TOKEN=<generate: python -c "import secrets; print(secrets.token_urlsafe(32))">
XCELSIOR_POSTGRES_DSN=postgresql://xcelsior:<strong-password>@localhost:5432/xcelsior
XCELSIOR_DB_BACKEND=postgres  # use dual only during a live SQLite -> Postgres migration

# Apply schema changes against the host Postgres instance
alembic upgrade head

# Deploy
docker compose up --build -d

# Check health
curl https://xcelsior.ca/healthz
```

**Systemd services (alternative to Docker):**

```bash
sudo cp xcelsior-api.service /etc/systemd/system/
sudo cp xcelsior-worker.service /etc/systemd/system/
sudo cp xcelsior-health.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now xcelsior-api xcelsior-health
```

**Nginx reverse proxy:**

Install the checked-in site files:

```bash
sudo cp nginx/xcelsior.conf /etc/nginx/sites-available/xcelsior
sudo cp nginx/headscale.conf /etc/nginx/sites-available/headscale
sudo cp nginx/headscale-http.conf /etc/nginx/sites-available/headscale-http
sudo ln -sf /etc/nginx/sites-available/xcelsior /etc/nginx/sites-enabled/xcelsior
sudo ln -sf /etc/nginx/sites-available/headscale /etc/nginx/sites-enabled/headscale
sudo ln -sf /etc/nginx/sites-available/headscale-http /etc/nginx/sites-enabled/headscale-http
sudo nginx -t && sudo systemctl reload nginx
```

`nginx/xcelsior.conf` proxies the API to `127.0.0.1:9500`. `nginx/headscale.conf` proxies `hs.xcelsior.ca` to `127.0.0.1:8443` with its own certificate at `/etc/letsencrypt/live/hs.xcelsior.ca/`.

**DNS (pending Cloudflare propagation):**

| Record | Name | Value |
|--------|------|-------|
| A | xcelsior.ca | 149.28.121.61 (proxied) |
| A | hs.xcelsior.ca | 149.28.121.61 (DNS-only, no proxy) |
| CNAME | www | xcelsior.ca |

---

## Worker Agent

The worker agent runs on each GPU host and handles the complete lifecycle:

### Startup Sequence

1. **Detect GPU** via `nvidia-smi`
2. **Register** with scheduler (`PUT /host` — starts as `pending`)
3. **Report versions** (`POST /agent/versions` — triggers admission)
4. **If admitted**: enter work loop (poll for jobs, execute in Docker, report telemetry)
5. **If rejected**: log reasons, retry after updating software

### Running

```bash
# Required
export XCELSIOR_HOST_ID=gpu-rig-01
export XCELSIOR_SCHEDULER_URL=https://xcelsior.ca
export XCELSIOR_API_TOKEN=your_token

# Optional
export XCELSIOR_COST_PER_HOUR=0.45
export XCELSIOR_PREFER_GVISOR=true

# Start
python worker_agent.py
```

### Security Layers (applied per-job)

1. **Version gating** — outdated runc/driver/CTK → no work
2. **Least-privilege Docker** — `--no-new-privileges`, cap-drop ALL, read-only rootfs, tmpfs, non-root
3. **Egress firewall** — iptables allowlist (PyPI, HuggingFace, Docker Hub only)
4. **Runtime isolation** — gVisor (`runsc`) for supported GPUs, hardened `runc` fallback

---

## Dashboard

The web dashboard at `/dashboard` provides a 6-tab SPA:

| Tab | Features |
|-----|----------|
| **Overview** | Host/job tables, marketplace, autoscale pool, billing, spot prices, XCU scores |
| **GPU Telemetry** | Live per-host GPU gauges (utilization, temp, memory, power, ECC errors), 5s auto-refresh |
| **Trust & Verification** | Host verification status, reputation leaderboard, trust tiers, jurisdiction overview |
| **Billing & Fund** | CAD reference pricing, cost estimator, AI Compute Fund attestation |
| **My Earnings** | Provider earnings by host, payout history, utilization rates |
| **Event Log** | Full SSE-powered event stream |

**Interactive elements:**
- **+ Add Host** button → registration form modal (host starts as pending)
- **+ Submit Job** button → job submission with tier selection
- **Export Rebate Docs** → CAF CSV download
- **First-visit onboarding** → role selection (GPU Provider / Job Submitter / Admin)

---

## CLI

```bash
# Jobs
python cli.py run MODEL VRAM_GB [--tier standard] [--priority 5]
python cli.py jobs [--status running]
python cli.py cancel JOB_ID
python cli.py process

# Hosts
python cli.py host-add --id ID --ip IP --gpu MODEL --vram GB --free-vram GB --rate RATE
python cli.py host-rm HOST_ID
python cli.py hosts [--all]
python cli.py check

# Billing
python cli.py bill JOB_ID
python cli.py bill-all
python cli.py revenue
python cli.py billing

# Marketplace
python cli.py list-rig --host ID --rate 0.75 --description "RTX 4090"
python cli.py unlist-rig --host ID
python cli.py marketplace

# Operations
python cli.py failover --host HOST_ID
python cli.py requeue --job JOB_ID
python cli.py autoscale
python cli.py canada-only --enable
```

---

## API Endpoints

**Base URL:** `https://xcelsior.ca` (production) or `http://localhost:8000` (dev)

**Auth:** `Authorization: Bearer <XCELSIOR_API_TOKEN>` (required when `XCELSIOR_ENV != dev`)

### Core

| Method | Path | Description |
|--------|------|-------------|
| `PUT` | `/host` | Register/update GPU host (admission-gated) |
| `DELETE` | `/host/{id}` | Remove host |
| `GET` | `/hosts` | List hosts (`?active_only=true`) |
| `POST` | `/hosts/check` | Health check all hosts |
| `POST` | `/job` | Submit compute job |
| `GET` | `/jobs` | List jobs (`?status=running`) |
| `GET` | `/job/{id}` | Job details |
| `PATCH` | `/job/{id}/status` | Update job status |
| `POST` | `/queue/process` | Process job queue |

### Billing & Marketplace

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/billing/bill-all` | Bill all completed jobs |
| `GET` | `/billing` | Billing records |
| `GET` | `/billing/revenue` | Total revenue |
| `POST` | `/marketplace/list` | List GPU on marketplace |
| `GET` | `/marketplace` | Browse listings |
| `GET` | `/spot-prices` | Current spot prices |

### v2.0 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/agent/versions` | Report agent versions → triggers admission |
| `POST` | `/agent/telemetry` | Push GPU telemetry |
| `GET` | `/api/telemetry/all` | All host telemetry |
| `GET` | `/api/verified-hosts` | Verification statuses |
| `GET` | `/api/reputation/leaderboard` | Reputation rankings |
| `GET` | `/api/trust-tiers` | Trust tier definitions |
| `GET` | `/api/pricing/reference` | CAD GPU pricing |
| `POST` | `/api/pricing/estimate` | Cost estimator |
| `GET` | `/api/billing/attestation` | AI Compute Fund attestation |
| `GET` | `/api/billing/export/caf/{id}` | Export CAF rebate CSV |
| `GET` | `/api/transparency/report` | Transparency report |
| `GET` | `/api/stream` | SSE event stream |
| `GET` | `/healthz` | Liveness probe |
| `GET` | `/readyz` | Readiness probe |
| `GET` | `/metrics` | Scheduler metrics |
| `GET` | `/dashboard` | Web dashboard |

---

## Security Model

### 4-Layer Defense-in-Depth (`security.py`)

| Layer | What | Enforcement Point |
|-------|------|-------------------|
| **1. Version Gating** | Min versions: runc ≥1.1.14, CTK ≥1.17.8, Docker ≥24.0, driver ≥535 | `POST /agent/versions` |
| **2. Least-Privilege Docker** | `--no-new-privileges`, cap-drop ALL, read-only rootfs, tmpfs /tmp, non-root user | `worker_agent.py` job execution |
| **3. Egress Firewall** | iptables allowlist: PyPI, HuggingFace, Docker Hub, apt repos | `worker_agent.py` job execution |
| **4. Runtime Isolation** | gVisor (`--runtime=runsc`) for supported GPUs, hardened runc fallback | `worker_agent.py` + `allocate()` tier check |

### Admission Enforcement

- `PUT /host` → sets `admitted=False, status=pending`
- `POST /agent/versions` → runs `admit_node()` → updates `admitted, status`
- `allocate()` → filters to `admitted=True` hosts only
- Sovereignty/regulated tiers → prefer gVisor/Kata runtime hosts

### Mining Detection

Worker agents run continuous heuristic detection:
- GPU utilization >95% sustained + PCIe TX <5MB/s for >5min → alert
- Configurable via `XCELSIOR_MINING_*` env vars

---

## Testing

```bash
# All tests
python -m pytest tests/test_scheduler.py tests/test_api.py tests/test_integration.py tests/test_worker_agent.py -v

# With coverage
python -m pytest -v --cov=. --cov-report=html

# Specific module
python -m pytest tests/test_scheduler.py -v
python -m pytest tests/test_api.py -v

# Lint
ruff check .
black --check .
```

See `docs_phase_plan.md` for the comprehensive phased testing plan.

---

## Troubleshooting

### Host stuck in "pending"

```bash
# Check agent versions
curl -s http://localhost:8000/hosts?active_only=false | python -m json.tool

# Verify minimum versions on the GPU host
nvidia-smi --query-gpu=driver_version --format=csv,noheader  # needs ≥535
runc --version                                                 # needs ≥1.1.14
nvidia-ctk --version                                           # needs ≥1.17.8
docker --version                                               # needs ≥24.0
```

### "401 Unauthorized"

```bash
# Generate token
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Set in .env: XCELSIOR_API_TOKEN=<token>
# Use: curl -H "Authorization: Bearer <token>" http://localhost:8000/hosts
```

### Worker agent can't reach scheduler

```bash
curl -s $XCELSIOR_SCHEDULER_URL/healthz
# If using Headscale: tailscale status
```

### Docker GPU access fails

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
# If fails: sudo systemctl restart docker && sudo nvidia-ctk runtime configure
```

---

## Roadmap

See `docs_phase_plan.md` for the complete phased plan including:
- Phase 1–6: Reliability, persistence, auth, observability, deployment, operator UX (DONE)
- Phase 7: Comprehensive testing plan (current)
- Future: Stripe Connect, SLA enforcement, NFS support, Slurm adapter

---

## License

MIT License. See [LICENSE](LICENSE).

---

**Built in Canada. 🍁 Ever upward. Xcelsior.**
