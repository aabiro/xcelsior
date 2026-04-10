<div align="center">

# 🍁 Xcelsior

### Sovereign GPU Compute for Canada

**Route AI workloads to admission-gated GPU hosts over a private mesh —
with PIPEDA compliance, 4-layer container security, and CAD-native billing.**

[![CI](https://github.com/aabiro/xcelsior/actions/workflows/ci.yml/badge.svg)](https://github.com/aabiro/xcelsior/actions/workflows/ci.yml)
[![Frontend CI](https://github.com/aabiro/xcelsior/actions/workflows/frontend.yml/badge.svg)](https://github.com/aabiro/xcelsior/actions/workflows/frontend.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript SDK](https://img.shields.io/badge/SDK-TypeScript-3178C6.svg)](https://docs.xcelsior.ca)
[![Docs](https://img.shields.io/badge/docs-xcelsior.ca-0891b2.svg)](https://docs.xcelsior.ca)

[Website](https://xcelsior.ca) · [API Docs](https://docs.xcelsior.ca) · [Blog](https://xcelsior.ca/blog) · [Pricing](https://xcelsior.ca/pricing)

</div>

---

## Architecture

```mermaid
graph TB
    subgraph clients["Clients"]
        dashboard["Dashboard · Next.js"]
        sdk["TypeScript SDK"]
        cli["CLI"]
    end

    subgraph platform["Xcelsior Platform"]
        api["FastAPI Gateway · 250+ endpoints"]
        scheduler["Scheduler · Job matching & compliance"]
        db[("PostgreSQL 16 + SQLite auxiliaries")]
        billing["Billing · CAD · Stripe Connect · BTC"]
        reputation["Reputation · 6-tier trust scoring"]
    end

    subgraph hosts["GPU Host Network"]
        mesh["Headscale Mesh · Private Overlay"]
        worker1["Worker Agent · RTX 4090"]
        worker2["Worker Agent · A100 80 GB"]
        worker3["Worker Agent · H100 80 GB"]
    end

    dashboard & sdk & cli -->|HTTPS| api
    api --> scheduler
    api --> db
    api --> billing
    api --> reputation
    scheduler -->|SSH over mesh| mesh
    mesh --- worker1 & worker2 & worker3
```

---

## Features

| | |
|---|---|
| **Data Sovereignty** | All compute stays in Canada. PIPEDA + Quebec Law 25 enforced at the scheduler level. |
| **4-Layer Security** | Version gating → least-privilege Docker → egress firewall → gVisor / Kata sandbox. |
| **CAD-Native Billing** | 13-province GST/HST, Stripe Connect payouts, AI Compute Fund (CAF) rebate export. |
| **Reputation Engine** | Multi-factor scoring with 7-day grace decay. Bronze → Silver → Gold → Platinum → Diamond → Sovereign tiers. |
| **Private Mesh** | Headscale overlay network — GPU workers never exposed to the public internet. |
| **Admission Gating** | Hosts must pass version checks + GPU fingerprinting before receiving any work. |

---

## Quick Start

```bash
git clone https://github.com/aabiro/xcelsior.git && cd xcelsior
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && cp .env.example .env

# API
uvicorn api:app --reload --port 8000

# Worker (separate terminal)
XCELSIOR_HOST_ID=my-gpu XCELSIOR_SCHEDULER_URL=http://localhost:8000 python worker_agent.py

# Submit a job
python cli.py run my-model 8.0
```

> **Dashboard** → `http://localhost:8000/dashboard`
> **Health** → `http://localhost:8000/healthz`
> **Worker Auth** → hosted/prod workers can use either `XCELSIOR_API_TOKEN` or `XCELSIOR_OAUTH_CLIENT_ID` + `XCELSIOR_OAUTH_CLIENT_SECRET`. If both are set, the worker prefers OAuth.

---

## Host Admission Flow

```mermaid
sequenceDiagram
    participant H as GPU Host
    participant A as API Gateway
    participant S as Security Engine

    H->>A: PUT /host (gpu_model, vram, province)
    A-->>H: status = pending, admitted = false

    H->>A: POST /agent/versions (driver, runc, ctk, docker)
    A->>S: admit_node()
    S->>S: check_node_versions()
    S->>S: recommend_runtime()
    S-->>A: admitted = true | rejection reasons

    alt Admitted
        A-->>H: status = active → enters allocation pool
        loop Every 5s
            H->>A: POST /agent/telemetry
        end
    else Rejected
        A-->>H: status = pending, reasons logged
    end
```

**Minimum versions** — runc ≥ 1.1.14 · nvidia-ctk ≥ 1.17.8 · Docker ≥ 24.0 · Driver ≥ 535

---

## Project Structure

```
api.py            FastAPI gateway (250+ endpoints, SSE, dashboard)
scheduler.py      Job queue, host allocation, spot pricing, preemption
worker_agent.py   Pull-based GPU agent, telemetry, Docker execution
security.py       4-layer defense: version gating → gVisor/Kata
billing.py        CAD pricing, 13-province tax, escrow, CAF export
events.py         Append-only event store, tamper-evident hashing
reputation.py     Multi-factor scoring, Bronze→Sovereign tiers
privacy.py        PIPEDA / Quebec Law 25, data retention, consent
verification.py   GPU fingerprint verification, re-verification scheduling
jurisdiction.py   Trust tiers, residency tracing, CLOUD Act analysis
artifacts.py      Two-tier B2/R2 storage, residency-aware routing
db.py             PostgreSQL ↔ SQLite dual-write, LISTEN/NOTIFY
cli.py            Full CLI for jobs, hosts, billing, marketplace
ai_assistant.py   Hexara AI assistant with tool-calling & onboarding wizards
routes/           Modular API route handlers (agent, instances, admin, etc.)
frontend/         Next.js 15 dashboard + marketing site (xcelsior.ca)
wizard/           Interactive host‑setup wizard (TypeScript)
fern/             API documentation (docs.xcelsior.ca)
scripts/          Deployment, install, and bootstrap scripts
tests/            1400+ backend tests (pytest) + 17 frontend tests (vitest)
```

---

## Deployment

### Docker Compose (production)

```bash
docker compose up --build -d    # API + Frontend + Scheduler
curl https://xcelsior.ca/healthz
```

### Systemd (alternative)

```bash
sudo cp xcelsior-*.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now xcelsior-api xcelsior-health
```

See `.env.example` for all 50+ configuration variables.

---

## Testing

```bash
# Backend (1400+ tests)
python -m pytest tests/ -v              # full suite
python -m pytest tests/ -v --cov=.      # with coverage

# Frontend (17 tests)
cd frontend && npm test                 # vitest

# Linting
ruff check . && black --check *.py routes/*.py tests/
```

CI runs automatically on push to `main` and on PRs — see [CI](.github/workflows/ci.yml) and [Frontend CI](.github/workflows/frontend.yml).

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Run the test suite — all tests must pass
4. Submit a PR with a clear description

See [docs.xcelsior.ca](https://docs.xcelsior.ca) for full API reference and integration guides.

---

<div align="center">

**MIT License** · [LICENSE](LICENSE)

Built with 🍁 in Canada — **Ever upward.**

[xcelsior.ca](https://xcelsior.ca)

</div>
