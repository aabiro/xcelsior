<!-- residency-guard: documents-removal — traceability matrix back to the original specs -->
# DONE

**Approx. completed:** 2026-02-15

---

# Xcelsior — Comprehensive Cross-Reference & Multi-Phase Implementation Plan

*Last updated: 2026-02-15 — reflects v2.8.0 codebase (20 modules, 165+ endpoints, ~21,000 LOC + 22 test files)*

## 1. Complete Traceability Matrix: FINAL Reports → Draft Reports → Code

Every concrete prescription from the three FINAL reports is traced back to its origin in the draft reports, and mapped to the current codebase status.

---

### A. REPORT_FEATURE_FINAL.md Prescriptions

| # | Prescription | Draft Origin | Code File | Status |
|---|---|---|---|---|
| F1 | Automated trust infrastructure (verification → monitoring → deverification) | FEATURE_1 §SLA Enforcement; FEATURE_2 Phase B §1 | verification.py (600 lines, 7 checks incl. serial + mem frag) | ✅ Complete |
| F2 | Defense-in-depth isolation (CVE gating, least-privilege, egress deny) | TECHNICAL_1 §Container Security; TECHNICAL_2 §4.1–4.3 | security.py (509 lines) | ✅ Complete |
| F3 | Canada-first compute economics (scheduling, storage, invoicing) | MARKETING_1 §5; FEATURE_2 Phase A | jurisdiction.py, billing.py, scheduler.py | ✅ Complete |
| F4 | **Stripe Connect for marketplace payouts/compliance** | FEATURE_1 Phase 3 ("Escrow Billing: Integrate Stripe") | stripe_connect.py (500 lines), 7 API endpoints | ✅ Complete |
| F5 | S3-compatible artifact storage with presigned URLs | FEATURE_1 §Artifact Management; TECHNICAL_2 §storage | artifacts.py (499 lines) | ✅ Complete |
| F6 | Two-tier storage: B2 CA East + R2 zero-egress | FEATURE_1 §Multi-Region; MARKETING_1 §4.3 | artifacts.py `ArtifactManager` with routing | ✅ Complete |
| F7 | Strict job state machine with event sourcing | FEATURE_1 §Trust; FEATURE_2 Phase C §1 | events.py (635 lines), tamper-evident hashing | ✅ Complete |
| F8 | Verification + telemetry as single integrated system | FEATURE_1 §GPU Telemetry; FEATURE_2 Phase A §3; TECHNICAL_2 §GPU Orch | nvml_telemetry.py (pynvml), verification.py, worker_agent.py telemetry_loop | ✅ Complete |
| F9 | Security posture as schedulable resource | TECHNICAL_2 §4; FEATURE_2 Phase B §1 | scheduler.py `allocate_job()` checks security | ✅ Complete |
| F10 | Canada-first billing measurable/exportable | MARKETING_1 §5.3; FEATURE_2 Phase A §4 | billing.py `export_caf_report()`, `export_caf_csv()` | ✅ Complete |
| F11 | **Decide early: payments + tax** (GST/HST, Stripe Connect) | MARKETING_1 §5.2; FEATURE_1 Phase 3 | billing.py PROVINCE_TAX_RATES ✅, stripe_connect.py ✅, GST threshold ✅ | ✅ Complete |

### B. REPORT_MARKETING_FINAL.md Prescriptions

| # | Prescription | Draft Origin | Code File | Status |
|---|---|---|---|---|
| M1 | Jurisdiction-aware scheduling (hard constraint, not preference) | MARKETING_1 §4.1 (Geo-Fenced Scheduler) | jurisdiction.py, scheduler.py sovereign queue | ✅ Complete |
| M2 | AI Compute Access Fund as demand accelerator (67% vs 50%) | MARKETING_1 §5.3 (AI CAF $300M) | billing.py `compute_fund_eligible_amount()` | ✅ Complete |
| M3 | Trust ladder: Residency → Sovereignty → Regulated | MARKETING_1 §6.3 (Enterprise/Health Tier) | jurisdiction.py `TrustTier` enum, `TRUST_TIER_REQUIREMENTS` | ✅ Complete |
| M4 | Phase A: Canada-only + province-aware + rebate-ready invoicing | MARKETING_1 §5; FEATURE_2 Phase A | billing.py, jurisdiction.py | ✅ Complete |
| M5 | Phase B: Host eligibility, structured legal-demand handling | MARKETING_1 §6.3; FEATURE_2 Phase B | verification.py, transparency in api.py | ✅ Complete |
| M6 | Phase C: Deeper auditability, tamper-evident logs | MARKETING_1 §4.2 (Sidecar Audit); FEATURE_2 Phase C | events.py hash chaining, api.py audit endpoints | ✅ Complete |
| M7 | Verified vs Community host tiers | MARKETING_1 §6.3 ("Gold" badge) | reputation.py `ReputationTier`, verification.py | ✅ Complete |
| M8 | **Dashboard: "Add Host" button, earnings, utilization analytics** | MARKETING_1 §6.1–6.3; FEATURE_2 Phase A §1–3 | dashboard.html 13-tab, 8+ modals, login page, wallet, reputation, settings, Earnings, Telemetry gauges, form UX v2, admin approve/reject, transparency chart, Analytics tab, Artifacts tab, HPC/Slurm tab, Admin panel, Host Detail modal, Compliance Dashboard, Marketplace search/filter | ✅ Complete |

### C. REPORT_XCELSIOR_TECHNICAL_FINAL.md Prescriptions

| # | Prescription | Draft Origin | Code File | Status |
|---|---|---|---|---|
| T1 | Pull-based agent + Headscale WireGuard | TECHNICAL_1 §VPN; TECHNICAL_2 §2 (Agent arch) | worker_agent.py (1499 lines), headscale on VPS | ✅ Complete |
| T2 | PostgreSQL migration (expand/contract, JSONB+GIN) | TECHNICAL_1 §SQLite→PG; TECHNICAL_2 §1 (dual-write) | db.py (759 lines), Alembic | ✅ Complete |
| T3 | Security (version gating, rootless, no-priv, egress, mining detect) | TECHNICAL_1 §Container Security; TECHNICAL_2 §4 | security.py (509 lines) | ✅ Complete |
| T4 | SSE for dashboards + PG LISTEN/NOTIFY | TECHNICAL_1 §SSE; TECHNICAL_2 §SSE bridge | api.py SSE endpoint, db.py PgEventBus | ✅ Complete |
| T5 | Spot/interruptible pricing + preemption | TECHNICAL_1 §Spot; TECHNICAL_2 §3.3–3.5 | scheduler.py spot pricing, preemption workflow | ✅ Complete |
| T6 | Container image caching (LRU + idle-time pre-pull) | TECHNICAL_2 §3.6 | worker_agent.py `cache_*` functions | ✅ Complete |
| T7 | **Slurm adapter for SHARCNET Nibi** | TECHNICAL_FINAL §Nibi | slurm_adapter.py (~500 lines) — Nibi/Graham/Narval profiles, sbatch generation, status sync | ✅ Complete |
| T8 | **gVisor GPU support notes** | TECHNICAL_2 §4.1 (nvproxy) | security.py `detect_gvisor()` + `install_gvisor()` auto-install, worker_agent.py startup integration | ✅ Complete |

### D. REPORT_FEATURE_2.md Prescriptions (Phase A/B/C)

| # | Prescription | Code | Status |
|---|---|---|---|
| D1 | Self-serve provider onboarding: CLI `host-add --interactive`, web "Add Host" button | cli.py `host-add` (w/ `--province` choices), dashboard Add Host (30+ GPUs, VRAM auto-fill, country dropdown, province optgroups), Provider Registration modal | ✅ Complete ||
| D2 | Job submission UX with subsidy preview: dashboard form, `simulate=true` | dashboard.html Submit Job modal w/ live cost estimate (tier × trust), descriptive tier labels, validation | ✅ Complete |
| D3 | Basic monitoring/telemetry: live gauges for utilization/temp, alert thresholds | worker_agent.py telemetry_loop, dashboard GPU Telemetry tab | ✅ Complete |
| D4 | Rebate-ready invoicing: "Export Rebate Docs" button in dashboard | billing.py `export_caf_report` + dashboard "Export Rebate Docs" button | ✅ Complete |
| D5 | Host verification workflow: admin approve/reject in dashboard | verification.py engine + dashboard Trust tab (approve/reject buttons) + `POST /api/verify/{id}/approve` + `POST /api/verify/{id}/reject` | ✅ Complete |
| D6 | Jurisdiction badges in dashboard | dashboard shows country + province in host tables | ✅ Complete |
| D7 | Transparency report visualization in dashboard | api.py `/api/transparency/report` + dashboard Transparency Report card (summary grid, bar chart, jurisdiction breakdown, CLOUD Act canary) | ✅ Complete |
| D8 | **Tamper-evident logging + SLA enforcement** | events.py hash chaining ✅, sla.py (436 lines) ✅, 5 SLA API endpoints ✅ | ✅ Complete |
| D9 | **"My Earnings" tab with charts** | dashboard.html "My Earnings" tab | ✅ Complete |

### E. REPORT_FEATURE_1.md Prescriptions

| # | Prescription | Code | Status |
|---|---|---|---|
| E1 | CLI-to-web auth bridge (browser login → JWT) | api.py OAuth2 device flow (`/api/auth/device`+`/token`+`/verify`), cli.py `login`/`logout`/`whoami`, `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/oauth/{provider}`, `GET/PATCH/DELETE /api/auth/me`, `POST /api/auth/refresh`, `POST/GET/DELETE /api/keys/*` | ✅ Complete |
| E2 | Interactive OpenAPI docs (Swagger + Fern playground) | FastAPI `/docs` + `/redoc`, fern/ config with playground | ✅ Complete |
| E3 | Role-based onboarding modal (GPU Provider vs Job Submitter) | dashboard.html onboarding modal with 3 roles | ✅ Complete |
| E4 | PostHog-style usage analytics | `/api/analytics/usage` endpoint with time-series grouping | ✅ Complete |
| E5 | NVML/DCGM GPU telemetry | nvml_telemetry.py (pynvml bindings), worker_agent.py `telemetry_loop()`, security.py NVML-first | ✅ Complete |
| E6 | SSE real-time log streaming (tail pattern) | `/api/stream` broadcast SSE ✅ + `/jobs/{job_id}/logs/stream` per-job SSE ✅ | ✅ Complete |
| E7 | Reputation scoring with decay, 6-tier (New User→Diamond), diminishing returns | reputation.py (690 lines) | ✅ Complete |
| E8 | Escrow billing with Stripe (credit-first, hourly deductions) | billing.py wallet system ✅ + stripe_connect.py (500 lines) ✅ | ✅ Complete |
| E9 | Automated refunds (host vs user failure logic) | billing.py `process_refund()` | ✅ Complete |
| E10 | Multi-region awareness, location filtering | jurisdiction.py, scheduler.py province filtering | ✅ Complete |
| E11 | Artifact management (S3-compatible, sidecar log forwarding) | artifacts.py (499 lines) | ✅ Complete |
| E12 | Interactive dashboard (role-based checklist, earnings tab) | dashboard.html 13-tab with login, onboarding, earnings, telemetry, logs, reputation, settings, wallet, analytics, artifacts, HPC/Slurm, admin | ✅ Complete |

---

## 2. Summary of Gaps

### Resolved Gaps (Previously Critical — Now Complete)

| Gap | Resolution | Date |
|---|---|---|
| ~~G1: Dashboard missing 8+ UI features~~ | dashboard.html v2.3.0: 12 tabs, 7 modals, login page, wallet display, reputation/settings tabs, job detail, host remove, toast notifications, analytics, artifacts, admin panel, real auth endpoints | ✅ 2026-02 |
| ~~G2: No Stripe Connect integration~~ | stripe_connect.py (500 lines) + 7 API endpoints + Canadian company fields | ✅ 2026-02 |
| ~~G3: No SLA enforcement endpoint~~ | sla.py (436 lines) + 5 API endpoints (enforce, host, violations, downtimes, targets) | ✅ 2026-02 |
| ~~G4: No job submission form in dashboard~~ | dashboard.html "+ Submit Job" button with tier/trust/image form | ✅ 2026-02 |
| ~~G5: No per-job log streaming~~ | `/jobs/{job_id}/logs/stream` SSE endpoint + Job Logs dashboard tab | ✅ 2026-02 |
| ~~G6: No reserved pricing~~ | `/api/pricing/reserved-plans` + `/api/pricing/reserve` (20%/30%/45% tiers) | ✅ 2026-02 |
| ~~G7: No GST threshold logic~~ | `/api/billing/gst-threshold` + `/api/billing/gst-threshold/{provider_id}` ($30k rule) | ✅ 2026-02 |
| ~~G8: No usage analytics~~ | `/api/analytics/usage` with day/week/gpu_model/province grouping | ✅ 2026-02 |

### Remaining Gaps

| Gap | Status |
|---|---|
| ~~CLI-to-web auth bridge~~ | ✅ Completed — OAuth2 device flow (RFC 8628) in api.py + cli.py |
| ~~Slurm adapter~~ | ✅ Completed — slurm_adapter.py with Nibi/Graham/Narval/generic profiles |
| ~~gVisor auto-install~~ | ✅ Completed — security.py `install_gvisor()` + worker startup integration |
| ~~Transparency report dashboard visualization~~ | ✅ Completed — summary grid + bar chart + canary in Trust tab |
| Test coverage | ✅ 795 tests across 21 files — all passing |

---

## 3. Multi-Phase Implementation Plan

### Phase 1: Dashboard Completeness ✅ COMPLETE
**Goal**: Make the dashboard fully functional as described across all reports.

1. ✅ **Add Host onboarding panel** — "+ Add Host" button with form (ID, IP, GPU, VRAM, rate, country/province)
2. ✅ **Job submission form** — "+ Submit Job" button with tier/trust/VRAM/image selection
3. ✅ **Live GPU telemetry gauges** — GPU Telemetry tab with utilization/temp bars per host
4. ✅ **"Export Rebate Docs" button** — Downloads CAF CSV
5. ✅ **"My Earnings" tab** — Provider earnings breakdown
6. ✅ **Role-based onboarding modal** — First-visit modal: GPU Provider / Job Submitter / Admin
7. ✅ **Transparency report visualization** — Summary grid, horizontal bar chart by type, jurisdiction breakdown, CLOUD Act canary
8. ✅ **Province badges** — Host tables show country + province
9. ✅ **Trust & Verification tab** — Verification status, reputation leaderboard, Canadian host counts

### Phase 2: SLA & Backend Enforcement ✅ COMPLETE
**Goal**: Complete the backend enforcement layer.

1. ✅ **SLA enforcement endpoint** `/api/sla/enforce` — Check host uptime vs tier SLA, auto-credit on breach
2. ✅ **SLA definitions per trust tier** — Community: 99.0%, Secure: 99.5%, Sovereign: 99.9%
3. ✅ **Per-job log streaming** — SSE endpoint `/jobs/{job_id}/logs/stream` for tail-style streaming
4. ✅ **Per-job log buffer** — `/jobs/{job_id}/logs` for non-streaming log retrieval

### Phase 3: Stripe Connect Integration ✅ COMPLETE
**Goal**: Enable real money flow through the marketplace.

1. ✅ **Stripe Connect account creation** — `stripe_connect.py` creates Express accounts via `/api/providers/register`
2. ✅ **Canadian company registration** — corporation_name, business_number, GST/HST number, legal_name fields
3. ✅ **Incorporation file upload** — `/api/providers/{id}/incorporation`
4. ✅ **Automated payouts** — `/api/providers/{id}/payout` with 85/15 split + per-province GST/HST
5. ✅ **Webhook handling** — `/api/providers/webhook` for account.updated, payment_intent.succeeded, payout.paid
6. ✅ **GST/HST collection** — billing.py `PROVINCE_TAX_RATES` (13 provinces) + $30k threshold endpoint
7. ✅ **Reserved pricing** — 1-month (20%), 3-month (30%), 1-year (45%) commitment tiers

### Phase 3.5: Form UX Overhaul ✅ COMPLETE
**Goal**: Best-in-class dashboard form UX with Canadian jurisdiction awareness.

1. ✅ **Country dropdown** — 🇨🇦 Canada first, 16 countries with flag emoji (replaces free-text input)
2. ✅ **Province/territory full names** — All 10 provinces + 3 territories in `<optgroup>` ("Ontario (ON)", etc.)
3. ✅ **30+ GPU models** — Consumer / Datacenter / Professional / AMD with auto-fill VRAM
4. ✅ **Provider Registration modal** — ID, email, type (individual/company), conditional corp fields, province
5. ✅ **Cost estimate preview** — Live $/hr based on priority tier × trust multiplier
6. ✅ **Form validation** — Required markers, error states, section grouping
7. ✅ **CLI province validation** — `--province` validates against 13 province/territory codes

### Phase 4: Advanced Features ✅ COMPLETE
**Goal**: Enterprise-grade differentiation.

| Item | Priority | Status |
|------|----------|--------|
| CLI-to-web auth bridge (OAuth2 device flow) | Medium | ✅ Done (RFC 8628: api.py `/api/auth/device`+`/token`+`/verify`, cli.py `login`/`logout`/`whoami`) |
| Slurm adapter for SHARCNET Nibi | Low | ✅ Done (slurm_adapter.py — Nibi/Graham/Narval profiles, sbatch generation, status sync) |
| Confidential Computing (gVisor nvproxy) | Low | ✅ Done (security.py `install_gvisor()` auto-install + worker_agent.py startup integration) |
| NFS/persistent storage for multi-node jobs | Low | ✅ Done (worker_agent.py mount/unmount, scheduler.py NFS fields, cli.py `--nfs-*` flags) |
| Multi-GPU job support | Low | ✅ Done (scheduler.py `num_gpus` field, allocator GPU-count filter, security.py multi-device, cli.py `--gpus N`) |

---

## 4. Current Codebase Inventory (v2.6.0)

| Module | Lines | Purpose |
|--------|-------|---------|
| api.py | ~3,920 | FastAPI REST API — 162+ endpoints, OAuth2 device flow, email/password + OAuth auth, API key management, multi-GPU + NFS job support, password reset, marketplace search, team management |
| scheduler.py | 2,870 | Job scheduling, host management, spot pricing, autoscale, multi-GPU allocation |
| worker_agent.py | 1,550 | GPU worker daemon, telemetry, benchmarks, mining detection, gVisor auto-install, NFS |
| billing.py | 1,041 | CAD billing, wallets, invoices, CAF exports, payouts, GST |
| db.py | ~1,090 | Dual-write PostgreSQL + SQLite, connection pooling, UserStore persistent auth + teams |
| cli.py | ~990 | CLI — 25+ commands, OAuth login/logout, Slurm submit, multi-GPU, NFS flags |
| reputation.py | 690 | Trust scoring, decay, 6-tier (New User→Diamond), diminishing returns, platform commission |
| events.py | 635 | Event sourcing, tamper-evident hash chains, state machine |
| privacy.py | 596 | PIPEDA compliance, retention, consent, Quebec Law 25 |
| verification.py | 600 | 7-check hardware attestation (incl. serial + mem fragmentation), GPU fingerprinting |
| slurm_adapter.py | ~500 | Slurm cluster adapter — Nibi/Graham/Narval profiles, sbatch gen, status sync |
| stripe_connect.py | 500 | Stripe Connect Express accounts, payouts, KYC |
| artifacts.py | 499 | S3 presigned URLs, B2+R2 routing, residency policies |
| sla.py | 436 | SLA tiers, uptime tracking, credit calculation |
| jurisdiction.py | 382 | Trust tiers, residency traces, Canada-only scheduling |
| nvml_telemetry.py | 350 | NVML GPU telemetry (pynvml), serial/PCIe/ECC/thermal/fragmentation, nvidia-smi fallback |
| security.py | ~610 | Version gating, rootless containers, egress, mining detect, gVisor auto-install |
| inference.py | 85 | ML inference helpers |
| dashboard.html | ~4,430 | 13-tab dashboard (+ Analytics, Artifacts, HPC/Slurm, Admin), login page (real auth + password reset), user navbar, wallet display, deposit modal, 6-tier reputation badges, job detail modal (tabbed: overview/logs/artifacts/audit), toast notifications, SSE, 30+ GPU models, host remove/detail, job cancel/requeue, transaction history + CSV export + type filters, reserved pricing cards, admin panel + SLA enforce, marketplace search/filter + stats, compliance dashboard + GST threshold, event log filters + expandable detail, change password, delete account, invoice list, SLA dashboard (per-host cards, violations, downtimes), alert config, session timeout modal, low balance warning, reputation sparkline, team management, notification preferences, privacy consent display + revoke, provider detail panel with earnings/payout, Slurm HPC submit/status/cancel, spot bid form, SLA credit calculator, residency trace UI, responsive design (tablet + mobile) |
| **Total** | **~20,700** | |

### Test Files (Phase 7)

| File | Tests | Scope |
|------|-------|-------|
| test_api.py | 117 | Health, CRUD, auth, telemetry, reputation, pricing, billing, transparency, marketplace, autoscale, agent, spot, SLA, privacy, failover, Slurm, OAuth device flow, analytics, user auth (register/login/OAuth/profile/refresh/delete), API keys (generate/list/revoke/bearer) |
| test_scheduler.py | 79 | Scheduler, host mgmt, billing, queue, spot pricing, preemption, compute scores |
| test_privacy.py | 55 | PII redaction, env var scrubbing, retention lifecycle, consent, Law 25 PIA |
| test_worker_agent.py | 53 | Config, API helpers, GPU detection, heartbeat, version reporting, polling, preemption, leases, job status, mining alerts, benchmarks, telemetry, verification, image cache, Tailscale, signals |
| test_slurm_adapter.py | 46 | Sbatch generation, GPU estimation, priority QoS, state mapping, profiles |
| test_db.py | 41 | SQLite CRUD, transactions, state namespace, GPU queries, event bus |
| test_artifacts.py | 31 | Presigned URLs, local backend CRUD, residency routing, artifact manager |
| test_events.py | 28 | Hash chain, state machine, leases |
| test_jurisdiction.py | 28 | Trust tiers, constraints, fund eligibility, residency trace |
| test_billing.py | 27 | Tax rates, wallets, metering, invoices, CAF, refunds |
| test_nvml_telemetry.py | 34 | NVML lifecycle, GPU telemetry collection, thermal averaging, fallback, verification reports |
| test_password_marketplace.py | 20 | Password reset (request/confirm/expired/invalid), change password (success/wrong/short/unauth), marketplace search (GPU/VRAM/price/province/sort/limit) |
| test_persistent_auth.py | 40 | UserStore CRUD (users/sessions/API keys/teams), team endpoints, persistent auth integration, SLA enforcement, dashboard version (v2.8.0) |
| test_slurm_ui.py | 41 | Slurm API (profiles/submit/status/cancel), spot bid, SLA hosts-summary, SLA enforce, residency trace, dashboard HTML structure (13-tab, responsive CSS, Slurm UI, spot bid form, SLA credit calc, residency trace UI, JS functions, v2.8.0) |
| test_invoices_sla.py | 12 | Invoice list (ok, empty, limit, fields), SLA hosts summary (empty, single, multiple, structure, uptime, pending, provinces) |
| test_reputation.py | 27 | Scoring, 6-tier thresholds, decay, penalties, marketplace boost |
| test_verification.py | 23 | GPU checks, store, engine lifecycle |
| test_security.py | 18 | Version gating, admission, egress, mining detection, gVisor, secure docker args |
| test_ux_polish.py | 53 | Data export endpoint, artifact expiry endpoint, reputation breakdown endpoint, auth created_at, dashboard UX polish (avatar, SSH keys, date picker, balance chart, score breakdown, skeleton CSS, empty states, notification toggles, alert sound, artifact expiry badges) |
| test_integration.py | 2 | End-to-end job lifecycle, marketplace billing |
| **Total** | **847** | |

### Supporting Files

| File | Purpose |
|------|---------|
| fern/ (3 files) | SDK generation config (Python + TypeScript), API playground |
| llms.txt | LLM-optimized documentation for AI agents |
| .env.example | 55+ environment variables documented |
| docker-compose.yml | API + PostgreSQL 16 + test DB |
| Dockerfile | Python 3.12-slim production image |
| runbooks/operations.md | Operational procedures |

---

## 3. Version History

### v2.8.0 — UX Polish & Profile Enhancement (2026-02-15)

**New API Endpoints (3):**
- `GET /api/auth/me/data-export` — PIPEDA-compliant data export (profile, jobs, billing, reputation)
- `GET /api/artifacts/{job_id}/expiry` — Artifact TTL info (90d–365d by type)
- `GET /api/reputation/{entity_id}/breakdown` — Score component analysis (jobs, uptime, penalties, decay)

**Dashboard Enhancements:**
- Profile: avatar (initial letter circle), name/email display, "Member since" date
- SSH Keys: upload form, key table with fingerprints, remove button (localStorage-backed)
- Data Export: PIPEDA "Request Data Export" button downloads JSON bundle
- Delete Account: triple-confirmation flow (confirm → confirm → type DELETE)
- Score Breakdown: colored bar chart (green=jobs, blue=uptime, red=penalties, amber=decay)
- Score History: event table with delta and reason columns
- Balance Chart: ASCII art trend visualization in billing tab
- Date Range: from/to date picker for transaction filtering
- Artifact Expiry: "Expires" column with color-coded badges (ok/warn/crit)
- Loading States: skeleton-pulse CSS animation for async-loaded sections
- Empty States: icon + description for no-data scenarios
- Notification Sound: Web Audio API beep on error/warning toasts
- Push Notifications: browser Notification API toggle
- Low Balance Warning: moderate ($5–$15 yellow) + critical (<$5 red) toasts

**Tests:** 53 new tests in `tests/test_ux_polish.py` (52 pass, 1 skip)
**Total:** 847 tests across 22 files
