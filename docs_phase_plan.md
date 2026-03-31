# Xcelsior Reliability Roadmap (Phased)

## Phase 1 — Reliability lock-in ✅
- CI on every PR/push: `pytest`, `ruff`, `black --check`.
- Integration tests for job lifecycle + billing + marketplace mixed cut aggregation.

## Phase 2 — Persistence hardening ✅
- Dual-write PostgreSQL + SQLite via `db.py`.
- JSONB payloads with GIN indexes for flexible querying.
- 4 dedicated SQLite databases for events, billing, reputation, privacy.
- Alembic migrations for Postgres schema versioning.

## Phase 3 — Auth + API safety ✅
- Token auth enforced when `XCELSIOR_ENV != dev`.
- Request validation via Pydantic v2 models.
- In-memory rate limiting (configurable via `XCELSIOR_RATE_LIMIT_*`).

## Phase 4 — Observability ✅
- Structured API error responses.
- Endpoints: `/healthz`, `/readyz`, `/metrics`.
- SSE event stream at `/api/stream`.
- GPU telemetry endpoints (push/pull).

## Phase 5 — Deployment baseline ✅
- Container-first: `Dockerfile` (Python 3.12), `docker-compose.yml`.
- PostgreSQL 16 with health checks, test DB init script.
- Dual-write data volume for SQLite + Postgres coexistence.
- Systemd service files for non-Docker deployments.

## Phase 6 — Operator UX ✅
- `.env.example` with 55+ variables documented.
- Runbooks: `runbooks/operations.md`.
- 7-tab dashboard with Add Host, Submit Job, Export, onboarding modals, telemetry gauges.

## Phase 6.5 — Report #1.B Feature Implementation ✅
*Implements all features from the Prioritized Feature Roadmap (Report #1.B).*

### Trust & Verification Layer ✅
- SLA enforcement engine: `sla.py` (436 lines), 3 tiers (Community 99.0%, Secure 99.5%, Sovereign 99.9%).
- Credit calculation: 95-99% → 10%, 90-95% → 25%, <90% → 100% refund.
- 5 SLA API endpoints: enforce, host status, violations, downtimes, targets.

### Stripe Connect & Provider Onboarding ✅
- `stripe_connect.py` (500 lines): Express accounts, KYC, payouts.
- 7 Provider API endpoints: register, get, list, incorporation upload, earnings, payout, webhook.
- Canadian company fields: corporation_name, business_number, GST/HST number, legal_name.
- 85/15 platform/provider payout split with per-province GST/HST.

### Interactive Documentation & SDK ✅
- OpenAPI tags on all 129 endpoints (19 tag groups).
- FastAPI Swagger UI (`/docs`) + ReDoc (`/redoc`) with "Try it out" sandbox.
- Fern SDK config: Python (`xcelsior` package) + TypeScript (`@xcelsior-gpu/sdk`).
- Fern API Playground with production + development environments.
- `llms.txt` (LLM-optimized docs per llmstxt.org spec).

### Per-Job SSE Log Streaming ✅
- `GET /jobs/{job_id}/logs/stream`: async generator, EventSource-compatible.
- `GET /jobs/{job_id}/logs`: non-streaming log buffer (last N lines).
- Dashboard "Job Logs" tab with live tail, stream/stop controls.
- Jobs table "Logs" button → opens streaming tab for that job.

### Financial Engineering ✅
- Reserved pricing tiers: 1-month (20%), 3-month (30%), 1-year (45% off).
- `GET /api/pricing/reserved-plans` + `POST /api/pricing/reserve`.
- $30,000 GST/HST small-supplier threshold: `GET /api/billing/gst-threshold`.
- Per-provider threshold check: `GET /api/billing/gst-threshold/{provider_id}`.
- Usage analytics: `GET /api/analytics/usage` (group by day/week/gpu_model/province).

### CLI v2.1 Commands ✅
- 11 new commands: reputation, verify, wallet, deposit, invoice, sla, provider-register, provider, leaderboard, compliance.

---

## Phase 7 — Comprehensive Testing Plan

### 7.1 — Unit Tests (Priority: HIGH)

Tests that can run without any external dependencies (no DB, no Docker, no network).

#### 7.1.1 — Scheduler Core (`test_scheduler.py`)

| Test Group | What to Test | Status |
|------------|-------------|--------|
| `TestAllocate` | VRAM filtering, admission gating (admitted=True only), isolation tier enforcement (gVisor preference for sovereign), empty host list, all hosts busy | Partial |
| `TestRegisterHost` | Register new host, update existing host, preserve `admitted` field, country/province persistence | Partial |
| `TestSubmitJob` | Valid submission, VRAM validation, tier validation, trust_tier field, priority ordering | Partial |
| `TestProcessQueue` | FIFO within same priority, priority ordering, skip queued when no capacity, preemption for urgent | Partial |
| `TestBilling` | Per-job billing calc, tier multipliers, duration accuracy, bill-all idempotency | Partial |
| `TestSpotPricing` | Supply/demand ratio, price floor/ceiling, sensitivity parameter, historical recording | ✅ |
| `TestPreemption` | Grace period, notification before kill, job re-queue, spot vs priority tier | ✅ |
| `TestMarketplace` | List rig, unlist, platform cut calculation, earnings aggregation | Partial |
| `TestAutoscale` | Pool add/remove, trigger thresholds, max pool size, Canada-only filtering | TODO |
| `TestComputeScores` | XCU calculation, score registration, score-based allocation | ✅ |

#### 7.1.2 — Security (`test_security.py` — ✅ CREATED)

| Test Group | What to Test | Status |
|------------|-------------|--------|
| `TestVersionGating` | Pass all versions, fail runc version, fail nvidia_ctk, fail docker, fail driver, edge cases (missing keys, empty dict) | ✅ |
| `TestAdmitNode` | Full admission pass, partial failure, runtime recommendation based on GPU model | ✅ |
| `TestRecommendRuntime` | gVisor-supported GPUs (RTX 4090, A100, H100), unsupported GPUs, edge cases | ✅ |
| `TestBuildSecureDockerArgs` | All security flags present (--no-new-privileges, cap-drop, read-only, tmpfs, non-root), GPU pass-through NOT --privileged | ✅ |
| `TestEgressRules` | Allowlist contains expected domains, iptables rule format, custom allowlist merge | ✅ |
| `TestMiningHeuristic` | Detect sustained high util + low PCIe, normal workload not flagged, configurable thresholds | ✅ |

#### 7.1.3 — Billing Engine (`test_billing.py` — ✅ CREATED)

| Test Group | What to Test | Status |
|------------|-------------|--------|
| `TestUsageMeters` | Record creation, Canadian compute flag, province detection | ✅ |
| `TestTaxRates` | All 13 provinces correct (ON=13%, QC=14.975%, AB=5%, etc.), unknown province defaults | ✅ |
| `TestInvoicing` | Line item aggregation, tax calculation, fund eligibility (CA=67%, intl=50%) | ✅ |
| `TestWallets` | Deposit, withdraw, insufficient funds, grace period, refunds | ✅ |
| `TestPayouts` | Platform fee deduction, provider payout amount, payout ledger recording | ✅ |
| `TestCAFExport` | CSV format correctness, all required columns, date filtering | ✅ |

#### 7.1.4 — Events & State Machine (`test_events.py` — ✅ CREATED)

| Test Group | What to Test | Status |
|------------|-------------|--------|
| `TestEventStore` | Append event, tamper-evident hash chain, query by entity, query by type | ✅ |
| `TestStateMachine` | Valid transitions (queued→running→completed), invalid transitions rejected, idempotent transitions | ✅ |
| `TestLeases` | Grant lease, renew lease, expire lease, lease duration tracking | ✅ |

#### 7.1.5 — Verification (`test_verification.py` — ✅ CREATED)

| Test Group | What to Test | Status |
|------------|-------------|--------|
| `TestGPUVerification` | Verify matching fingerprint, detect spoofed GPU, re-verification scheduling | ✅ |
| `TestVerificationHistory` | State transitions logged, failure count tracking | ✅ |
| `TestJobFailureLog` | Failures recorded, host deverification on threshold | ✅ |

#### 7.1.6 — Reputation (`test_reputation.py` — ✅ CREATED)

| Test Group | What to Test | Status |
|------------|-------------|--------|
| `TestScoring` | Job completion adds points, failure deducts, verification bonus | ✅ |
| `TestDecay` | No decay within 7-day grace, 1pt/day after grace, floor at 0 | ✅ |
| `TestTiers` | 6-tier thresholds (New User/Bronze/Silver/Gold/Platinum/Diamond), tier upgrade/downgrade | ✅ |
| `TestSearchBoost` | Higher tier = higher marketplace visibility multiplier | ✅ |

#### 7.1.7 — Jurisdiction (`test_jurisdiction.py` — ✅ CREATED)

| Test Group | What to Test | Status |
|------------|-------------|--------|
| `TestTrustTiers` | Community/Residency/Sovereignty/Regulated definitions and requirements | ✅ |
| `TestResidencyTrace` | Canadian host → CA trace, non-CA host → foreign trace | ✅ |
| `TestFundEligibility` | CA compute at 67%, non-CA at 50%, mixed workload calculation | ✅ |
| `TestProvinceCompliance` | Quebec PIA trigger, all 13 provinces recognized | ✅ |

#### 7.1.8 — Artifacts (`test_artifacts.py` — ✅ CREATED, 31 tests)

| Test Group | What to Test | Status |
|------------|-------------|--------|
| `TestStorageConfig` | Defaults, from_env, custom prefix | ✅ |
| `TestEnums` | ResidencyPolicy, ArtifactType, ArtifactState values | ✅ |
| `TestArtifactMeta` | Dataclass fields, to_dict, from_dict | ✅ |
| `TestLocalBackend` | Upload/download URLs, head, delete, list with prefix filtering | ✅ |
| `TestArtifactManager` | _make_key, residency routing (CANADA_ONLY→primary, ANY→cache), download cache preference/fallback, job artifacts, cleanup | ✅ |
| `TestArtifactManagerNoCache` | Fallback to primary when no cache configured | ✅ |

#### 7.1.9 — Privacy (`test_privacy.py` — ✅ CREATED, 55 tests)

| Test Group | What to Test | Status |
|------------|-------------|--------|
| `TestRedactPII` | Email, phone, SIN, credit card, IP, api_key, empty, none, no PII, multiple, custom patterns | ✅ |
| `TestRedactEnvVars` | Known secrets, keyword patterns, empty, ALWAYS_REDACT, case insensitive | ✅ |
| `TestRedactJobRecord` | Strict commands, permissive, IP, PII in logs, location strip/keep | ✅ |
| `TestSanitizeLogOutput` | PII removal from log lines | ✅ |
| `TestPrivacyConfig` | Law 25 defaults, to_dict | ✅ |
| `TestRetentionPolicies` | All categories, JOB_PAYLOAD=0s, BILLING=7yr, LOGS=7d | ✅ |
| `TestDataLifecycleManager` | track_data, default/override retention, expired records, mark_purged, purge_expired | ✅ |
| `TestConsent` | Record, no consent default, revoke, get_consents, consent with details | ✅ |
| `TestConfigPersistence` | Save/load, unknown org strict defaults, overwrite | ✅ |
| `TestRetentionSummary` | Summary dict structure | ✅ |
| `TestQuebecPIA` | Non-QC, QC no PI, QC-to-QC, QC-to-ON with PI, case insensitive | ✅ |

#### 7.1.10 — Database Layer (`test_db.py` — ✅ CREATED, 41 tests)

| Test Group | What to Test | Status |
|------------|-------------|--------|
| `TestSQLiteConnection` | Table creation, WAL mode | ✅ |
| `TestSQLiteTransaction` | Commit, rollback | ✅ |
| `TestPayloadEncoding` | Decode dict/json/none/invalid, encode dict/string | ✅ |
| `TestJobCRUD` | Upsert/get, nonexistent, update, load all/by status, delete/delete all, empty ID skip | ✅ |
| `TestHostCRUD` | Upsert/get, nonexistent, load all/active only, delete/delete all, empty ID skip | ✅ |
| `TestStateNamespace` | Upsert/get, nonexistent, overwrite, list payload, string payload | ✅ |
| `TestQueryHostsByGPU` | Filter by model, min vram, both, no match, no filters | ✅ |
| `TestDualWriteEngineSQLite` | Connection, transaction | ✅ |
| `TestPgEventBusInMemory` | Add/remove listener, notify inmemory, remove nonexistent | ✅ |

#### 7.1.11 — Slurm Adapter (`test_slurm_adapter.py` — ✅ CREATED, 46 tests)

| Test Group | What to Test | Status |
|------------|-------------|--------|
| `TestClusterProfiles` | Required profiles, keys, nibi, graham, get_profile default/by name | ✅ |
| `TestGPUVRAMTable` | Common GPUs, a100=40, rtx_4090=24 | ✅ |
| `TestEstimateGPUs` | Zero/negative→1, exact fit, needs 2, small/large model, unknown GPU default | ✅ |
| `TestPriorityToQoS` | Premium≥90, normal≥50, low≥10, default<10 | ✅ |
| `TestEstimateWalltime` | Large 24h, medium 12h, small 6h, unknown default, custom default | ✅ |
| `TestSlurmStateMap` | All states mapped, all values valid, specific mappings | ✅ |
| `TestJobTranslation` | Sbatch header, metadata, apptainer, docker, GPU count, walltime, env vars, multi-node, memory, QoS, no image | ✅ |
| `TestSubmitDryRun` | Dry-run returns script without subprocess | ✅ |
| `TestSlurmCLI` | CLI commands formatted correctly | ✅ |
| `TestSlurmJobMap` | Job ID mapping persistence | ✅ |
| `TestSlurmAvailability` | is_available returns bool | ✅ |

#### 7.1.12 — NVML Telemetry (`test_nvml_telemetry.py` — ✅ CREATED, 34 tests)

| Test Group | What to Test | Status |
|------------|-------------|--------|
| `TestNVMLInit` | Init/shutdown lifecycle, double-init idempotent, is_nvml_available | ✅ |
| `TestCollectGPUTelemetry` | All metrics via mock pynvml (model, util, memory, temp, power, PCIe, serial, ECC, compute cap, driver/CUDA version, timestamp) | ✅ |
| `TestCollectAllGPUs` | Multi-GPU list, fallback when not initialized | ✅ |
| `TestGetGPUInfoNVML` | GPU info convenience function, returns None when not initialized | ✅ |
| `TestBuildVerificationReport` | Report has all verification fields, values match telemetry | ✅ |
| `TestThermalHistory` | Rolling average, per-GPU history, window cap | ✅ |
| `TestNVMLFallback` | nvidia-smi fallback parsing, empty list without nvidia-smi | ✅ |

### 7.2 — API Tests (Priority: HIGH) ✅ COMPLETE — 100 tests

Tests using FastAPI's `TestClient` (no real server needed).

#### 7.2.1 — Host Registration Flow (`test_api.py`) ✅

| Test | What to Test | Status |
|------|-------------|--------|
| `test_put_host_creates_pending` | PUT /host → status=pending, admitted=False | ✅ |
| `test_put_host_with_valid_versions` | PUT /host with versions → admit_node runs → admitted=True if pass | ✅ |
| `test_put_host_with_failing_versions` | PUT /host with old runc → admitted=False, status=pending | ✅ |
| `test_put_host_preserves_country_province` | Country=CA, province=ON persisted correctly | ✅ |
| `test_agent_versions_admits_host` | POST /agent/versions with good versions → host becomes active | ✅ |
| `test_agent_versions_rejects_host` | POST /agent/versions with bad versions → host stays pending | ✅ |
| `test_admitted_host_receives_work` | Submit job → process queue → only admitted hosts get work | ✅ |
| `test_pending_host_excluded` | Pending host with enough VRAM still doesn't get jobs | ✅ |
| `test_company_fields` | Provider company metadata persisted on host | ✅ |

#### 7.2.2 — Core CRUD ✅

| Test | What to Test | Status |
|------|-------------|--------|
| `test_submit_job` | POST /job returns job_id, status=queued | ✅ |
| `test_list_jobs_filter` | GET /jobs?status=running filters correctly | ✅ |
| `test_list_hosts_active` | GET /hosts?active_only=true excludes dead/pending | ✅ |
| `test_delete_host` | DELETE /host/{id} removes host | ✅ |
| `test_update_job_status` | PATCH /job/{id}/status transitions correctly | ✅ |

#### 7.2.3 — Auth ✅

| Test | What to Test | Status |
|------|-------------|--------|
| `test_dev_mode_no_token_ok` | XCELSIOR_ENV=dev → no token needed | ✅ |
| `test_public_paths` | Public paths accessible without auth | ✅ |
| `test_token_generate` | POST /api/auth/generate-token returns token | ✅ |

#### 7.2.4 — v2.0 Endpoints ✅

| Test | What to Test | Status |
|------|-------------|--------|
| `test_telemetry_push_pull` | POST telemetry → GET returns it | ✅ |
| `test_verified_hosts` | GET /api/verified-hosts returns list | ✅ |
| `test_reputation_leaderboard` | GET /api/reputation/leaderboard returns sorted | ✅ |
| `test_pricing_reference` | GET /api/pricing/reference returns GPU rates | ✅ |
| `test_pricing_estimate` | POST /api/pricing/estimate returns cost | ✅ |
| `test_billing_attestation` | GET /api/billing/attestation returns fund data | ✅ |
| `test_transparency_report` | GET /api/transparency/report returns report | ✅ |
| `test_healthz_readyz` | GET /healthz and /readyz return 200 | ✅ |

**Additional v2.0 tests (83 more):** marketplace, autoscale, agent, spot pricing, compute scores, jurisdiction, SLA, privacy, failover, Slurm, OAuth device flow, analytics — all ✅

### 7.3 — Integration Tests (Priority: MEDIUM) ✅ COMPLETE — 27 tests

Tests that exercise multiple modules together. Uses TestClient with temp files.

#### 7.3.1 — Full Job Lifecycle (`test_integration.py`)

| Test | What to Test |
|------|-------------|
| `test_host_register_admit_assign_complete_bill` | Register host → admit → submit job → process queue → complete → bill → verify billing record |
| `test_sovereign_job_routing` | Canadian host + sovereignty tier → job routed to CA host only |
| `test_spot_pricing_lifecycle` | List spot → submit spot job → preemption if higher priority |
| `test_marketplace_full_cycle` | List rig → browse → submit job → bill → platform cut deducted |
| `test_failover_job_reassignment` | Host goes dead → failover → job re-queued → assigned to new host |
| `test_autoscale_up_down` | Queue pressure → autoscale up → queue drained → autoscale down |

#### 7.3.2 — Billing + Jurisdiction

| Test | What to Test |
|------|-------------|
| `test_canadian_compute_fund_flow` | CA host job → billed → fund eligibility calculated → CAF export correct |
| `test_province_tax_application` | Ontario job → 13% HST, Quebec → 14.975% QST+GST |
| `test_wallet_escrow_lifecycle` | Deposit → job submitted → escrow hold → job completes → deduct → payout |

#### 7.3.3 — Security + Admission

| Test | What to Test |
|------|-------------|
| `test_version_gating_blocks_scheduling` | Host with old runc → register → submit job → queue process → job NOT assigned |
| `test_gvisor_preference_sovereign` | Sovereign tier job → prefers gVisor host over runc host |
| `test_reputation_affects_marketplace` | Low-rep host → lower marketplace visibility |

### 7.4 — Worker Agent Tests (Priority: MEDIUM) ✅ COMPLETE — 53 tests

Tests for worker_agent.py using mocked HTTP calls and subprocess.

#### 7.4.1 — `test_worker_agent.py` ✅

| Test | What to Test | Status |
|------|-------------|--------|
| `test_gpu_detection` | Mock nvidia-smi → correct VRAM parsing (5 cases) | ✅ |
| `test_registration_flow` | Heartbeat registers → correct payload/auth headers | ✅ |
| `test_work_polling` | Agent polls /agent/work → receives jobs or empty | ✅ |
| `test_telemetry_push` | Mock GPU metrics → POST /agent/telemetry → correct payload | ✅ |
| `test_mining_detection_alert` | Mining alert sent to scheduler, silent on error | ✅ |
| `test_image_cache_eviction` | Cache exceeds max → oldest images evicted, LRU tracking | ✅ |
| `test_headscale_enrollment` | Mock tailscale up → successful mesh join, authkey, login-server | ✅ |
| `test_config_validation` | Missing HOST_ID/SCHEDULER_URL → sys.exit | ✅ |
| `test_api_helpers` | _api_headers, _api_url correctness | ✅ |
| `test_lease_management` | Claim/renew/release lease lifecycle | ✅ |
| `test_version_reporting` | Report versions → admitted/rejected results | ✅ |
| `test_preemption_check` | Check preemption returns job IDs | ✅ |
| `test_job_status_reporting` | Report job status success/failure | ✅ |
| `test_benchmark_reporting` | Send benchmark score (XCU conversion) | ✅ |
| `test_verification_report` | Submit verification report to scheduler | ✅ |
| `test_popular_images` | Fetch popular images list | ✅ |
| `test_signal_handler` | SIGTERM sets shutdown event | ✅ |
| `test_host_ip_detection` | hostname -I, tailscale IP, fallback | ✅ |

### 7.5 — Database Migration Tests (Priority: LOW) ✅ COMPLETE — 29 tests

| Test | What to Test | Status |
|------|-------------|--------|
| `TestSQLiteAutoInit` (8 tests) | Each module's SQLite tables created on first access | ✅ |
| `TestDualWriteEngine` (6 tests) | Dual-write engine init + fallback behavior | ✅ |
| `TestDualWriteConsistency` (8 tests) | Write via dual → read from SQLite matches | ✅ |
| `TestEventBusSQLiteFallback` (4 tests) | EventBus SQLite persistence fallback | ✅ |
| `TestAlembicMigrationFiles` (3 tests) | Migration files exist and are importable | ✅ |

### 7.6 — End-to-End Tests (Priority: LOW) ✅ COMPLETE — 19 tests

E2E tests using TestClient (no external services needed).

| Test | What to Test | Status |
|------|-------------|--------|
| `TestDashboardLoads` (5 tests) | GET /dashboard 200, all tabs present, status cards | ✅ |
| `TestSSEStreamEvents` (1 test) | broadcast_sse callable | ✅ |
| `TestAddHostViaAPI` (3 tests) | Register host → appears in list, heartbeat updates | ✅ |
| `TestExportCAFCSV` (4 tests) | GET /api/billing/export/caf → valid CSV | ✅ |
| `TestHealthProbes` (3 tests) | /healthz, /readyz, /metrics return correct data | ✅ |
| `TestFullJobLifecycleE2E` (3 tests) | Submit → process → complete → billing record | ✅ |

### 7.7 — Test Execution Plan

#### Phase 7a — Immediate (before production deploy)
```bash
# Run existing tests
python -m pytest tests/test_scheduler.py tests/test_api.py -v

# Add security + admission tests
python -m pytest tests/test_security.py -v

# Add billing engine tests
python -m pytest tests/test_billing.py -v
```

#### Phase 7b — Pre-launch (xcelsior.ca)
```bash
# Integration tests with PostgreSQL
# Start a PostgreSQL 16 instance reachable at localhost:5432
alembic upgrade head
python -m pytest tests/test_integration.py -v

# Worker agent tests
python -m pytest tests/test_worker_agent.py -v
```

#### Phase 7c — CI Pipeline
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: xcelsior_test
          POSTGRES_USER: xcelsior
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: black --check .
      - run: python -m pytest -v --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v4
```

#### Phase 7d — Coverage Targets

| Module | Current | Target |
|--------|---------|--------|
| `scheduler.py` | ~75% | 85% |
| `api.py` | ~70% | 80% |
| `security.py` | ~80% | 90% |
| `billing.py` | ~65% | 85% |
| `events.py` | ~70% | 80% |
| `verification.py` | ~60% | 75% |
| `reputation.py` | ~65% | 80% |
| `jurisdiction.py` | ~70% | 75% |
| `artifacts.py` | ~60% | 70% |
| `privacy.py` | ~70% | 70% |
| `db.py` | ~65% | 80% |
| `slurm_adapter.py` | ~75% | 75% |
| `worker_agent.py` | ~55% | 70% |
| **Overall** | **~65%** | **80%** |

---

## Phase 8 — Production Launch Checklist

### Pre-deploy
- [x] All Phase 7a tests pass
- [x] DNS propagated: `xcelsior.ca` → Cloudflare → VPS
- [x] DNS propagated: `hs.xcelsior.ca` → VPS (DNS-only, no proxy)
- [x] Headscale running and accepting pre-auth keys
- [x] PostgreSQL 16 running with Alembic migrations applied
- [ ] Nginx configured with SSL (Let's Encrypt or Cloudflare origin cert)
- [x] `.env` production values set (token, PG password, DB_BACKEND=dual)
- [ ] `docker compose up --build -d` succeeds
- [ ] `curl https://xcelsior.ca/healthz` returns 200

### Post-deploy
- [ ] Submit test job via dashboard
- [ ] Verify GPU worker can register and get admitted
- [ ] Verify SSE stream delivers events to dashboard
- [ ] Verify billing records created for completed jobs
- [ ] Monitor for 24h — check logs, memory usage, DB size

---

## Phase 8.5 — Form UX Overhaul ✅

**Completed:** Dashboard forms rebuilt for best-in-class UX.

- [x] **Country dropdown** — 🇨🇦 Canada first, 16 additional countries with flag emoji
- [x] **Province/territory full names** — All 10 provinces + 3 territories with `<optgroup>` separation
- [x] **GPU model expanded** — 30+ models (Consumer, Datacenter, Professional, AMD) with VRAM specs
- [x] **VRAM auto-fill** — Selecting a GPU auto-populates VRAM from `data-vram` attributes
- [x] **Conditional province** — Province field hidden when country ≠ CA
- [x] **Cost estimate preview** — Submit Job modal shows live $/hr estimate based on tier × trust
- [x] **Provider Registration modal** — New modal with conditional corporation fields (company vs individual)
- [x] **Form validation** — Required field markers, error states with red borders and hints
- [x] **Form sections** — Logical grouping with section titles and dividers
- [x] **CLI province choices** — `--province` arg validates against 13 codes (AB..YT)
- [x] **CLI host-add updated** — Now accepts `--country` and `--province` with validation
- [x] **Submit Job descriptions** — Tier options show pricing and SLA commitments inline

---

## Phase 8.6 — Dashboard UI Expansion ✅

**Completed:** Massive UI overhaul closing ~85 endpoint gaps. Dashboard grew from 1,470 → 2,091 lines (v2.2.1).

### Authentication & Login
- [x] **Login page** — Full login page with Google, GitHub, HuggingFace OAuth buttons (SVG icons) + email/password form
- [x] **Dev mode bypass** — "Continue without login" link for development/testing
- [x] **Session management** — `sessionStorage`-based auth gating; main app hidden until login
- [x] **User navbar** — Avatar, dropdown menu (Settings, API Keys, My Reputation, Billing, Sign Out)
- [x] **Logout flow** — Clears session, returns to login page

### Wallet & Credits
- [x] **Wallet display** — Persistent navbar balance with color coding (green >$50, yellow $10–50, red <$5)
- [x] **Deposit modal** — 6 preset amounts ($10/$25/$50/$100/$250/$500) + custom input
- [x] **Deposit processing** — Calls `POST /api/billing/wallet/{id}/deposit` with dev-mode fallback
- [x] **Balance auto-refresh** — Fetches wallet balance on login and after deposits

### Reputation Tab (NEW)
- [x] **6-tier badge system** — CSS classes + emoji icons for all tiers (🔘 new_user, 🥉 bronze, 🥈 silver, 🥇 gold, 💎 platinum, 👑 diamond)
- [x] **Tier progress bar** — Visual progress toward next tier with percentage
- [x] **Tier benefits table** — All 6 tiers with search boost, max premium, commission rates
- [x] **Verification badges grid** — Email, phone, hardware, datacenter, incorporation, gov ID status
- [x] **Score history placeholder** — Ready for chart integration

### Settings Tab (NEW)
- [x] **Profile form** — Name, email (readonly), customer/provider IDs (readonly), role select, country/province
- [x] **API Keys management** — Generate with name, copy-to-clipboard, key preview table
- [x] **SSH Keys management** — Generate key pair, show public key
- [x] **Privacy & Consent toggles** — Telemetry, cross-border data, profiling opt-in/out

### Job Management Enhancements
- [x] **Job detail modal** — Clickable job ID opens detail view (ID, name, status, tier, VRAM, host, docker image, timestamps)
- [x] **Cancel button** — Cancel running/queued jobs via `PATCH /job/{id}` with `{status: 'cancelled'}`
- [x] **Requeue button** — Requeue failed jobs via `POST /job/{id}/requeue`

### Host Management Enhancements
- [x] **Remove button** — "✕" per host row with confirmation dialog → `DELETE /host/{hostId}`

### Trust Tab Additions
- [x] **SLA targets card** — Community 99.0%, Secure 99.5%, Sovereign 99.9% with refresh button

### Infrastructure
- [x] **Toast notification system** — Bottom-right stack, success/error/warning variants, auto-dismiss 4s
- [x] **Provider registration URL fix** — Corrected `/api/stripe/provider/register` → `/api/providers/register`
- [x] **Tab bar expanded** — 7 → 9 tabs (+ Reputation, ⚙️ Settings)
- [x] **6 modals** — Login, Deposit, Job Detail added to existing Submit Job, Provider Registration, Withdraw

---

## Phase 8.7 — Real Auth Backend + Dashboard v2.3.0 ✅

**Completed:** Real authentication endpoints, 3 new dashboard tabs, billing enhancements, admin panel. Dashboard grew from 2,091 → 2,658 lines. 682 tests (17 new auth tests).

### Duplicate Test File Cleanup
- [x] **Identified 4 stale root-level test files** — `test_api.py`, `test_scheduler.py`, `test_worker_agent.py`, `test_integration.py` in root (pyproject.toml `testpaths = ["tests"]` confirmed root files dead)
- [x] **Removed all 4 duplicates** — root `test_api.py` had outdated 4-tier assertion, root `test_integration.py` was 105 lines vs 584 in `tests/`

### Real Authentication Backend (12 new endpoints in api.py ~350 lines)
- [x] **User registration** — `POST /api/auth/register` with PBKDF2-HMAC-SHA256 password hashing (100k iterations)
- [x] **Email/password login** — `POST /api/auth/login` with constant-time password comparison (`hmac.compare_digest`)
- [x] **OAuth login** — `POST /api/auth/oauth/{provider}` for Google/GitHub/HuggingFace (dev-mode user creation)
- [x] **User profile** — `GET /api/auth/me` (auth required), `PATCH /api/auth/me` (update name/role/country/province)
- [x] **Token refresh** — `POST /api/auth/refresh` (invalidates old session, creates new 30-day token)
- [x] **Account deletion** — `DELETE /api/auth/me` (removes user + all sessions + all API keys)
- [x] **API key management** — `POST /api/keys/generate` (named `xc-` prefixed keys), `GET /api/keys` (redacted list), `DELETE /api/keys/{key_preview}` (revoke)
- [x] **API key auth** — Bearer token authentication via API keys for all auth-protected endpoints
- [x] **In-memory stores** — `_users_db`, `_sessions`, `_api_keys` with `_user_lock` threading protection
- [x] **Pydantic models** — `RegisterRequest`, `LoginRequest`, `ProfileUpdateRequest`

### Analytics Tab (NEW)
- [x] **Usage analytics** — Calls `GET /api/analytics/usage` with group-by (day/week/gpu_model/province) + days selector (7/30/90)
- [x] **Bar chart** — Gradient-colored CSS bar chart for usage visualization
- [x] **Summary stats** — Total Jobs, Total Spend, GPU Hours, Avg GPU Utilization
- [x] **Breakdown table** — Period, Jobs, GPU Hours, Spend, Canadian columns
- [x] **Canada ratio** — Canada vs International compute ratio bar

### Artifacts Tab (NEW)
- [x] **Job lookup** — Search by Job ID, calls `GET /api/artifacts/{jobId}`
- [x] **Artifacts table** — ID, Type, Job, Size, Residency, Created, Actions/download
- [x] **Download** — Calls `POST /api/artifacts/download` for presigned URLs
- [x] **Upload form** — Job ID, type (job_output/model_checkpoint/dataset/log_bundle), residency policy, size
- [x] **Storage info** — B2 CA-East Montreal, R2 Global Cloudflare, auto-routing explanation

### Admin Tab (NEW — Role-Gated)
- [x] **Role gating** — Admin tab only visible when user role = admin
- [x] **Canada-Only toggle** — Switch control calling `GET/PUT /canada`
- [x] **Maintenance Mode toggle** — Switch control for maintenance state
- [x] **Spot/preemption controls** — Update spot prices, trigger preemption cycle, scale up/down
- [x] **System metrics** — Fetch and display `GET /metrics` data
- [x] **Audit chain** — Verify chain integrity via `GET /api/audit/verify-chain`
- [x] **Legal requests** — Record new via `POST /api/transparency/legal-request` with type/agency/description
- [x] **Image builder** — Build Docker images via `POST /build`, view recent builds via `GET /builds`

### Billing Tab Enhancements
- [x] **Transaction history** — Table with Date/Description/Amount/Balance, calls `GET /api/billing/wallet/{id}/history`
- [x] **CSV export** — Export transaction history as CSV blob download
- [x] **Reserved pricing cards** — 3 cards (1-month 20%, 3-month 30%, 1-year 45%) with Subscribe buttons calling `POST /api/pricing/reserve`

### Auth Flow Integration (Dashboard JS)
- [x] **Real OAuth** — `oauthLogin()` calls `POST /api/auth/oauth/{provider}` with dev-mode fallback
- [x] **Real email login** — `emailLogin()` calls `POST /api/auth/login` with dev-mode fallback
- [x] **Real registration** — `toggleSignup()` calls `POST /api/auth/register` with password validation
- [x] **Admin gating** — `showApp()` shows/hides admin tab based on role, loads admin panel for admins
- [x] **Auth-aware API keys** — `generateApiKey()` tries auth endpoint first, falls back to legacy `POST /token/generate`

### Tests (17 new — 682 total)
- [x] `TestUserAuth` (12 tests) — register, duplicate email, short password, login success/wrong password/nonexistent, OAuth 3 providers + invalid, get profile/unauth, update profile, refresh token, delete account
- [x] `TestApiKeys` (5 tests) — generate+list, API key as bearer auth, revoke, unauthenticated generate

---

## Phase 8.8 — Password Reset, Marketplace Search, Host Detail & Compliance Dashboard (v2.4.0)

### Backend — New API Endpoints (4)
- [x] `POST /api/auth/password-reset` — Generates `secrets.token_urlsafe(32)` reset token, stores in `_users_db` with 1-hour expiry. Returns same message regardless of email existence (security). Returns token in test env.
- [x] `POST /api/auth/password-reset/confirm` — Validates token + expiry, hashes new password, invalidates all user sessions. Message: "Password updated. Please log in again."
- [x] `POST /api/auth/change-password` — Auth-required, validates current password via `hmac.compare_digest`, sets new hash+salt. Message: "Password changed successfully"
- [x] `GET /marketplace/search` — 7 query params (`gpu_model`, `min_vram`, `max_price`, `province`, `country`, `min_reputation`, `sort_by`) + `limit` (default 50). Filters `get_marketplace(active_only=True)`. Returns `{listings, total, filters_applied}`.

### Backend — Bug Fixes (4)
- [x] Fixed `_hmac.compare_digest` → `hmac.compare_digest` (wrong module alias)
- [x] Fixed `_hash_password()` tuple unpacking at 3 locations — was assigning `(hash, salt)` tuple directly to `password_hash` instead of `hash, _ = _hash_password(...)`

### Dashboard — CSS (~40 new rules)
- [x] `.btn-danger` — red destructive action button
- [x] `.host-detail-panel`, `.host-detail-grid`, `.host-stat` — host detail modal layout
- [x] Severity colors: `.sev-info` (teal), `.sev-warning` (orange), `.sev-error` (red), `.sev-critical` (purple)
- [x] `.sev-badge` — severity indicator badge
- [x] `.compliance-grid`, `.compliance-cell` — province compliance matrix layout
- [x] `.detail-tabs`, `.detail-tab` — job detail tabbed interface
- [x] `.filter-bar` — marketplace/event log filter bar layout
- [x] `.collapsible-header`, `.collapsible-body` — advanced options toggle
- [x] `.log-viewer` — log display with monospace font

### Dashboard — HTML Structure (+621 lines → 3,280 total)
- [x] **Login page** — "Forgot password?" link + password reset form (email) + reset confirmation form (token + new password)
- [x] **Host Detail Modal** — Full modal with `host-detail-content`, marketplace toggle button
- [x] **Event Log** — Filter bar: severity dropdown, type dropdown, search input, clear button, event count
- [x] **Trust/Compliance** — Province matrix, tax rates table, trust tier requirements table, Quebec PIA button + result
- [x] **Settings** — Change Password card (current/new/confirm) + Danger Zone card (Delete Account `.btn-danger`)
- [x] **Submit Job** — GPU Count selector (1/2/4/8), Pricing Mode (On-Demand/Spot + hint), Advanced Options collapsible (NFS mount config + env vars)
- [x] **Marketplace** — Filter bar (GPU model, min VRAM, max price, province, sort, search)
- [x] **Host table** — Host IDs now clickable → `showHostDetail(hostId)`
- [x] **Job Detail** — Tabbed (Overview/Logs/Artifacts/Audit), cost breakdown, log viewer, artifacts table, audit trail

### Dashboard — JavaScript (~350 new lines)
- [x] `showHostDetail(hostId)` — Promise.allSettled for host/SLA/compute-score, stat grid + detail rows
- [x] `toggleHostMarketplace()` — POST to marketplace toggle
- [x] `showForgotPassword()`, `hideForgotPassword()`, `requestPasswordReset()`, `confirmPasswordReset()` — password reset flow
- [x] `changePassword()` — validates match, calls change-password endpoint
- [x] `deleteAccount()` — double confirmation, DELETE /api/auth/me
- [x] `searchMarketplace()` — URLSearchParams builder, calls GET /marketplace/search
- [x] `showJobDetailTab(tab)` — tab switching (overview/logs/artifacts/audit)
- [x] `fetchJobLogsInDetail(jobId)`, `streamJobLogsInDetail(jobId)` — log fetch + EventSource streaming
- [x] `fetchJobArtifactsInDetail(jobId)`, `fetchJobAuditInDetail(jobId)` — artifacts + audit trail
- [x] `formatBytes(b)` — file size utility
- [x] `addEventLogEntry(entry)`, `filterEventLog()`, `clearEventLog()` — event log filtering
- [x] `toggleAdvancedJob()` — collapsible advanced options toggle
- [x] `fetchComplianceData()` — provinces + tax rates + tier requirements from 3 endpoints
- [x] `runQuebecPIA()` — POST to /api/compliance/quebec-pia-check
- [x] `updateSpotHint()` — spot pricing mode hint

### Tests (20 new — 702 total)
- [x] `TestPasswordReset` (7 tests) — request, unknown email, confirm, old password fails, invalid token, short password, expired token
- [x] `TestChangePassword` (4 tests) — success, wrong current, short new, unauthenticated
- [x] `TestMarketplaceSearch` (9 tests) — all, by GPU, by VRAM, by price, by province, sort price, filters_applied, limit, empty result

---

## Phase 8.9 — Invoices, SLA Dashboard, Session Management & Alert Config (v2.5.0)

### Backend (2 new endpoints)
- [x] `GET /api/billing/invoices/{customer_id}` — Monthly invoice list with tax/CAF breakdown (limit param)
- [x] `GET /api/sla/hosts-summary` — Batch SLA status for all registered hosts (uptime %, violations)
- [x] Route ordering fix: `/api/sla/hosts-summary` before `/{host_id}` to prevent wildcard capture

### Dashboard CSS (~30 new rules)
- [x] `.tooltip-wrap` + `.tooltip-text` — Hover tooltips for reputation badges
- [x] `.sla-card-grid` + `.sla-card` + `.sla-uptime.ok/warn/crit` — Per-host SLA status cards with color-coded uptime
- [x] `.inv-status` + `.inv-paid` + `.inv-pending` — Invoice status badges
- [x] `.job-progress` + `.job-progress-bar` + `.job-progress-fill` — Running job elapsed time indicator
- [x] `.session-modal-overlay` + `.session-modal` — Session timeout warning modal
- [x] `.low-balance-banner` — Wallet low balance warning banner
- [x] `.txn-filter-bar` + `.txn-pill` — Transaction type filter pills
- [x] `.event-row` + `.event-detail` — Event log expandable detail rows

### Dashboard HTML (7 structural additions)
- [x] Low balance banner (`#low-balance-banner`) — Warning when wallet < $5
- [x] Transaction filter pills — All/Deposits/Charges/Refunds type filtering
- [x] Invoice table — 8-column table with amount, tax, CAF, status, download
- [x] Marketplace stats summary — Listings, Active, Avg $/hr, Total VRAM
- [x] SLA section expansion — Per-host cards, violations table, active downtimes table
- [x] Alert configuration — Telegram bot token/chat ID, mining threshold, save/load
- [x] Session timeout modal — "Session Expired" overlay with re-login button
- [x] GST threshold display — Progress bar + status in compliance section
- [x] Reputation trend chart — Unicode sparkline in reputation tab

### Dashboard JavaScript (~20 new functions)
- [x] `fetchInvoices()` — Invoice list with table rendering
- [x] `fetchTransactionHistoryFiltered()` + `filterTransactions()` + `renderFilteredTransactions()` — Type-filtered transaction display
- [x] `fetchSLAHostCards()` — Per-host SLA status cards with uptime coloring
- [x] `fetchSLAViolations()` + `fetchSLADowntimes()` — SLA violation/downtime tables
- [x] `fetchMarketplaceStats()` — Marketplace overview stats
- [x] `repTooltip()` — Reputation badge hover tooltips
- [x] `fetchReputationHistory()` — Unicode block sparkline trend chart
- [x] `jobProgress()` — Elapsed time indicator for running jobs
- [x] `checkLowBalance()` — Low balance detection + warning toast
- [x] `startSessionCheck()` + `refreshAuthToken()` — Session management + auto JWT refresh
- [x] `loadAlertConfig()` + `saveAlertConfig()` — Telegram alert configuration
- [x] `fetchGSTThreshold()` — GST registration progress display
- [x] `toggleEventDetail()` — Expandable event log detail rows
- [x] `handleMiningAlert()` + `handleSLAViolation()` — Enhanced SSE alert handlers

### Wiring (5 modified functions)
- [x] `fetchReputation()` — Tooltip wrapping for tier badges
- [x] `fetchJobs()` — Job progress indicator after status badge
- [x] `updateWalletDisplay()` — Low balance check integration
- [x] `connectSSE()` — Mining alert + SLA violation handlers
- [x] `refresh()` — Added filtered transactions, marketplace stats, invoices
- [x] `showApp()` — Session check initialization
- [x] `emailLogin()` + `oauthLogin()` — Login timestamp for session tracking
- [x] `fetchMyReputation()` — Reputation history chart call
- [x] `fetchComplianceData()` — GST threshold fetch call

### Tests (12 new, 714 total)
- [x] `tests/test_invoices_sla.py` — 12 tests:
  - `TestInvoiceList` (5 tests) — returns ok, empty no usage, limit param, default limit 12, field validation
  - `TestSLAHostsSummary` (7 tests) — empty, single host, multiple hosts, field structure, uptime range, pending status, province info

---

## Phase 10 — Persistent Auth, Teams & Provider UX (v2.6.0) ✅ COMPLETE

**Goal**: Replace in-memory auth stores with SQLite-backed persistence, add team/org management, improve provider UX with earnings panel & payout in host detail, add admin SLA enforcement button, notification preferences, and enhanced privacy consent UI.

### Backend Changes
1. ✅ **db.py — UserStore class** (~290 lines): Persistent auth via `data/auth.db` (5 tables: `users`, `sessions`, `api_keys`, `teams`, `team_members`, 7 indexes)
2. ✅ **api.py — Auth rewiring**: 16+ auth functions dual-pathed (persistent or in-memory via `XCELSIOR_PERSISTENT_AUTH` env var)
3. ✅ **api.py — 7 team management endpoints**: `POST /api/teams`, `GET /api/teams/me`, `GET /api/teams/{team_id}`, `POST /api/teams/{team_id}/members`, `DELETE /api/teams/{team_id}/members/{email}`, `DELETE /api/teams/{team_id}`
4. ✅ **api.py — API key persistence bug fix**: `POST /api/keys/generate` now writes to UserStore when persistent auth is enabled

### Dashboard Changes (v2.5.0 → v2.6.0)
5. ✅ **Team management UI** in Settings tab: create team (name + plan), view members, add/remove members, delete team
6. ✅ **Provider detail panel** in Host Detail modal: earnings, pending payout, jobs completed, uptime, "Request Payout" button
7. ✅ **"Enforce SLA" admin button** in System Controls card
8. ✅ **Notification preferences** in Settings: 6 notification categories + digest frequency
9. ✅ **Enhanced privacy consent** in Settings: data retention table, "Revoke All Consent" button
10. ✅ **~30 CSS rules** for teams, provider stats, notifications, privacy/retention

### Tests (test_persistent_auth.py — 40 tests)
- `TestUserStoreCRUD` (10 tests) — create/get/exists/update/delete users, cascading deletes, list
- `TestUserStoreSessions` (5 tests) — create/get/delete/cleanup sessions, expiry
- `TestUserStoreAPIKeys` (4 tests) — create/get/list/delete API keys
- `TestUserStoreTeams` (8 tests) — create/add/remove/delete teams, capacity enforcement, user teams
- `TestTeamEndpoints` (8 tests) — API integration: create/view/members/delete + auth + access control
- `TestPersistentAuthEndpoints` (4 tests) — register/login/me/key persistence
- `TestSLAEnforcement` (1 test) — SLA enforce endpoint validation

---

## Phase 11 — Slurm HPC Tab, Responsive Design & UX Polish (v2.7.0) ✅ COMPLETE

**Goal**: Add HPC/Slurm management UI tab, responsive design for tablet/mobile, spot bid form, SLA credit calculator display, and residency trace UI. Also fix pre-existing residency trace API bug.

### Dashboard Changes (v2.6.0 → v2.7.0)
1. ✅ **HPC/Slurm tab** (new 13th tab): Cluster profiles grid, submit job form (name/VRAM/priority/tier/GPUs/image/profile/dry-run), job status lookup, cancel button, dry-run script preview, recent submissions list
2. ✅ **Spot bid form** in Overview tab: place spot bid with name/VRAM/max bid/priority, integrated below spot prices table
3. ✅ **SLA credit calculator** display in Overview tab: uptime-to-credit table (10%/25%/100%), SLA hosts summary with per-host uptime %
4. ✅ **Residency trace UI** in Overview tab: job ID lookup, formatted trace display with all jurisdiction fields
5. ✅ **Responsive CSS** with 3 breakpoints:
   - 1024px: single-column grid, form layout
   - 768px: tab bar horizontal scroll, mobile tables, compact cards, mobile-friendly modals
   - 480px: full-width buttons, stacked layout

### API Fixes
6. ✅ **Fixed residency trace endpoint** (`GET /api/jurisdiction/residency-trace/{job_id}`): Now correctly constructs `HostJurisdiction` from host data and passes proper arguments to `generate_residency_trace()`
7. ✅ **Added `fetchSlurmProfiles()`** to dashboard refresh cycle

### Tests (test_slurm_ui.py — 41 tests)
- `TestSlurmProfiles` (2 tests) — profiles endpoint returns dict with known clusters
- `TestSlurmSubmit` (5 tests) — dry run, profile selection, validation (name/VRAM), multi-GPU
- `TestSlurmStatus` (2 tests) — status lookup, endpoint existence
- `TestSlurmCancel` (2 tests) — cancel nonexistent, endpoint existence
- `TestSpotBid` (5 tests) — submit bid, validation, spot prices, price update
- `TestSLAHostsSummary` (2 tests) — summary structure, uptime fields
- `TestSLAEnforce` (1 test) — credit info response
- `TestResidencyTrace` (2 tests) — nonexistent job 404, existing job trace
- `TestDashboardHTML` (20 tests) — version v2.7.0, Slurm tab (button/div/profiles/form/status/recent/script/switcher), spot bid form, SLA credit calc, residency trace UI, responsive media queries, JS functions, refresh cycle

---

## Phase 9 — Future Work

| Item | Priority | Status | Effort |
|------|----------|--------|--------|
| ~~Stripe Connect integration (payouts)~~ | ~~High~~ | ✅ Done (stripe_connect.py) | ~~2 days~~ |
| ~~SLA enforcement endpoint~~ | ~~High~~ | ✅ Done (sla.py + 5 endpoints) | ~~1 day~~ |
| ~~Per-job SSE log streaming~~ | ~~High~~ | ✅ Done (/jobs/{id}/logs/stream) | ~~0.5 days~~ |
| ~~Reserved pricing model~~ | ~~High~~ | ✅ Done (3 tiers) | ~~0.5 days~~ |
| ~~GST/HST $30k threshold~~ | ~~Medium~~ | ✅ Done (2 endpoints) | ~~0.5 days~~ |
| ~~Usage analytics endpoint~~ | ~~Medium~~ | ✅ Done (/api/analytics/usage) | ~~0.5 days~~ |
| ~~Fern SDK + llms.txt~~ | ~~Medium~~ | ✅ Done (fern/ config + llms.txt) | ~~0.5 days~~ |
| ~~Form UX overhaul~~ | ~~High~~ | ✅ Done (Phase 8.5) | ~~0.5 days~~ |
| ~~GitHub Actions CI pipeline~~ | ~~High~~ | ✅ Done (.github/workflows/ci.yml — lint+test, PG 16, Py 3.12, codecov) | ~~0.5 days~~ |
| gVisor auto-install in worker agent | Medium | ✅ Done (security.py `install_gvisor()` + worker_agent.py auto-install on startup) | 1 day |
| ~~Admin verification approve/reject panel~~ | ~~Medium~~ | ✅ Done (2 API endpoints + dashboard approve/reject buttons) | ~~0.5 days~~ |
| ~~Transparency report dashboard chart~~ | ~~Low~~ | ✅ Done (summary grid + bar chart + canary in Trust tab) | ~~0.5 days~~ |
| ~~Phase 7 test files~~ | ~~High~~ | ✅ Done (17 test files: 682 tests total — scheduler, API, security, billing, events, verification, reputation, jurisdiction, artifacts, privacy, db, slurm_adapter, worker_agent, integration, db_migrations, e2e, nvml_telemetry) | ~~2 days~~ |
| NFS mount support for shared model storage | Medium | ✅ Done (worker_agent.py mount/unmount, scheduler.py job fields, cli.py `--nfs-*`) | 1 day |
| Slurm cluster adapter | Low | ✅ Done (slurm_adapter.py — Nibi/Graham/Narval profiles, sbatch gen, status sync) | 3 days |
| CLI-to-web OAuth flow | Low | ✅ Done (RFC 8628 device flow: api.py endpoints, cli.py `login`/`logout`/`whoami`) | 2 days |
| Multi-GPU job support | Low | ✅ Done (scheduler.py `num_gpus`, allocator GPU-count filter, cli.py `--gpus N`) | 3 days |
| NVML GPU telemetry (pynvml) | High | ✅ Done (nvml_telemetry.py, 6-tier reputation, serial check, mem fragmentation) | 1 day |
| ~~Dashboard UI expansion~~ | ~~High~~ | ✅ Done (Phase 8.6 — login, wallet, settings, reputation, job detail, host remove, toasts) | ~~1 day~~ |
| ~~Dashboard v2.3.0 expansion~~ | ~~High~~ | ✅ Done (Phase 8.7 — real auth, analytics, artifacts, admin, billing enhancements) | ~~1 day~~ |
| ~~Dashboard v2.4.0 expansion~~ | ~~High~~ | ✅ Done (Phase 8.8 — password reset, marketplace search, host detail, compliance) | ~~1 day~~ |
| ~~Dashboard v2.5.0 expansion~~ | ~~High~~ | ✅ Done (Phase 8.9 — invoices, SLA dashboard, marketplace stats, session mgmt, alerts, GST threshold) | ~~1 day~~ |
| ~~UI/UX roadmap document~~ | ~~Medium~~ | ✅ Done (docs/UI_ROADMAP.md — 180 items, 14 phases, competitor analysis) | ~~0.5 days~~ |
| Persistent auth storage | Medium | ✅ Done (Phase 10 — db.py UserStore + auth.db, dual-path in api.py) | 1 day |
| ~~Password reset flow~~ | ~~Medium~~ | ✅ Done (Phase 8.8 — POST /api/auth/password-reset + /confirm + /change-password) | ~~0.5 days~~ |
| Real OAuth credentials | Medium | TODO — Configure Google/GitHub/HuggingFace OAuth app credentials | 0.5 days |
| Responsive mobile design | Low | ✅ Done (Phase 11 — media queries for 1024/768/480px, tab bar scroll, table scroll, mobile modals) | 1 day |
