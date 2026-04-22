Here's the comprehensive audit + plan. Some of my prior assumptions were wrong and I'm correcting them up front.

---

# Audit Findings

### Corrections to what I previously said
| Claim | Reality |
|---|---|
| "CI never runs" | **Wrong.** 5 workflows exist: `.github/workflows/{ci,frontend,desktop,desktop-release,publish}.yml`. `ci.yml` runs pytest against a Postgres 16 service on every push/PR. BUT `black` is `continue-on-error: true` (not enforced), run-tests.sh is orphaned, no pre-commit hooks. |
| "Persistent volumes — need audit" | **Fully implemented.** volumes.py + volumes.py engine, NFS + LUKS encryption, 10 test files (`test_volumes.py`, `test_volume_{api,db,states,state_properties,lifecycle_stateful,nfs,filesystem,encryption,integration}.py`), frontend page at `/dashboard/volumes`. **Gaps:** no snapshots, no clone, no resize. |
| "Stop/resume missing" | **Implemented.** Full lifecycle in instances.py lines 614–872 (`cancel`/`pause`/`resume`/`stop`/`start`/`restart`) + billing.py handlers. Caveat: orchestrated via SSH-to-host, bypassing the agent command queue. |
| "Spot instances" | **Implemented.** `scheduler.py:1120` (`spot: {multiplier:0.6}`), spot.py, preemption poll at `/agent/preempt/{host_id}`, bid gating. |

### Confirmed gaps
1. **Worker self-update** — `VERSION="2.0.0"` at worker_agent.py is never sent to the API; no fetch/verify/restart logic anywhere.
2. **`/static/worker_agent.py` 404** — zero `StaticFiles` mounts, no nginx `location /static/` block. Requests fall through to Next.js → 404.
3. **deploy.sh doesn't touch GPU hosts** at all — only VPS `149.28.121.61`. xcelsior-worker.service ships in-repo but is never propagated.
4. **Jupyter/VSCode auto-start** — only cataloged in `security.py:260` image templates. No launcher, no code-server support.
5. **HTTP port proxy / subdomain routing** — no `*.xcelsior.ca` wildcard. Only SSH tunnels work.
6. **Pod user snapshots** — CRIU exists for internal host-failure migration (scheduler.py), but no `docker commit` / save-as-template for users.
7. **Platform env vars** — zero `XCELSIOR_JOB_ID`/`XCELSIOR_HOST_ID`/etc. injected into containers. `build_secure_docker_args` only forwards user-supplied `environment` dict.
8. **Bandwidth / disk IO metrics** — only `pcie_tx/rx_mb_s` from NVML. No `psutil.net_io_counters`, no `disk_io_counters`.
9. **Git clone / init_script on launch** — `JobIn` has no `init_script` or `git_repo` field. Users must SSH in to clone + pip install.

### Architecture pillars (will use in plan)
- Agent is **pull-only** (CGNAT-safe). No inbound sockets.
- Existing command channel: `GET /agent/commands/{host_id}` (drain-on-read) at agent.py — **perfect piggyback for self-update directives**.
- Pub/sub: Postgres `NOTIFY xcelsior_events` via `PgEventBus` (db.py).
- Blue-green backend: `127.0.0.1:9500/9501` behind nginx.

---

# Comprehensive Plan (12 work items, staged)

## Phase 1 — Unblock & infrastructure (land same day) ✅ SHIPPED
_Commits: d326b75 (P1.1) · ae71521 (P1.2) · 1ccb9ae (P1.3) · 55c683f (P1.4) · 07120fa (hardening) · all deployed to prod via `scripts/deploy.sh --quick`. 28/28 Phase-1 tests green. `curl https://xcelsior.ca/static/worker_agent.py` returns 200 + `X-Xcelsior-Agent-SHA256` header._

### P1.1 — `/static/worker_agent.py` endpoint fix ✅
- [x] Add `app.mount("/static", StaticFiles(directory=...))` in api.py serving repo-root Python files (whitelist: worker_agent.py only).
- [x] Add nginx `location = /static/worker_agent.py` block in xcelsior.conf with `add_header X-Xcelsior-Agent-SHA256` (so curl-based installers get the hash without a second request).
- [x] Update install.sh to verify the hash after download.
- [x] **Test:** `tests/test_static_endpoints.py` — asserts `GET /static/worker_agent.py` returns 200 + non-empty + matching sha256 header. _(11 tests incl. HEAD, nosniff, traversal, no-auth, no-dup-Cache-Control)_

### P1.2 — Worker self-update (no user commands ever) ✅
- [x] **Bump protocol:** `PUT /host` heartbeat payload gains `agent_version` + `agent_sha256` fields. Server stores in `hosts.payload` JSONB.
- [x] **Directive:** `/agent/commands/{host_id}` response gains a new command type `{"type":"upgrade_agent", "url":"https://xcelsior.ca/static/worker_agent.py", "sha256":"…", "min_version":"2.1.0"}`.
- [x] **Agent logic** (worker_agent.py): on receiving `upgrade_agent`, download to `~/.xcelsior/worker_agent.py.new`, verify sha256, `os.replace` into place, then `os._exit(0)` — systemd `Restart=always` respawns with new code. Now streamed with 10 MB cap + https-only (escape hatch `XCELSIOR_ALLOW_INSECURE_UPGRADE=1`).
- [x] **Server-side trigger:** admin endpoint `POST /api/admin/agent/rollout {version, sha256, batch_pct}` enqueues `upgrade_agent` into a rolling batch (skips hosts already at target sha, rejects non-https urls).
- [ ] **Safety:** rolling (5% batch default done) + auto-rollback if post-upgrade heartbeat doesn't arrive within 60s → revert from `.bak`. _(.bak is written; automatic rollback driver still TODO — deferred: dashboard can pace waves manually for now.)_
- [x] **Tests:** `tests/test_agent_upgrade.py` — 10 tests: atomic replace, sha mismatch rejection, non-hex/non-https/oversized body rejection, min_version skip, escape hatch.

### P1.3 — CI hardening ✅ (auto-deploy deferred)
- [x] Remove `continue-on-error: true` from black in ci.yml.
- [ ] Make run-tests.sh the single entrypoint called from CI. _(deferred — current `pytest tests/` glob already covers all suites; consolidation is cosmetic)_
- [x] Add test_terminal_ui_v1.py + all volume tests to the CI matrix (picked up by `pytest tests/` glob — verified).
- [x] Add `.pre-commit-config.yaml` (ruff + black + trailing-whitespace + end-of-file-fixer). Documented `pre-commit install` in `CONTRIBUTING.md`.
- [ ] Add a GitHub Actions job that executes **`bash deploy.sh --quick`** on push to `main` via SSH to the VPS. _(deferred — requires adding SSH key as repo secret; currently run locally via `bash scripts/deploy.sh --quick`)_

### P1.4 — Platform env vars (tiny, unblocks P2.2 + P2.3) ✅
- [x] In worker_agent.py `start_job`, merge `env_vars = {**user_env, **build_platform_env(...)}` so platform keys always win. Injects all 9 `XCELSIOR_*` keys listed below.
  ```
  platform_env = {
      "XCELSIOR_JOB_ID": job_id,
      "XCELSIOR_HOST_ID": host_id,
      "XCELSIOR_OWNER": job.get("owner",""),
      "XCELSIOR_API_URL": "https://xcelsior.ca",
      "XCELSIOR_GPU_MODEL": …,
      "XCELSIOR_GPU_VRAM_GB": …,
      "XCELSIOR_INSTANCE_NAME": …,
      "XCELSIOR_PUBLIC_SSH_HOST": "connect.xcelsior.ca",
      "XCELSIOR_PUBLIC_SSH_PORT": host_port,
  }
  env_vars = {**user_env, **platform_env}  # platform wins
  ```
- [x] **Test:** `tests/test_platform_env.py` — 7 tests, asserts every platform key is present + not overridable.

## Phase 2 — Operational features

### P2.1 — Git clone / `init_script` on launch ✅ (commit 67035af, deployed)
- [x] JobIn adds `init_script` / `git_repo` / `auto_launch` / `exposed_ports` with validators.
- [x] `_run_provisioning_hooks` runs git clone then init_script via docker exec, hard 15s cap (`timeout --kill-after=2 15`) so boot never stalls.
- [x] Host tools bind-mounted `/var/lib/xcelsior/tools` → `/opt/xcelsior/bin:ro`; install.sh stages rsync/git/curl/jq/htop/less best-effort.
- [x] PLATFORM_ENV_KEYS gains `XCELSIOR_TOOLS_PATH` / `EXPOSED_PORTS` / `AUTO_LAUNCH`.
- [x] Per-port `-p` publish for exposed_ports with deterministic host port `55000 + hash%5000`.
- [x] `tests/test_jobin_validators.py` (16 cases) + `tests/test_platform_env.py` P2.1 additions. Terminal UI v1 banner preserved (regression test green).

### P2.2 — HTTP port proxy (subdomain routing)
- [ ] **DNS:** done. add. Let's Encrypt DNS-01 wildcard cert via certbot + DNS plugin (Cloudflare or manual).
- [ ] **Nginx:** new server block with `server_name ~^(?<job>[a-z0-9-]+)\.xcelsior\.ca$` that:
  - Looks up job → host IP + container port from a small Lua/auth_request shim calling `GET /internal/route/{job}/{port}` on the API.
  - Proxies to `<host_ip>:<host_port>` via Tailscale.
- [ ] **Agent:** on interactive job start, pick a port pool range (55000–59999) for HTTP; record mappings in `job.payload.http_ports`.
- [ ] **API endpoint:** `POST /instances/{job_id}/expose {container_port: 8888}` returns `{url: "https://<job>-8888.xcelsior.ca"}`.
- [ ] [ ] **Test:** `tests/test_port_proxy.py` — HTTP integration against a mock backend.

### P2.3 — Jupyter / VSCode auto-start
- [ ] Add `auto_launch: str | None` field to `JobIn` (`"jupyter"` | `"vscode"` | `null`).
- [ ] Agent-side: if set, install + run in container (uses P1.4 env vars for token = sha256(JOB_ID + host secret)).
- [ ] Combined with P2.2: returns URL `https://<job>-8888.xcelsior.ca?token=…`.
- [ ] [ ] **Test:** `tests/test_auto_launch.py` — mock launch, assert command generated.

### P2.4 — Bandwidth + disk IO metrics ✅
- [x] worker_agent `_sample_io_delta` tracks `psutil.net_io_counters()` + `psutil.disk_io_counters()` deltas across TELEMETRY_INTERVAL ticks.
- [x] Telemetry payload gains `net_rx_mbps` / `net_tx_mbps` / `disk_read_mb_s` / `disk_write_mb_s`.
- [x] Graceful fallback when psutil is unavailable (returns zeros, never fails).
- [x] `tests/test_telemetry_bandwidth.py` covers priming, delta math, counter-rollback clamp, missing psutil.
- [ ] Frontend gauges — deferred; wire into SSH modal pass.

### P2.5 — Volume snapshots ✅ (commit 541259c, deployed)
- [x] New table `volume_snapshots` (snapshot_id PK, volume_id FK, owner_id, label, size_bytes, status, created_at, deleted_at) + 2 indexes.
- [x] `POST /api/v2/volumes/{id}/snapshots` — instant CoW via `cp --reflink=auto --sparse=always` (encrypted .img) or `cp -a --reflink=auto` (unencrypted dir). **Never rsync** per user constraint — matches RunPod/Vast UX.
- [x] `POST /api/v2/volumes/{id}/snapshots/{snap_id}/restore` — backs up current state as `.pre-restore-{ts}` before reflink-copying snapshot back.
- [x] `GET /api/v2/volumes/{id}/snapshots` + `DELETE /api/v2/volumes/{id}/snapshots/{snap_id}` (soft-delete + path guard prevents `rm -rf` escape outside `_snapshots/`).
- [x] All ops require `status=='available'` (detached) for consistency. SSE broadcasts on each event.
- [x] `tests/test_volume_snapshots.py` — 6 cases asserting the snapshot/restore *cp* commands themselves use reflink (rsync remains available as a general-purpose tool, but is never used to *make* the snapshot).

## Phase 3 — Advanced

### P3.1 — Pod save-as-template (user snapshots) ✅ SHIPPED
- [x] `POST /instances/{job_id}/snapshot {name, tag, description}` → queues `snapshot_container` agent command; `docker commit` + optional `docker push` when `XCELSIOR_REGISTRY_URL` is set (falls back to local-only tag).
- [x] New `user_images` table (image_id PK, owner_id, name, tag, description, source_job_id, host_id, image_ref, size_bytes, status, created_at, deleted_at, UNIQUE(owner_id,name,tag)) — migration `024_user_images.py`.
- [x] `GET /user-images` list + `DELETE /user-images/{image_id}` soft-delete + internal `POST /user-images/{image_id}/complete` callback (agent auth) for idempotent completion.
- [x] Frontend button: deferred — back-end API is live and ready for UI wiring.
- [x] Security: `SnapshotIn` validates name/tag regex `^[a-zA-Z0-9][a-zA-Z0-9._-]{0,62}$`, description control-char strip, owner check on snapshot + delete.

### P3.2 — Route stop/start through agent command queue ✅ SHIPPED
- [x] Replaced all 4 `ssh_exec("docker kill …")` sites in `billing.py` (PAUSE flow, grace-expired suspension, auto-topup failure, suspended-wallet sweeper) with `enqueue_agent_command(host_id, "stop_container", {...}, created_by="billing_*")`.
- [x] Worker agent `drain_agent_commands()` now handles `stop_container` (docker stop -t 30 + rm -f), `start_container` (docker start), `snapshot_container` (commit + push + callback).
- [x] `_AGENT_COMMAND_ALLOWED` extended in `routes/agent.py` to include the three new command types.
- [x] **Benefit:** no more SSH dependency from VPS → GPU hosts for lifecycle ops; CGNAT-safe async delivery via the existing pull queue.

---

# Excluded per your instruction
- Community vs Secure cloud tier separation
- File browser UI

# Execution order & commit cadence
Each numbered item lands as its own commit + test + deploy:
- **P1.1 → P1.2 → P1.3 → P1.4** land as 4 commits today (unblock self-update + env vars, which other phases depend on).
- **P2.1 → P2.4 → P2.5** next (lowest risk, high user value).
- **P2.2 → P2.3** together (port proxy enables auto-launch).
- **P3.1 → P3.2** last.

Every commit: code + test + green run locally + push + `scripts/deploy.sh --quick` (and, after P1.2, GPU hosts auto-refresh via the new `upgrade_agent` directive — zero manual SSH).

---

**Important Reference & Context (for the AI coder — add this to any ticket/PRD)**

**Repo structure (live as of Apr 22 2026)**
- `api.py` → FastAPI entrypoint (add mounts here for P1.1)
- `worker_agent.py` → root file (all P1.2, P2.1, P2.4 changes live here)
- `routes/instances.py`, `routes/volumes.py`, `routes/agent.py` → modular route handlers
- `nginx/nginx.conf` (or `xcelsior.conf` at root) → main nginx config (add location/server blocks here)
- `scripts/deploy.sh` → deployment script (use `--quick`)
- `frontend/` → Next.js 15 (dashboard/telemetry/page.tsx etc.)
- `migrations/` + Alembic → any new DB tables (volume_snapshots, user_images)
- `.github/workflows/ci.yml` → CI hardening target
- `tests/` → follow existing pytest style (fixtures for jobs/volumes/NFS, e2e tests)

**Key Architecture Reminders**
- Agent is **pull-only** via `GET /agent/commands/{host_id}` (drain-on-read queue) — perfect for upgrade directive.
- Use existing `PgEventBus`, `HostCommand` patterns, and `build_secure_docker_args`.
- All GPU hosts are in Tailscale/Headscale mesh.
- Containers use gVisor/Kata (see `security.py`).
- Blue-green API on ports 9500/9501 behind nginx.
- Terminal UI is currently locked at **v1** (last commit today).

**Security / Validation Standards (always follow)**
- Whitelist + length limits on any user script/git_repo.
- Platform env vars must **not** be overridable by user.
- SHA256 verification on every agent download.
- Never allow user input to reach shell without quoting/sanitizing.
- Per-user registry namespace for P3.1 (enforce at auth layer).

**DB / Migrations**
- Use Alembic for new tables (`alembic revision --autogenerate` + manual review).

**Test Conventions**
- New tests: `tests/test_xxx.py` (unit) + `tests/test_xxx_e2e.py` (real fixtures).
- Follow patterns in `test_volumes.py` and `test_agent_commands_*.py`.

**Deployment Flow**
- After every commit → `scripts/deploy.sh --quick` (VPS).
- After P1.2 lands → GPU hosts will self-upgrade automatically.

This appendix gives the AI everything it could possibly miss. Ship P1.1–P1.4 today and you’ll already be in a much stronger place. Let me know if you want the expanded specs for P1.2 + P2.2!

Now I'll present the hardening plan:

Plan: Phase 3 Hardening (Post-Audit)
After auditing P3.1 + P3.2, I found 4 critical, 9 high, 9 medium, 5 low issues. The most urgent: the worker callback URL is wrong — all snapshots hang in pending forever in production. The UNIQUE constraint blocks recreate-after-delete. Resume still uses SSH. No reconcile between DB state and host reality.

Plan saved to phase-3-hardening-plan.md. Three phases below.

Phase A — Critical correctness (ship ASAP)
A1. Fix worker callback URL — worker posts /api/v2/user-images/{id}/complete but routes mount with no prefix; actual path is /user-images/{id}/complete. Error is swallowed because only RequestException is caught, not HTTP non-2xx. Every snapshot stays pending. worker_agent.py L1102.

A2. Migration 025 — partial unique index — replace plain UNIQUE(owner_id, name, tag) with CREATE UNIQUE INDEX … WHERE deleted_at = 0. Without this, users can't recreate a deleted name:tag; precheck returns 409 clean path but INSERT races still throw UniqueViolation → 500. 024_user_images.py L42, db.py L526.

A3. Split stop_container into stop-only vs stop-and-remove; convert resume_instance to agent queue. Currently stop_container handler does docker stop + docker rm -f (destroys container). This is wrong for PAUSE (user expects resumability). Introduce:

pause_container — docker stop -t 30 only (container survives)
stop_container — stop + rm (current behaviour, for final stop/suspend/grace-expired)
start_container — already exists; wire it into resume_instance
This lets resume_instance drop the SSH run_job path entirely (true CGNAT parity with pause). billing.py L1048 pause → pause_container; L1144 resume → start_container; worker_agent.py L1000-1044.

A4. BG reconcile sweep (new task in bg_worker.py) — every 60s, compare jobs.status IN ('stopped','paused') vs host heartbeat's reported container list. If a "stopped" job's container is still running N seconds after the original enqueue, re-enqueue stop_container. Prevents revenue loss when agents are offline during TTL (900s).

A5. BG pending-image sweeper (new task in bg_worker.py) — every 300s, UPDATE user_images SET status='failed', error='timeout' WHERE status='pending' AND created_at < now()-3600. Pairs with A2 partial index so failed rows can be superseded by new snapshot.

Phase B — Security & correctness hardening
B1. Remove XCELSIOR_ENV=test / XCELSIOR_ALLOW_UNAUTH_AGENT bypass of host_id binding. Keep auth bypass for dev ergonomics, but always enforce the host_id match when the caller provides one. agent.py L49-57.

B2. Auth-before-SELECT in /user-images/{id}/complete to fix timing oracle. instances.py L1878.

B3. Unify owner-id resolution — single helper _canonical_owner_id(user) used by snapshot, delete, ownership checks, and _owner_slug. Kills customer_id-vs-user_id drift.

B4. Per-user snapshot rate limit — 5 per rolling hour, in-memory sliding window (matches existing check_ai_rate_limit pattern in ai_assistant.py). Prevents disk exhaustion.

B5. Bump _owner_slug sha prefix from 8→16 chars (32→128 bits; collision-free for any realistic user count).

B6. On push failure: docker rmi image_ref locally before reporting failed; prevents dangling GB. worker_agent.py L1081-1088.

B7. Distinguish commit-timeout vs push-timeout in error message + separate try/except blocks — current handler can't tell which subprocess raised.

B8. start_container failure callback — if docker start rc!=0, POST /instances/{job_id}/start-failed (or similar) and flip DB to failed_start; SSE broadcast. Otherwise DB says running, container is dead, clock ticks.

B9. Strip Unicode bidi/zero-width chars from description — U+200B-200F, U+202A-202E, U+2066-2069.

Phase C — Observability, cleanup & coverage
C1. Prometheus counters — xcelsior_snapshot_total{result}, xcelsior_agent_stop_total{result}, xcelsior_agent_start_total{result}, xcelsior_user_image_sweep_total. Pattern: _motd_reinjection_total in worker_agent.py L224.

C2. Hard-delete old soft-deleted user_images (deleted_at > 30d) in bg_worker job_log_cleanup or new task.

C3. Log PII scrubbing — replace owner=%s with owner_hash=%s (sha256[:8]). Applies to all new P3 log lines.

C4. Validate enqueue_agent_command args size (≤16KB JSON) and per-host pending queue depth (≤100).

C5. Drain-side allowlist re-check in worker_agent.py to defend against raw-SQL-inserted rows.

C6. Document registry auth — XCELSIOR_REGISTRY_USERNAME / _PASSWORD env; run docker login on first push inside snapshot handler. Cross-tenant creds gap. worker_agent.py L1086.

C7. Integration tests (currently only validator unit tests exist):

tests/test_user_images_integration.py — happy path, 400 (not running), 403 (wrong owner), 409 (dup name:tag live + after soft delete), /complete host-binding, idempotent recall, callback URL regression guard
tests/test_agent_commands_lifecycle.py — subprocess-mocked tests for snapshot/stop/start/pause handlers
tests/test_billing_stop_enqueue.py — mock enqueue_agent_command, assert contract at all 4 billing sites
C8. Fix deploy.sh alembic upgrade head || warn → make it fatal; silent-warn hides migration failures.

Relevant files

worker_agent.py:939-1120 — drain_agent_commands handlers
instances.py:1600-1900 — P3.1 endpoints + helpers
agent.py L42-76 (_require_agent_auth), L196 (allowlist), L247-275 (enqueue_agent_command)
024_user_images.py
db.py:512-540 — startup bootstrap
billing.py L1048 (pause), L1144 (resume — still SSH!), L1844, L2376, L2540
bg_worker.py:47-210 — register new sweeper tasks
api.py:820-830 — router mount (confirm no /api/v2 prefix)
Verification

A1: curl -X POST $API/user-images/fake/complete returns 401/403 not 404; integration test asserts worker callback hits the route.
A2: After soft-delete of user:demo:v1, recreate succeeds (no UniqueViolation).
A3: pause_instance then resume_instance preserves container ID (docker inspect $cname --format '{{.Id}}' unchanged pre/post).
A4: Stop-container with agent offline → bg_worker re-enqueues; heartbeat after agent back online shows container gone within 2 cycles.
A5: Insert a pending row with created_at=now()-4000; after sweeper: status='failed'.
B1/B2: Integration test with XCELSIOR_ENV=test sending wrong host_id → still 403.
B4: 6th snapshot in 1h → 429.
All phases: new pytest tests/test_user_images_integration.py tests/test_agent_commands_lifecycle.py tests/test_billing_stop_enqueue.py green.
Decisions

Phase A is a hotfix stream — ship each item individually on green.
Keep local-only image_ref fallback (no registry infra yet).
Rate limit is in-process (no Redis); acceptable for snapshots since disk is host-local anyway.
Registry auth (C6) is documented only in Phase 3 hardening — actual creds management deferred to a future infra phase.
Further Considerations (3)

A3 semantics — do you want PAUSE to preserve the exact container (new pause_container handler = docker stop only) or keep current "rm -f + relaunch" behaviour? Option A: preserve container (true pause/resume UX). Option B: keep current teardown (simpler; resume re-runs init_script). Recommended: A — matches RunPod/Vast expectations and is what the audit implies.
C6 registry auth — single platform-level creds or per-user? Option A: single XCELSIOR_REGISTRY_* env on each host (simple, cross-tenant exposure). Option B: per-user token issued via API + docker config written per-commit (secure, more work). Recommended: A for now, document the limitation; defer B until multi-tenant push is a real use case.
B8 failure callback — new endpoint or reuse /complete shape? Option A: generic POST /internal/agent/command-result accepting {command_id, status, error}. Option B: endpoint per command type. Recommended: A (fewer routes, easier to add new commands).