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

## Phase 1 — Unblock & infrastructure (land same day)

### P1.1 — `/static/worker_agent.py` endpoint fix
- [ ] Add `app.mount("/static", StaticFiles(directory=...))` in api.py serving repo-root Python files (whitelist: worker_agent.py only).
- [ ] Add nginx `location = /static/worker_agent.py` block in xcelsior.conf with `add_header X-Xcelsior-Agent-SHA256` (so curl-based installers get the hash without a second request).
- [ ] Update install.sh to verify the hash after download.
- [ ] [ ] **Test:** `tests/test_static_endpoints.py` — asserts `GET /static/worker_agent.py` returns 200 + non-empty + matching sha256 header.

### P1.2 — Worker self-update (no user commands ever)
- [ ] **Bump protocol:** `PUT /host` heartbeat payload gains `agent_version` + `agent_sha256` fields. Server stores in `hosts.payload` JSONB.
- [ ] **Directive:** `/agent/commands/{host_id}` response gains a new command type `{"type":"upgrade_agent", "url":"https://xcelsior.ca/static/worker_agent.py", "sha256":"…", "min_version":"2.1.0"}`.
- [ ] **Agent logic** (worker_agent.py): on receiving `upgrade_agent`, download to `~/.xcelsior/worker_agent.py.new`, verify sha256, `os.replace` into place, then `os.execv` itself via systemd restart hook (`systemctl restart xcelsior-worker` from the root-granted ExecStartPre, OR emit a marker file + exit — systemd's `Restart=always` respawns with new code).
- [ ] **Server-side trigger:** admin endpoint `POST /admin/agent/roll-out {version, sha256}` enqueues `upgrade_agent` into every active host's command queue. Also called automatically when a new worker_agent.py lands in the deployed `api` container (compare running file's sha256 to last-announced).
- [ ] **Safety:** rolling (5% batch) + auto-rollback if post-upgrade heartbeat doesn't arrive within 60s → revert from `.bak`.
- [ ] [ ] **Tests:** `tests/test_agent_upgrade.py` — mock directive, mock filesystem, verify atomic replace + sha mismatch rejection + rollback path.

### P1.3 — CI hardening
- [ ] Remove `continue-on-error: true` from black in ci.yml.
- [ ] Make run-tests.sh the single entrypoint called from CI (fixes the duplication noted in audit).
- [ ] Add test_terminal_ui_v1.py + all volume tests to the CI matrix (already picked up by `pytest tests/` glob — verify).
- [ ] Add `.pre-commit-config.yaml` (ruff + black + trailing-whitespace + end-of-file-fixer). Document `pre-commit install` in `CONTRIBUTING.md`.
- [ ] [ ] Add a GitHub Actions job that executes **`bash deploy.sh --quick`** on push to `main` via SSH to the VPS (so I never have to run it locally again).

### P1.4 — Platform env vars (tiny, unblocks P2.2 + P2.3)
- [ ] In worker_agent.py `start_job`, merge before `build_secure_docker_args`:
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
  env_vars = {**platform_env, **user_env}  # user cannot override
  ```
- [ ] [ ] **Test:** `tests/test_platform_env.py` — asserts every platform key is present + not overridable.

## Phase 2 — Operational features

### P2.1 — Git clone / `init_script` on launch
- [ ] Add `init_script: str | None` + `git_repo: str | None` to `JobIn` in instances.py (length ≤ 4 KiB, ASCII-printable, whitelist regex; store in `job.payload`).
- [ ] In the interactive init_script (v1-locked banner first, then **appended** after `Setting up SSH…`):
  - If `git_repo` set: `git clone <url> /workspace && cd /workspace` (shell-quoted).
  - If `init_script` set: run it under `bash -e` with output streamed to the same log channel.
- [ ] **Test:** `tests/test_init_script.py` — submits job with fake repo URL, asserts clone attempted, rejects invalid scripts (shell metachars, too-long).
- [ ] [ ] **Terminal UI:** bumps to **v2** in lockstep (new lines after v1 banner are additions — the v1 lock test gets a new section + `TERMINAL_UI_VERSION="v2"`).

### P2.2 — HTTP port proxy (subdomain routing)
- [ ] **DNS:** add `*.xcelsior.ca` → `149.28.121.61`. Let's Encrypt DNS-01 wildcard cert via certbot + DNS plugin (Cloudflare or manual).
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

### P2.4 — Bandwidth + disk IO metrics
- [ ] In worker_agent.py `telemetry_loop`, add `psutil.net_io_counters(pernic=True)` + `psutil.disk_io_counters(perdisk=False)` deltas (bytes/sec since last tick).
- [ ] Extend telemetry payload: `{net_rx_mbps, net_tx_mbps, disk_read_mbps, disk_write_mbps}`.
- [ ] Frontend: 2 new gauges in dashboard/telemetry/page.tsx/dashboard/telemetry/page.tsx).
- [ ] [ ] **Test:** `tests/test_telemetry_bandwidth.py` + `frontend/e2e/telemetry-bandwidth.spec.ts`.

### P2.5 — Volume snapshots
- [ ] New table `volume_snapshots` (id, volume_id, created_at, size_bytes, status).
- [ ] `POST /api/v2/volumes/{id}/snapshots` → agent command to `rsync` the mounted volume to NFS snapshot dir with content-addressable naming.
- [ ] `POST /api/v2/volumes/{id}/restore?snapshot_id=…` → clone back.
- [ ] **Tests:** extend test_volumes.py with `test_snapshot_create`/`restore`/`delete`/`list`.
- [ ] [ ] **E2E:** `tests/test_volume_snapshots_e2e.py` using real local NFS fixture.

## Phase 3 — Advanced

### P3.1 — Pod save-as-template (user snapshots)
- [ ] `POST /instances/{job_id}/snapshot {name, description}` → queues agent command `docker commit <container> registry.xcelsior.ca/<owner>/<name>:<ts>` + push to per-user registry.
- [ ] Stored in new `user_images` table (owner_id, image_ref, size, source_job_id).
- [ ] Frontend: "Save as template" button in instance detail page.
- [ ] Security: per-user image namespace enforced at registry auth (Caddy or nginx auth-request).
- [ ] [ ] **Test:** `tests/test_user_snapshots.py`.

### P3.2 — Route stop/start through agent command queue
- [ ] Currently `billing.py:1016` does `ssh_exec(host["ip"], "docker kill …")` — replace with enqueue `{"type":"stop_container", "container": "..."}` into `/agent/commands/{host_id}`.
- [ ] Agent handles `stop_container` / `start_container` as new command types.
- [ ] **Benefit:** no more SSH dependency from VPS → GPU hosts; agent owns lifecycle.
- [ ] [ ] **Test:** `tests/test_agent_commands_lifecycle.py`.

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