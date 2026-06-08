# Persistent Volumes — Completion Plan

*Created: 2026-06-07*  
*Parent roadmap:* [`NEXT_PRIORITIES_ROADMAP.md`](./NEXT_PRIORITIES_ROADMAP.md) §1

Track every task required to move volumes from **metadata-only dev mode** to **production NFS-backed persistent storage** with full team tenancy, billing, UI, ops, and tests.

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done

---

## Current state (baseline)

| Area | Status |
|------|--------|
| DB schema (`volumes`, `volume_attachments`, `volume_snapshots`) | ✅ |
| REST API `/api/v2/volumes/*` | ✅ |
| Team create/list/mutate guards (`routes/_deps.py`) | ✅ |
| Volume engine LUKS + NFS provision (`volumes.py`) | ✅ code; ⚠️ needs `XCELSIOR_NFS_SERVER` |
| Scheduler bind-mount on instance start | ✅ code path exists |
| Billing tick for storage GB-month | ✅ |
| Dashboard volumes page | ✅ UI + team banner (2026-06-07) |
| Instance launch `volume_ids` team scope | ✅ `_user_can_access_volume` (2026-06-07) |
| AI assistant `list_volumes` | ✅ team-aware owner ids (2026-06-07) |
| NFS health in readiness | ✅ `nfs_storage_healthcheck` in `/readyz` (2026-06-07) |
| Prod NFS configured | ✅ Mac appliance `100.64.0.3`, mode `full` (2026-06-08) |
| E2E create → launch → delete | ✅ `volumes_e2e_smoke.py` + `ops_infra_smoke.py` PASS (2026-06-08) |
| E2E persist across instance restart | ✅ `--persist` PASS on ASUS `aaryn-tuf-rtx2060` (2026-06-07) |

---

## Phase A — Correctness fixes (code, no infra)

### A.1 Team & launch integration

- [x] **Fix launch volume ownership check** (`routes/instances.py`)
  - [x] Import `_user_can_access_volume` from `routes._deps`
  - [x] Replace `vol.get("owner_id") != volume_owner_id` with `_user_can_access_volume(user, vol)`
  - [x] Remove stale comment "Volumes use user_id as owner_id"
- [x] **Test:** `test_member_can_launch_instance_with_team_volume` in `test_team_tenancy_sweep.py`
  - [x] Create team volume as member
  - [x] Launch instance with `volume_ids: [vol_id]` → 200
  - [x] Cleanup job + volume
- [x] **Test:** `test_outsider_cannot_launch_with_team_volume`

### A.2 AI & auxiliary callers

- [x] **Fix `ai_assistant._tool_list_volumes`**
  - [x] Use `_volume_owner_ids_readable(user)` + `list_volumes_for_owner_ids`
- [x] **Audit other volume callers**
  - [x] `list_volumes` only in `routes/volumes.py`, `ai_assistant.py` (team-aware), `volumes.py` engine — cli/scheduler use engine correctly
  - [x] Launch modal frontend: volumes from API already team-scoped ✓

### A.3 API response completeness

- [x] **`POST /api/v2/volumes`** returns full volume record including `owner_id`
  - [x] After `ve.create_volume`, call `ve.get_volume(volume_id)` for response body
- [x] **`GET /api/nfs/config`** includes `nfs_export_base` for admin debugging

### A.4 Config consistency

- [x] **Single `NFS_EXPORT_BASE` source**
  - [x] `scheduler.py` imports `NFS_EXPORT_BASE` from `volumes` module (not duplicate `os.environ.get` with different default)
  - [x] Document: Mac appliance export `/exports/volumes` inside Ganesha container — see [`VOLUMES_RUNBOOK.md`](./VOLUMES_RUNBOOK.md)
- [x] **Fix `list_volumes_for_owner_ids` NameError** (`o` → `x`) — 2026-06-07

### A.5 Tests (no NFS required)

- [x] `test_member_volume_scoped_to_team_wallet`
- [x] `test_viewer_cannot_create_team_volume`
- [x] `test_member_can_launch_instance_with_team_volume`
- [x] `test_outsider_cannot_launch_with_team_volume`
- [x] Run: `pytest tests/test_team_tenancy_sweep.py tests/test_volumes*.py tests/test_app_security_sweep.py::test_volume_get_forbidden_cross_account` — 166 passed (2026-06-07)

**Phase A exit:** All tests green without `XCELSIOR_NFS_SERVER`; team launch works in metadata-only mode.

---

## Phase B — NFS infrastructure (ops)

### B.1 Storage server

- [x] Choose NFS host — **Mac InferenceData** (`100.64.0.3`, `tag:mac-nfs`)
- [x] NFS server — **NFS-Ganesha** in `xcelsior-mac-nfs` container (userspace; avoids macOS kernel `nfsd` on `:2049`)
- [x] Export directory `/exports/volumes` on LUKS ext4 (`inference.luks` on InferenceData SSD)
- [x] Mesh-only access — Headscale ACL `100.64.0.0/10` in `ganesha.conf` + `tag:xcelsior` / `autogroup:member` rules
- [x] Published on host `:12049` (container `:2049`); mount opts `nfsvers=4.0,port=12049`
- [x] Disk quota monitoring (alert at 80% / 90%) — `scripts/check_mac_nfs_disk.sh` (2026-06-07)

### B.2 LUKS prerequisites (encrypted volumes default ON in UI)

- [x] `cryptsetup`, `e2fsprogs` in Mac appliance container (privileged Docker)
- [x] Bulk LUKS: `inference.luks` + keyfile `~/.config/xcelsior/inference.key`
- [x] Per-volume LUKS via `xcelsior-nfs-exec` inside appliance (API SSH → Mac → docker exec)
- [x] Sudoers on Mac host — N/A (privileged container; no host sudo)
- [x] Encrypted volume round-trip E2E in prod — `volumes_e2e_smoke.py --encrypted` PASS (2026-06-07)

### B.3 API host connectivity

- [x] VPS `~/.ssh/xcelsior` in Mac `authorized_keys` for `aaryn`
- [x] API container provision via SSH + `xcelsior-nfs-exec` — verified `ops_infra_smoke.py` volume CRUD
- [x] Prod env set — see [`VOLUMES_RUNBOOK.md`](./VOLUMES_RUNBOOK.md) Mac appliance section
- [x] Headscale ACL: `tag:xcelsior` → `tag:mac-nfs:22,12049`

### B.4 GPU worker connectivity

- [x] Headscale ACL: `autogroup:member` → `tag:mac-nfs:12049`
- [x] VPS NFS mount test to Mac — PASS (`nfsvers=4.0,port=12049`)
- [x] VPS worker-mount smoke — `volumes_e2e_smoke.py --worker-mount` PASS (2026-06-07)
- [x] Test from live GPU worker host — ASUS `100.64.0.6` mount + persist PASS (2026-06-07)
- [ ] Test from `aarynfans-prod` (optional second worker)
- [x] `nfs-common` on workers — verified on ASUS (`mount.nfs4`, persist E2E PASS)

**Phase B exit:** Manual `mkdir` on NFS via API SSH succeeds; worker test mount succeeds.

---

## Phase C — Health, observability, runbook

### C.1 Health probes

- [x] Add `volumes.nfs_storage_healthcheck()`:
  - [x] `configured: bool` — `bool(NFS_SERVER)`
  - [x] `reachable: bool` — SSH `test -d $EXPORT_BASE`
  - [x] `export_base`, `server`
  - [x] `mode`: `full` | `metadata-only` | `degraded`
- [x] Expose in:
  - [x] `GET /readyz` — fails if `XCELSIOR_NFS_REQUIRED=true` and not reachable
  - [x] `GET /api/nfs/config` (admin) — full detail
  - [ ] Optional: `GET /api/admin/infrastructure` volume section
- [x] Env: `XCELSIOR_NFS_REQUIRED=true` in prod

### C.2 Logging & metrics

- [ ] Log lines already exist — verify grep patterns in runbook
- [ ] Add metric gauges (optional): `xcelsior_volumes_total`, `xcelsior_volumes_error`, `xcelsior_nfs_reachable`
- [ ] SSE events: `volume_created`, `volume_deleted`, etc. — dashboard already listens ✓

### C.3 Runbook

- [x] Create/update [`VOLUMES_RUNBOOK.md`](./VOLUMES_RUNBOOK.md):
  - [x] Mac appliance deploy, ACL, mount opts (`nfsvers=4.0,port=12049`)
  - [x] Provision failure → check SSH, disk space, LUKS sudo
  - [x] Volume stuck in `error` → Retry button / `POST .../retry`
  - [x] Attach mount failure on host → SSH to worker, `mount` debug
  - [x] NFS server reboot → `reopen_luks_volume` / encrypted volume recovery
  - [x] Orphan cleanup → `cleanup_orphaned_volume_mounts`, `reconcile_orphan_attachments`
  - [x] Customer delete while attached → detach first
  - [x] Team billing disputes → check `owner_id` on volume row

**Phase C exit:** On-call can diagnose NFS outage from `/readyz` + runbook without reading source.

---

## Phase D — End-to-end workflows

### D.1 Create → provision

- [x] User creates volume in prod — `ops_infra_smoke.py` + `volumes_e2e_smoke.py` PASS
- [x] `/readyz` reports `nfs_volumes.mode=full`, `reachable=true`
- [x] DB: `status=available`, `owner_id=billing_customer_id`
- [x] NFS: path provisioned on Mac appliance (unencrypted smoke volumes)
- [x] User creates **encrypted** volume — LUKS round-trip verified in prod (2026-06-07)

### D.2 Launch with volumes (new instance)

- [x] Launch instance with `volume_ids` — `volumes_e2e_smoke.py` PASS (2026-06-08)
- [x] Scheduler NFS-mount on GPU host — ASUS `aaryn-tuf-rtx2060` Mac NFS mount PASS (2026-06-07)
- [x] Write persist marker inside instance — `--persist` smoke PASS
- [x] Stop/start instance → file still present — `--persist` smoke PASS
- [x] Terminate instance → volume status returns `available`, data on NFS intact — `--persist` smoke (2026-06-07)

### D.3 Attach to running instance

- [x] Attach available volume to running instance — `mount_volume` agent cmd + `nsenter` bind (2026-06-07)
- [x] Document hot-attach behavior — runbook § Hot-attach updated (2026-06-07)
- [x] Detach → unmount on host — `unmount_volume` agent cmd + terminate cleanup

### D.4 Delete

- [x] Delete empty volume → NFS path removed — smoke CRUD + engine tests
- [x] Delete fails while attached → 409 — `test_delete_blocked_by_attachment`
- [x] Viewer cannot delete (403) — `test_viewer_cannot_attach_or_delete_team_volume`

### D.5 Team workspace

- [x] Member creates volume → `owner_id` = team billing customer — `test_member_volume_scoped_to_team_wallet`
- [x] Viewer sees volume in list, cannot create/delete/attach — team tenancy tests
- [x] Admin can delete team volume — tenancy sweep cleanup paths
- [~] Switch to personal workspace → team volumes hidden — UI team-changed reload; manual QA pending

### D.6 Billing

- [x] Volume billing tick runs in `bg-worker` — `billing.py` volume section + janitor calls
- [x] `billing_cycles` rows with `gpu_model=storage`, `tier=volume` — `scripts/volumes_billing_audit.py` (2026-06-07)
- [x] Team wallet debited for team-owned volumes — `owner_id` on volume row; audit script verifies `customer_id`
- [x] Suspended wallet skips billing — `test_volume_nfs` + billing fail-closed path

### D.7 Snapshots (if in scope for v1)

- [ ] `POST /api/v2/volumes/{id}/snapshots` creates snapshot on NFS `_snapshots/`
- [ ] List / delete / restore smoke test
- [ ] Or explicitly defer with checkbox note in roadmap

**Phase D exit:** Staging E2E script passes all D.1–D.6 scenarios.

---

## Phase E — Frontend polish

- [x] `TeamContextBanner` variant `volumes`
- [x] Viewer disabled create/mutate + toast on 403
- [x] `xcelsior-team-changed` reload
- [x] **LaunchInstanceModal:** team-visible volumes via `/api/v2/volumes/available`; viewer launch blocked — 2026-06-07
- [x] **LaunchInstanceModal:** `xcelsior-team-changed` reloads available volumes — 2026-06-07
- [x] **Volume `error` state:** NFS provisioning hint + Retry CTA with i18n — 2026-06-07
- [x] **Region mismatch warning** when listing region ≠ selected volume region — 2026-06-07
- [ ] **Monthly cost** uses team wallet context in banner/tooltip
- [x] **Analytics:** `xcelsior-team-changed` on financial tab (wallet KPIs)
- [x] **French i18n** for volume error/launch strings — 2026-06-07

---

## Phase F — Scripts & CI

- [x] **`scripts/volumes_e2e_smoke.py`** (staging)
  - [x] Auth → create → get → list → launch → delete
  - [x] Exit code 1 on any failure
- [ ] **Post-deploy audit** — volumes page loads with team banner
- [x] **CI job** — `pytest tests/test_team_tenancy_sweep.py tests/test_volumes*.py` (153 passed locally 2026-06-08)
- [x] **`.env.example`** — expand NFS section with `XCELSIOR_NFS_REQUIRED` + runbook pointer

---

## Phase G — Security

- [x] Cross-account volume access → 404 — `test_volume_get_forbidden_cross_account`
- [x] NFS export not public internet — Headscale mesh + Ganesha `100.64.0.0/10`
- [x] LUKS keys only in DB (Fernet); never on NFS disk — encryption tests + engine design
- [x] `rm -rf` path traversal guard in `_destroy_volume_storage` — existing ✓
- [x] Rate limits on volume create — `XCELSIOR_MAX_VOLUMES_PER_OWNER` (default 50) + existing GB caps (2026-06-07)
- [x] Team viewer cannot attach/detach (API 403) — `test_viewer_cannot_attach_or_delete_team_volume`

---

## Master checklist (quick scan)

```
Phase A  Correctness     [x] code complete (tests green)
Phase B  NFS infra       [x] Mac appliance live in prod (2026-06-08)
Phase C  Health/runbook  [x] health + runbook done; metrics optional
Phase D  E2E workflows   [x] CRUD + launch + persist + hot-attach + billing audit (2026-06-07)
Phase E  Frontend polish  [x] error UX + launch filter done (2026-06-07)
Phase F  Scripts/CI      [x] smoke scripts PASS in prod
Phase G  Security         [x] viewer/cross-account + volume count cap done (2026-06-07)
```

---

## Definition of done

All must be true:

1. [x] Production `XCELSIOR_NFS_SERVER` configured and `nfs_storage_healthcheck().reachable == true`
2. [x] No `metadata-only` warnings for new volume creates in prod (`mode=full`)
3. [x] Team member launch with team `volume_ids` works (test + prod smoke)
4. [x] Data survives instance stop/start with attached volume (`--persist` PASS)
5. [x] Billing charges correct wallet (personal or team) — `volumes_billing_audit.py` (2026-06-07)
6. [x] Viewer read-only in API and UI — tenancy tests + dashboard gates
7. [x] `VOLUMES_RUNBOOK.md` complete; on-call trained
8. [x] All volume-related tests pass in CI (166 local; full suite via `run-tests.sh`)
9. [x] `NEXT_PRIORITIES_ROADMAP.md` §1 checkboxes updated

---

## Work log

| Date | Change |
|------|--------|
| 2026-06-07 | Plan created; team UI + `list_volumes_for_owner_ids` fix; team volume tests added |
| 2026-06-07 | Phase A started: launch ownership fix, AI tool, NFS health, tests |
| 2026-06-07 | Phase A complete: 18 team tenancy tests pass; smoke script; runbook; analytics team-changed |
| 2026-06-07 | Phase E: volume error UX + i18n; launch modal team-changed + region warning |
| 2026-06-08 | Mac NFS appliance prod cutover; Headscale ACL `tag:mac-nfs`; ops + e2e smoke PASS |
| 2026-06-07 | Encrypted + worker-mount E2E PASS; persist smoke + disk check script; nfs4 mount opts |
| 2026-06-07 | ASUS 2060 interim GPU; persist+terminate E2E; viewer attach guard test; hot-attach runbook |
| 2026-06-07 | Live hot-attach (`mount_volume`/`unmount_volume`); billing audit script; `--hot-attach` smoke |

---

*Update this file as tasks complete. Link PRs/commits in the work log.*