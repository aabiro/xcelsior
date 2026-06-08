# Persistent Volumes — Operations Runbook

*Companion to [`VOLUMES_COMPLETION_PLAN.md`](./VOLUMES_COMPLETION_PLAN.md)*

---

## Architecture (30-second version)

1. **API** (`volumes.VolumeEngine`) provisions storage on the **NFS server** via SSH (`XCELSIOR_NFS_SERVER`).
2. **Metadata** lives in PostgreSQL (`volumes`, `volume_attachments`, `volume_snapshots`).
3. On **instance start**, the **scheduler** NFS-mounts `{EXPORT_BASE}/{volume_id}` on the GPU host, then bind-mounts into the container.
4. **Billing** (`billing.py` bg tick) charges `owner_id` wallet per GB-month (`VOLUME_PRICE_PER_GB_MONTH_CAD = 0.03`).
5. **Encrypted volumes** use LUKS2 loopback images; keys are Fernet-encrypted in DB only — never written to NFS.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `XCELSIOR_NFS_SERVER` | Mesh IP workers mount (prod: `100.64.0.1` VPS) |
| `XCELSIOR_NFS_SSH_HOST` | SSH target for API provision/destroy (`127.0.0.1` when colocated on VPS) |
| `XCELSIOR_NFS_SSH_USER` | SSH user for remote provision (`root` on colocated VPS) |
| `XCELSIOR_NFS_SSH_CMD_WRAP` | Optional remote command wrapper (legacy Mac only) |
| `XCELSIOR_NFS_EXPORT_BASE` | Export path prefix (must match server export), e.g. `/exports/volumes` |
| `XCELSIOR_NFS_PATH` | Legacy per-job shared NFS path (models/datasets) |
| `XCELSIOR_NFS_MOUNT` | Default mount point on workers for legacy NFS |
| `XCELSIOR_NFS_MOUNT_OPTS` | Override mount options; VPS uses kernel NFS on `:2049` (default opts OK) |
| `XCELSIOR_NFS_REQUIRED` | If `true`, `/readyz` fails when NFS unreachable |
| `XCELSIOR_MAX_VOLUME_GB` | Per-volume size cap (default 2000) |
| `XCELSIOR_MAX_TOTAL_STORAGE_GB` | Per-owner total cap (default 2000) |
| `XCELSIOR_MAX_VOLUMES_PER_OWNER` | Max volume count per owner (default 50) |
| `XCELSIOR_HOT_MOUNT_TIMEOUT_SEC` | Hot-attach poll timeout in API (default 45) |

**Critical:** `XCELSIOR_NFS_EXPORT_BASE` on API must match the path exported by the NFS server. Mismatch → provision succeeds on wrong path or mount fails on workers.

### Production — VPS colocated NFS (current)

NFS export on **pixelenhance-labs** (`100.64.0.1`). API containers SSH to `127.0.0.1` for provision/destroy; GPU workers mount `100.64.0.1:/exports/volumes/{volume_id}`.

**Why not Mac:** Docker Desktop port-forwarding mangled NFS RPC; Ganesha on `:12049` was unreliable. Moved to kernel NFS on the VPS (2026-06-08).

**Production `.env` (on VPS `/opt/xcelsior/.env`):**

```
XCELSIOR_NFS_SERVER=100.64.0.1
XCELSIOR_NFS_SSH_HOST=127.0.0.1
XCELSIOR_NFS_SSH_USER=root
XCELSIOR_NFS_EXPORT_BASE=/exports/volumes
XCELSIOR_NFS_SSH_CMD_WRAP=
XCELSIOR_NFS_REQUIRED=true
```

**Setup:**

1. `sudo bash scripts/setup_nfs_vps.sh` (if not already done)
2. `docker-compose.yml` bind-mounts `/exports:/exports` on API + scheduler-worker
3. Worker `~/.xcelsior/worker.env`: `XCELSIOR_NFS_SERVER=100.64.0.1` (no `port=12049`)

**GPU worker deploy** (`worker_agent.py`):

```bash
# From dev machine (mesh SSH works as aaryn@, not xcelsior@):
cd /path/to/xcelsior
bash scripts/deploy_worker_agent.sh --host 100.64.0.6 --user aaryn

# From VPS (repo is /opt/xcelsior — not ~/storage/projects/xcelsior):
cd /opt/xcelsior
bash scripts/deploy_worker_agent.sh --host 100.64.0.6 --user aaryn
```

VPS → worker `:22` may time out; deploy from a workstation with direct mesh SSH to the worker.

### Deprecated — Mac InferenceData appliance

LUKS + NFS-Ganesha on Mac (`100.64.0.3`, `:12049`) — **retired** due to Docker Desktop NFS RPC issues. Scripts remain for reference: `deploy_mac_nfs_appliance.sh`, `switch_prod_nfs_mac.sh`, `check_mac_nfs_disk.sh`.

---

## Health checks

```bash
# Liveness (no NFS)
curl -s https://xcelsior.ca/healthz

# Readiness (Postgres + optional NFS)
curl -s https://xcelsior.ca/readyz | jq .

# Admin NFS config
curl -s -H "Authorization: Bearer $ADMIN_TOKEN" https://xcelsior.ca/api/nfs/config | jq .

# VPS export disk usage
df -h /exports/volumes
```

**Disk quota monitoring:** `df -h /exports/volumes` on VPS; alert at 80%/90% used. Legacy Mac check: `scripts/check_mac_nfs_disk.sh`.

**Log grep patterns:**

| Pattern | Meaning |
|---------|---------|
| `metadata-only` | `XCELSIOR_NFS_SERVER` unset — volumes are DB-only |
| `NFS storage created for volume` | Provision OK |
| `NFS volume provision failed` | SSH or remote command failed |
| `Managed volume .* mount failed` | Worker could not NFS-mount at instance start |
| `LUKS luksFormat failed` | Encrypted provision failed (sudo/disk) |

---

## Incident: volumes stuck in `error`

**Symptoms:** User sees `error` badge; Retry button in UI.

**Steps:**

1. Get `volume_id` from user or DB:
   ```sql
   SELECT volume_id, owner_id, name, size_gb, encrypted, status, created_at
   FROM volumes WHERE status = 'error' ORDER BY created_at DESC LIMIT 20;
   ```
2. Check API logs for that `volume_id` at create time.
3. Verify NFS from API host:
   ```bash
   ssh -i /path/to/key user@$XCELSIOR_NFS_SERVER "test -d $XCELSIOR_NFS_EXPORT_BASE && echo OK"
   ```
4. If SSH fails → fix mesh connectivity / authorized_keys.
5. If LUKS fails → check `sudo cryptsetup` on NFS server.
6. User clicks **Retry** or ops calls `POST /api/v2/volumes/{id}/retry` as team member/admin.

---

## Hot-attach vs launch-time attach

| Path | Behavior |
|------|----------|
| **Launch with `volume_ids`** | Worker NFS-mounts at job start and bind-mounts into container (`/workspace` by default). Verified on ASUS `aaryn-tuf-rtx2060`. |
| **`POST /api/v2/volumes/{id}/attach`** | Hot-attach: enqueues `mount_volume` on the instance host; worker NFS-mounts at `/mnt/xcelsior-volumes/{volume_id}` and `nsenter` bind-mounts into the running container. Poll timeout `XCELSIOR_HOT_MOUNT_TIMEOUT_SEC` (default 45s). |
| **Detach** | `POST .../detach` enqueues `unmount_volume`; worker unbinds and lazy-unmounts host NFS. Terminate also runs `detach_all_for_instance`. |

---

## Incident: instance started but volume empty / not mounted

**Symptoms:** Job running; `/workspace` empty; logs show `mount failed on host`.

**Steps:**

1. Confirm `volume_ids` on job row / instance detail API.
2. SSH to GPU host (mesh):
   ```bash
   mount | grep xcelsior-volumes
   ls -la /mnt/xcelsior-volumes/
   ```
3. Manual mount test (VPS — kernel NFS on `:2049`):
   ```bash
   mkdir -p /mnt/xcelsior-volumes/vol-XXXXXXXXXXXX
   mount -t nfs4 -o hard,timeo=600,retrans=3,rsize=1048576,wsize=1048576,noatime,nosuid,nodev,_netdev,tcp,nfsvers=4 \
     $XCELSIOR_NFS_SERVER:$XCELSIOR_NFS_EXPORT_BASE/vol-XXXXXXXXXXXX \
     /mnt/xcelsior-volumes/vol-XXXXXXXXXXXX
   ```
4. If mount fails → NFS export permissions, `rpcbind`, firewall, wrong export path.
5. Restart instance after fixing host-side NFS client.

---

## Incident: NFS server reboot (encrypted volumes)

Encrypted volumes need LUKS reopened after NFS host reboot.

**API method:** `VolumeEngine.reopen_luks_volume(volume_id)` — retrieves key from DB, `luksOpen`, mount.

**Ops:** Batch reopen all encrypted volumes:

```bash
docker compose exec -T api-blue python scripts/volumes_reopen_luks.py
docker compose exec -T api-blue python scripts/volumes_reopen_luks.py --volume-id vol-XXXXXXXXXXXX
docker compose exec -T api-blue python scripts/volumes_reopen_luks.py --dry-run
```

---

## Incident: orphan attachments

**Symptoms:** Volume shows `attached` but instance terminated.

**Automated:** `bg-worker` calls `reconcile_orphan_attachments()` during billing tick.

**Manual:**
```sql
SELECT * FROM volume_attachments WHERE detached_at = 0;
-- Compare instance_id to jobs table terminal states
```

---

## Incident: worker orphan mounts

**Worker agent** runs `cleanup_orphaned_volume_mounts()` periodically.

**Manual on host:**
```bash
umount -l /mnt/xcelsior-volumes/vol-*   # careful — verify not in use
```

---

## Team billing

- `volumes.owner_id` = `billing_customer_id` (team wallet when in team workspace).
- Verify:
  ```sql
  SELECT v.volume_id, v.owner_id, t.billing_customer_id, t.name
  FROM volumes v
  LEFT JOIN teams t ON t.billing_customer_id = v.owner_id
  WHERE v.status != 'deleted' LIMIT 20;
  ```
- Viewer 403 on create/mutate is expected.

---

## Safe delete procedure

1. Detach from all instances (UI or `POST .../detach`).
2. Confirm `status = available`.
3. `DELETE /api/v2/volumes/{id}` — destroys NFS storage + DB row (soft delete).
4. Encrypted: confirm LUKS image removed on NFS server.

**Never** manually `rm -rf` on NFS without updating DB — causes billing/orphan drift.

---

## Disk capacity & per-owner caps

| Limit | Env var | Default | Notes |
|-------|---------|---------|-------|
| Per-volume size | `XCELSIOR_MAX_VOLUME_GB` | 2000 | UI create modal caps at 2000 GB |
| Total GB per owner | `XCELSIOR_MAX_TOTAL_STORAGE_GB` | 2000 | Team wallet shares one cap |
| Volume count per owner | `XCELSIOR_MAX_VOLUMES_PER_OWNER` | 50 | Abuse guard for many 1 GB volumes |

**Mac appliance disk:** Monitor InferenceData SSD with `scripts/check_mac_nfs_disk.sh` (warn 80%, crit 90%).

**Suggested cron** (workstation or monitoring host with mesh SSH to Mac):

```cron
# Daily 06:00 — Mac NFS disk check
0 6 * * * /path/to/xcelsior/scripts/check_mac_nfs_disk.sh >> /var/log/xcelsior-mac-nfs-disk.log 2>&1
```

---

## Log grep patterns (ops)

| Symptom | Grep (API / worker logs) |
|---------|--------------------------|
| Provision failure | `NFS volume provision failed` / `LUKS luksFormat failed` |
| Hot-attach | `Hot-attached volume` / `Hot mount timed out` / `mount_volume` |
| Hot-detach | `Hot-unmount enqueued` / `unmount_volume` |
| Billing tick | `AUTO-BILLING:.*volumes` |
| Orphan reconcile | `Orphan volume reconciliation` |
| Worker NFS skip | `Managed volume.*mount failed` |

---

## Staging smoke

```bash
# Fund audit wallet for launch smoke (idempotent):
docker compose exec -T api-blue python scripts/fund_audit_wallet.py

# NFS CRUD only (no wallet/GPU):
python3 scripts/volumes_e2e_smoke.py --base-url https://xcelsior.ca --infra-only

# LUKS encrypted volume provision + delete:
python3 scripts/volumes_e2e_smoke.py --infra-only --encrypted

# VPS→Mac NFS mount (simulates GPU worker mesh mount):
python3 scripts/volumes_e2e_smoke.py --infra-only --worker-mount

# Optional second GPU worker mount (SKIP if host not registered, e.g. aarynfans-prod):
python3 scripts/volumes_e2e_smoke.py --infra-only --worker-mount --worker-host 100.64.0.6
python3 scripts/volumes_worker_mount_smoke.py --host $AARYNFANS_PROD_HOST

# Volume snapshots on NFS _snapshots/ (detached volume):
python3 scripts/volumes_e2e_smoke.py --infra-only --snapshots

# Persist across stop/start (requires admitted GPU worker, e.g. ASUS aaryn-tuf-rtx2060):
python3 scripts/volumes_e2e_smoke.py --infra-only --persist

# Hot-attach to running instance (requires GPU worker + mesh SSH for verify):
python3 scripts/volumes_e2e_smoke.py --hot-attach

# Volume billing tick audit (prod):
docker compose exec -T api-blue python scripts/volumes_billing_audit.py

# LUKS reopen after Mac NFS reboot:
docker compose exec -T api-blue python scripts/volumes_reopen_luks.py --dry-run

# Interim prod GPU worker (until tower-server returns): ASUS RTX 2060 at 100.64.0.6
# Worker env must include Mac NFS settings — see ~/.xcelsior/worker.env on the host.

# Full path including instance launch (requires funded wallet):
python3 scripts/volumes_e2e_smoke.py --base-url https://xcelsior.ca

# Combined PayPal + NFS infra:
python3 scripts/ops_infra_smoke.py
```

---

## Contacts & escalation

| Level | Action |
|-------|--------|
| L1 | User Retry; check `/readyz` NFS section |
| L2 | SSH to NFS + worker; verify exports and mounts |
| L3 | Eng on-call — `volumes.py` provision path, scheduler mount path |

---

*Last updated: 2026-06-07*