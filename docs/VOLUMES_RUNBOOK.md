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
| `XCELSIOR_NFS_SERVER` | Mesh IP workers mount (e.g. `100.64.0.3` Mac appliance) |
| `XCELSIOR_NFS_SSH_HOST` | SSH target for API provision/destroy |
| `XCELSIOR_NFS_SSH_USER` | SSH user for remote provision (`aaryn` on Mac; `root` on colocated VPS) |
| `XCELSIOR_NFS_SSH_CMD_WRAP` | Optional remote command wrapper (Mac: `/Users/aaryn/.local/bin/xcelsior-nfs-exec`) |
| `XCELSIOR_NFS_EXPORT_BASE` | Export path prefix (must match server export), e.g. `/exports/volumes` |
| `XCELSIOR_NFS_PATH` | Legacy per-job shared NFS path (models/datasets) |
| `XCELSIOR_NFS_MOUNT` | Default mount point on workers for legacy NFS |
| `XCELSIOR_NFS_MOUNT_OPTS` | Override mount options; **Mac appliance requires** `nfsvers=4.0,port=12049` |
| `XCELSIOR_NFS_REQUIRED` | If `true`, `/readyz` fails when NFS unreachable |
| `XCELSIOR_MAX_VOLUME_GB` | Per-volume size cap (default 2000) |
| `XCELSIOR_MAX_TOTAL_STORAGE_GB` | Per-owner total cap (default 2000) |

**Critical:** `XCELSIOR_NFS_EXPORT_BASE` on API must match the path exported by the NFS server. Mismatch → provision succeeds on wrong path or mount fails on workers.

### Production — Mac InferenceData appliance (current)

LUKS-backed ext4 inside Docker + NFS-Ganesha on the Mac external SSD (`aaryns-macbook-pro`, `100.64.0.3`).

**Deploy (from workstation):**

```bash
bash scripts/deploy_mac_nfs_appliance.sh   # builds xcelsior-mac-nfs on Mac
bash scripts/switch_prod_nfs_mac.sh        # points VPS .env at Mac + restarts API
```

**Production `.env` (on VPS):**

```
XCELSIOR_NFS_SERVER=100.64.0.3
XCELSIOR_NFS_SSH_HOST=100.64.0.3
XCELSIOR_NFS_SSH_USER=aaryn
XCELSIOR_NFS_EXPORT_BASE=/exports/volumes
XCELSIOR_NFS_SSH_CMD_WRAP=/Users/aaryn/.local/bin/xcelsior-nfs-exec
XCELSIOR_NFS_MOUNT_OPTS=hard,timeo=600,retrans=3,rsize=1048576,wsize=1048576,noatime,nosuid,nodev,_netdev,tcp,nfsvers=4.0,port=12049
XCELSIOR_NFS_REQUIRED=true
```

**Architecture notes:**

- Bulk LUKS: `/Volumes/InferenceData/inference.luks` (80G sparse) opened inside `xcelsior-mac-nfs`.
- Export path inside container: `/exports/volumes/{volume_id}`.
- Ganesha listens on container `:2049`, published as host **`:12049`** (macOS `nfsd` owns `:2049`).
- Clients **must** use `nfsvers=4.0` — Ganesha 3.5 does not support NFSv4.2 negotiation.
- API SSH runs commands on Mac; `xcelsior-nfs-exec` wraps them as `docker exec xcelsior-mac-nfs`.

**Headscale ACL** (`infra/headscale/acl.hujson`):

- Mac node tagged `tag:mac-nfs`.
- `tag:xcelsior` (VPS) → `tag:mac-nfs:22,12049` (SSH provision + NFS).
- `autogroup:member` → `tag:mac-nfs:12049` (GPU workers mount exports).

Apply: `bash scripts/headscale_apply_acl.sh`

**Per-volume LUKS:** Encrypted volumes use `cryptsetup` inside the appliance container (privileged). Requires Linux device-mapper in the container — works on Docker Desktop Mac; not native macOS cryptsetup.

### Alternate — VPS colocated NFS

When API and NFS run on the same VPS:

1. `sudo bash scripts/setup_nfs_vps.sh`
2. `.env`: `XCELSIOR_NFS_SERVER=100.64.0.1`, `XCELSIOR_NFS_SSH_HOST=127.0.0.1`, `XCELSIOR_NFS_SSH_USER=root`
3. `docker-compose.yml` bind-mounts `/exports:/exports` on API + scheduler-worker.

---

## Health checks

```bash
# Liveness (no NFS)
curl -s https://xcelsior.ca/healthz

# Readiness (Postgres + optional NFS)
curl -s https://xcelsior.ca/readyz | jq .

# Admin NFS config
curl -s -H "Authorization: Bearer $ADMIN_TOKEN" https://xcelsior.ca/api/nfs/config | jq .

# Mac InferenceData disk usage (warn 80%, crit 90% — exit 1/2)
bash scripts/check_mac_nfs_disk.sh
```

**Disk quota monitoring:** Run `check_mac_nfs_disk.sh` from cron or a monitoring agent (SSH to Mac required). Exit codes: `0` OK, `1` warn (≥80%), `2` crit (≥90%). Override thresholds with `WARN_PCT` / `CRIT_PCT`.

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

## Incident: instance started but volume empty / not mounted

**Symptoms:** Job running; `/workspace` empty; logs show `mount failed on host`.

**Steps:**

1. Confirm `volume_ids` on job row / instance detail API.
2. SSH to GPU host (mesh):
   ```bash
   mount | grep xcelsior-volumes
   ls -la /mnt/xcelsior-volumes/
   ```
3. Manual mount test (Mac appliance — note `nfsvers=4.0,port=12049`):
   ```bash
   mkdir -p /mnt/xcelsior-volumes/vol-XXXXXXXXXXXX
   mount -t nfs4 -o hard,timeo=600,retrans=3,rsize=1048576,wsize=1048576,noatime,nosuid,nodev,_netdev,tcp,nfsvers=4.0,port=12049 \
     $XCELSIOR_NFS_SERVER:$XCELSIOR_NFS_EXPORT_BASE/vol-XXXXXXXXXXXX \
     /mnt/xcelsior-volumes/vol-XXXXXXXXXXXX
   ```
4. If mount fails → NFS export permissions, `rpcbind`, firewall, wrong export path.
5. Restart instance after fixing host-side NFS client.

---

## Incident: NFS server reboot (encrypted volumes)

Encrypted volumes need LUKS reopened after NFS host reboot.

**API method:** `VolumeEngine.reopen_luks_volume(volume_id)` — retrieves key from DB, `luksOpen`, mount.

**Ops:** Trigger retry on affected volumes or run batch reopen script (TODO: add `scripts/volumes_reopen_luks.py`).

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

# Persist across stop/start (requires admitted GPU worker, e.g. ASUS aaryn-tuf-rtx2060):
python3 scripts/volumes_e2e_smoke.py --infra-only --persist

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