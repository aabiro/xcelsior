"""Routes: volumes."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Literal

from routes._deps import (
    _effective_billing_customer_id,
    _get_current_user,
    _require_volume_read,
    _require_volume_write,
    _require_volume_write_role,
    _volume_owner_ids_readable,
    _volume_scope_owner_id,
    append_user_audit_event,
    broadcast_sse,
    log,
)
from scheduler import (
    log,
)
from volumes import get_volume_engine, VOLUME_PRICE_PER_GB_MONTH_CAD

router = APIRouter()


# ── Model: VolumeCreate ──


class VolumeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    size_gb: int = Field(default=50, ge=1, le=2000)
    region: str = "ca-east"
    encrypted: bool = True


@router.post("/api/v2/volumes", tags=["Volumes"])
def api_volume_create(body: VolumeCreate, request: Request):
    """Create a new persistent volume. Billed in real-time from credits."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    _require_volume_write_role(user)
    owner_id = _volume_scope_owner_id(user)
    ve = get_volume_engine()
    try:
        created = ve.create_volume(
            owner_id=owner_id,
            name=body.name,
            size_gb=body.size_gb,
            region=body.region,
            encrypted=body.encrypted,
        )
        vol = ve.get_volume(created["volume_id"]) or created
        vol["price_per_gb_month_cad"] = VOLUME_PRICE_PER_GB_MONTH_CAD
        vol["estimated_monthly_cost_cad"] = round(body.size_gb * VOLUME_PRICE_PER_GB_MONTH_CAD, 2)
        broadcast_sse(
            "volume.created",
            {"volume_id": vol["volume_id"], "name": vol["name"], "size_gb": vol["size_gb"]},
        )
        append_user_audit_event(
            "user.volume.created",
            "volume",
            vol["volume_id"],
            user,
            data={"name": vol["name"], "size_gb": vol["size_gb"], "owner_id": owner_id},
        )
        return {"ok": True, "volume": vol}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, "Volume encryption is misconfigured — contact support.") from e


@router.get("/api/v2/volumes", tags=["Volumes"])
def api_volume_list(request: Request):
    """List volumes owned by the current user."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:read")
    ve = get_volume_engine()
    owner_ids = sorted(_volume_owner_ids_readable(user))
    volumes = ve.list_volumes_for_owner_ids(owner_ids)
    for v in volumes:
        v["price_per_gb_month_cad"] = VOLUME_PRICE_PER_GB_MONTH_CAD
        v["monthly_cost_cad"] = round(v.get("size_gb", 0) * VOLUME_PRICE_PER_GB_MONTH_CAD, 2)
    return {"ok": True, "volumes": volumes}


@router.get("/api/v2/volumes/available", tags=["Volumes"])
def api_volumes_available(request: Request):
    """List volumes available for attachment (status=available) for the current user."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:read")
    ve = get_volume_engine()
    owner_ids = sorted(_volume_owner_ids_readable(user))
    volumes = ve.list_volumes_for_owner_ids(owner_ids)
    available = [
        {
            "volume_id": v["volume_id"],
            "name": v.get("name", ""),
            "size_gb": v.get("size_gb", 0),
            "region": v.get("region", ""),
            "encrypted": v.get("encrypted", False),
        }
        for v in volumes
        if v.get("status") == "available"
    ]
    return {"ok": True, "volumes": available}


@router.get("/api/v2/volumes/{volume_id}", tags=["Volumes"])
def api_volume_get(volume_id: str, request: Request):
    """Get volume details."""
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:read")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    _require_volume_read(user, vol)
    return {"ok": True, "volume": vol}


# ── Model: VolumeRename ──


class VolumeRename(BaseModel):
    name: str = Field(min_length=1, max_length=128)


@router.patch("/api/v2/volumes/{volume_id}", tags=["Volumes"])
def api_volume_rename(volume_id: str, body: VolumeRename, request: Request):
    """Rename a volume."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    _require_volume_write(user, vol)
    try:
        result = ve.rename_volume(volume_id, vol["owner_id"], body.name)
        broadcast_sse("volume.renamed", {"volume_id": volume_id, "name": result["name"]})
        return {"ok": True, "volume": result}
    except PermissionError:
        raise HTTPException(403, "Not authorised to rename this volume")
    except ValueError as e:
        raise HTTPException(400, str(e))


# ── Model: VolumeAttachRequest ──


class VolumeAttachRequest(BaseModel):
    instance_id: str
    mount_path: str = Field(
        default="/workspace", pattern=r"^/(workspace|mnt/[a-zA-Z0-9._-]+|data)$"
    )
    mode: Literal["rw", "ro"] = "rw"


@router.post("/api/v2/volumes/{volume_id}/attach", tags=["Volumes"])
def api_volume_attach(volume_id: str, body: VolumeAttachRequest, request: Request):
    """Attach a volume to a running instance."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    _require_volume_write(user, vol)
    try:
        region_warning = ve.attach_region_warning(volume_id, body.instance_id)
        att = ve.attach_volume(volume_id, body.instance_id, body.mount_path, body.mode)
        if not att:
            raise HTTPException(409, "Volume not available for attachment")
        broadcast_sse("volume.attached", {"volume_id": volume_id, "instance_id": body.instance_id})
        resp: dict = {"ok": True, "attachment": att}
        if region_warning:
            resp["region_warning"] = region_warning
        return resp
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/api/v2/volumes/{volume_id}/detach", tags=["Volumes"])
def api_volume_detach(volume_id: str, request: Request):
    """Detach a volume from its current instance."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    _require_volume_write(user, vol)
    # Detach using atomic FOR UPDATE inside detach_volume
    if vol.get("status") != "attached":
        raise HTTPException(400, "Volume is not attached to any instance")
    try:
        ve.detach_volume(volume_id, instance_id=None)
        broadcast_sse("volume.detached", {"volume_id": volume_id})
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/api/v2/volumes/{volume_id}", tags=["Volumes"])
def api_volume_delete(volume_id: str, request: Request):
    """Delete a volume. Must not have active attachments."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    _require_volume_write(user, vol)
    try:
        result = ve.delete_volume(volume_id, owner_id=vol["owner_id"])
        broadcast_sse("volume.deleted", {"volume_id": volume_id})
        append_user_audit_event(
            "user.volume.deleted",
            "volume",
            volume_id,
            user,
            data={"owner_id": vol["owner_id"]},
        )
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(409, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@router.post("/api/v2/volumes/{volume_id}/retry", tags=["Volumes"])
def api_volume_retry_provision(volume_id: str, request: Request):
    """Retry provisioning for a volume stuck in 'error' status."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    _require_volume_write(user, vol)
    try:
        result = ve.retry_provision(volume_id, owner_id=vol["owner_id"])
        broadcast_sse("volume.retried", {"volume_id": volume_id})
        return {"ok": True, "volume": result}
    except PermissionError:
        raise HTTPException(403, "Not authorised to retry this volume")
    except ValueError as e:
        msg = str(e)
        if "decrypt" in msg.lower() or "encryption key" in msg.lower():
            raise HTTPException(409, "Volume encryption key is invalid — delete and recreate the volume.") from e
        raise HTTPException(400, msg)
    except RuntimeError as e:
        if "encryption" in str(e).lower() or "secrets_key" in str(e).lower():
            raise HTTPException(503, "Volume encryption is misconfigured — contact support.") from e
        raise HTTPException(502, str(e))


@router.post("/api/v2/admin/volumes/reopen-encrypted", tags=["Volumes", "Admin"])
def api_admin_reopen_encrypted_volumes(request: Request):
    """Reopen all encrypted volumes after NFS server reboot.

    Iterates encrypted volumes in 'available' or 'attached' status,
    reopens their LUKS devices, and re-mounts them. Admin-only.
    """
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "admin")
    ve = get_volume_engine()
    with ve._conn() as conn:
        rows = conn.execute(
            "SELECT volume_id FROM volumes WHERE encrypted = TRUE "
            "AND status IN ('available', 'attached') AND key_ciphertext != ''",
        ).fetchall()
    results = {"reopened": [], "failed": []}
    for row in rows:
        vid = row["volume_id"]
        ok = ve.reopen_luks_volume(vid)
        if ok:
            results["reopened"].append(vid)
        else:
            results["failed"].append(vid)
    log.info(
        "Admin reopen encrypted volumes: %d reopened, %d failed",
        len(results["reopened"]),
        len(results["failed"]),
    )
    return {"ok": True, **results}


# ── P2.5 Snapshots ────────────────────────────────────────────────────


class SnapshotCreate(BaseModel):
    label: str = Field(default="", max_length=128)


@router.post("/api/v2/volumes/{volume_id}/snapshots", tags=["Volumes"])
def api_volume_snapshot_create(volume_id: str, body: SnapshotCreate, request: Request):
    """Take an instant CoW snapshot of a detached volume."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    _require_volume_write(user, vol)
    try:
        snap = ve.create_snapshot(volume_id, vol["owner_id"], body.label)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    broadcast_sse(
        "volume.snapshot.created",
        {"volume_id": volume_id, "snapshot_id": snap["snapshot_id"]},
    )
    return {"ok": True, "snapshot": snap}


@router.get("/api/v2/volumes/{volume_id}/snapshots", tags=["Volumes"])
def api_volume_snapshot_list(volume_id: str, request: Request):
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:read")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    _require_volume_read(user, vol)
    try:
        snaps = ve.list_snapshots(volume_id, vol["owner_id"])
    except PermissionError:
        raise HTTPException(404, "Volume not found")
    return {"ok": True, "snapshots": snaps}


@router.post("/api/v2/volumes/{volume_id}/snapshots/{snapshot_id}/restore", tags=["Volumes"])
def api_volume_snapshot_restore(volume_id: str, snapshot_id: str, request: Request):
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    _require_volume_write(user, vol)
    try:
        result = ve.restore_snapshot(volume_id, vol["owner_id"], snapshot_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    broadcast_sse(
        "volume.snapshot.restored",
        {"volume_id": volume_id, "snapshot_id": snapshot_id},
    )
    return {"ok": True, **result}


@router.delete("/api/v2/volumes/{volume_id}/snapshots/{snapshot_id}", tags=["Volumes"])
def api_volume_snapshot_delete(volume_id: str, snapshot_id: str, request: Request):
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    _require_volume_write(user, vol)
    try:
        result = ve.delete_snapshot(volume_id, vol["owner_id"], snapshot_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    broadcast_sse(
        "volume.snapshot.deleted",
        {"volume_id": volume_id, "snapshot_id": snapshot_id},
    )
    return {"ok": True, **result}


@router.get("/api/v2/volumes/{volume_id}/promotions/preview", tags=["Volumes"])
def api_volume_promotion_preview(volume_id: str, request: Request, job_id: str = ""):
    """What promoting `job_id` onto this volume would copy. Copies nothing.

    A0 of `docs/artifact-promotion-plan.md`. The manifest exists before the
    machinery that acts on it so the shape can be reviewed — and so the tool
    layer, when it arrives, has something truthful to show a user *before* a
    40 GB copy starts rather than after.

    Both objects are authorised separately and in the right order. The volume is
    checked first because it is the one named in the path; the artifacts are
    scoped by tenant inside the query rather than filtered afterwards, since a
    manifest is a list of file names and sizes and resolving before checking is
    how a read becomes a disclosure.
    """
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:read")
    if not job_id:
        raise HTTPException(422, "job_id is required")

    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    _require_volume_read(user, vol)

    from artifacts import get_artifact_manager

    manifest = get_artifact_manager().resolve_promotion_manifest(
        job_id, tenant_id=_effective_billing_customer_id(user)
    )
    if not manifest.get("ok"):
        raise HTTPException(503, "Artifact catalog unavailable")

    # A job with no promotable artifacts is a 200 with an empty manifest, not a
    # 404: the job may exist and simply have produced nothing yet, and the
    # caller needs to tell "nothing to promote" from "no such job" without being
    # told which other tenants' jobs exist.
    return {
        "ok": True,
        "volume_id": volume_id,
        "job_id": job_id,
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest_sha256": manifest["manifest_sha256"],
        "earliest_retain_until": manifest["earliest_retain_until"],
        "unverifiable": manifest["unverifiable"],
        # `object_key` is deliberately absent. It is a storage-internal locator,
        # it is not needed to decide whether to promote, and this response is
        # destined for a model's context.
        "files": [
            {
                "artifact_id": f["artifact_id"],
                "logical_name": f["logical_name"],
                "size_bytes": f["size_bytes"],
                "artifact_type": f["artifact_type"],
            }
            for f in manifest["files"]
        ],
    }


def _pick_promotion_host(region: str) -> str:
    """The least-loaded active host in `region`, or "" if there is none.

    §3.4 of `docs/artifact-promotion-plan.md`. Region is a **hard filter, not a
    preference**: volumes are NFS exports and a host outside the region either
    cannot reach the export or pays a cross-region transfer for every byte of a
    checkpoint. Silently falling back to "any host" would turn an unroutable
    promotion into an expensive one, and the user would see neither.

    Load is counted as live jobs on the host, so a promotion lands on the box
    with the most spare I/O rather than the first one returned. The plan's cost
    note stands and is worth restating: a busy GPU host now does I/O for someone
    else's promotion, and the moment that competes with training it should move
    to a dedicated worker. It is measurable before it is a problem.

    Returning "" is a real outcome — no host in that region — and the caller
    creates the promotion anyway rather than refusing. The row is the durable
    record; a sweep can place it when capacity appears. Refusing here would
    throw away a request the user made for a reason.
    """
    from control_plane.db import control_plane_transaction

    try:
        with control_plane_transaction() as conn:
            row = conn.execute(
                """
                SELECT h.host_id
                  FROM hosts h
                  LEFT JOIN jobs j
                    ON j.host_id = h.host_id
                   AND j.status IN ('running', 'starting', 'provisioning')
                 WHERE h.status = 'active'
                   AND (%s = '' OR h.region = %s)
              GROUP BY h.host_id
              ORDER BY count(j.job_id) ASC, h.host_id ASC
                 LIMIT 1
                """,
                (region, region),
            ).fetchone()
    except Exception as exc:  # pragma: no cover - placement must never 500
        log.warning("promotion host placement failed for region %r: %s", region, exc)
        return ""
    return str(row[0]) if row else ""


class PromotionCreate(BaseModel):
    job_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(default="", max_length=160)


@router.post("/api/v2/volumes/{volume_id}/promotions", tags=["Volumes"])
def api_volume_promotion_create(volume_id: str, body: PromotionCreate, request: Request):
    """Start copying a job's artifacts onto this volume. Idempotent.

    A1 of `docs/artifact-promotion-plan.md`: one host, an already-attached
    volume, no resume. The row is created here and the copy is performed by the
    agent, which fetches the manifest itself — the command carries only the
    promotion id.

    A replay returns the existing row with `replayed: true` rather than
    appearing to start a second copy. `charge_saved_card` says the same thing
    for the same reason: a caller that retried a timeout needs to know whether
    its first attempt landed.
    """
    import time as _time
    import uuid as _uuid

    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")

    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    _require_volume_write(user, vol)

    tenant_id = _effective_billing_customer_id(user)
    from artifacts import get_artifact_manager

    manifest = get_artifact_manager().resolve_promotion_manifest(body.job_id, tenant_id=tenant_id)
    if not manifest.get("ok"):
        raise HTTPException(503, "Artifact catalog unavailable")
    if manifest["file_count"] == 0:
        # Not a 404: the job may exist and have produced nothing. Refusing here
        # rather than creating an empty promotion that would "succeed" having
        # copied nothing, which reads to a model as "your weights are saved".
        raise HTTPException(409, "This job has no artifacts available to promote")

    # The digest is the default key, so a retry with no key still converges —
    # and a job that has produced new artifacts since is a different promotion
    # rather than a replay of a stale file list.
    idem = body.idempotency_key or manifest["manifest_sha256"]

    from control_plane.db import control_plane_transaction

    promotion_id = str(_uuid.uuid4())
    with control_plane_transaction() as conn:
        cur = conn.execute(
            """INSERT INTO volume_promotions
                 (promotion_id, tenant_id, owner_user_id, job_id, volume_id,
                  idempotency_key, manifest_sha256, file_count, total_bytes, state)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending')
               ON CONFLICT (tenant_id, job_id, idempotency_key) DO NOTHING""",
            (
                promotion_id, tenant_id, user.get("user_id"), body.job_id, volume_id,
                idem, manifest["manifest_sha256"], manifest["file_count"],
                manifest["total_bytes"],
            ),
        )
        created = cur.rowcount == 1
        if created:
            # §3.3: before the first byte moves, not after. Inside the same
            # transaction as the insert, so a promotion can never exist in a
            # state where it is copying artifacts the janitor may still delete.
            from artifacts import take_promotion_hold

            held = take_promotion_hold(conn, body.job_id, tenant_id)
            log.info(
                "promotion %s: held %d artifact(s) for job %s",
                promotion_id, held, body.job_id,
            )
        row = conn.execute(
            "SELECT promotion_id, state, file_count, total_bytes, volume_id "
            "FROM volume_promotions "
            "WHERE tenant_id = %s AND job_id = %s AND idempotency_key = %s",
            (tenant_id, body.job_id, idem),
        ).fetchone()

    existing_id = str(row[0])
    if not created:
        return {
            "ok": True, "replayed": True, "promotion_id": existing_id,
            "state": row[1], "file_count": row[2], "total_bytes": row[3],
            "volume_id": row[4],
        }

    # Enqueued only for a genuinely new row, so a replay cannot queue a second
    # copy of the same bytes.
    #
    # An attached volume has an obvious host — the instance's, which can already
    # see the mount. An unattached one has none, and §3.4 of the promotion plan
    # calls that the genuinely open question. Its answer, chosen over "any
    # healthy host" and over a dedicated promotion worker: the least-loaded
    # active host **in the volume's region**, reusing the mount commands the
    # agent already has. No new deployable, no new network path.
    host_id = str(vol.get("host_id") or "")
    if not host_id:
        host_id = _pick_promotion_host(str(vol.get("region") or ""))
    enqueued = None
    if host_id:
        from routes.agent import enqueue_agent_command

        enqueued = enqueue_agent_command(
            host_id, "promote_artifacts", {"promotion_id": existing_id},
            created_by=str(user.get("user_id") or ""), ttl_sec=3600,
        )

    append_user_audit_event(
        "user.volume.promotion_started", "volume", volume_id, user,
        data={
            "promotion_id": existing_id, "job_id": body.job_id,
            "file_count": manifest["file_count"], "total_bytes": manifest["total_bytes"],
        },
    )
    return {
        "ok": True, "replayed": False, "promotion_id": existing_id,
        "state": "pending", "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"], "volume_id": volume_id,
        "command_id": enqueued,
        "started_at_utc": _time.time(),
    }


@router.get("/api/v2/volumes/{volume_id}/promotions/{promotion_id}", tags=["Volumes"])
def api_volume_promotion_get(volume_id: str, promotion_id: str, request: Request):
    """State of one promotion. A foreign id is not-found, never forbidden."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:read")

    from control_plane.db import control_plane_transaction

    tenant_id = _effective_billing_customer_id(user)
    with control_plane_transaction() as conn:
        row = conn.execute(
            "SELECT promotion_id, job_id, volume_id, state, file_count, total_bytes, "
            "       failure_code, created_at, completed_at "
            "  FROM volume_promotions WHERE promotion_id = %s AND tenant_id = %s",
            (promotion_id, tenant_id),
        ).fetchone()
    if not row or str(row[2] or "") != volume_id:
        raise HTTPException(404, "Promotion not found")
    return {
        "ok": True,
        "promotion_id": str(row[0]), "job_id": row[1], "volume_id": row[2],
        "state": row[3], "file_count": row[4], "total_bytes": row[5],
        "failure_code": row[6],
        "created_at": row[7].isoformat() if row[7] else None,
        "completed_at": row[8].isoformat() if row[8] else None,
    }


@router.get("/api/v1/promotions/{promotion_id}/manifest", tags=["Volumes"])
def api_promotion_manifest_for_agent(promotion_id: str, request: Request):
    """The file list and read grants for one promotion. Worker-authenticated.

    Separate from the customer-facing preview, and deliberately so: this is the
    only response that carries **presigned URLs**, which are time-limited read
    grants for a tenant's weights. That is why the command row holds nothing but
    `{promotion_id}` — a queued, logged, retained row is the wrong place for a
    credential — and why this endpoint is reachable only by a platform worker.

    The manifest is re-resolved from the catalog rather than stored on the row,
    so a promotion cannot hand out URLs for artifacts that have since been
    deleted, and the tenant is taken from the promotion rather than the caller.
    """
    from routes._deps import _require_auth, _require_platform_worker

    user = _require_auth(request)
    _require_platform_worker(user)

    from control_plane.db import control_plane_transaction

    with control_plane_transaction() as conn:
        row = conn.execute(
            "SELECT tenant_id, job_id, volume_id, state, manifest_sha256 "
            "  FROM volume_promotions WHERE promotion_id = %s",
            (promotion_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "Promotion not found")
    tenant_id, job_id, volume_id, state, expected_digest = row[0], row[1], row[2], row[3], row[4]
    if state in ("succeeded", "abandoned"):
        raise HTTPException(409, f"Promotion is {state}; nothing to fetch")

    from artifacts import get_artifact_manager

    am = get_artifact_manager()
    manifest = am.resolve_promotion_manifest(job_id, tenant_id=tenant_id)
    if not manifest.get("ok"):
        raise HTTPException(503, "Artifact catalog unavailable")
    if manifest["manifest_sha256"] != expected_digest:
        # The artifact set changed after the promotion was created. Refusing is
        # the only honest answer: copying the new set would deliver something
        # the user did not ask for, and copying a subset would be a partial
        # volume reported as complete.
        raise HTTPException(
            409,
            "The artifact set changed since this promotion was created; "
            "start a new promotion",
        )

    # §3.5: a resumed promotion skips what is already verified. Read here rather
    # than trusted from the agent — the agent is the party that crashed.
    with control_plane_transaction() as conn:
        done_rows = conn.execute(
            "SELECT artifact_id FROM volume_promotion_files "
            " WHERE promotion_id = %s AND state = 'done' AND sha256_verified",
            (promotion_id,),
        ).fetchall()
    already_done = {str(r[0]) for r in done_rows}

    files = []
    for f in manifest["files"]:
        if f["artifact_id"] in already_done:
            # No presigned URL for a file that will not be fetched. A read grant
            # nobody uses is still a read grant that was issued and logged.
            files.append({
                "artifact_id": f["artifact_id"],
                "logical_name": f["logical_name"],
                "size_bytes": f["size_bytes"],
                "sha256": f["sha256"],
                "url": "",
                "already_done": True,
            })
            continue
        try:
            grant = am.primary.generate_download_url(f["object_key"])
        except Exception:
            raise HTTPException(503, "Storage backend unavailable")
        files.append({
            "artifact_id": f["artifact_id"],
            "logical_name": f["logical_name"],
            "size_bytes": f["size_bytes"],
            "sha256": f["sha256"],
            "url": grant.get("url") if isinstance(grant, dict) else grant,
            "already_done": f["artifact_id"] in already_done,
        })
    return {
        "ok": True,
        "promotion_id": promotion_id,
        "volume_id": volume_id,
        "manifest_sha256": expected_digest,
        "files": files,
    }


class PromotionResult(BaseModel):
    state: Literal["succeeded", "failed"]
    failure_code: str = Field(default="", max_length=64)
    bytes_written: int = Field(default=0, ge=0)


@router.post("/api/v1/promotions/{promotion_id}/result", tags=["Volumes"])
def api_promotion_result_from_agent(promotion_id: str, body: PromotionResult, request: Request):
    """The agent reporting what happened. Worker-authenticated, terminal-once.

    A promotion already in a terminal state is not overwritten: a retried
    report must not turn a recorded failure into a success, and the state
    machine's `succeeded` check requires a volume and a completion time, so a
    row that reached success has already been proven complete.
    """
    from routes._deps import _require_auth, _require_platform_worker

    user = _require_auth(request)
    _require_platform_worker(user)

    from control_plane.db import control_plane_transaction

    with control_plane_transaction() as conn:
        row = conn.execute(
            "SELECT state FROM volume_promotions WHERE promotion_id = %s FOR UPDATE",
            (promotion_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Promotion not found")
        if row[0] in ("succeeded", "failed", "abandoned"):
            return {"ok": True, "state": row[0], "already_terminal": True}
        conn.execute(
            "UPDATE volume_promotions "
            "   SET state = %s, failure_code = NULLIF(%s, ''), "
            "       completed_at = clock_timestamp(), updated_at = clock_timestamp() "
            " WHERE promotion_id = %s",
            (body.state, body.failure_code, promotion_id),
        )
        # Released on failure as well as success. A promotion that failed has
        # no further claim on the artifacts, and holding them would turn one
        # bad copy into indefinite retention.
        owner = conn.execute(
            "SELECT job_id, tenant_id FROM volume_promotions WHERE promotion_id = %s",
            (promotion_id,),
        ).fetchone()
        if owner:
            from artifacts import release_promotion_hold

            release_promotion_hold(conn, owner[0], owner[1])
    return {"ok": True, "state": body.state}


class PromotionFileResult(BaseModel):
    artifact_id: str = Field(min_length=1, max_length=64)
    logical_name: str = Field(default="", max_length=255)
    size_bytes: int = Field(default=0, ge=0)
    sha256_verified: bool = False
    state: Literal["done", "failed"] = "done"
    failure_code: str = Field(default="", max_length=64)


@router.post("/api/v1/promotions/{promotion_id}/files", tags=["Volumes"])
def api_promotion_file_result(promotion_id: str, body: PromotionFileResult, request: Request):
    """One file finished. Worker-authenticated.

    §3.5 exists because retries are certain: "a promotion that restarts from
    zero after a failure at 38 GB will be retried by a human who then watches it
    fail again". This is the row that lets the next attempt skip it.

    **`done` requires `sha256_verified`** — enforced by a CHECK, not only here.
    The resume path skips `done` files, so a file marked done without
    verification would be skipped forever and the volume would carry an
    unverified copy that looks like a backup. An agent claiming `done` without
    verification is refused rather than trusted.
    """
    from routes._deps import _require_auth, _require_platform_worker

    user = _require_auth(request)
    _require_platform_worker(user)

    if body.state == "done" and not body.sha256_verified:
        raise HTTPException(
            422,
            "a file cannot be recorded done without sha256 verification — the "
            "resume path skips done files, so this would leave an unverified "
            "copy nobody re-checks",
        )

    from control_plane.db import control_plane_transaction

    with control_plane_transaction() as conn:
        exists = conn.execute(
            "SELECT 1 FROM volume_promotions WHERE promotion_id = %s", (promotion_id,)
        ).fetchone()
        if not exists:
            raise HTTPException(404, "Promotion not found")
        conn.execute(
            """INSERT INTO volume_promotion_files
                 (promotion_id, tenant_id, artifact_id, logical_name, size_bytes,
                  bytes_written, sha256_verified, state, failure_code, updated_at)
               SELECT %s, vp.tenant_id, %s, %s, %s, %s, %s, %s, NULLIF(%s, ''),
                      clock_timestamp()
                 FROM volume_promotions vp WHERE vp.promotion_id = %s
               ON CONFLICT (promotion_id, artifact_id) DO UPDATE
                 SET state = EXCLUDED.state,
                     bytes_written = EXCLUDED.bytes_written,
                     sha256_verified = EXCLUDED.sha256_verified,
                     failure_code = EXCLUDED.failure_code,
                     updated_at = clock_timestamp()""",
            (
                promotion_id, body.artifact_id, body.logical_name, body.size_bytes,
                body.size_bytes if body.state == "done" else 0,
                body.sha256_verified, body.state, body.failure_code,
                promotion_id,
            ),
        )
        # Touch the promotion so the stale sweep measures progress rather than
        # start time. Without this a long multi-file copy is abandoned mid-way
        # for making steady progress.
        conn.execute(
            "UPDATE volume_promotions SET updated_at = clock_timestamp() "
            " WHERE promotion_id = %s",
            (promotion_id,),
        )
    return {"ok": True, "artifact_id": body.artifact_id, "state": body.state}
