"""Residual job-scoped agent commands: attempt-owned vs pure-legacy.

Drives the shipped residual enqueue entry points / helpers against real
Postgres:

- SSH reinject (``routes.ssh._trigger_reinject_for_user``)
- admin reinject / reset (``routes.instances`` helpers via resolve + enqueue)
- volume hot mount/unmount (``VolumeEngine._enqueue_hot_*``)
- snapshot identity (``control_plane.job_targets`` + API resolver path)
- pure policy classifier for host residual inventory

Attempt-owned work must never enqueue unfenced wrong-identity
``xcl-{job_id}`` container commands. Pure-legacy keeps ``xcl-{job_id}``.
Host-scoped admin rollout (upgrade_agent / rollback_agent) stays out of
job attempt authority.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _has_attempt = (
            _c.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name='jobs' AND column_name='active_attempt_id'"
            ).fetchone()
            is not None
        )
        _has_job_attempts = (
            _c.execute("SELECT to_regclass('public.job_attempts')").fetchone()[0]
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not (_has_attempt and _has_job_attempts):  # pragma: no cover
        pytestmark = pytest.mark.skip("control-plane attempt tables missing")


@pytest.fixture
def cleanup():
    ids = {"jobs": [], "hosts": [], "commands": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute(
                "DELETE FROM agent_commands WHERE args->>'job_id' = %s OR job_id = %s",
                (jid, jid),
            )
            conn.execute("DELETE FROM placement_leases WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM job_attempts WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mk_host(cleanup, host_id: str) -> str:
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload)
               VALUES (%s, 'active', %s, %s)
               ON CONFLICT (host_id) DO NOTHING""",
            (host_id, time.time(), json.dumps({"host_id": host_id})),
        )
        conn.commit()
    cleanup["hosts"].append(host_id)
    return host_id


def _mk_running_legacy(cleanup, *, host_id: str, owner: str, interactive: bool = True) -> str:
    job_id = f"j-res-leg-{uuid.uuid4().hex[:8]}"
    payload = {
        "job_id": job_id,
        "name": job_id,
        "owner": owner,
        "status": "running",
        "container_name": f"xcl-{job_id}",
        "interactive": interactive,
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload)
               VALUES (%s, 'running', 0, %s, %s, %s)""",
            (job_id, time.time(), host_id, json.dumps(payload)),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id


def _mk_running_attempt_owned(
    cleanup, *, host_id: str, owner: str, interactive: bool = True
) -> tuple[str, str]:
    job_id = f"j-res-att-{uuid.uuid4().hex[:8]}"
    attempt_id = str(uuid.uuid4())
    # Stale legacy name in payload is the hazard we must not honor.
    payload = {
        "job_id": job_id,
        "name": job_id,
        "owner": owner,
        "status": "running",
        "container_name": f"xcl-{job_id}",
        "interactive": interactive,
    }
    with _pool.connection() as conn:
        fence = conn.execute(
            "SELECT nextval('placement_fencing_token_seq')"
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload,
                    active_attempt_id)
               VALUES (%s, 'running', 0, %s, %s, %s, NULL)""",
            (job_id, time.time(), host_id, json.dumps(payload)),
        )
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token, job_generation)
               VALUES (%s, %s, 1, 'running', %s, %s, 1)""",
            (attempt_id, job_id, host_id, fence),
        )
        conn.execute(
            "UPDATE jobs SET active_attempt_id=%s WHERE job_id=%s",
            (attempt_id, job_id),
        )
        conn.execute(
            """INSERT INTO placement_leases
                   (job_id, attempt_id, host_id, fencing_token, status,
                    claim_deadline, claimed_at, last_renewed_at, expires_at)
               VALUES (%s, %s, %s, %s, 'active',
                       clock_timestamp() + interval '10 minutes',
                       clock_timestamp(), clock_timestamp(),
                       clock_timestamp() + interval '10 minutes')""",
            (job_id, attempt_id, host_id, fence),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id, attempt_id


def _mk_fenced_history_no_active(
    cleanup, *, host_id: str, owner: str, status: str = "stopped"
) -> str:
    job_id = f"j-res-fh-{uuid.uuid4().hex[:8]}"
    attempt_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "name": job_id,
        "owner": owner,
        "status": status,
        "container_name": f"xcl-{job_id}",
        "interactive": True,
    }
    with _pool.connection() as conn:
        fence = conn.execute(
            "SELECT nextval('placement_fencing_token_seq')"
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload,
                    active_attempt_id)
               VALUES (%s, %s, 0, %s, %s, %s, NULL)""",
            (job_id, status, time.time(), host_id, json.dumps(payload)),
        )
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token, job_generation)
               VALUES (%s, %s, 1, 'lost', %s, %s, 1)""",
            (attempt_id, job_id, host_id, fence),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id


def _commands_for_job(job_id: str) -> list[dict]:
    with _pool.connection() as conn:
        rows = conn.execute(
            """SELECT command, args, host_id, created_by, attempt_id
                 FROM agent_commands
                WHERE args->>'job_id' = %s OR job_id = %s
                ORDER BY id""",
            (job_id, job_id),
        ).fetchall()
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(r)
        else:
            out.append(
                {
                    "command": r[0],
                    "args": r[1],
                    "host_id": r[2],
                    "created_by": r[3],
                    "attempt_id": r[4],
                }
            )
    return out


# ── Pure policy ────────────────────────────────────────────────────────


def test_classify_and_attempt_container_name_match_worker():
    from control_plane.job_targets import (
        attempt_container_name,
        classify_job_target,
        is_fenced_history_job,
    )

    job_id = "job-abc"
    attempt_id = "abcdef12-3456-7890-abcd-ef1234567890"
    assert attempt_container_name(job_id, attempt_id) == f"xcl-{job_id}-abcdef12"

    leg = classify_job_target(
        job_id=job_id,
        host_id="h1",
        status="running",
        active_attempt_id=None,
        has_fenced_history=False,
        payload_container_name=None,
    )
    assert leg.class_ == "legacy"
    assert leg.container_name == f"xcl-{job_id}"
    assert leg.allows_unfenced_container_command is True

    att = classify_job_target(
        job_id=job_id,
        host_id="h1",
        status="running",
        active_attempt_id=attempt_id,
        has_fenced_history=True,
        payload_container_name=f"xcl-{job_id}",  # stale must be ignored
    )
    assert att.class_ == "attempt_owned"
    assert att.container_name == f"xcl-{job_id}-abcdef12"
    assert att.allows_unfenced_container_command is True

    fh = classify_job_target(
        job_id=job_id,
        host_id="h1",
        status="stopped",
        active_attempt_id=None,
        has_fenced_history=True,
    )
    assert fh.class_ == "fenced_history"
    assert fh.allows_unfenced_container_command is False

    assert is_fenced_history_job(active_attempt_id=None, has_fenced_history=False) is False
    assert is_fenced_history_job(active_attempt_id="a", has_fenced_history=False) is True
    assert is_fenced_history_job(active_attempt_id=None, has_fenced_history=True) is True


def test_resolve_job_command_target_real_pg(cleanup):
    from control_plane.job_targets import (
        attempt_container_name,
        resolve_job_command_target,
    )

    host_id = f"h-res-resolve-{uuid.uuid4().hex[:6]}"
    _mk_host(cleanup, host_id)
    owner = f"owner-{uuid.uuid4().hex[:6]}@test"

    leg = _mk_running_legacy(cleanup, host_id=host_id, owner=owner)
    att_job, attempt_id = _mk_running_attempt_owned(
        cleanup, host_id=host_id, owner=owner
    )
    fh = _mk_fenced_history_no_active(cleanup, host_id=host_id, owner=owner)

    t_leg = resolve_job_command_target(leg)
    assert t_leg is not None
    assert t_leg.class_ == "legacy"
    assert t_leg.container_name == f"xcl-{leg}"
    assert t_leg.allows_unfenced_container_command is True

    t_att = resolve_job_command_target(att_job)
    assert t_att is not None
    assert t_att.class_ == "attempt_owned"
    assert t_att.container_name == attempt_container_name(att_job, attempt_id)
    assert t_att.container_name != f"xcl-{att_job}"

    t_fh = resolve_job_command_target(fh)
    assert t_fh is not None
    assert t_fh.class_ == "fenced_history"
    assert t_fh.allows_unfenced_container_command is False


# ── SSH reinject ───────────────────────────────────────────────────────


def test_ssh_reinject_legacy_enqueues_xcl_job_id(cleanup, monkeypatch):
    from routes import ssh as ssh_mod

    host_id = f"h-res-ssh-l-{uuid.uuid4().hex[:6]}"
    _mk_host(cleanup, host_id)
    owner = f"ssh-leg-{uuid.uuid4().hex[:6]}@test"
    job_id = _mk_running_legacy(cleanup, host_id=host_id, owner=owner)

    user = {"email": owner, "customer_id": owner}
    monkeypatch.setattr(
        "routes._deps._customer_ids_accessible_by_user",
        lambda _u: [owner],
    )

    n = ssh_mod._trigger_reinject_for_user(user)
    assert n >= 1
    cmds = _commands_for_job(job_id)
    reinject = [c for c in cmds if c["command"] == "reinject_shell"]
    assert len(reinject) == 1
    assert reinject[0]["args"]["container_name"] == f"xcl-{job_id}"
    assert reinject[0]["args"]["job_id"] == job_id


def test_ssh_reinject_attempt_owned_uses_attempt_container_name(
    cleanup, monkeypatch
):
    from control_plane.job_targets import attempt_container_name
    from routes import ssh as ssh_mod

    host_id = f"h-res-ssh-a-{uuid.uuid4().hex[:6]}"
    _mk_host(cleanup, host_id)
    owner = f"ssh-att-{uuid.uuid4().hex[:6]}@test"
    job_id, attempt_id = _mk_running_attempt_owned(
        cleanup, host_id=host_id, owner=owner
    )
    expected = attempt_container_name(job_id, attempt_id)

    user = {"email": owner, "customer_id": owner}
    monkeypatch.setattr(
        "routes._deps._customer_ids_accessible_by_user",
        lambda _u: [owner],
    )

    n = ssh_mod._trigger_reinject_for_user(user)
    assert n >= 1
    cmds = _commands_for_job(job_id)
    reinject = [c for c in cmds if c["command"] == "reinject_shell"]
    assert len(reinject) == 1
    assert reinject[0]["args"]["container_name"] == expected
    # Never the stale legacy identity while attempt-owned.
    assert reinject[0]["args"]["container_name"] != f"xcl-{job_id}"


def test_ssh_reinject_fenced_history_skips_enqueue(cleanup, monkeypatch):
    from routes import ssh as ssh_mod

    host_id = f"h-res-ssh-f-{uuid.uuid4().hex[:6]}"
    _mk_host(cleanup, host_id)
    owner = f"ssh-fh-{uuid.uuid4().hex[:6]}@test"
    # Fenced history with status=running is pathological; if it appears, skip.
    job_id = _mk_fenced_history_no_active(
        cleanup, host_id=host_id, owner=owner, status="running"
    )

    user = {"email": owner, "customer_id": owner}
    monkeypatch.setattr(
        "routes._deps._customer_ids_accessible_by_user",
        lambda _u: [owner],
    )

    n = ssh_mod._trigger_reinject_for_user(user)
    assert n == 0
    cmds = _commands_for_job(job_id)
    assert [c for c in cmds if c["command"] == "reinject_shell"] == []


# ── Reset / admin reinject (resolve + real enqueue) ────────────────────


def test_reset_enqueue_legacy_and_attempt_owned(cleanup):
    from control_plane.job_targets import (
        attempt_container_name,
        resolve_job_command_target,
    )
    from routes.agent import enqueue_agent_command

    host_id = f"h-res-rst-{uuid.uuid4().hex[:6]}"
    _mk_host(cleanup, host_id)
    owner = f"rst-{uuid.uuid4().hex[:6]}@test"

    leg = _mk_running_legacy(cleanup, host_id=host_id, owner=owner)
    att_job, attempt_id = _mk_running_attempt_owned(
        cleanup, host_id=host_id, owner=owner
    )
    fh = _mk_fenced_history_no_active(
        cleanup, host_id=host_id, owner=owner, status="running"
    )

    # Mirror the shipped gate in api_reset_instance / admin reinject.
    for job_id, expect_cname, allow in (
        (leg, f"xcl-{leg}", True),
        (att_job, attempt_container_name(att_job, attempt_id), True),
        (fh, None, False),
    ):
        target = resolve_job_command_target(job_id)
        assert target is not None
        if not allow:
            assert target.allows_unfenced_container_command is False
            continue
        assert target.allows_unfenced_container_command is True
        assert target.container_name == expect_cname
        enqueue_agent_command(
            host_id=target.host_id,
            command="reset_container",
            args={"job_id": job_id, "container_name": target.container_name},
            created_by="test_reset",
        )

    leg_cmds = [
        c for c in _commands_for_job(leg) if c["command"] == "reset_container"
    ]
    assert len(leg_cmds) == 1
    assert leg_cmds[0]["args"]["container_name"] == f"xcl-{leg}"

    att_cmds = [
        c for c in _commands_for_job(att_job) if c["command"] == "reset_container"
    ]
    assert len(att_cmds) == 1
    assert att_cmds[0]["args"]["container_name"] == attempt_container_name(
        att_job, attempt_id
    )
    assert att_cmds[0]["args"]["container_name"] != f"xcl-{att_job}"

    fh_cmds = [
        c for c in _commands_for_job(fh) if c["command"] == "reset_container"
    ]
    assert fh_cmds == []


# ── Volumes ────────────────────────────────────────────────────────────


def test_volume_hot_mount_unmount_legacy_and_attempt_owned(cleanup):
    from control_plane.job_targets import attempt_container_name
    from volumes import VolumeEngine

    host_id = f"h-res-vol-{uuid.uuid4().hex[:6]}"
    _mk_host(cleanup, host_id)
    owner = f"vol-{uuid.uuid4().hex[:6]}@test"
    leg = _mk_running_legacy(cleanup, host_id=host_id, owner=owner)
    att_job, attempt_id = _mk_running_attempt_owned(
        cleanup, host_id=host_id, owner=owner
    )

    eng = VolumeEngine()
    eng._enqueue_hot_mount(
        host_id, leg, "vol-leg-1", "/workspace", "rw"
    )
    eng._enqueue_hot_mount(
        host_id, att_job, "vol-att-1", "/workspace", "rw"
    )
    eng._enqueue_hot_unmount(host_id, leg, "vol-leg-1", "/workspace")
    eng._enqueue_hot_unmount(host_id, att_job, "vol-att-1", "/workspace")

    leg_mount = [
        c
        for c in _commands_for_job(leg)
        if c["command"] in ("mount_volume", "unmount_volume")
    ]
    assert len(leg_mount) == 2
    for c in leg_mount:
        assert c["args"]["container_name"] == f"xcl-{leg}"

    att_mount = [
        c
        for c in _commands_for_job(att_job)
        if c["command"] in ("mount_volume", "unmount_volume")
    ]
    assert len(att_mount) == 2
    expected = attempt_container_name(att_job, attempt_id)
    for c in att_mount:
        assert c["args"]["container_name"] == expected
        assert c["args"]["container_name"] != f"xcl-{att_job}"


def test_volume_hot_mount_fenced_history_fails_closed(cleanup):
    from volumes import VolumeEngine

    host_id = f"h-res-vol-f-{uuid.uuid4().hex[:6]}"
    _mk_host(cleanup, host_id)
    owner = f"vol-f-{uuid.uuid4().hex[:6]}@test"
    fh = _mk_fenced_history_no_active(
        cleanup, host_id=host_id, owner=owner, status="running"
    )

    eng = VolumeEngine()
    with pytest.raises(ValueError, match="no residual container target"):
        eng._enqueue_hot_mount(host_id, fh, "vol-fh-1", "/workspace", "rw")

    assert [
        c for c in _commands_for_job(fh) if c["command"] == "mount_volume"
    ] == []


# ── Snapshot identity ──────────────────────────────────────────────────


def test_snapshot_identity_attempt_owned_not_legacy_name(cleanup):
    from control_plane.job_targets import (
        attempt_container_name,
        resolve_job_command_target,
    )
    from routes.agent import enqueue_agent_command

    host_id = f"h-res-snap-{uuid.uuid4().hex[:6]}"
    _mk_host(cleanup, host_id)
    owner = f"snap-{uuid.uuid4().hex[:6]}@test"
    leg = _mk_running_legacy(cleanup, host_id=host_id, owner=owner)
    att_job, attempt_id = _mk_running_attempt_owned(
        cleanup, host_id=host_id, owner=owner
    )

    for job_id in (leg, att_job):
        target = resolve_job_command_target(job_id)
        assert target is not None
        assert target.allows_unfenced_container_command
        enqueue_agent_command(
            target.host_id,
            "snapshot_container",
            {
                "job_id": job_id,
                "image_id": f"img-{uuid.uuid4().hex[:8]}",
                "container_name": target.container_name,
                "image_ref": "registry.example/x:tag",
            },
            created_by=owner,
            ttl_sec=3600,
        )

    leg_s = [
        c for c in _commands_for_job(leg) if c["command"] == "snapshot_container"
    ]
    assert len(leg_s) == 1
    assert leg_s[0]["args"]["container_name"] == f"xcl-{leg}"

    att_s = [
        c
        for c in _commands_for_job(att_job)
        if c["command"] == "snapshot_container"
    ]
    assert len(att_s) == 1
    assert att_s[0]["args"]["container_name"] == attempt_container_name(
        att_job, attempt_id
    )
    assert att_s[0]["args"]["container_name"] != f"xcl-{att_job}"


# ── Host-scoped residual inventory ─────────────────────────────────────


def test_host_scoped_admin_rollout_not_job_dual_writer():
    """upgrade_agent / rollback_agent stay host-level (no job_id authority)."""
    root = Path(__file__).resolve().parents[1]
    admin_src = (root / "routes" / "admin.py").read_text()
    bg_src = (root / "bg_worker.py").read_text()

    # Admin rollout enqueues upgrade_agent by host_id only.
    assert ' "upgrade_agent"' in admin_src or '"upgrade_agent"' in admin_src
    assert "upgrade_agent" in admin_src
    # Must not bind job attempt authority on the rollout path.
    assert "enqueue_current_attempt_command" not in admin_src
    assert "active_attempt_id" not in admin_src.split("upgrade_agent")[0][-500:]

    # Watchdog rollback is host-scoped args only.
    assert "rollback_agent" in bg_src
    # The rollback enqueue block must not pass job_id as residual target.
    rb_idx = bg_src.find('"rollback_agent"')
    assert rb_idx > 0
    rb_snippet = bg_src[rb_idx : rb_idx + 200]
    assert "job_id" not in rb_snippet


def test_enqueue_site_inventory_classifies_job_vs_host():
    """Static inventory: residual job sites gated; host sites separate."""
    root = Path(__file__).resolve().parents[1]
    sources = {
        "ssh": (root / "routes" / "ssh.py").read_text(),
        "instances": (root / "routes" / "instances.py").read_text(),
        "volumes": (root / "volumes.py").read_text(),
        "bg_worker": (root / "bg_worker.py").read_text(),
        "admin": (root / "routes" / "admin.py").read_text(),
        "billing": (root / "billing.py").read_text(),
        "serverless": (root / "serverless" / "service.py").read_text(),
    }

    # Job-scoped residuals must import / use the shared target resolver.
    assert "classify_job_target" in sources["ssh"] or "resolve_job_command_target" in sources["ssh"]
    assert "resolve_job_command_target" in sources["instances"]
    assert "resolve_job_command_target" in sources["volumes"]
    assert "resolve_job_command_target" in sources["bg_worker"]

    # Host-scoped: upgrade/rollback and serverless prepull stay unfenced host.
    assert "upgrade_agent" in sources["admin"]
    assert "prepull_image" in sources["serverless"]
    assert "resolve_job_command_target" not in sources["admin"]
    assert "resolve_job_command_target" not in sources["serverless"]

    # Billing lifecycle already fenced (not residual dual-writer of reinject etc).
    assert "request_fenced_stop" in sources["billing"]
