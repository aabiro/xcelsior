"""CVP-01 T-01: customer NFS fields must not reach the host's root mount helper.

T1 (reports/cvp-01-evidence/T1/RESULTS-2026-11-09.md) showed a customer could
set ``nfs_server``/``nfs_path``/``nfs_mount_point`` on ``POST /instance`` and
have a provider host mount another tenant's volume, or the whole export tree,
read-write into their container. Four layers now refuse that, and each is
asserted on its own so a regression in one is visible even if another holds:

1. the REST route refuses the fields for non-operators;
2. the launch-plan validator refuses them in any canonical spec;
3. the worker refuses a legacy mount point under the managed-volume root;
4. the root helper refuses a volume mount whose export is not that volume's
   own directory, and refuses to report success for a mount point that
   already carries a different source.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

from control_plane.launch.canonicalize import canonicalize  # noqa: E402
from control_plane.launch.validation import validate_canonical_spec  # noqa: E402

import worker_privileged as wp  # noqa: E402

NFS_FIELDS = {
    "nfs_server": "100.64.0.4",
    "nfs_path": "/exports/volumes/vol-victim000001",
    "nfs_mount_point": "/mnt/xcelsior-volumes/vol-victim000001",
}


# ── 1. REST route ───────────────────────────────────────────────────────


@pytest.fixture
def instance_client(monkeypatch):
    from fastapi.testclient import TestClient

    import routes.instances as instances
    from api import app

    monkeypatch.setattr(instances, "_require_scope", lambda *a, **k: None)
    monkeypatch.setattr(instances, "_require_team_instance_write", lambda *a, **k: None)
    return TestClient(app), instances


@pytest.mark.parametrize("field", sorted(NFS_FIELDS))
def test_route_refuses_nfs_fields_for_customers(instance_client, monkeypatch, field):
    client, instances = instance_client
    customer = {"email": "c@example.test", "user_id": "user-c", "role": "submitter", "is_admin": 0}
    monkeypatch.setattr(instances, "_require_auth", lambda request: customer)
    body = {"name": "x", "interactive": True, "num_gpus": 1, field: NFS_FIELDS[field]}
    r = client.post("/instance", json=body)
    assert r.status_code == 403, r.text
    assert "volume_ids" in r.text


def test_route_still_lets_an_operator_past_the_gate(instance_client, monkeypatch):
    """The gate itself is the only thing under test: the rest of the route is
    stubbed at the suspension check so this stays database-free."""
    client, instances = instance_client
    operator = {"email": "op@example.test", "user_id": "user-op", "role": "admin", "is_admin": 1}
    monkeypatch.setattr(instances, "_require_auth", lambda request: operator)
    sentinel = RuntimeError("past the NFS gate")
    with mock.patch("abuse_policy.account_suspended", side_effect=sentinel):
        with pytest.raises(RuntimeError, match="past the NFS gate"):
            client.post("/instance", json={"name": "x", "interactive": True, **NFS_FIELDS})


# ── 2. launch-plan validator ───────────────────────────────────────────


@pytest.mark.parametrize("field", sorted(NFS_FIELDS))
def test_canonical_spec_with_nfs_fields_is_a_problem(field):
    spec = canonicalize({"name": "x", "num_gpus": 1, field: NFS_FIELDS[field]})
    problems = validate_canonical_spec(spec)
    assert [p.field for p in problems if p.code == "nfs_operator_only"] == [field]


def test_canonical_spec_without_nfs_fields_has_no_nfs_problem():
    spec = canonicalize({"name": "x", "num_gpus": 1, "volume_ids": ["vol-abc123def456"]})
    assert not [p for p in validate_canonical_spec(spec) if p.code == "nfs_operator_only"]


# ── 3. worker: legacy mount point ───────────────────────────────────────


def test_worker_refuses_a_legacy_mount_under_the_managed_root(monkeypatch):
    import worker_agent

    def _run_authority():
        return {
            "job-1": {
                "attempt_id": "att-1",
                "fencing_token": 7,
                "container_name": "xcl-job-1-att-1",
            }
        }

    job = {
        "job_id": "job-1",
        "name": "job-1",
        "image": "image:test",
        "_v2_attempt_id": "att-1",
        "_v2_fencing_token": 7,
        "_v2_container_name": "xcl-job-1-att-1",
        **NFS_FIELDS,
    }
    with (
        mock.patch.object(worker_agent, "_v2_attempts", _run_authority()),
        mock.patch.object(worker_agent, "get_gpu_info", return_value={"gpu_model": "RTX 2060"}),
        mock.patch.object(worker_agent, "build_platform_env", return_value={}),
        mock.patch.object(worker_agent, "_kill_container"),
        mock.patch.object(worker_agent, "_remove_container"),
        mock.patch.object(worker_agent.time, "sleep"),
        mock.patch.object(worker_agent, "report_job_status", return_value=True) as report,
        mock.patch.object(worker_agent, "_push_log_lines"),
        mock.patch.object(worker_agent, "_v2_resource_intent"),
        mock.patch.object(worker_agent, "_v2_cleanup_resources"),
        mock.patch.object(worker_agent, "_mount_nfs", return_value=True) as mount,
        mock.patch.object(worker_agent, "build_secure_docker_args") as build,
    ):
        worker_agent.run_job(job)
    mount.assert_not_called()
    build.assert_not_called()
    assert any(
        "legacy NFS mount point is reserved" in str(c.kwargs.get("error_message"))
        for c in report.call_args_list
    )


# ── 4. root helper ─────────────────────────────────────────────────────


@pytest.fixture
def helper_conf(tmp_path):
    return {
        **wp.load_config(str(tmp_path / "missing.conf")),
        "VOLUME_DIR": str(tmp_path / "volumes"),
        "VOLUME_KEY_DIR": str(tmp_path / "keys"),
    }


@pytest.fixture
def helper_mount(monkeypatch):
    calls = []
    monkeypatch.setattr(wp.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(
        wp.subprocess,
        "run",
        lambda cmd, **k: calls.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=b""),
    )
    return calls


def _req(**args):
    return {"v": wp.PROTOCOL, "op": "nfs_mount", "args": args}


@pytest.mark.parametrize(
    "export",
    ["/exports/volumes", "/exports", "/exports/volumes/vol-other000001", "/exports/volumes/v1/sub"],
)
def test_helper_refuses_a_volume_export_that_is_not_the_volume(
    helper_conf, helper_mount, export, monkeypatch
):
    monkeypatch.setattr(wp, "_is_mountpoint", lambda p: False)
    reply = wp.handle(_req(server="100.64.0.1", export=export, volume_id="v1"), helper_conf)
    assert reply["refused"], reply
    assert helper_mount == []


def test_helper_accepts_the_volume_own_directory(helper_conf, helper_mount, monkeypatch):
    monkeypatch.setattr(wp, "_is_mountpoint", lambda p: False)
    reply = wp.handle(
        _req(server="100.64.0.1", export="/exports/volumes/v1", volume_id="v1"), helper_conf
    )
    assert reply["ok"] and reply["mount_point"] == "/mnt/xcelsior-volumes/v1"


def test_helper_refuses_to_reuse_a_mount_with_a_different_source(
    helper_conf, helper_mount, monkeypatch
):
    monkeypatch.setattr(wp, "_is_mountpoint", lambda p: True)
    monkeypatch.setattr(wp, "_mount_source", lambda p: "100.64.0.1:/exports/volumes/v1")
    reply = wp.handle(
        _req(server="100.64.0.9", export="/exports/volumes/v1", volume_id="v1"), helper_conf
    )
    assert reply["refused"] and "different source" in reply["error"]
    assert helper_mount == []  # no mount, and no success for someone else's data


def test_helper_reuses_a_mount_with_the_same_source(helper_conf, helper_mount, monkeypatch):
    monkeypatch.setattr(wp, "_is_mountpoint", lambda p: True)
    monkeypatch.setattr(wp, "_mount_source", lambda p: "100.64.0.1:/exports/volumes/v1")
    reply = wp.handle(
        _req(server="100.64.0.1", export="/exports/volumes/v1", volume_id="v1"), helper_conf
    )
    assert reply["ok"] and reply["already"] is True


def test_helper_unreadable_source_is_not_a_match(helper_conf, helper_mount, monkeypatch):
    monkeypatch.setattr(wp, "_is_mountpoint", lambda p: True)
    monkeypatch.setattr(wp, "_mount_source", lambda p: "")
    reply = wp.handle(_req(server="100.64.0.1", export="/exports/models"), helper_conf)
    assert reply["refused"]


def test_mount_source_reads_findmnt(monkeypatch):
    seen = []

    def run(cmd, **k):
        seen.append(cmd)
        return SimpleNamespace(returncode=0, stdout="100.64.0.4:/exports/volumes/v1\n", stderr="")

    monkeypatch.setattr(wp.subprocess, "run", run)
    assert wp._mount_source("/mnt/xcelsior-volumes/v1") == "100.64.0.4:/exports/volumes/v1"
    assert seen[0][:1] == ["findmnt"] and "/mnt/xcelsior-volumes/v1" in seen[0]
    monkeypatch.setattr(
        wp.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="x")
    )
    assert wp._mount_source("/mnt/xcelsior-volumes/v1") == ""


def test_the_production_export_setup_is_the_root_cause_and_is_documented():
    """Not a behaviour test: a pin so the mesh-wide export is not forgotten.

    Application-layer refusals above stop the customer path. The storage host
    still exports every volume to every mesh address with ``no_root_squash``
    (api/scripts/setup_nfs_vps.sh), so a provider host that is compromised in
    any other way can still reach every volume. Narrowing that export is the
    remaining remediation item in the CVP brief (T-01, step 3).
    """
    script = Path(__file__).resolve().parents[1] / "api" / "scripts" / "setup_nfs_vps.sh"
    assert "no_root_squash" in script.read_text()


# ── D1: single-row lookups where a whole-table scan used to be ──────────


def test_agent_ssh_keys_looks_the_job_up_by_id(monkeypatch):
    """A whole-table ``list_jobs()`` here timed out on a 137k-row database and
    made the worker skip key injection silently (T1 defect D1)."""
    from fastapi.testclient import TestClient

    import routes.agent as agent
    import scheduler
    from api import app

    monkeypatch.setattr(agent, "_require_agent_auth", lambda *a, **k: {"host_id": "h"})
    monkeypatch.setattr(agent, "list_jobs", mock.Mock(side_effect=AssertionError("scanned")))
    monkeypatch.setattr(scheduler, "get_job", lambda job_id: None)
    assert TestClient(app).get("/agent/ssh-keys/nope").status_code == 404
    monkeypatch.setattr(scheduler, "get_job", lambda job_id: {"job_id": job_id, "owner": ""})
    assert TestClient(app).get("/agent/ssh-keys/j1").json() == {"ok": True, "keys": []}


def test_terminal_ticket_looks_the_instance_up_by_id(monkeypatch):
    from fastapi.testclient import TestClient

    import routes.terminal as terminal
    import scheduler
    from api import app

    monkeypatch.setattr(terminal, "_require_auth", lambda request: {"email": "c@example.test"})
    monkeypatch.setattr(terminal, "_require_scope", lambda *a, **k: None)
    monkeypatch.setattr(terminal, "list_jobs", mock.Mock(side_effect=AssertionError("scanned")))
    seen = []

    def check(user, instance, instance_id=None):
        seen.append((instance, instance_id))
        from fastapi import HTTPException

        raise HTTPException(404, "stop here")

    monkeypatch.setattr(terminal, "_check_terminal_access", check)
    monkeypatch.setattr(scheduler, "get_job", lambda job_id: {"job_id": job_id})
    r = TestClient(app).post("/api/terminal/ticket", json={"instance_id": "abc12345"})
    assert r.status_code == 404 and seen == [({"job_id": "abc12345"}, "abc12345")]
