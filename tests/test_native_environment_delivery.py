"""Execution secrets stay encrypted until the exact native lease is claimed."""

import json
import os
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from control_plane.job_specs import canonical_spec_hash
from serverless.env_secrets import decrypt_env_for_worker, encrypt_env_for_storage
from tests.test_serverless_native_lifecycle import native  # noqa: F401
from tests.test_worker_claim_handoff import agent, client_post, command  # noqa: F401


def bind_environment(x, env):
    with x.pool.connection() as conn:
        spec = conn.execute("SELECT spec FROM jobs WHERE job_id=%s", (x.jid,)).fetchone()[0]
        spec["environment"] = env
        conn.execute(
            "UPDATE jobs SET spec=%s,spec_hash=%s WHERE job_id=%s",
            (Jsonb(spec), canonical_spec_hash(spec), x.jid),
        )


def claim_body(r):
    return dict(
        lease_id=r.lease_id,
        job_id=r.job_id,
        attempt_id=r.attempt_id,
        host_id=r.host_id,
        fencing_token=r.fencing_token,
        worker_session_id="native-test",
    )


def test_provisioning_never_persists_decrypted_credentials(native):
    secret = "fixture-only-endpoint-secret"
    env = encrypt_env_for_storage({"HF_TOKEN": secret, "MODEL": "test"})
    native.repo.patch_endpoint(native.ep["endpoint_id"], native.owner, {"env": env})
    w = native.svc.provision_worker(native.ep["endpoint_id"], billable=False)
    assert w
    jid = w["scheduler_job_id"]
    try:
        with native.pool.connection() as conn:
            row = conn.execute("SELECT spec,payload FROM jobs WHERE job_id=%s", (jid,)).fetchone()
            assert row[0]["environment"] == env
            assert secret not in json.dumps(row)
    finally:
        with native.pool.connection() as conn:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM serverless_workers WHERE scheduler_job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))


def test_claim_delivers_admitted_environment_without_plaintext_storage(native):
    import api

    secret = "fixture-only-claim-secret"
    env = encrypt_env_for_storage({"HF_TOKEN": secret, "MODEL": "frozen"})
    bind_environment(native, env)
    r = native.reserve()
    # Endpoint edits after admission cannot change what this worker executes.
    native.repo.patch_endpoint(native.ep["endpoint_id"], native.owner, {"env": {"MODEL": "new"}})
    headers = {"Authorization": f"Bearer {os.environ['XCELSIOR_API_TOKEN']}"}
    with TestClient(api.app) as client:
        response = client.post("/agent/v2/leases/claim", json=claim_body(r), headers=headers)
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        grant = response.json()
        assert grant["runtime_environment"] == {"HF_TOKEN": secret, "MODEL": "frozen"}
        with native.pool.connection() as conn:
            row = conn.execute(
                "SELECT spec,spec_hash FROM jobs WHERE job_id=%s", (native.jid,)
            ).fetchone()
            assert row[0]["environment"] == env
            assert grant["runtime_environment_spec_hash"] == row[1]
            commands = conn.execute(
                "SELECT args FROM agent_commands WHERE job_id=%s", (native.jid,)
            ).fetchall()
            assert secret not in json.dumps(commands)
        # The claiming session may replay a lost response; any other session is
        # refused and never sees the environment.
        replay = client.post(
            "/agent/v2/leases/claim",
            json={**claim_body(r), "worker_session_id": "another-worker"},
            headers=headers,
        )
        assert replay.status_code == 409
        assert secret not in replay.text


@pytest.mark.parametrize("fault", ["stale_fence", "decrypt_failure", "spec_tamper"])
def test_failed_claim_never_decrypts_or_leaves_partial_authority(native, monkeypatch, fault):
    import api

    bind_environment(native, encrypt_env_for_storage({"HF_TOKEN": "fixture-only-secret"}))
    r = native.reserve()
    body = claim_body(r)
    decrypt = mock.Mock(side_effect=ValueError("private diagnostic"))
    monkeypatch.setattr("serverless.env_secrets.decrypt_env_for_worker", decrypt)
    if fault == "stale_fence":
        body["fencing_token"] += 1
    elif fault == "spec_tamper":
        with native.pool.connection() as conn:
            conn.execute(
                "UPDATE jobs SET spec=jsonb_set(spec,'{environment,HF_TOKEN}','\"tampered\"') WHERE job_id=%s",
                (native.jid,),
            )
    with TestClient(api.app) as client:
        response = client.post(
            "/agent/v2/leases/claim",
            json=body,
            headers={"Authorization": f"Bearer {os.environ['XCELSIOR_API_TOKEN']}"},
        )
    assert response.status_code == 409, response.text
    assert "private diagnostic" not in response.text
    assert decrypt.call_count == (1 if fault == "decrypt_failure" else 0)
    with native.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT status FROM placement_leases WHERE lease_id=%s", (r.lease_id,)
            ).fetchone()[0]
            == "offered"
        )
        assert (
            conn.execute(
                "SELECT status FROM job_attempts WHERE attempt_id=%s", (r.attempt_id,)
            ).fetchone()[0]
            == "reserved"
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM serverless_attempt_credentials WHERE attempt_id=%s",
                (r.attempt_id,),
            ).fetchone()[0]
            == 0
        )


def test_broken_ciphertext_is_not_replaced_with_an_empty_value():
    with pytest.raises(ValueError, match="runtime environment decryption failed"):
        decrypt_env_for_worker({"HF_TOKEN": "enc:not-a-valid-ciphertext"})


@pytest.mark.parametrize("grant_fault", [None, "missing", "wrong_hash", "missing_key"])
def test_worker_receives_runtime_secrets_without_mutating_spec_or_journal(
    agent, monkeypatch, grant_fault
):
    import api
    import worker_agent as worker

    secret = "fixture-only-worker-secret"
    bind_environment(agent, encrypt_env_for_storage({"HF_TOKEN": secret, "MODEL": "test"}))
    cmd = command(agent)
    original = json.dumps(cmd, sort_keys=True, default=str)
    run = mock.Mock()
    monkeypatch.setattr(worker, "run_job", run)
    monkeypatch.setattr(worker, "v2_ack_command", mock.Mock(return_value=True))
    nack = mock.Mock()
    monkeypatch.setattr(worker, "v2_nack_command", nack)
    with TestClient(api.app) as client:

        def post(path, body):
            code, result = client_post(client, path, body)
            if path == "/agent/v2/leases/claim" and code == 200:
                if grant_fault == "missing":
                    result.pop("runtime_environment")
                elif grant_fault == "wrong_hash":
                    result["runtime_environment_spec_hash"] = "sha256:wrong"
                elif grant_fault == "missing_key":
                    result["runtime_environment"].pop("MODEL")
            return code, result

        monkeypatch.setattr(worker, "_v2_post", post)
        with mock.patch.object(worker.threading, "Thread"):
            worker.handle_start_attempt(cmd)
    assert json.dumps(cmd, sort_keys=True, default=str) == original
    if grant_fault:
        run.assert_not_called()
        nack.assert_called_once_with(
            cmd["command_id"], "runtime_environment_missing", retryable=False
        )
        with agent.pool.connection() as conn:
            assert (
                conn.execute(
                    "SELECT status FROM placement_leases WHERE lease_id=%s", (agent.r.lease_id,)
                ).fetchone()[0]
                == "released"
            )
    else:
        assert run.call_args.args[0]["environment"] == {"HF_TOKEN": secret, "MODEL": "test"}
    from pathlib import Path

    assert secret not in Path(worker._V2_JOURNAL_PATH).read_text()
