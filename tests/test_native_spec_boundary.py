"""Execution entry points never promote JSON observations into admission."""

import ast
from pathlib import Path

import pytest
from psycopg.types.json import Jsonb

from control_plane.db import run_transaction
from control_plane.job_specs import canonical_spec_hash
from control_plane.leases import claim_lease, LeaseClaimRejected
from tests.test_serverless_native_lifecycle import native  # noqa: F401


@pytest.mark.parametrize("evidence", ["missing", "recorded_execution", "wrong_hash"])
def test_an_old_offer_cannot_claim_unadmitted_or_changed_execution(native, evidence):
    reservation = native.reserve()
    with native.pool.connection() as conn:
        if evidence == "missing":
            conn.execute("UPDATE jobs SET spec=NULL WHERE job_id=%s", (native.jid,))
        elif evidence == "recorded_execution":
            spec = {"image": "historical-command", "job_type": "serverless_worker"}
            conn.execute(
                "UPDATE jobs SET spec=%s,spec_hash=%s WHERE job_id=%s",
                (Jsonb(spec), canonical_spec_hash(spec), native.jid),
            )
        else:
            conn.execute("UPDATE jobs SET spec_hash='changed' WHERE job_id=%s", (native.jid,))
    with pytest.raises(LeaseClaimRejected):
        run_transaction(
            lambda conn: claim_lease(
                conn,
                lease_id=reservation.lease_id,
                job_id=native.jid,
                attempt_id=reservation.attempt_id,
                host_id=native.hid,
                fencing_token=reservation.fencing_token,
                worker_session_id="old-offer",
            )
        )
    with native.pool.connection() as conn:
        assert conn.execute(
            "SELECT status,claimed_at FROM placement_leases WHERE lease_id=%s",
            (reservation.lease_id,),
        ).fetchone() == ("offered", None)
        assert conn.execute(
            "SELECT status FROM job_attempts WHERE attempt_id=%s", (reservation.attempt_id,)
        ).fetchone() == ("reserved",)


def test_unadmitted_observation_cannot_create_wallet_charges(native):
    from billing import get_billing_engine

    engine = get_billing_engine()
    engine.deposit(native.owner, 100, description="isolated unadmitted observation test")
    with native.pool.connection() as conn:
        conn.execute(
            """UPDATE jobs SET spec=NULL,spec_hash=NULL,status='running',phase='running',
            host_id=%s,owner_id=%s,payload=payload||jsonb_build_object(
                'owner',%s::text,'started_at',EXTRACT(EPOCH FROM clock_timestamp())-3600)
            WHERE job_id=%s""",
            (native.hid, native.owner, native.owner, native.jid),
        )
        before = conn.execute(
            "SELECT balance_micros FROM wallets WHERE customer_id=%s", (native.owner,)
        ).fetchone()
    assert engine.bill_running_period(native.jid)["billed"] is False
    engine.auto_billing_cycle()
    with native.pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM billing_cycles WHERE job_id=%s", (native.jid,)
        ).fetchone() == (0,)
        assert (
            conn.execute(
                "SELECT balance_micros FROM wallets WHERE customer_id=%s", (native.owner,)
            ).fetchone()
            == before
        )


def test_runtime_has_no_execution_payload_fallback():
    root = Path(__file__).resolve().parents[1]
    files = [
        "control_plane/attempt_billing.py",
        "control_plane/attempts.py",
        "control_plane/leases.py",
        "control_plane/serverless_credentials.py",
        "control_plane/serverless_workers.py",
        "control_plane/scheduler/claim.py",
        "control_plane/scheduler/reservation.py",
        "control_plane/scheduler/service.py",
        "control_plane/scheduler/snapshot.py",
        "api/routes/agent_v2.py",
        "api/serverless/service.py",
        "billing/billing.py",
    ]
    for name in files:
        for node in ast.walk(ast.parse((root / name).read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                compact = "".join(node.value.lower().split())
                assert "coalesce(spec,payload)" not in compact, name
                assert "coalesce(j.spec,j.payload)" not in compact, name


def test_explicit_fixture_admission_is_never_automatic():
    root = Path(__file__).resolve().parents[1]
    assert "admit_fixture_job" not in (root / "tests/conftest.py").read_text()
    for directory in ("control_plane", "workers", "api", "core", "billing"):
        for path in (root / directory).rglob("*.py"):
            if "migrations" in path.parts:
                continue
            assert "tests.native_jobs" not in path.read_text(), path
