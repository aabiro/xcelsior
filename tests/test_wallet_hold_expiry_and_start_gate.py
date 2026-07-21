"""Start/restart available-balance gates + durable wallet-hold expiry.

Drives shipped ``wallet_has_available_funds``, ``start_instance`` /
``restart_instance`` fund checks, and ``expire_stale_wallet_holds``
against real Postgres.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _has_holds = (
            _c.execute("SELECT to_regclass('public.wallet_holds')").fetchone()[0]
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
else:
    if not _has_holds:  # pragma: no cover
        pytestmark = pytest.mark.skip(
            "wallet_holds missing — run alembic upgrade head"
        )


@pytest.fixture
def cleanup():
    ids = {"customers": [], "jobs": [], "holds": [], "hosts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute(
                "UPDATE jobs SET wallet_hold_id = NULL WHERE job_id=%s", (jid,)
            )
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["holds"]:
            try:
                conn.execute(
                    "DELETE FROM wallet_holds WHERE hold_id=%s::uuid", (hid,)
                )
            except Exception:
                pass
        for cid in ids["customers"]:
            conn.execute("DELETE FROM wallet_holds WHERE customer_id=%s", (cid,))
            conn.execute(
                "DELETE FROM wallet_transactions WHERE customer_id=%s", (cid,)
            )
            conn.execute("DELETE FROM wallets WHERE customer_id=%s", (cid,))
        for host in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (host,))
        conn.commit()


def _customer(cleanup) -> str:
    cid = f"exp-cust-{uuid.uuid4().hex[:10]}"
    cleanup["customers"].append(cid)
    return cid


def _mk_stopped_job(cleanup, *, owner: str, host_id: str) -> str:
    job_id = f"j-exp-{uuid.uuid4().hex[:8]}"
    now = time.time()
    import json

    payload = {
        "owner": owner,
        "status": "stopped",
        "container_name": f"xcl-{job_id}",
        "started_at": now - 1000,
        "stopped_at": now - 100,
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload)
               VALUES (%s, 'active', %s, %s)
               ON CONFLICT (host_id) DO NOTHING""",
            (host_id, now, json.dumps({"host_id": host_id})),
        )
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload)
               VALUES (%s, 'stopped', 0, %s, %s, %s)""",
            (job_id, now - 1000, host_id, json.dumps(payload)),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    cleanup["hosts"].append(host_id)
    return job_id


# ── Available gate ─────────────────────────────────────────────────────


def test_wallet_has_available_funds_blocked_when_fully_held(cleanup):
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 5.0, description="seed")
    h = eng.create_wallet_hold(cid, 5.0, idempotency_key=f"full-{cid}")
    assert h["held"] is True
    cleanup["holds"].append(h["hold_id"])

    gate = eng.wallet_has_available_funds(cid)
    assert gate["ok"] is False
    assert gate["reason"] == "insufficient_available"
    assert float(gate["balance_cad"]) == pytest.approx(5.0)
    assert float(gate["available_cad"]) == pytest.approx(0.0)
    assert float(gate["held_cad"]) == pytest.approx(5.0)


def test_wallet_has_available_funds_ok_when_partial_hold(cleanup):
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 10.0, description="seed")
    h = eng.create_wallet_hold(cid, 3.0, idempotency_key=f"part-{cid}")
    cleanup["holds"].append(h["hold_id"])

    gate = eng.wallet_has_available_funds(cid)
    assert gate["ok"] is True
    assert float(gate["available_cad"]) == pytest.approx(7.0)


def test_start_instance_fails_when_ledger_positive_but_fully_held(cleanup):
    """Shipped start_instance refuses when available is zero despite balance>0."""
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 4.0, description="seed")
    h = eng.create_wallet_hold(cid, 4.0, idempotency_key=f"start-block-{cid}")
    cleanup["holds"].append(h["hold_id"])

    host_id = f"h-exp-s-{uuid.uuid4().hex[:6]}"
    job_id = _mk_stopped_job(cleanup, owner=cid, host_id=host_id)

    result = eng.start_instance(job_id)
    assert result.get("started") is False
    assert result.get("reason") == "insufficient_balance"
    assert float(result.get("available_cad") or 0) == pytest.approx(0.0)


def test_restart_instance_fails_when_fully_held(cleanup):
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 3.0, description="seed")
    h = eng.create_wallet_hold(cid, 3.0, idempotency_key=f"rst-block-{cid}")
    cleanup["holds"].append(h["hold_id"])

    host_id = f"h-exp-r-{uuid.uuid4().hex[:6]}"
    job_id = _mk_stopped_job(cleanup, owner=cid, host_id=host_id)

    result = eng.restart_instance(job_id)
    assert result.get("restarted") is False
    assert result.get("reason") == "insufficient_balance"


def test_start_gate_passes_when_available_positive(cleanup, monkeypatch):
    """When available > 0, fund gate allows start (enqueue may still fail without agent)."""
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 10.0, description="seed")
    h = eng.create_wallet_hold(cid, 2.0, idempotency_key=f"start-ok-{cid}")
    cleanup["holds"].append(h["hold_id"])

    host_id = f"h-exp-ok-{uuid.uuid4().hex[:6]}"
    job_id = _mk_stopped_job(cleanup, owner=cid, host_id=host_id)

    # Avoid needing a real agent — just prove fund gate is not the failure.
    monkeypatch.setattr(
        "routes.agent.enqueue_agent_command",
        lambda *a, **k: 1,
    )

    result = eng.start_instance(job_id)
    # May be started True (legacy path enqueued) or fail for other reasons,
    # but must not be insufficient_balance when available > 0.
    assert result.get("reason") != "insufficient_balance"
    if result.get("started"):
        assert result.get("status") in ("running", "queued", "restarting")


# ── Expiry ─────────────────────────────────────────────────────────────


def test_expire_stale_wallet_holds_frees_available(cleanup):
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 8.0, description="seed")
    h = eng.create_wallet_hold(
        cid, 5.0, idempotency_key=f"exp-{cid}", expires_in_sec=60
    )
    hold_id = h["hold_id"]
    cleanup["holds"].append(hold_id)

    # Force past expiry without waiting.
    with _pool.connection() as conn:
        conn.execute(
            """UPDATE wallet_holds
                  SET expires_at = %s
                WHERE hold_id = %s::uuid""",
            (time.time() - 10, hold_id),
        )
        conn.commit()

    # Before global expire, lazy read should already expire via _active_holds_total
    # but force the durable path explicitly.
    n = eng.expire_stale_wallet_holds(limit=100)
    assert n >= 1

    w = eng.get_wallet(cid)
    assert float(w["held_cad"]) == pytest.approx(0.0)
    assert float(w["available_cad"]) == pytest.approx(8.0)

    with _pool.connection() as conn:
        st = conn.execute(
            "SELECT status FROM wallet_holds WHERE hold_id=%s::uuid", (hold_id,)
        ).fetchone()
    status = st[0] if not isinstance(st, dict) else st["status"]
    assert status == "expired"

    # Second expire is a no-op for this hold.
    n2 = eng.expire_stale_wallet_holds(limit=100)
    # May expire other fixtures' holds; this hold stays expired once.
    with _pool.connection() as conn:
        st2 = conn.execute(
            "SELECT status FROM wallet_holds WHERE hold_id=%s::uuid", (hold_id,)
        ).fetchone()
    status2 = st2[0] if not isinstance(st2, dict) else st2["status"]
    assert status2 == "expired"
    assert n2 >= 0


def test_after_expiry_start_gate_passes(cleanup, monkeypatch):
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 6.0, description="seed")
    h = eng.create_wallet_hold(
        cid, 6.0, idempotency_key=f"exp-start-{cid}", expires_in_sec=60
    )
    hold_id = h["hold_id"]
    cleanup["holds"].append(hold_id)

    with _pool.connection() as conn:
        conn.execute(
            "UPDATE wallet_holds SET expires_at=%s WHERE hold_id=%s::uuid",
            (time.time() - 5, hold_id),
        )
        conn.commit()

    eng.expire_stale_wallet_holds()
    gate = eng.wallet_has_available_funds(cid)
    assert gate["ok"] is True
    assert float(gate["available_cad"]) == pytest.approx(6.0)

    host_id = f"h-exp-es-{uuid.uuid4().hex[:6]}"
    job_id = _mk_stopped_job(cleanup, owner=cid, host_id=host_id)
    monkeypatch.setattr("routes.agent.enqueue_agent_command", lambda *a, **k: 1)
    result = eng.start_instance(job_id)
    assert result.get("reason") != "insufficient_balance"


def test_inventory_start_gate_and_expiry_registration():
    root = Path(__file__).resolve().parents[1]
    billing = (root / "billing.py").read_text()
    assert "def expire_stale_wallet_holds" in billing
    assert "def wallet_has_available_funds" in billing
    assert "wallet_has_available_funds" in billing
    # start/restart use available gate
    assert billing.count("wallet_has_available_funds") >= 3

    instances = (root / "routes" / "instances.py").read_text()
    assert "wallet_has_available_funds" in instances
    assert 'wallet["balance_cad"] <= 0' not in instances or instances.count(
        "available"
    ) > 0
    # API start/restart must not only check raw balance_cad
    start_src = instances[instances.find("def api_start_instance") :]
    start_src = start_src[: start_src.find("def api_restart_instance")]
    assert "wallet_has_available_funds" in start_src
    restart_src = instances[instances.find("def api_restart_instance") :]
    restart_src = restart_src[: restart_src.find("def api_admin_reinject_shell")]
    assert "wallet_has_available_funds" in restart_src

    bg = (root / "bg_worker.py").read_text()
    assert 'register_task("wallet_hold_expiry"' in bg
    assert "expire_stale_wallet_holds" in bg
