"""Concurrency-safe wallet holds — available balance, dual-launch race, release.

Drives shipped BillingEngine hold APIs and available-balance accounting
against real Postgres.
"""

from __future__ import annotations

import concurrent.futures
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
    ids = {"customers": [], "jobs": [], "holds": []}
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
            conn.execute("DELETE FROM wallet_holds WHERE hold_id=%s::uuid", (hid,))
        for cid in ids["customers"]:
            conn.execute("DELETE FROM wallet_holds WHERE customer_id=%s", (cid,))
            conn.execute(
                "DELETE FROM wallet_transactions WHERE customer_id=%s", (cid,)
            )
            conn.execute("DELETE FROM wallets WHERE customer_id=%s", (cid,))
        conn.commit()


def _customer(cleanup) -> str:
    cid = f"hold-cust-{uuid.uuid4().hex[:10]}"
    cleanup["customers"].append(cid)
    return cid


def test_available_balance_subtracts_active_holds(cleanup):
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 10.0, description="seed")
    w = eng.get_wallet(cid)
    assert float(w["balance_cad"]) == pytest.approx(10.0)
    assert float(w["available_cad"]) == pytest.approx(10.0)
    assert float(w["held_cad"]) == pytest.approx(0.0)

    h = eng.create_wallet_hold(
        cid, 4.0, idempotency_key=f"t1-{cid}", expires_in_sec=600
    )
    assert h["held"] is True
    cleanup["holds"].append(h["hold_id"])

    w2 = eng.get_wallet(cid)
    assert float(w2["balance_cad"]) == pytest.approx(10.0)
    assert float(w2["held_cad"]) == pytest.approx(4.0)
    assert float(w2["available_cad"]) == pytest.approx(6.0)
    assert eng.available_balance_cad(cid) == pytest.approx(6.0)


def test_second_hold_fails_when_available_insufficient(cleanup):
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 5.0, description="seed")
    h1 = eng.create_wallet_hold(cid, 4.0, idempotency_key=f"a-{cid}")
    assert h1["held"] is True
    cleanup["holds"].append(h1["hold_id"])

    h2 = eng.create_wallet_hold(cid, 2.0, idempotency_key=f"b-{cid}")
    assert h2["held"] is False
    assert h2["reason"] == "insufficient_available"
    assert float(h2["available_cad"]) == pytest.approx(1.0)


def test_concurrent_dual_hold_race_only_one_succeeds(cleanup):
    """Two connections each try to hold 6 of 10 — only one may succeed."""
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 10.0, description="seed")

    results: list[dict] = []

    def _try_hold(tag: str) -> dict:
        local = BillingEngine()
        return local.create_wallet_hold(
            cid,
            6.0,
            idempotency_key=f"race-{tag}-{cid}",
            expires_in_sec=600,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_try_hold, "a")
        f2 = pool.submit(_try_hold, "b")
        results = [f1.result(timeout=15), f2.result(timeout=15)]

    held = [r for r in results if r.get("held")]
    failed = [r for r in results if not r.get("held")]
    assert len(held) == 1, f"expected exactly one hold success: {results}"
    assert len(failed) == 1
    assert failed[0].get("reason") == "insufficient_available"
    cleanup["holds"].append(held[0]["hold_id"])

    w = eng.get_wallet(cid)
    assert float(w["held_cad"]) == pytest.approx(6.0)
    assert float(w["available_cad"]) == pytest.approx(4.0)


def test_hold_idempotent_key_replays(cleanup):
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 20.0, description="seed")
    key = f"idem-{cid}"
    h1 = eng.create_wallet_hold(cid, 3.0, idempotency_key=key)
    h2 = eng.create_wallet_hold(cid, 3.0, idempotency_key=key)
    assert h1["held"] and h2["held"]
    assert h1["hold_id"] == h2["hold_id"]
    assert h2.get("idempotent_replay") is True
    cleanup["holds"].append(h1["hold_id"])

    w = eng.get_wallet(cid)
    assert float(w["held_cad"]) == pytest.approx(3.0)


def test_release_once_then_idempotent(cleanup):
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 8.0, description="seed")
    h = eng.create_wallet_hold(cid, 5.0, idempotency_key=f"rel-{cid}")
    hold_id = h["hold_id"]
    cleanup["holds"].append(hold_id)

    r1 = eng.release_wallet_hold(hold_id, reason="test")
    r2 = eng.release_wallet_hold(hold_id, reason="test_again")
    assert r1.get("released") is True
    assert r1.get("already_terminal") is not True
    assert r2.get("released") is True
    assert r2.get("already_terminal") is True

    w = eng.get_wallet(cid)
    assert float(w["held_cad"]) == pytest.approx(0.0)
    assert float(w["available_cad"]) == pytest.approx(8.0)
    assert float(w["balance_cad"]) == pytest.approx(8.0)


def test_link_hold_to_job_and_release_for_job(cleanup):
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 12.0, description="seed")
    h = eng.create_wallet_hold(cid, 2.5, idempotency_key=f"job-{cid}")
    hold_id = h["hold_id"]
    cleanup["holds"].append(hold_id)

    job_id = f"j-hold-{uuid.uuid4().hex[:8]}"
    now = time.time()
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload)
               VALUES (%s, 'queued', 0, %s, %s)""",
            (job_id, now, '{"owner":"%s"}' % cid),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)

    assert eng.link_wallet_hold_to_job(hold_id, job_id) is True
    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT wallet_hold_id::text FROM jobs WHERE job_id=%s", (job_id,)
        ).fetchone()
    linked = row[0] if not isinstance(row, dict) else row["wallet_hold_id"]
    assert str(linked) == hold_id

    rel = eng.release_wallet_hold_for_job(job_id, reason="job_cancelled")
    assert rel.get("released") is True
    # Second release is idempotent.
    rel2 = eng.release_wallet_hold_for_job(job_id, reason="job_cancelled")
    assert rel2.get("released") is True

    w = eng.get_wallet(cid)
    assert float(w["available_cad"]) == pytest.approx(12.0)


def test_wallet_preflight_creates_hold(cleanup, monkeypatch):
    """Shipped instance preflight entry creates a durable hold."""
    from billing import BillingEngine
    from routes import instances as inst

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 15.0, description="seed")

    hold_id = inst._wallet_preflight(
        cid,
        pricing_mode="on_demand",
        gpu_model="RTX 4090",
        num_gpus=1,
        idempotency_key=f"pref-{cid}",
    )
    assert hold_id
    cleanup["holds"].append(hold_id)

    w = eng.get_wallet(cid)
    assert float(w["held_cad"]) > 0
    assert float(w["available_cad"]) < float(w["balance_cad"])


def test_same_idempotency_key_dual_preflight_does_not_double_hold(cleanup):
    """Production-shaped: two preflights with the SAME key share one hold.

    With a unique per-submit key this only applies to true retries. With a
    shared key (bad), both would return the same hold_id and held_cad is
    one estimate — concurrent same-key must not create two holds.
    """
    from billing import BillingEngine
    from routes import instances as inst

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 10.0, description="seed")
    shared_key = f"shared-launch-{cid}"

    h1 = inst._wallet_preflight(
        cid, pricing_mode="on_demand", num_gpus=1, idempotency_key=shared_key
    )
    h2 = inst._wallet_preflight(
        cid, pricing_mode="on_demand", num_gpus=1, idempotency_key=shared_key
    )
    assert h1 and h2
    assert h1 == h2
    cleanup["holds"].append(h1)

    w = eng.get_wallet(cid)
    # One hold amount only (~1h default), not 2×.
    assert float(w["held_cad"]) == pytest.approx(
        eng.estimate_launch_hold_cad(pricing_mode="on_demand", num_gpus=1)
    )


def test_create_after_release_same_key_succeeds(cleanup):
    """After release, same idempotency key must mint a new hold (not UniqueViolation)."""
    from billing import BillingEngine

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 20.0, description="seed")
    key = f"relaunch-{cid}"

    h1 = eng.create_wallet_hold(cid, 3.0, idempotency_key=key)
    assert h1["held"] is True
    cleanup["holds"].append(h1["hold_id"])
    assert eng.release_wallet_hold(h1["hold_id"])["released"] is True

    h2 = eng.create_wallet_hold(cid, 3.0, idempotency_key=key)
    assert h2["held"] is True, h2
    assert h2["hold_id"] != h1["hold_id"]
    cleanup["holds"].append(h2["hold_id"])

    w = eng.get_wallet(cid)
    assert float(w["held_cad"]) == pytest.approx(3.0)


def test_preflight_fail_closed_no_soft_pass_on_hold_error(cleanup, monkeypatch):
    """create_wallet_hold exception must not soft-return None when available>0."""
    from billing import BillingEngine
    from routes import instances as inst
    import fastapi

    eng = BillingEngine()
    cid = _customer(cleanup)
    eng.deposit(cid, 50.0, description="seed")

    def _boom(*a, **k):
        raise RuntimeError("simulated hold failure")

    monkeypatch.setattr(BillingEngine, "create_wallet_hold", _boom)

    with pytest.raises(fastapi.HTTPException) as ei:
        inst._wallet_preflight(
            cid,
            pricing_mode="on_demand",
            num_gpus=1,
            idempotency_key=f"boom-{cid}",
        )
    assert ei.value.status_code == 402
    # Funds still fully available — no silent unguarded launch path.
    w = eng.get_wallet(cid)
    assert float(w["held_cad"]) == pytest.approx(0.0)
    assert float(w["available_cad"]) == pytest.approx(50.0)


def test_production_submit_key_is_unique_per_attempt():
    """api_submit_instance must use a per-attempt uuid in the hold key."""
    src = (Path(__file__).resolve().parents[1] / "routes" / "instances.py").read_text()
    assert 'f"launch:{customer_id}:{uuid.uuid4().hex}"' in src
    # Old deterministic shape must not be the production key.
    assert "j.name}:{j.pricing_mode}" not in src


def test_inventory_wallet_hold_sites():
    root = Path(__file__).resolve().parents[1]
    billing = (root / "billing.py").read_text()
    assert "def create_wallet_hold" in billing
    assert "def release_wallet_hold" in billing
    assert "def available_balance_cad" in billing
    assert "FOR UPDATE" in billing
    assert "status <> 'held'" in billing  # terminal prior frees key

    instances = (root / "routes" / "instances.py").read_text()
    assert "create_wallet_hold" in instances or "link_wallet_hold_to_job" in instances
    assert "_wallet_preflight" in instances
    assert "Unable to reserve wallet funds" in instances

    sched = (root / "scheduler.py").read_text()
    assert "release_wallet_hold_for_job" in sched

    mig = (root / "migrations" / "versions" / "063_wallet_holds.py").read_text()
    assert "CREATE TABLE IF NOT EXISTS wallet_holds" in mig
    assert "uq_wallet_holds_idempotency" in mig
