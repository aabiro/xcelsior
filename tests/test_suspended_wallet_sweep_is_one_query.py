"""The suspended-wallet sweep reads its work in one query, not one per wallet.

`stop_jobs_for_suspended_wallets` runs every 30 seconds. It used to select
every empty or suspended wallet and then scan `jobs` once per wallet on an
unindexed owner expression. Most empty wallets own nothing running, so the
cost grew with the number of empty accounts, not with the work to do.

The join must keep the same answer: running native jobs whose admitted owner
has an empty or suspended wallet, with the exact authority to fence.
"""

from __future__ import annotations

import uuid

from billing import get_billing_engine
from db import _get_pg_pool
from tests.fixtures.native_runtime import running_instance


class _CountingConnection:
    def __init__(self, inner, counter):
        self._inner = inner
        self._counter = counter

    def execute(self, *args, **kwargs):
        self._counter.append(args[0] if args else "")
        return self._inner.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name in ("_inner", "_counter"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)


class _CountingPool:
    def __init__(self, inner, counter):
        self._inner = inner
        self._counter = counter

    def connection(self):
        pool, counter = self._inner, self._counter

        class _Ctx:
            def __enter__(self):
                self._cm = pool.connection()
                return _CountingConnection(self._cm.__enter__(), counter)

            def __exit__(self, *exc):
                return self._cm.__exit__(*exc)

        return _Ctx()


def test_the_sweep_is_one_query_however_many_wallets_are_empty(monkeypatch):
    import db

    engine = get_billing_engine()
    empty = [f"sweep-empty-{uuid.uuid4().hex[:10]}" for _ in range(5)]
    for cid in empty:
        engine.get_wallet(cid)  # a zero-balance wallet with nothing running
    owner = f"sweep-owner-{uuid.uuid4().hex[:10]}"
    engine.get_wallet(owner)
    stopped: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        engine,
        "stop_instance",
        lambda job_id, *, reason, expected_authority: (
            stopped.append((job_id, expected_authority)) or {"stopped": True}
        ),
    )
    queries: list[str] = []
    real_pool = _get_pg_pool()
    try:
        with running_instance(owner=owner, cost_per_hour=1.0) as native:
            monkeypatch.setattr(db, "_get_pg_pool", lambda: _CountingPool(real_pool, queries))
            assert engine.stop_jobs_for_suspended_wallets() >= 1
            monkeypatch.setattr(db, "_get_pg_pool", lambda: real_pool)
            assert len(queries) == 1, (
                f"the sweep issued {len(queries)} queries; it must not scan jobs once "
                "per empty wallet"
            )
            mine = [authority for job_id, authority in stopped if job_id == native["job_id"]]
            assert mine == [
                {
                    "attempt_id": str(native["authority"]["attempt_id"]),
                    "generation": mine[0]["generation"],
                    "owner": owner,
                }
            ]
            assert isinstance(mine[0]["generation"], int)
            assert not any(job_id.startswith("sweep-empty-") for job_id, _ in stopped)
    finally:
        with real_pool.connection() as conn:
            for cid in [*empty, owner]:
                conn.execute("DELETE FROM wallets WHERE customer_id=%s", (cid,))


def test_a_funded_active_wallet_is_not_swept(monkeypatch):
    engine = get_billing_engine()
    owner = f"sweep-funded-{uuid.uuid4().hex[:10]}"
    engine.deposit(owner, 10)
    stopped: list[str] = []
    monkeypatch.setattr(
        engine,
        "stop_instance",
        lambda job_id, *, reason, expected_authority: (stopped.append(job_id) or {"stopped": True}),
    )
    try:
        with running_instance(owner=owner, cost_per_hour=1.0) as native:
            engine.stop_jobs_for_suspended_wallets()
            assert native["job_id"] not in stopped
            with _get_pg_pool().connection() as conn:
                conn.execute("UPDATE wallets SET status='suspended' WHERE customer_id=%s", (owner,))
            engine.stop_jobs_for_suspended_wallets()
            assert native["job_id"] in stopped, "a suspended wallet with a balance is swept"
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM wallet_transactions WHERE customer_id=%s", (owner,))
            conn.execute("DELETE FROM wallets WHERE customer_id=%s", (owner,))
