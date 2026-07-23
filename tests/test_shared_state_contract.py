"""Shared-state contracts for Lightning and Slurm (Track B B9.3b-2 / B9.3c).

Migration `060_shared_state_to_pg` moved these two out of SQLite and a JSON
file but kept their file-era shapes: no tenant, no currency, nothing
stopping two rows from claiming one payment or one external Slurm job, and
a sync loop that *deleted* a mapping the moment it finished — destroying
the only record that an external job ever ran.

Migration `067` supplies the structural contract; `lightning.py` and
`slurm_adapter.py` supply the behavioural half. Every test here drives real
PostgreSQL.

Companion §10.1; Track B §B9.3.
"""

from __future__ import annotations

import time
import uuid

import psycopg
import pytest

import slurm_adapter
from db import pg_transaction

# ─────────────────────────── helpers ───────────────────────────


@pytest.fixture
def ln_rows():
    created: list[str] = []

    def _make(*, status: str = "pending", customer_id: str | None = None,
              payment_hash: str | None = None, amount_cad: float = 25.0) -> str:
        deposit_id = f"ct-{uuid.uuid4().hex[:12]}"
        cust = customer_id or f"cust-{uuid.uuid4().hex[:8]}"
        with pg_transaction() as conn:
            conn.execute(
                """INSERT INTO ln_deposits
                   (deposit_id, customer_id, tenant_id, label, bolt11,
                    payment_hash, amount_msat, amount_sats, amount_btc,
                    amount_cad, btc_cad_rate, status, created_at, expires_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    deposit_id, cust, cust, f"lbl-{deposit_id}", "bolt",
                    payment_hash or f"hash-{deposit_id}",
                    1_000_000, 1000, 0.00001, amount_cad, 100_000.0, status,
                    time.time(), time.time() + 3600,
                ),
            )
        created.append(deposit_id)
        return deposit_id

    yield _make

    if created:
        with pg_transaction() as conn:
            conn.execute(
                "DELETE FROM ln_deposits WHERE deposit_id = ANY(%s)", (created,)
            )


@pytest.fixture
def slurm_rows():
    created: list[str] = []

    def _track(job_id: str) -> str:
        created.append(job_id)
        return job_id

    yield _track

    if created:
        with pg_transaction() as conn:
            conn.execute(
                "DELETE FROM slurm_job_mappings WHERE xcelsior_job_id = ANY(%s)",
                (created,),
            )


def _row(sql: str, params: tuple):
    with pg_transaction() as conn:
        return conn.execute(sql, params).fetchone()


# ───────────── ln_deposits: tenancy, currency, ledger link ─────────────


def test_deposit_carries_tenant_and_currency(ln_rows):
    """An amount without a currency is not money (companion §10.1)."""
    deposit_id = ln_rows()

    row = _row(
        "SELECT tenant_id, currency FROM ln_deposits WHERE deposit_id = %s",
        (deposit_id,),
    )
    assert row is not None
    tenant_id, currency = row
    assert tenant_id, "every deposit must name the tenant that is paying"
    assert currency == "CAD"


def test_credited_deposit_links_its_ledger_entry(ln_rows):
    """"credited but no ledger entry" must be a query, not an audit."""
    import lightning as ln

    deposit_id = ln_rows(status="paid")

    assert ln.mark_credited(deposit_id, wallet_ledger_entry_id="TX-abc123") is True

    row = _row(
        "SELECT status, wallet_ledger_entry_id FROM ln_deposits WHERE deposit_id = %s",
        (deposit_id,),
    )
    assert row == ("credited", "TX-abc123")


def test_sweep_records_the_ledger_entry_the_callback_wrote(ln_rows, monkeypatch):
    """The credit callback's return value is threaded into the deposit."""
    import lightning as ln

    monkeypatch.setattr(ln, "LN_ENABLED", True)
    deposit_id = ln_rows(status="paid")

    ln.process_ln_deposits(
        credit_callback=lambda c, a, d: {"tx_id": "TX-from-callback"}
    )

    row = _row(
        "SELECT status, wallet_ledger_entry_id FROM ln_deposits WHERE deposit_id = %s",
        (deposit_id,),
    )
    assert row == ("credited", "TX-from-callback")


def test_callback_returning_none_still_credits(ln_rows, monkeypatch):
    """Not every caller can return a ledger row; that must not break."""
    import lightning as ln

    monkeypatch.setattr(ln, "LN_ENABLED", True)
    deposit_id = ln_rows(status="paid")

    ln.process_ln_deposits(credit_callback=lambda c, a, d: None)

    row = _row(
        "SELECT status, wallet_ledger_entry_id FROM ln_deposits WHERE deposit_id = %s",
        (deposit_id,),
    )
    assert row is not None
    assert row[0] == "credited"
    assert row[1] is None


# ───────────── ln_deposits: uniqueness and immutability ─────────────


def test_one_payment_hash_cannot_fund_two_deposits(ln_rows):
    """Two rows claiming one provider payment is a double credit."""
    shared_hash = f"hash-shared-{uuid.uuid4().hex[:8]}"
    ln_rows(payment_hash=shared_hash)

    with pytest.raises(psycopg.errors.UniqueViolation):
        ln_rows(payment_hash=shared_hash)


def test_status_check_rejects_an_invented_state(ln_rows):
    deposit_id = ln_rows()
    with pytest.raises(psycopg.errors.CheckViolation):
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE ln_deposits SET status = 'teleporting' WHERE deposit_id = %s",
                (deposit_id,),
            )


@pytest.mark.parametrize(
    "column,value",
    [
        ("customer_id", "someone-else"),
        ("amount_msat", 1),
        ("payment_hash", "different-hash"),
        ("currency", "USD"),
    ],
)
def test_payment_terms_are_immutable(ln_rows, column, value):
    """The terms a customer paid against cannot be edited afterwards.

    Companion §10.1 calls for an "immutable expected amount/currency and
    expiry" — an UPDATE that changes them rewrites history under a settled
    payment, and there is no legitimate reason to do it.
    """
    deposit_id = ln_rows(status="paid")

    with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
        with pg_transaction() as conn:
            conn.execute(
                f"UPDATE ln_deposits SET {column} = %s WHERE deposit_id = %s",
                (value, deposit_id),
            )


def test_lifecycle_transitions_are_still_allowed(ln_rows):
    """Immutability must not freeze the whole row — status still moves."""
    import lightning as ln

    deposit_id = ln_rows(status="paid")
    assert ln.mark_credited(deposit_id) is True
    row = _row("SELECT status FROM ln_deposits WHERE deposit_id = %s", (deposit_id,))
    assert row == ("credited",)


# ───────────── slurm_job_mappings: identity and uniqueness ─────────────


def test_registered_mapping_carries_the_contract_columns(slurm_rows):
    job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(
        job_id, "5551", cluster_id="nibi", tenant_id="tenant-1"
    )

    row = _row(
        "SELECT cluster_id, tenant_id, desired_state, observed_state, "
        "submit_idempotency_key, version, submitted_at IS NOT NULL, "
        "terminal_at IS NULL FROM slurm_job_mappings WHERE xcelsior_job_id = %s",
        (job_id,),
    )
    assert row is not None
    cluster, tenant, desired, observed, key, version, submitted, open_ = row
    assert cluster == "nibi"
    assert tenant == "tenant-1"
    assert desired == "running"
    assert observed == "pending"
    assert key, "a submit must carry an idempotency key"
    assert version == 1
    assert submitted is True
    assert open_ is True


def test_one_slurm_job_cannot_serve_two_xcelsior_jobs(slurm_rows):
    """Both would act on its status, and both would bill for it."""
    first = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    second = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(first, "7777", cluster_id="nibi")

    with pytest.raises(psycopg.errors.UniqueViolation):
        slurm_adapter.register_slurm_job(second, "7777", cluster_id="nibi")


def test_same_job_id_on_a_different_cluster_is_distinct(slurm_rows):
    """Slurm ids are only unique within a cluster."""
    first = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    second = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(first, "8888", cluster_id="nibi")
    slurm_adapter.register_slurm_job(second, "8888", cluster_id="graham")

    row = _row(
        "SELECT count(*) FROM slurm_job_mappings WHERE slurm_job_id = %s", ("8888",)
    )
    assert row == (2,)


def test_re_registering_bumps_version(slurm_rows):
    job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(job_id, "9001", cluster_id="nibi")
    slurm_adapter.register_slurm_job(job_id, "9002", cluster_id="nibi")

    row = _row(
        "SELECT slurm_job_id, version FROM slurm_job_mappings "
        "WHERE xcelsior_job_id = %s",
        (job_id,),
    )
    assert row == ("9002", 2)


def test_observed_state_check_rejects_an_invented_state(slurm_rows):
    job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(job_id, "9100", cluster_id="nibi")

    with pytest.raises(psycopg.errors.CheckViolation):
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_job_mappings SET observed_state = 'vibing' "
                "WHERE xcelsior_job_id = %s",
                (job_id,),
            )


# ───────────── slurm: bidirectional reconcile ─────────────


def _fake_status(state: str):
    def _get(slurm_job_id):
        return {"slurm_job_id": slurm_job_id, "xcelsior_state": state}

    return _get


def test_sync_persists_the_observation(slurm_rows, monkeypatch):
    job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(job_id, "9200", cluster_id="nibi")
    monkeypatch.setattr(slurm_adapter, "get_slurm_job_status", _fake_status("running"))

    changes = slurm_adapter.sync_slurm_statuses()

    assert (job_id, "9200", "pending", "running") in changes
    row = _row(
        "SELECT observed_state, last_observed_at IS NOT NULL, terminal_at "
        "FROM slurm_job_mappings WHERE xcelsior_job_id = %s",
        (job_id,),
    )
    assert row is not None
    assert row[0] == "running"
    assert row[1] is True, "last_observed_at must reflect real freshness"
    assert row[2] is None, "a running job is not terminal"


def test_terminal_mapping_is_retained_not_deleted(slurm_rows, monkeypatch):
    """The regression that destroyed audit history.

    The previous sync loop ran `DELETE FROM slurm_job_mappings` when a job
    reached a terminal state, erasing the only record that an external job
    ever ran for a tenant. Companion §10.1: "Historical mappings remain
    queryable and auditable."
    """
    job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(job_id, "9300", cluster_id="nibi")
    monkeypatch.setattr(
        slurm_adapter, "get_slurm_job_status", _fake_status("completed")
    )

    slurm_adapter.sync_slurm_statuses()

    row = _row(
        "SELECT observed_state, terminal_at IS NOT NULL FROM slurm_job_mappings "
        "WHERE xcelsior_job_id = %s",
        (job_id,),
    )
    assert row is not None, (
        "the mapping was deleted on completion; terminal history must be "
        "retained and marked, not destroyed"
    )
    assert row[0] == "completed"
    assert row[1] is True


def test_terminal_mappings_are_not_polled_again(slurm_rows, monkeypatch):
    job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(job_id, "9400", cluster_id="nibi")
    monkeypatch.setattr(slurm_adapter, "get_slurm_job_status", _fake_status("failed"))
    slurm_adapter.sync_slurm_statuses()

    polled: list[str] = []

    def _tracking(slurm_job_id):
        polled.append(slurm_job_id)
        return {"xcelsior_state": "failed"}

    monkeypatch.setattr(slurm_adapter, "get_slurm_job_status", _tracking)
    slurm_adapter.sync_slurm_statuses()

    assert "9400" not in polled, "a settled mapping must drop out of the poll set"


def test_desired_cancelled_drives_a_cancel_request(slurm_rows, monkeypatch):
    """desired → cluster, the direction that did not exist before.

    Previously a cancellation could only be expressed by deleting the row
    and hoping the external job noticed.
    """
    job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(job_id, "9500", cluster_id="nibi")
    assert slurm_adapter.set_slurm_desired_state(job_id, "cancelled") is True

    monkeypatch.setattr(slurm_adapter, "get_slurm_job_status", _fake_status("running"))
    cancelled: list[str] = []

    slurm_adapter.sync_slurm_statuses(cancel_callback=cancelled.append)

    assert cancelled == ["9500"], (
        "a mapping whose desired_state is 'cancelled' while the external job "
        "is still running must produce a cancel request"
    )


def test_no_cancel_request_for_a_job_already_terminal(slurm_rows, monkeypatch):
    job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(job_id, "9600", cluster_id="nibi")
    slurm_adapter.set_slurm_desired_state(job_id, "cancelled")
    monkeypatch.setattr(
        slurm_adapter, "get_slurm_job_status", _fake_status("completed")
    )
    cancelled: list[str] = []

    slurm_adapter.sync_slurm_statuses(cancel_callback=cancelled.append)

    assert cancelled == [], "a finished job does not need cancelling"


def test_set_desired_state_rejects_unknown_values(slurm_rows):
    job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(job_id, "9700", cluster_id="nibi")
    with pytest.raises(ValueError):
        slurm_adapter.set_slurm_desired_state(job_id, "paused")


def test_poll_failure_on_one_mapping_does_not_stop_the_rest(
    slurm_rows, monkeypatch
):
    """Per-mapping isolation, same rule as the Lightning sweep."""
    bad = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    good = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(bad, "9800", cluster_id="nibi")
    slurm_adapter.register_slurm_job(good, "9801", cluster_id="nibi")

    def _flaky(slurm_job_id):
        if slurm_job_id == "9800":
            raise RuntimeError("cluster unreachable")
        return {"xcelsior_state": "running"}

    monkeypatch.setattr(slurm_adapter, "get_slurm_job_status", _flaky)
    slurm_adapter.sync_slurm_statuses()

    row = _row(
        "SELECT observed_state FROM slurm_job_mappings WHERE xcelsior_job_id = %s",
        (good,),
    )
    assert row == ("running",), "a healthy mapping was starved by a failing one"


def test_load_map_excludes_terminal_and_survives_db_error(slurm_rows, monkeypatch):
    """The in-memory map is a snapshot of *open* work, never authority."""
    open_job = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    done_job = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(open_job, "9900", cluster_id="nibi")
    slurm_adapter.register_slurm_job(done_job, "9901", cluster_id="nibi")
    with pg_transaction() as conn:
        conn.execute(
            "UPDATE slurm_job_mappings SET terminal_at = clock_timestamp(), "
            "observed_state = 'completed' WHERE xcelsior_job_id = %s",
            (done_job,),
        )

    loaded = slurm_adapter._load_slurm_map()
    assert open_job in loaded
    assert done_job not in loaded, "settled mappings are history, not work"


def test_no_json_file_authority_remains():
    """Companion §10.1: a JSON file cannot coordinate several processes."""
    assert not hasattr(slurm_adapter, "SLURM_MAP_FILE"), (
        "SLURM_MAP_FILE is dead file-authority configuration; the mapping is "
        "owned by PostgreSQL (migration 060)"
    )
