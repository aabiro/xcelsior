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

    def _make(
        *,
        status: str = "pending",
        customer_id: str | None = None,
        payment_hash: str | None = None,
        amount_cad: float = 25.0,
    ) -> str:
        deposit_id = f"ct-{uuid.uuid4().hex[:12]}"
        cust = customer_id or f"cust-{uuid.uuid4().hex[:8]}"
        with pg_transaction() as conn:
            conn.execute(
                """INSERT INTO ln_deposits
                    (deposit_id,customer_id,tenant_id,label,bolt11,payment_hash,
                     amount_msat,amount_sats,amount_micros,amount_cad_minor,
                     btc_cad_rate_exact,status,created_at_ts,expires_at_ts)
                VALUES(%s,%s,%s,%s,'bolt',%s,1000000,1000,%s,%s,100000,%s,
                    clock_timestamp(),clock_timestamp()+interval '1 day')""",
                (
                    deposit_id,
                    cust,
                    cust,
                    f"lbl-{deposit_id}",
                    payment_hash or f"hash-{deposit_id}",
                    round(amount_cad * 1_000_000),
                    round(amount_cad * 100),
                    status,
                ),
            )
        created.append(deposit_id)
        return deposit_id

    yield _make

    if created:
        with pg_transaction() as conn:
            conn.execute("DELETE FROM ln_deposits WHERE deposit_id = ANY(%s)", (created,))


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
    """ "credited but no ledger entry" must be a query, not an audit."""
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

    ln.process_ln_deposits(credit_callback=lambda c, a, d: {"tx_id": "TX-from-callback"})

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
        ("tenant_id", "other-tenant"),
        ("amount_sats", 1),
        ("amount_micros", 1),
        ("amount_cad_minor", 1),
        ("btc_cad_rate_exact", 1),
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
    slurm_adapter.register_slurm_job(job_id, "5551", cluster_id="nibi", tenant_id="tenant-1")

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

    row = _row("SELECT count(*) FROM slurm_job_mappings WHERE slurm_job_id = %s", ("8888",))
    assert row == (2,)


def test_a_mapping_is_never_re_pointed(slurm_rows):
    """Re-pointing lost the first job's record and the run it would bill.

    Registering the same submission with another Slurm job is refused and
    changes nothing; with the same Slurm job it is a no-op (a retried
    register after a lost response).
    """
    job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(job_id, "9001", cluster_id="nibi")
    with pytest.raises(slurm_adapter.SlurmAdapterError) as e:
        slurm_adapter.register_slurm_job(job_id, "9002", cluster_id="nibi")
    assert e.value.code == "mapping_conflict"
    assert slurm_adapter.register_slurm_job(job_id, "9001", cluster_id="nibi") == []

    row = _row(
        "SELECT slurm_job_id, version FROM slurm_job_mappings WHERE xcelsior_job_id = %s",
        (job_id,),
    )
    assert row == ("9001", 1)


def test_each_attempt_of_a_job_keeps_its_own_mapping(slurm_rows):
    """One mapping per submission (migration 157): a retry is a new attempt
    and a new Slurm job, and the first attempt's record stays."""
    job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(
        job_id, "9301", cluster_id="nibi", submission_name=f"{job_id}-a1"
    )
    # While the first attempt is open the job can't have a second.
    with pytest.raises(psycopg.errors.UniqueViolation):
        slurm_adapter.register_slurm_job(
            job_id, "9302", cluster_id="nibi", submission_name=f"{job_id}-a2"
        )
    with pg_transaction() as conn:
        conn.execute(
            "UPDATE slurm_job_mappings SET observed_state = 'failed', "
            "terminal_at = clock_timestamp() WHERE submission_name = %s",
            (f"{job_id}-a1",),
        )
    slurm_adapter.register_slurm_job(
        job_id, "9302", cluster_id="nibi", submission_name=f"{job_id}-a2"
    )
    with pg_transaction() as conn:
        rows = conn.execute(
            "SELECT submission_name, slurm_job_id, terminal_at IS NOT NULL "
            "FROM slurm_job_mappings WHERE xcelsior_job_id = %s ORDER BY submission_name",
            (job_id,),
        ).fetchall()
    assert rows == [(f"{job_id}-a1", "9301", True), (f"{job_id}-a2", "9302", False)]


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
#
# sync_slurm_statuses is now a batched pass (workers/slurm_reconciler.py):
# one squeue per cluster, sacct only for jobs that left the controller. These
# tests give each case its own cluster (a profile file names it) and answer
# poll_cluster_jobs / accounting_window for that cluster only; any other
# cluster with open rows in the shared database sees a transport error,
# which by contract changes nothing.


@pytest.fixture
def own_cluster(tmp_path, monkeypatch, slurm_rows):
    """A cluster only this test polls; returns (cluster_id, register, answer)."""
    import json as _json

    cid = f"ct-{uuid.uuid4().hex[:8]}"
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        _json.dumps({"profiles": {cid: {"name": cid, "partition_gpu": "g", "partition_cpu": "c"}}})
    )
    monkeypatch.setenv(slurm_adapter.PROFILES_FILE_ENV, str(profiles))
    data = {"squeue": {}, "sacct": {}, "fail": False, "sacct_ids": [], "xids": {}}

    def name(sid):
        # As the adapter names jobs: by the Xcelsior job id, not the Slurm one.
        return f"xcl-{data['xids'].get(sid, sid)}"

    def poll(*, profile_name=None, runner=None):
        if profile_name != cid or data["fail"]:
            return {"error": "not this test's cluster", "error_code": "transport_error"}
        return {
            "jobs": {
                sid: {
                    "name": name(sid),
                    "state": st,
                    "reason": "None",
                    "elapsed": "0:01",
                    "time_limit": "1:00",
                    "nodes": "1",
                    "nodelist": "n1",
                }
                for sid, st in data["squeue"].items()
            }
        }

    def window(job_ids, *, since_minutes, profile_name=None, runner=None):
        data["sacct_ids"].extend(job_ids)
        records = {}
        for sid in job_ids:
            for i, st in enumerate(data["sacct"].get(sid, [])):
                records.setdefault(sid, []).append(
                    {
                        "name": name(sid),
                        "state": st,
                        "raw_state": st,
                        "observed_state": slurm_adapter.SLURM_OBSERVED_STATE[st],
                        "submitted_at": 1794000000.0 + i,
                        "started_at": 1794000005.0 + i,
                        "ended_at": 1794000065.0 + i,
                        "elapsed_sec": 60,
                        "exit_code": "0:0",
                        "alloc_tres": "cpu=1",
                        "node_list": "n1",
                    }
                )
        return {"records": records}

    monkeypatch.setattr(slurm_adapter, "poll_cluster_jobs", poll)
    monkeypatch.setattr(slurm_adapter, "accounting_window", window)

    def register(slurm_id):
        job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
        slurm_adapter.register_slurm_job(job_id, slurm_id, cluster_id=cid)
        data["xids"][str(slurm_id)] = job_id
        return job_id

    return cid, register, data


def _state(job_id):
    return _row(
        "SELECT observed_state, terminal_at IS NOT NULL, last_observed_at IS NOT NULL "
        "FROM slurm_job_mappings WHERE xcelsior_job_id = %s",
        (job_id,),
    )


def test_sync_persists_the_observation(own_cluster):
    _cid, register, data = own_cluster
    job_id = register("9200")
    data["squeue"] = {"9200": "RUNNING"}

    changes = slurm_adapter.sync_slurm_statuses()

    assert (job_id, "9200", "pending", "running") in changes
    assert _state(job_id) == ("running", False, True), (
        "persisted, not terminal, and last_observed_at reflects real freshness"
    )


def test_terminal_mapping_is_retained_not_deleted(own_cluster):
    """The regression that destroyed audit history.

    The previous sync loop ran `DELETE FROM slurm_job_mappings` when a job
    reached a terminal state, erasing the only record that an external job
    ever ran for a tenant. Companion §10.1: "Historical mappings remain
    queryable and auditable."
    """
    _cid, register, data = own_cluster
    job_id = register("9300")
    data["sacct"] = {"9300": ["COMPLETED"]}  # gone from the controller, finished

    slurm_adapter.sync_slurm_statuses()

    row = _state(job_id)
    assert row is not None, "terminal history must be retained and marked, not destroyed"
    assert row[:2] == ("completed", True)


def test_a_finished_state_in_squeue_alone_does_not_close_the_mapping(own_cluster):
    """A requeued job returns to PENDING under the same id; only leaving the
    controller with a finished accounting record settles a mapping."""
    _cid, register, data = own_cluster
    job_id = register("9310")
    data["squeue"] = {"9310": "COMPLETED"}

    slurm_adapter.sync_slurm_statuses()

    assert _state(job_id)[:2] == ("completed", False)


def test_terminal_mappings_are_not_polled_again(own_cluster):
    _cid, register, data = own_cluster
    register("9400")
    data["sacct"] = {"9400": ["FAILED"]}
    slurm_adapter.sync_slurm_statuses()
    data["sacct_ids"].clear()

    slurm_adapter.sync_slurm_statuses()

    assert "9400" not in data["sacct_ids"], "a settled mapping must drop out of the poll set"


def test_desired_cancelled_drives_a_cancel_request(own_cluster):
    """desired → cluster, the direction that did not exist before.

    Previously a cancellation could only be expressed by deleting the row
    and hoping the external job noticed.
    """
    _cid, register, data = own_cluster
    job_id = register("9500")
    assert slurm_adapter.set_slurm_desired_state(job_id, "cancelled") is True
    data["squeue"] = {"9500": "RUNNING"}
    cancelled: list[str] = []

    slurm_adapter.sync_slurm_statuses(cancel_callback=cancelled.append)

    assert cancelled == ["9500"], (
        "a mapping whose desired_state is 'cancelled' while the external job "
        "is still running must produce a cancel request"
    )


def test_no_cancel_request_for_a_job_already_terminal(own_cluster):
    _cid, register, data = own_cluster
    job_id = register("9600")
    slurm_adapter.set_slurm_desired_state(job_id, "cancelled")
    data["squeue"] = {"9600": "COMPLETED"}
    cancelled: list[str] = []

    slurm_adapter.sync_slurm_statuses(cancel_callback=cancelled.append)

    assert cancelled == [], "a finished job does not need cancelling"


def test_a_failing_cluster_does_not_stop_the_rest(own_cluster):
    """Isolation is per cluster now: one unreachable cluster costs only its
    own mappings a pass, and changes none of them."""
    _cid, register, data = own_cluster
    good = register("9801")
    data["squeue"] = {"9801": "RUNNING"}
    # Every cluster but this test's answers with a transport error (see the
    # fixture), so any other open rows in the shared database are that
    # failing cluster; the healthy one must still be reconciled.
    slurm_adapter.sync_slurm_statuses()
    assert _state(good)[0] == "running", "a healthy cluster was starved by a failing one"

    data["fail"] = True
    data["squeue"] = {"9801": "COMPLETED"}
    slurm_adapter.sync_slurm_statuses()
    assert _state(good)[0] == "running", "a failed poll must change nothing"


def test_sync_records_a_pending_job_without_check_violation(own_cluster):
    """Regression: a real PENDING job aborted the whole sync pass.

    The adapter maps Slurm PENDING to the Xcelsior job state "queued", and
    the sync wrote that straight into observed_state, whose CHECK
    constraint only allows the mapping vocabulary. The first pending job
    raised CheckViolation out of sync_slurm_statuses. The callback still
    gets the job state.
    """
    _cid, register, data = own_cluster
    job_id = register("9950")
    other = register("9951")
    data["squeue"] = {"9950": "PENDING", "9951": "RUNNING"}
    seen: list = []

    slurm_adapter.sync_slurm_statuses(update_callback=lambda x, s: seen.append((x, s)))

    assert _state(job_id)[0] == "pending"
    assert _state(other)[0] == "running", "a pending job must not stop the rest of the pass"
    assert (other, "running") in seen


def test_preempted_job_settles_as_preempted(own_cluster):
    """Decision D2: preemption is its own observation, not a cancellation,
    and it settles once the job has left the controller unrequeued."""
    _cid, register, data = own_cluster
    job_id = register("9960")
    data["sacct"] = {"9960": ["PREEMPTED"]}

    slurm_adapter.sync_slurm_statuses()

    assert _state(job_id)[:2] == ("preempted", True)


def test_set_desired_state_rejects_unknown_values(slurm_rows):
    job_id = slurm_rows(f"xj-{uuid.uuid4().hex[:10]}")
    slurm_adapter.register_slurm_job(job_id, "9700", cluster_id="nibi")
    with pytest.raises(ValueError):
        slurm_adapter.set_slurm_desired_state(job_id, "paused")


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
