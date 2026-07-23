"""Lightning deposit reconciliation (Track B B9.3b-3, companion §10.1).

The four ways a deposit and the wallet can disagree, each of which is
somebody's money. Every test drives the real sweep against real
PostgreSQL.

`paid_not_credited` deserves special note: it is the exact condition the
pre-2026-07-22 Lightning module produced on *every* deposit, and it is
invisible without a check like this one because nothing errors — the row
just sits in `paid` forever while the customer waits.
"""

from __future__ import annotations

import time
import uuid

import pytest

from control_plane.billing_reconcile import (
    FINDING_AMOUNT_MISMATCH,
    FINDING_CREDITED_WITHOUT_LEDGER,
    FINDING_PAID_NOT_CREDITED,
    FINDING_STUCK_PENDING,
    RESOURCE_TYPE,
    reconcile_ln_deposits,
)
from db import pg_transaction


@pytest.fixture
def deposit():
    """Seed a deposit with precise control over its typed timestamps."""
    created: list[str] = []

    def _make(
        *,
        status: str = "pending",
        amount_cad: float = 25.0,
        amount_minor: int | None = None,
        ledger_entry_id: str | None = None,
        paid_age_sec: float | None = None,
        expiry_age_sec: float | None = None,
    ) -> str:
        deposit_id = f"rec-{uuid.uuid4().hex[:12]}"
        cust = f"cust-{uuid.uuid4().hex[:8]}"
        now = time.time()
        expires_at = now - expiry_age_sec if expiry_age_sec else now + 3600
        with pg_transaction() as conn:
            conn.execute(
                """INSERT INTO ln_deposits
                   (deposit_id, customer_id, tenant_id, label, bolt11,
                    payment_hash, amount_msat, amount_sats, amount_btc,
                    amount_cad, amount_cad_minor, btc_cad_rate, status,
                    wallet_ledger_entry_id, created_at, expires_at,
                    expires_at_ts)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, to_timestamp(%s))""",
                (
                    deposit_id, cust, cust, f"lbl-{deposit_id}", "bolt",
                    f"hash-{deposit_id}", 1_000_000, 1000, 0.00001,
                    amount_cad,
                    amount_minor if amount_minor is not None
                    else int(round(amount_cad * 100)),
                    100_000.0, status, ledger_entry_id, now, expires_at,
                    expires_at,
                ),
            )
            if paid_age_sec is not None:
                conn.execute(
                    "UPDATE ln_deposits SET paid_at_ts = "
                    "clock_timestamp() - make_interval(secs => %s) "
                    "WHERE deposit_id = %s",
                    (paid_age_sec, deposit_id),
                )
        created.append(deposit_id)
        return deposit_id

    yield _make

    with pg_transaction() as conn:
        if created:
            conn.execute(
                "DELETE FROM reconciliation_findings WHERE resource_type = %s "
                "AND resource_id = ANY(%s)",
                (RESOURCE_TYPE, created),
            )
            conn.execute(
                "DELETE FROM ln_deposits WHERE deposit_id = ANY(%s)", (created,)
            )


def _findings(deposit_id: str) -> list[tuple[str, str, str | None]]:
    with pg_transaction() as conn:
        rows = conn.execute(
            "SELECT finding_type, severity, resolved_at::text "
            "FROM reconciliation_findings "
            "WHERE resource_type = %s AND resource_id = %s",
            (RESOURCE_TYPE, deposit_id),
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def _run() -> None:
    with pg_transaction() as conn:
        reconcile_ln_deposits(conn)


# ───────────────────── the four conditions ─────────────────────


def test_paid_but_not_credited_is_an_error_finding(deposit):
    """A customer paid and the wallet never moved."""
    deposit_id = deposit(status="paid", paid_age_sec=1800)

    _run()

    found = _findings(deposit_id)
    assert (FINDING_PAID_NOT_CREDITED, "error", None) in found, (
        f"expected an open paid_not_credited finding, got {found}"
    )


def test_recently_paid_deposit_is_not_yet_a_finding(deposit):
    """The watcher runs every 5s; do not alarm on work in flight."""
    deposit_id = deposit(status="paid", paid_age_sec=5)

    _run()

    assert _findings(deposit_id) == []


def test_credited_without_a_ledger_entry_is_flagged(deposit):
    deposit_id = deposit(status="credited", ledger_entry_id=None)

    _run()

    types = [f[0] for f in _findings(deposit_id)]
    assert FINDING_CREDITED_WITHOUT_LEDGER in types


def test_credited_with_a_ledger_entry_is_clean(deposit):
    deposit_id = deposit(status="credited", ledger_entry_id="TX-1")

    _run()

    assert _findings(deposit_id) == []


def test_pending_past_expiry_is_flagged(deposit):
    deposit_id = deposit(status="pending", expiry_age_sec=7200)

    _run()

    types = [f[0] for f in _findings(deposit_id)]
    assert FINDING_STUCK_PENDING in types


def test_pending_before_expiry_is_clean(deposit):
    deposit_id = deposit(status="pending")

    _run()

    assert _findings(deposit_id) == []


def test_amount_mismatch_between_representations_is_flagged(deposit):
    """The exact and legacy columns describe one payment; they must agree."""
    deposit_id = deposit(status="pending", amount_cad=25.0, amount_minor=9999)

    _run()

    types = [f[0] for f in _findings(deposit_id)]
    assert FINDING_AMOUNT_MISMATCH in types


def test_sub_half_cent_difference_is_not_a_mismatch(deposit):
    """Float representation noise is not a discrepancy."""
    deposit_id = deposit(
        status="pending", amount_cad=19.989999999999998, amount_minor=1999
    )

    _run()

    assert _findings(deposit_id) == []


# ───────────────────── sweep behaviour ─────────────────────


def test_findings_are_deduplicated_across_sweeps(deposit):
    """A five-minute sweep must not produce a finding every five minutes."""
    deposit_id = deposit(status="paid", paid_age_sec=1800)

    for _ in range(3):
        _run()

    open_findings = [f for f in _findings(deposit_id) if f[2] is None]
    assert len(open_findings) == 1, (
        f"expected one open finding after three sweeps, got {open_findings}"
    )


def test_finding_auto_resolves_once_the_condition_clears(deposit):
    """Otherwise the operator view fills with solved problems."""
    deposit_id = deposit(status="paid", paid_age_sec=1800)
    _run()
    assert any(f[2] is None for f in _findings(deposit_id))

    with pg_transaction() as conn:
        conn.execute(
            "UPDATE ln_deposits SET status = 'credited', "
            "wallet_ledger_entry_id = 'TX-late' WHERE deposit_id = %s",
            (deposit_id,),
        )

    _run()

    assert all(f[2] is not None for f in _findings(deposit_id)), (
        "the finding should be resolved once the deposit was credited"
    )


def test_reconciler_never_moves_money(deposit):
    """Report-only (B0.3 rule 17, `DA§8.7`).

    Crediting from a reconciler would be a second, untested money path
    competing with process_ln_deposits. The sweep makes a discrepancy
    visible; a human decides.
    """
    deposit_id = deposit(status="paid", paid_age_sec=1800)

    with pg_transaction() as conn:
        before = conn.execute(
            "SELECT status, wallet_ledger_entry_id, amount_cad_minor "
            "FROM ln_deposits WHERE deposit_id = %s",
            (deposit_id,),
        ).fetchone()

    _run()

    with pg_transaction() as conn:
        after = conn.execute(
            "SELECT status, wallet_ledger_entry_id, amount_cad_minor "
            "FROM ln_deposits WHERE deposit_id = %s",
            (deposit_id,),
        ).fetchone()

    assert before == after, (
        "the reconciler mutated the deposit; it is report-only and must "
        "record findings without touching money state"
    )


def test_one_deposit_can_raise_several_findings(deposit):
    """Independent conditions are independent findings."""
    deposit_id = deposit(
        status="credited", ledger_entry_id=None, amount_cad=25.0, amount_minor=1
    )

    _run()

    types = {f[0] for f in _findings(deposit_id)}
    assert FINDING_CREDITED_WITHOUT_LEDGER in types
    assert FINDING_AMOUNT_MISMATCH in types


def test_terminal_statuses_are_not_scanned(deposit):
    """An expired deposit is settled, not drift."""
    deposit_id = deposit(status="pending", expiry_age_sec=7200)
    with pg_transaction() as conn:
        conn.execute(
            "UPDATE ln_deposits SET status = 'expired' WHERE deposit_id = %s",
            (deposit_id,),
        )

    _run()

    assert _findings(deposit_id) == []


def test_findings_carry_the_tenant(deposit):
    """Findings are tenant-filtered in the admin UI."""
    deposit_id = deposit(status="paid", paid_age_sec=1800)

    _run()

    with pg_transaction() as conn:
        row = conn.execute(
            "SELECT tenant_id FROM reconciliation_findings "
            "WHERE resource_type = %s AND resource_id = %s",
            (RESOURCE_TYPE, deposit_id),
        ).fetchone()
    assert row is not None and row[0], "finding must name the tenant"


def test_task_entry_point_is_registered_durably():
    """A process timer would not survive a restart or dedupe across replicas."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "bg_worker.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    names = {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "register_task"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }
    assert "lightning_reconcile" in names, (
        "the reconciler must be a durable scheduled_tasks entry so it "
        "survives restarts and only one replica runs it per interval"
    )


def test_bounded_scan_does_not_resolve_unscanned_findings(deposit):
    """The scan is bounded; absence from one pass is not "condition cleared".

    Resolving on absence alone would silently close real findings the
    moment the backlog exceeded the scan limit — precisely when they
    matter most. This drives that path with scan_limit=1.
    """
    first = deposit(status="paid", paid_age_sec=1800)
    second = deposit(status="paid", paid_age_sec=1800)

    # Two full sweeps: both deposits get an open finding.
    _run()
    assert any(f[2] is None for f in _findings(first))
    assert any(f[2] is None for f in _findings(second))

    # Now sweep with room for only one row.
    with pg_transaction() as conn:
        result = reconcile_ln_deposits(conn, scan_limit=1)
    assert result.truncated is True

    still_open = [
        deposit_id
        for deposit_id in (first, second)
        if any(f[2] is None for f in _findings(deposit_id))
    ]
    assert len(still_open) == 2, (
        "a truncated sweep resolved a finding for a deposit it never "
        "looked at; resolution must be scoped to scanned rows"
    )
