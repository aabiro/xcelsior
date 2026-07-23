"""Lightning deposit reconciliation (Track B B9.3b-3, companion §10.1).

Companion §10.1 requires "reconciliation for `paid but not credited`,
`credited but provider unknown`, expired, and amount mismatch". Those are
the four ways a deposit and the wallet can disagree, and each one is
somebody's money:

- **paid_not_credited** — the customer paid and the wallet never moved.
  This is the failure mode the pre-2026-07-22 Lightning module produced on
  every single deposit (the sweep raised before crediting anything), and
  it is invisible without a check like this one because nothing errors:
  the row simply sits in `paid` forever.
- **credited_without_ledger_entry** — the deposit says credited but names
  no `wallet_ledger_entry_id`, so the credit cannot be traced to a ledger
  posting. Either the link was lost or the money was never posted.
- **stuck_pending_past_expiry** — the invoice expired and nothing moved
  the row out of `pending`, so it is polled forever.
- **amount_mismatch** — the exact minor-unit amount disagrees with the
  legacy float column beyond a half cent, meaning the two representations
  of the same payment have diverged.

**Findings are recorded, never auto-repaired** (Track B B0.3 rule 17).
Crediting a wallet from a reconciler would be a second, untested money
path competing with `process_ln_deposits`; the point here is to make a
discrepancy *visible* so a human decides. `DA§8.7` states the same rule
for the warehouse: "BigQuery may detect discrepancies; it never repairs
money directly."

Findings land in the existing `reconciliation_findings` table so they
surface in the admin UI Track A Phase 9 already built, rather than in a
second findings authority nobody looks at.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

log = logging.getLogger("xcelsior.control_plane.billing_reconcile")

RESOURCE_TYPE = "ln_deposit"

FINDING_PAID_NOT_CREDITED = "paid_not_credited"
FINDING_CREDITED_WITHOUT_LEDGER = "credited_without_ledger_entry"
FINDING_STUCK_PENDING = "stuck_pending_past_expiry"
FINDING_AMOUNT_MISMATCH = "amount_mismatch"

# How long a deposit may sit in `paid` before it is a finding. The watcher
# runs every 5s, so anything past a few minutes is stuck, not in flight.
DEFAULT_CREDIT_GRACE_SEC = float(
    os.environ.get("XCELSIOR_LN_CREDIT_GRACE_SEC", "300")
)
# Grace after invoice expiry before an un-swept pending row is a finding.
DEFAULT_EXPIRY_GRACE_SEC = float(
    os.environ.get("XCELSIOR_LN_EXPIRY_GRACE_SEC", "3600")
)


@dataclass
class ReconcileResult:
    opened: list[str] = field(default_factory=list)
    resolved: int = 0
    scanned: int = 0
    truncated: bool = False


def _open_finding(
    conn: Connection,
    *,
    resource_id: str,
    tenant_id: str | None,
    finding_type: str,
    severity: str,
    summary: str,
    desired: dict[str, Any] | None = None,
    observed: dict[str, Any] | None = None,
) -> str | None:
    """Record a finding unless an identical one is already open.

    Same dedupe contract as `control_plane.reconcile._open_finding`:
    one open finding per (resource, type), so a sweep running every
    minute does not produce a finding per minute.
    """
    existing = conn.execute(
        """
        SELECT 1 FROM reconciliation_findings
         WHERE resource_type = %s AND resource_id = %s
           AND finding_type = %s AND resolved_at IS NULL
         LIMIT 1
        """,
        (RESOURCE_TYPE, resource_id, finding_type),
    ).fetchone()
    if existing is not None:
        return None

    row = conn.execute(
        """
        INSERT INTO reconciliation_findings
            (resource_type, resource_id, tenant_id, finding_type, severity,
             summary, desired, observed, action_taken)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'report_only')
        RETURNING finding_id
        """,
        (
            RESOURCE_TYPE,
            resource_id,
            tenant_id,
            finding_type,
            severity,
            summary,
            Jsonb(desired) if desired else None,
            Jsonb(observed) if observed else None,
        ),
    ).fetchone()
    log.warning(
        "ln reconcile finding [%s] %s %s: %s",
        severity, finding_type, resource_id, summary,
    )
    return str(row[0]) if row is not None else None


def _resolve_cleared(
    conn: Connection,
    still_open: set[tuple[str, str]],
    scanned_ids: set[str],
) -> int:
    """Close findings whose condition no longer holds.

    Without this a deposit that was stuck and then credited keeps an open
    finding forever, and the operator view fills with resolved problems
    until nobody reads it.

    Scoped to `scanned_ids` on purpose. The scan is bounded, so "not in
    still_open" does **not** mean "condition cleared" — it can also mean
    "this sweep never looked at it". Resolving on absence alone would
    silently close real findings the moment the backlog exceeded the scan
    limit, which is precisely when they matter most.
    """
    if not scanned_ids:
        return 0

    rows = conn.execute(
        """
        SELECT finding_id, resource_id, finding_type
          FROM reconciliation_findings
         WHERE resource_type = %s AND resolved_at IS NULL
           AND resource_id = ANY(%s)
        """,
        (RESOURCE_TYPE, list(scanned_ids)),
    ).fetchall()

    resolved = 0
    for finding_id, resource_id, finding_type in rows:
        if (str(resource_id), str(finding_type)) in still_open:
            continue
        conn.execute(
            "UPDATE reconciliation_findings SET resolved_at = clock_timestamp() "
            "WHERE finding_id = %s",
            (finding_id,),
        )
        resolved += 1
    return resolved


# Bound the work one sweep does in a single transaction. The predicate is
# already limited to non-terminal deposits, but "non-terminal" is not a
# guaranteed-small set — a stalled watcher is exactly what this reconciler
# exists to detect, and that is also the condition that grows the backlog.
# An unbounded scan inside one transaction is how a reconciler becomes the
# oldest transaction in the database (blueprint §23.1).
DEFAULT_SCAN_LIMIT = int(os.environ.get("XCELSIOR_LN_RECONCILE_SCAN_LIMIT", "5000"))


def reconcile_ln_deposits(
    conn: Connection,
    *,
    credit_grace_sec: float = DEFAULT_CREDIT_GRACE_SEC,
    expiry_grace_sec: float = DEFAULT_EXPIRY_GRACE_SEC,
    scan_limit: int = DEFAULT_SCAN_LIMIT,
) -> ReconcileResult:
    """Compare deposit state against the wallet ledger and record drift.

    Read-only with respect to money. The only writes are to
    `reconciliation_findings`.
    """
    result = ReconcileResult()
    still_open: set[tuple[str, str]] = set()

    rows = conn.execute(
        """
        SELECT deposit_id, tenant_id, customer_id, status, amount_cad,
               amount_cad_minor, wallet_ledger_entry_id,
               EXTRACT(EPOCH FROM (clock_timestamp() - paid_at_ts)) AS paid_age,
               EXTRACT(EPOCH FROM (clock_timestamp() - expires_at_ts)) AS expiry_age
          FROM ln_deposits
         WHERE status IN ('pending', 'paid', 'credited')
         ORDER BY created_at_ts NULLS FIRST
         LIMIT %s
        """,
        (max(1, scan_limit),),
    ).fetchall()
    result.scanned = len(rows)
    result.truncated = len(rows) >= max(1, scan_limit)
    scanned_ids: set[str] = {str(r[0]) for r in rows}

    for row in rows:
        (
            deposit_id, tenant_id, customer_id, status, amount_cad,
            amount_minor, ledger_entry_id, paid_age, expiry_age,
        ) = row
        deposit_id = str(deposit_id)

        # 1. paid but not credited past the grace window.
        if status == "paid" and paid_age is not None and float(paid_age) > credit_grace_sec:
            still_open.add((deposit_id, FINDING_PAID_NOT_CREDITED))
            opened = _open_finding(
                conn,
                resource_id=deposit_id,
                tenant_id=tenant_id or customer_id,
                finding_type=FINDING_PAID_NOT_CREDITED,
                severity="error",
                summary=(
                    f"deposit paid {float(paid_age):.0f}s ago and the wallet "
                    f"has not been credited"
                ),
                desired={"status": "credited"},
                observed={"status": status, "paid_age_sec": float(paid_age)},
            )
            if opened:
                result.opened.append(opened)

        # 2. credited with no traceable ledger posting.
        if status == "credited" and not ledger_entry_id:
            still_open.add((deposit_id, FINDING_CREDITED_WITHOUT_LEDGER))
            opened = _open_finding(
                conn,
                resource_id=deposit_id,
                tenant_id=tenant_id or customer_id,
                finding_type=FINDING_CREDITED_WITHOUT_LEDGER,
                severity="warning",
                summary=(
                    "deposit is credited but names no wallet ledger entry, so "
                    "the credit cannot be traced to a posting"
                ),
                desired={"wallet_ledger_entry_id": "<a ledger tx id>"},
                observed={"wallet_ledger_entry_id": None},
            )
            if opened:
                result.opened.append(opened)

        # 3. pending well past its invoice expiry.
        if (
            status == "pending"
            and expiry_age is not None
            and float(expiry_age) > expiry_grace_sec
        ):
            still_open.add((deposit_id, FINDING_STUCK_PENDING))
            opened = _open_finding(
                conn,
                resource_id=deposit_id,
                tenant_id=tenant_id or customer_id,
                finding_type=FINDING_STUCK_PENDING,
                severity="info",
                summary=(
                    f"invoice expired {float(expiry_age):.0f}s ago but the "
                    f"deposit is still pending"
                ),
                desired={"status": "expired"},
                observed={"status": status, "expiry_age_sec": float(expiry_age)},
            )
            if opened:
                result.opened.append(opened)

        # 4. the two money representations disagree.
        if amount_minor is not None and amount_cad is not None:
            drift = abs((float(amount_minor) / 100) - float(amount_cad))
            if drift > 0.005:
                still_open.add((deposit_id, FINDING_AMOUNT_MISMATCH))
                opened = _open_finding(
                    conn,
                    resource_id=deposit_id,
                    tenant_id=tenant_id or customer_id,
                    finding_type=FINDING_AMOUNT_MISMATCH,
                    severity="error",
                    summary=(
                        f"exact and legacy amounts differ by ${drift:.4f}; the "
                        f"two representations of one payment have diverged"
                    ),
                    desired={"amount_cad_minor": int(amount_minor)},
                    observed={"amount_cad": float(amount_cad)},
                )
                if opened:
                    result.opened.append(opened)

    result.resolved = _resolve_cleared(conn, still_open, scanned_ids)
    return result


def reconcile_ln_deposits_task() -> None:
    """Durable `scheduled_tasks` entry point.

    Registered rather than run on a process timer so it survives restarts
    and only one replica runs it per interval (Track A's
    `claim_and_run_tasks` uses SKIP LOCKED).
    """
    from db import pg_transaction

    with pg_transaction() as conn:
        result = reconcile_ln_deposits(conn)
    if result.opened or result.resolved:
        log.info(
            "ln reconcile: scanned=%d opened=%d resolved=%d",
            result.scanned, len(result.opened), result.resolved,
        )
