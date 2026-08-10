"""Writing the placement decision down. C1's second half.

Gate P5 clause 3: *"preference is honoured in the audit trail: the chosen host's
reputation and SLA at time of placement are recorded."*

`preference.placement_evidence()` produces the evidence; this persists it, along
with the decision it justified. Two properties matter more than the schema:

**Refusals are recorded, not just placements.** A preference that refused *was*
honoured — by the refusal. A table holding only successes would carry no
evidence of the behaviour the gate exists to produce, and could not answer "why
did nothing launch last Tuesday", which is the question an operator actually
arrives with.

**The row cannot be changed.** `placement_decisions` carries a WORM trigger, so
this module has no update path and could not acquire one without a migration.
Copied evidence is worth exactly what it costs to rewrite.

## What this table is not

It records **other hosts' prices and states** in `candidates`, because a refusal
is only interpretable against the field it refused over — "no host met 99.5%" is
unreadable without knowing what the field offered. That makes it an *internal*
audit record. Exposing it to a tenant verbatim would hand them a priced snapshot
of the fleet, and the tenant-facing view has to project it down to the chosen
host plus aggregates. That projection belongs to C2's surface, and until it
exists no route reads this table.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from control_plane.scheduler.preference import (
    PlacementChoice,
    PlacementPreference,
    PlacementRefused,
    host_tier,
    host_uptime_pct,
    usable_price,
    verification_status,
)

#: Cents per hour → micros of CAD per hour. Money is integer micros; the premium
#: is recomputed from two exact integers rather than stored as a rounded
#: percentage that cannot be checked against anything.
MICROS_PER_CENT = 10_000


def price_micros(cents_per_hour: float | None) -> int | None:
    """Round to micros, or None when there is no usable price.

    `None` rather than `0`: the table's CHECK rejects a zero price precisely
    because a zero baseline is what silently turns every premium into 0%.
    """
    if cents_per_hour is None:
        return None
    value = round(float(cents_per_hour) * MICROS_PER_CENT)
    return value if value > 0 else None


def preference_as_dict(preference: PlacementPreference | None) -> dict:
    """What the user asked for, verbatim, so a decision can be re-read against
    the preference that produced it rather than against whatever the defaults
    became later."""
    if preference is None:
        return {}
    return {
        "min_uptime_pct": preference.min_uptime_pct,
        "min_tier": preference.min_tier,
        "require_verified": preference.require_verified,
        "max_premium_pct": preference.max_premium_pct,
    }


def candidate_summary(host: dict) -> dict:
    """One line per host the choice was made among.

    Deliberately a summary and not the row: the projected row carries raw
    timestamps and scores for every host in the shortlist, and the reason to keep
    candidates at all is to make a refusal interpretable, not to snapshot the
    fleet.
    """
    return {
        "host_id": str(host.get("host_id") or ""),
        "price_micros": price_micros(usable_price(host)),
        "uptime_pct": host_uptime_pct(host),
        "tier": host_tier(host),
        "verification_status": verification_status(host),
        # Carried because it is the one field that distinguishes "this host is
        # not verified" from "nobody could read whether it is".
        "verification_unavailable": bool(host.get("verification_unavailable")),
    }


def record_placement(
    conn,
    *,
    tenant_id: str,
    decision: PlacementChoice | PlacementRefused,
    candidates: Sequence[dict],
    preference: PlacementPreference | None = None,
    job_id: str | None = None,
) -> str:
    """Append one decision and return its id.

    `candidates` is the list the decision was made over — the same list handed to
    `choose_host`, after `attach_placement_evidence`.
    """
    if not str(tenant_id or "").strip():
        # A row nobody owns is a row no tenant-scoped read will ever return, and
        # the table's index leads with tenant_id. Failing here beats writing an
        # unreachable record.
        raise ValueError("placement decisions must be attributed to a tenant")

    summaries = [candidate_summary(h) for h in candidates]
    row: dict[str, Any] = {
        "tenant_id": str(tenant_id),
        "job_id": job_id,
        "asked": preference_as_dict(preference),
        "candidate_count": len(summaries),
        "candidates": summaries,
    }

    if isinstance(decision, PlacementRefused):
        row.update(
            outcome="refused",
            host_id=None,
            refusal_code=decision.code,
            refusal_detail=decision.detail,
            evidence={"asked": decision.asked, "best_available": decision.best_available},
            baseline_price_micros=None,
            chosen_price_micros=None,
        )
    else:
        row.update(
            outcome="placed",
            host_id=str(decision.host.get("host_id") or ""),
            refusal_code=None,
            refusal_detail=None,
            evidence=decision.evidence,
            baseline_price_micros=price_micros(decision.baseline_price),
            chosen_price_micros=price_micros(decision.chosen_price),
        )

    return str(
        conn.execute(
            """INSERT INTO placement_decisions
                 (tenant_id, job_id, host_id, outcome, refusal_code, refusal_detail,
                  asked, evidence, candidate_count, candidates,
                  baseline_price_micros, chosen_price_micros)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING decision_id""",
            (
                row["tenant_id"], row["job_id"], row["host_id"], row["outcome"],
                row["refusal_code"], row["refusal_detail"],
                json.dumps(row["asked"], default=str),
                json.dumps(row["evidence"], default=str),
                row["candidate_count"],
                json.dumps(row["candidates"], default=str),
                row["baseline_price_micros"], row["chosen_price_micros"],
            ),
        ).fetchone()[0]
    )


def read_placement(conn, decision_id: str, *, tenant_id: str) -> dict | None:
    """One decision, scoped to its tenant.

    `tenant_id` is required rather than optional: an audit record keyed by a uuid
    is enumerable, and a read that forgets to scope is the defect that makes it
    so.
    """
    record = conn.execute(
        """SELECT decision_id, tenant_id, job_id, host_id, outcome, refusal_code,
                  refusal_detail, asked, evidence, candidate_count, candidates,
                  baseline_price_micros, chosen_price_micros, decided_at
             FROM placement_decisions
            WHERE decision_id = %s AND tenant_id = %s""",
        (decision_id, tenant_id),
    ).fetchone()
    if record is None:
        return None
    keys = (
        "decision_id", "tenant_id", "job_id", "host_id", "outcome", "refusal_code",
        "refusal_detail", "asked", "evidence", "candidate_count", "candidates",
        "baseline_price_micros", "chosen_price_micros", "decided_at",
    )
    out = {k: record[i] for i, k in enumerate(keys)}
    out["decision_id"] = str(out["decision_id"])
    out["premium_pct"] = premium_pct(
        out["baseline_price_micros"], out["chosen_price_micros"]
    )
    return out


def premium_pct(baseline_micros: int | None, chosen_micros: int | None) -> float | None:
    """Recomputed from the two stored integers rather than stored.

    A stored percentage is a rounded number nobody can check; two exact integers
    can be re-divided by anyone reading the row.
    """
    if not baseline_micros or not chosen_micros or baseline_micros <= 0:
        return None
    return (chosen_micros - baseline_micros) / baseline_micros * 100.0


def list_placements(conn, *, tenant_id: str, limit: int = 50) -> list[dict]:
    """Most recent first, for the tenant that made them."""
    rows = conn.execute(
        """SELECT decision_id, job_id, host_id, outcome, refusal_code,
                  candidate_count, baseline_price_micros, chosen_price_micros, decided_at
             FROM placement_decisions
            WHERE tenant_id = %s
         ORDER BY decided_at DESC
            LIMIT %s""",
        (tenant_id, int(limit)),
    ).fetchall()
    keys = (
        "decision_id", "job_id", "host_id", "outcome", "refusal_code",
        "candidate_count", "baseline_price_micros", "chosen_price_micros", "decided_at",
    )
    out = []
    for record in rows:
        item = {k: record[i] for i, k in enumerate(keys)}
        item["decision_id"] = str(item["decision_id"])
        item["premium_pct"] = premium_pct(
            item["baseline_price_micros"], item["chosen_price_micros"]
        )
        out.append(item)
    return out
