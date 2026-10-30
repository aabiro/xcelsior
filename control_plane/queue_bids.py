"""Paid queue ordering evidence. Aging never changes billing or isolation."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from psycopg.types.json import Jsonb

from service_tier_pricing import QUEUE_BID_POLICY_VERSION, SERVICE_TIERS


def ordering(snapshot: dict[str, Any], queued_seconds: float) -> dict[str, Any]:
    """Reference policy, also expressed in SQL by the SKIP LOCKED claimant.

    Purchased tiers stay ahead of promoted work. Promoted jobs use FIFO by
    promotion qualification time, never a bid comparison with paying peers.
    """
    rank = SERVICE_TIERS[snapshot.get("tier") or "standard"]["priority"]
    seconds = max(Decimal(0), Decimal(str(queued_seconds)))
    steps = int(seconds // 900)
    bid = int(snapshot.get("queue_bid_micros") or 0)
    credit = steps * bid // 10
    gap = snapshot.get("next_tier_gap_micros")
    promoted = bool(rank < 3 and bid > 0 and gap is not None and bid + credit >= gap)
    promotion_step = ((max(0, int(gap or 0) - bid) * 10 + bid - 1) // bid) if promoted else None
    return {
        "policy_version": QUEUE_BID_POLICY_VERSION,
        "original_tier": rank,
        "effective_tier": rank + int(promoted),
        "promoted": promoted,
        "aging_steps": steps,
        "aging_credit_micros": credit,
        "effective_bid_micros": bid + credit,
        "promotion_after_seconds": promotion_step * 900 if promotion_step is not None else None,
    }


def record_submission(conn, job: dict) -> None:
    quote = job.get("service_tier_quote")
    if not quote:
        return
    evidence = ordering(quote, 0)
    conn.execute(
        """INSERT INTO queue_bid_events
              (tenant_id,job_id,event_type,service_tier,effective_tier,bid_price_id,
               bid_units,bid_micros,policy_version,pricing_snapshot,ordering_snapshot)
           VALUES(%s,%s,'submitted',%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT(job_id,event_type,claim_number) DO NOTHING""",
        (
            job["owner"],
            job["job_id"],
            quote["tier"],
            evidence["effective_tier"],
            quote.get("queue_bid_price_id"),
            quote.get("queue_bid_units", 0),
            quote.get("queue_bid_micros", 0),
            QUEUE_BID_POLICY_VERSION,
            Jsonb(quote),
            Jsonb(evidence),
        ),
    )


def record_claim(
    conn,
    *,
    job_id: str,
    owner: str,
    claim_number: int,
    quote: dict[str, Any],
    effective_tier: int,
    promoted: bool,
    aging_steps: int,
    aging_credit: int,
    queue_seconds: Any,
    effective_bid: int,
    promoted_at: Any,
) -> None:
    """The claimant's SQL rank, kept as evidence for ordering experiments."""
    conn.execute(
        """INSERT INTO queue_bid_events
              (tenant_id,job_id,event_type,claim_number,service_tier,effective_tier,
               bid_price_id,bid_units,bid_micros,aging_steps,aging_credit_micros,
               queue_seconds,policy_version,pricing_snapshot,ordering_snapshot)
           VALUES(%s,%s,'claimed',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT(job_id,event_type,claim_number) DO NOTHING""",
        (
            owner,
            job_id,
            claim_number,
            quote["tier"],
            effective_tier,
            quote.get("queue_bid_price_id"),
            quote.get("queue_bid_units", 0),
            quote.get("queue_bid_micros", 0),
            aging_steps,
            aging_credit,
            queue_seconds,
            QUEUE_BID_POLICY_VERSION,
            Jsonb(quote),
            Jsonb(
                {
                    "policy_version": QUEUE_BID_POLICY_VERSION,
                    "promoted": promoted,
                    "effective_bid_micros": effective_bid,
                    "promoted_at": str(promoted_at) if promoted_at else None,
                }
            ),
        ),
    )
