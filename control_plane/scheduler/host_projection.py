"""The query that assembles what `preference.choose_host` reads.

Until now every field `choose_host` consults was assembled by nobody, so every
field *name* in that module was an assumption — the same class of assumption
that produced the invented tier vocabulary. Six of the twelve are aliases or
derivations rather than columns:

| what `choose_host` reads | where it actually lives |
|---|---|
| `verification_state` | `host_verifications.state` |
| `reputation_tier` | `reputation_scores.tier` |
| `reputation_score` | `reputation_scores.`**`final_score`** |
| `sla_total_seconds` | `sla_monthly.total_seconds`, summed and clamped |
| `sla_downtime_seconds` | `sla_monthly.downtime_seconds`, summed |
| `price_cents_per_hour` | `gpu_offers.ask_cents_per_hour`, or `cost_per_hour` × 100 |

The other six — `host_id`, `verified_at`, `deverified_at`, `last_check_at`,
`next_check_at`, `verification_unavailable` — match their columns, or are set
here outright.

## Why this module validates its own output

**Every absent field in `choose_host` fails closed.** That is right on its own,
and it means one wrong name here produces a gate that refuses every request —
*indistinguishable from the correct behaviour* this phase spent five rounds
building. Fail-closed defaults camouflage broken plumbing.

So a missing key is an **error**, not a `None`: "this host has no evidence" and
"this query did not run" are different facts, and only the first should ever
reach a refusal.

## `final_score`, not `raw_score`

`reputation_scores` carries both. `reputation.py` computes
`final = raw * reliability` and then `tier = score_to_tier(final)`, and its own
query already aliases `final_score AS score`. `raw_score` is before penalties
and the reliability weighting, so projecting it would shift hosts a tier while
`host_tier()` derives from it. Both names read plausibly, which is why this is
stated rather than left to whoever edits the SELECT next.

## Which failures are tolerated, and why only one

The verification read is wrapped; the reputation and SLA reads are not.

That asymmetry is deliberate. Verification has an explicit fail-*open* flag —
`verification_unavailable` — that `choose_host` consults, so an unreadable
verification store produces a *stated* condition: constrained requests refuse
with `verification_unreadable`, unconstrained ones proceed, which is the
cold-start tolerance `scheduler.py` already has. Reputation and SLA have no such
flag; swallowing their errors would yield zeros, and zeros read as "no history"
and "no tier" — a universally-refusing gate that looks exactly like a working
one. So those propagate.
"""

from __future__ import annotations

import calendar
import logging
import time
from datetime import datetime
from typing import Any, Iterable, Sequence

from control_plane.scheduler.preference import OBSERVATION_WINDOW_DAYS

log = logging.getLogger(__name__)

#: Every key `choose_host` may read, other than the price. The shape check
#: requires all of them, so a field dropped from the SELECT fails here rather
#: than becoming a silent refusal downstream.
REQUIRED_EVIDENCE_FIELDS = (
    "host_id",
    "verification_state",
    "verified_at",
    "deverified_at",
    "last_check_at",
    "next_check_at",
    "verification_unavailable",
    "reputation_tier",
    "reputation_score",
    "sla_total_seconds",
    "sla_downtime_seconds",
)

#: The one price key `usable_price` treats as canonical. `attach_placement_evidence`
#: normalises into it; see `normalise_price_cents`.
PRICE_FIELD = "price_cents_per_hour"


class ProjectionError(RuntimeError):
    """The projection produced a row the gate cannot safely read."""


def assert_evidence_shape(hosts: Sequence[dict], *, require_price: bool = False) -> None:
    """Every required key present on every row, or raise.

    Deliberately **not** a filter and **not** a default-filler. A row missing
    `sla_total_seconds` is refused by the gate as "no measured history", which is
    a statement about the host; if the column was simply never selected, that
    statement is false and the operator is sent to look at the wrong thing.
    """
    required = REQUIRED_EVIDENCE_FIELDS + ((PRICE_FIELD,) if require_price else ())
    for index, host in enumerate(hosts):
        missing = [f for f in required if f not in host]
        if missing:
            raise ProjectionError(
                f"projected host row {index} "
                f"({host.get('host_id', '<no host_id>')}) is missing {missing}. "
                "Every absent field fails closed in the gate, so this would "
                "surface as a preference that refuses everything — which looks "
                "exactly like the gate working."
            )
        if not isinstance(host.get("verification_unavailable"), bool):
            raise ProjectionError(
                f"projected host row {index} "
                f"({host.get('host_id', '<no host_id>')}) has "
                f"verification_unavailable="
                f"{host.get('verification_unavailable')!r}; it must be True or "
                "False. It is the one field here that fails open, so a null or "
                "absent value disables the unread-evidence refusal silently."
            )


def _month_bounds(month: str) -> tuple[float, float]:
    """Epoch start and end of a `YYYY-MM` row, **the way `sla.py` computes them**.

    `sla.py` writes `total_seconds = days_in_month * 86400` from
    `datetime.strptime(month, "%Y-%m").timestamp()`, which is local time on the
    writer. Recomputing it identically here — rather than in SQL, where the
    session timezone would silently differ — keeps the clamp consistent with the
    numbers it is clamping.
    """
    start = datetime.strptime(month, "%Y-%m")
    days = calendar.monthrange(start.year, start.month)[1]
    begin = start.timestamp()
    return begin, begin + days * 86400.0


def _cutoff_month(now: float, window_days: int) -> str:
    return datetime.fromtimestamp(now - window_days * 24 * 3600).strftime("%Y-%m")


def project_placement_evidence(
    conn,
    host_ids: Iterable[str],
    *,
    now: float | None = None,
    window_days: int = OBSERVATION_WINDOW_DAYS,
) -> dict[str, dict]:
    """Assemble placement evidence for `host_ids`, keyed by host id.

    One query per source rather than one join: `sla_monthly` needs aggregating
    over a trailing window while the others are point lookups, and a single join
    would either fan the SLA rows out or need a subquery that reads worse than
    three statements.
    """
    ids = sorted({str(h) for h in host_ids if h})
    if not ids:
        return {}
    now = time.time() if now is None else float(now)

    verification: dict[str, dict] = {}
    verification_readable = True
    try:
        for record in conn.execute(
            """SELECT host_id, state, verified_at, deverified_at,
                      last_check_at, next_check_at
                 FROM host_verifications
                WHERE host_id = ANY(%s)""",
            (ids,),
        ).fetchall():
            verification[str(record[0])] = {
                "verification_state": record[1],
                "verified_at": record[2],
                "deverified_at": record[3],
                "last_check_at": record[4],
                "next_check_at": record[5],
            }
    except Exception as exc:
        # Recorded, not swallowed — see the module docstring. A request carrying
        # `require_verified` refuses on this; one without it proceeds. Both need
        # the flag to be *present*, which is why it is set on every row below
        # rather than only on the failure path.
        verification_readable = False
        log.warning("placement evidence: verification store unreadable: %s", exc)

    reputation: dict[str, dict] = {}
    for record in conn.execute(
        # `entity_type = 'host'` is necessary and **not sufficient**, and the
        # difference is measured rather than assumed. `reputation.py` defaults
        # the column to `"host"` at every layer — the dataclass field,
        # `_ensure_entity`, `_get_or_create_score_record`, `record_event` — so
        # rows for entities that are plainly not hosts still carry `host`. The
        # predicate is kept because a correctly-typed user row must never be
        # read as a host's, but what actually keeps non-hosts out of a placement
        # is the caller's shortlist being built from hosts and offers. Fixing
        # the writer is a reputation-module change, not one made here.
        """SELECT entity_id, tier, final_score
             FROM reputation_scores
            WHERE entity_id = ANY(%s)
              AND entity_type = 'host'""",
        (ids,),
    ).fetchall():
        reputation[str(record[0])] = {
            "reputation_tier": record[1],
            "reputation_score": record[2],
        }

    #: Rows are per calendar month, so a 90-day window is summed as the calendar
    #: months that *overlap* it — up to ~120 days of evidence. Prorating a
    #: month's downtime to the window boundary would mean inventing when the
    #: downtime happened, and the whole module exists to stop unmeasured things
    #: being treated as measured. Erring toward more real evidence is the safe
    #: direction for a floor.
    sla: dict[str, dict] = {}
    for record in conn.execute(
        """SELECT host_id, month, total_seconds, downtime_seconds
             FROM sla_monthly
            WHERE host_id = ANY(%s)
              AND month >= %s""",
        (ids, _cutoff_month(now, window_days)),
    ).fetchall():
        host_id, month, total, downtime = (
            str(record[0]), str(record[1]), record[2], record[3]
        )
        try:
            begin, _end = _month_bounds(month)
        except ValueError:
            log.warning("placement evidence: unparsable sla_monthly.month %r", month)
            continue
        # **The in-progress month counts time that has not happened yet.**
        # `sla.py` writes the full calendar month into `total_seconds` while
        # capping downtime at `now`, so on the 10th of a 31-day month a host
        # that was down for a day reads 96.8% instead of 90%. Every host's
        # uptime is inflated for most of every month, against a gate whose
        # entire job is `min_uptime_pct`. Clamped to elapsed time here.
        elapsed = max(0.0, now - begin)
        counted = min(float(total or 0.0), elapsed)
        bucket = sla.setdefault(host_id, {"sla_total_seconds": 0.0, "sla_downtime_seconds": 0.0})
        bucket["sla_total_seconds"] += counted
        bucket["sla_downtime_seconds"] += min(float(downtime or 0.0), counted)

    projected: dict[str, dict] = {}
    for host_id in ids:
        row: dict[str, Any] = {
            "host_id": host_id,
            # Always present, True or False, never absent. This is the module's
            # only fail-*open* field — everything else refuses when missing,
            # while this one refuses only when set — so if it could go missing,
            # the unread-evidence refusal would never fire and the silent
            # fallback returns at the layer that just closed it. The field that
            # catches missing evidence must not itself be able to go missing.
            "verification_unavailable": not verification_readable,
            "verification_state": None,
            "verified_at": None,
            "deverified_at": None,
            "last_check_at": None,
            "next_check_at": None,
            "reputation_tier": None,
            "reputation_score": None,
            "sla_total_seconds": 0.0,
            "sla_downtime_seconds": 0.0,
        }
        row.update(verification.get(host_id, {}))
        row.update(reputation.get(host_id, {}))
        row.update(sla.get(host_id, {}))
        projected[host_id] = row

    assert_evidence_shape(list(projected.values()))
    return projected


def normalise_price_cents(candidate: dict) -> float | None:
    """Cents per hour, from whichever key the candidate carries.

    Three names are live for one number, and **one of them is in a different
    unit**: `gpu_offers.ask_cents_per_hour` is cents, while the host dicts
    `scheduler.allocate_best_host` ranks carry `cost_per_hour` in dollars. A
    premium computed across a mixed list would be wrong by 100×, and
    `usable_price` reads neither of the dollar names — so on real candidate rows
    every request would have refused with `no_priced_hosts`.

    Normalising here means `usable_price` sees one key in one unit, and the
    conversion is written down once instead of being implied by whichever list a
    caller happened to build.
    """
    for key in (PRICE_FIELD, "ask_cents_per_hour"):
        raw = candidate.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    raw = candidate.get("cost_per_hour")
    if raw is not None:
        try:
            return float(raw) * 100.0
        except (TypeError, ValueError):
            return None
    return None


def attach_placement_evidence(
    conn,
    candidates: Sequence[dict],
    *,
    now: float | None = None,
    window_days: int = OBSERVATION_WINDOW_DAYS,
) -> list[dict]:
    """Return copies of `candidates` carrying everything `choose_host` reads.

    The candidates come from whatever built the shortlist — they own the price
    and the hardware facts. This adds the evidence and normalises the price, then
    checks the merged shape, so a caller cannot hand `choose_host` a row that
    fails closed for a reason nobody chose.
    """
    for index, candidate in enumerate(candidates):
        if not str(candidate.get("host_id") or "").strip():
            # Not a fact about a host — a malformed shortlist. Filling it with
            # no-evidence defaults would let an unidentifiable row compete for a
            # placement, and refusing it silently would hide the caller's bug
            # behind a refusal that reads as policy.
            raise ProjectionError(
                f"candidate {index} has no host_id, so no evidence can be read "
                f"for it: {candidate!r}"
            )

    evidence = project_placement_evidence(
        conn, [c["host_id"] for c in candidates], now=now, window_days=window_days
    )
    merged = []
    for candidate in candidates:
        host_id = str(candidate["host_id"])
        row = dict(candidate)
        row.update(evidence[host_id])
        row[PRICE_FIELD] = normalise_price_cents(candidate)
        merged.append(row)
    assert_evidence_shape(merged, require_price=True)
    return merged
