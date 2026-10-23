"""Apply admitted placement constraints at scoring and authoritative binding."""

from dataclasses import asdict
from typing import Any

from control_plane.scheduler.preference import (
    PlacementPreference,
    PlacementRefused,
    pct_to_bps,
    bps_to_pct,
)


def normalize_preference(raw: dict) -> dict:
    if not isinstance(raw, dict) or set(raw) - set(PlacementPreference.__dataclass_fields__):
        raise ValueError("invalid placement preference")
    values = asdict(PlacementPreference(**raw))
    for key, maximum in [("min_uptime_pct", 10000), ("max_premium_pct", 1000000)]:
        bps = pct_to_bps(values[key])
        if bps is not None and not 0 <= bps <= maximum:
            raise ValueError("placement preference exceeds its boundary")
        values[key] = bps_to_pct(bps)
    values["min_tier"] = str(values["min_tier"] or "").strip().lower() or None
    if type(values["require_verified"]) is not bool:
        raise ValueError("verification preference must be boolean")
    return values


def preference_for(spec):
    raw = spec.get("placement_preference")
    if not raw:
        return None
    preference = PlacementPreference(**normalize_preference(raw))
    return None if preference.is_empty() else preference


def choose_preferred(conn, spec, hosts, *, now=None):
    from control_plane.scheduler.constrained_placement import evaluate_preference

    preference = preference_for(spec)
    if preference is None:
        return None
    decision, _ = evaluate_preference(conn, hosts, preference, now=now)
    return decision


def require_preferred_host(
    conn, *, spec, host_id, scope_host_ids=None, freshness_sec=300, hypothetical_host=None
):
    """Run inside the same serializable reservation/planning transaction.

    Predicate reads include current fleet prices, capacity and evidence, so a
    conflicting writer cannot silently invalidate the premium baseline. No
    additional identity-table write privilege is granted to the scheduler.
    """
    from control_plane.scheduler.reservation import HostNotEligible

    try:
        preference = preference_for(spec)
    except ValueError as exc:
        raise HostNotEligible("invalid admitted placement preference") from exc
    if preference is None:
        return None
    level = conn.execute("SHOW transaction_isolation").fetchone()
    if level is None or level[0] != "serializable":
        raise HostNotEligible("constrained placement requires serializable revalidation")
    from control_plane.scheduler.snapshot import read_fleet
    from control_plane.scheduler.filters import filter_hosts, FilterContext

    fleet = read_fleet(conn, host_freshness_timeout_sec=freshness_sec)
    hosts = [h for h in fleet.hosts if scope_host_ids is None or h["host_id"] in scope_host_ids]
    if hypothetical_host:
        hosts = [hypothetical_host if h["host_id"] == host_id else h for h in hosts]
    eligible, _ = filter_hosts(spec, hosts, FilterContext(stale_host_ids=fleet.stale_host_ids))
    decision = choose_preferred(conn, spec, eligible, now=fleet.taken_at.timestamp())
    if decision is None:
        raise HostNotEligible("preference disappeared during revalidation")
    if isinstance(decision, PlacementRefused):
        raise HostNotEligible("placement preference refused", preference=decision.as_dict())
    if str(decision.host["host_id"]) != host_id:
        raise HostNotEligible(
            "placement preference chose another host", selected_host_id=decision.host["host_id"]
        )
    return decision.as_dict()
