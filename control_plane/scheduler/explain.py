"""Placement explanation payloads (blueprint §3.2 / Phase 3 exit gate).

Every decision — and every *non*-decision — gets a JSON-serializable
explanation an operator can read: which hosts failed which hard filters,
how the eligible ones ranked, and why the winner won. The same payload
shape feeds ``job_attempts.placement_explanation`` at the Phase 4
cutover and `explain_instance_placement` in MCP, so shadow-mode evidence
and production evidence stay directly comparable.

Payloads are bounded: per-host detail is capped and the remainder is
summarized in aggregate, so a 500-host fleet cannot bloat a decision row.
"""

from __future__ import annotations

from typing import Any

from control_plane.scheduler.filters import (
    FILTER_POLICY_VERSION,
    FilterReason,
    aggregate_reason,
)
from control_plane.scheduler.scoring import SCORING_POLICY_VERSION, ScoreBreakdown

EXPLAIN_VERSION = "explain/v1"

DEFAULT_MAX_REJECTIONS = 25
DEFAULT_MAX_RANKED = 10


def _reason_dict(reason: FilterReason) -> dict[str, Any]:
    return {"code": reason.code, "message": reason.message, "details": reason.details}


def _score_dict(score: ScoreBreakdown) -> dict[str, Any]:
    return {
        "host_id": score.host_id,
        "total": score.total,
        "components": score.components,
        "tie_break": score.tie_break,
    }


def build_explanation(
    *,
    job: dict[str, Any],
    host_count: int,
    rejections: dict[str, list[FilterReason]],
    ranked: list[ScoreBreakdown],
    selected_host_id: str | None,
    queue_reason_code: str | None = None,
    max_rejections: int = DEFAULT_MAX_REJECTIONS,
    max_ranked: int = DEFAULT_MAX_RANKED,
) -> dict[str, Any]:
    """One self-contained, bounded explanation for a placement decision.

    ``ranked`` is best-first (the winner, if any, is ``ranked[0]``);
    ``rejections`` is every ineligible host's full typed reason list.
    """
    request = {
        "gpu_model": job.get("gpu_model"),
        "num_gpus": job.get("num_gpus", 1),
        "vram_needed_gb": job.get("vram_needed_gb"),
        "region": job.get("region"),
        "max_price_per_hour": job.get("max_price_per_hour"),
        "tier": job.get("tier"),
    }
    payload: dict[str, Any] = {
        "explain_version": EXPLAIN_VERSION,
        "filter_policy_version": FILTER_POLICY_VERSION,
        "scoring_policy_version": SCORING_POLICY_VERSION,
        "request": {k: v for k, v in request.items() if v not in (None, "")},
        "hosts_considered": host_count,
        "hosts_eligible": len(ranked),
        "hosts_rejected": len(rejections),
        # Aggregate constraint-failure counts always cover ALL rejections,
        # even when per-host detail below is truncated.
        "rejection_summary": aggregate_reason(rejections),
    }

    detail = {
        host_id: [_reason_dict(r) for r in reasons]
        for host_id, reasons in list(sorted(rejections.items()))[:max_rejections]
    }
    payload["rejections"] = detail
    if len(rejections) > max_rejections:
        payload["rejections_truncated"] = len(rejections) - max_rejections

    payload["ranked"] = [_score_dict(s) for s in ranked[:max_ranked]]
    if len(ranked) > max_ranked:
        payload["ranked_truncated"] = len(ranked) - max_ranked

    if selected_host_id is not None:
        payload["selected_host_id"] = selected_host_id
    else:
        payload["queue_reason_code"] = queue_reason_code or "no_eligible_host"
    return payload
