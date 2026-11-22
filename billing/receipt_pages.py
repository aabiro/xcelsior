"""Public workload pages built from real, shared run receipts.

A page answers "what does <workload> cost on <GPU>?" — for example vLLM on an
RTX 3060 — from runs that actually happened and were billed:

* Only runs whose owner shared them (``receipt_shares``), or operator runs on
  the real fleet shared the same way. Nothing is estimated or made up.
* Only finished, reconciled runs that started from a quick-start template
  (``launch_template_id``), on a known GPU model, that ran at least a minute
  and ended normally (completed, cancelled or terminated; a failed or
  preempted run is not what the workload costs).
* A page appears once :data:`MIN_RUNS` such runs exist for one template and
  GPU model, and only while that GPU model has live, admitted supply a launch
  can get — a page never advertises a GPU nobody can rent.
* A page never carries a person, account, job, host, image reference or exact
  timestamp: only counts, medians and ranges (:data:`PAGE_FIELDS`, held by a
  test), and the month of the newest run.
"""

from __future__ import annotations

import re
import statistics
import time
from datetime import datetime, timezone
from typing import Any

MIN_RUNS = 3
MIN_BILLED_SECONDS = 60
PAGE_STATUSES = frozenset({"completed", "cancelled", "terminated"})

#: Every key a public page carries. Anything else is a leak; a test holds the
#: builder to this set.
PAGE_FIELDS = (
    "slug",
    "template_id",
    "template_label",
    "template_description",
    "category",
    "gpu_model",
    "runs",
    "customer_runs",
    "operator_runs",
    "median_billed_seconds",
    "min_billed_seconds",
    "max_billed_seconds",
    "median_total_micros",
    "min_total_micros",
    "max_total_micros",
    "median_rate_micros_per_hour",
    "pricing_modes",
    "median_gpu_utilization",
    "utilization_runs",
    "latest_month",
)

_CACHE_SECONDS = 300
_cache: dict[str, Any] = {"at": 0.0, "pages": None}


def gpu_slug(gpu_model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", gpu_model.lower()).strip("-")


def page_slug(template_id: str, gpu_model: str) -> str:
    """``vllm-on-rtx-3060``: the workload, then the GPU."""
    return f"{template_id}-on-{gpu_slug(gpu_model)}"


def templates() -> dict[str, dict]:
    from security import get_image_templates

    return {str(t["id"]): t for t in get_image_templates()}


def share_blocker(receipt: dict) -> str | None:
    """Why this run cannot count on a page, or None when it can.

    ``not_finished`` and ``not_reconciled`` clear on their own (the run ends,
    the billing loop catches up); the others are permanent for this run.
    """
    from host_metadata import normalize_gpu_model

    if not receipt.get("final"):
        return "not_finished"
    if receipt.get("status") not in PAGE_STATUSES:
        return "ended_abnormally"
    template_id = receipt.get("launch_template_id")
    if not template_id or str(template_id) not in templates():
        return "no_template"
    if not normalize_gpu_model(str(receipt.get("gpu_model") or "")):
        return "no_gpu"
    if float(receipt.get("billed_seconds") or 0) < MIN_BILLED_SECONDS:
        return "too_short"
    if not receipt.get("reconciled"):
        return "not_reconciled"
    return None


def share_state(conn, job_id: str) -> dict | None:
    """The run's share row (dict cursor), or None if it was never shared."""
    row = conn.execute(
        """SELECT source, EXTRACT(EPOCH FROM created_at) AS created_at,
                  EXTRACT(EPOCH FROM revoked_at) AS revoked_at
             FROM receipt_shares WHERE job_id = %s""",
        (job_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "source": row["source"],
        "created_at": float(row["created_at"]),
        "revoked_at": float(row["revoked_at"]) if row["revoked_at"] is not None else None,
    }


def set_shared(conn, *, job_id: str, owner_id: str, shared_by: str, source: str, on: bool) -> None:
    """Turn the run's share on (again) or off. ``conn`` is a dict cursor."""
    if on:
        conn.execute(
            """INSERT INTO receipt_shares (job_id, owner_id, shared_by, source)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (job_id) DO UPDATE
                  SET shared_by = EXCLUDED.shared_by, source = EXCLUDED.source,
                      created_at = CASE WHEN receipt_shares.revoked_at IS NULL
                                        THEN receipt_shares.created_at ELSE now() END,
                      revoked_at = NULL""",
            (job_id, owner_id, shared_by, source),
        )
    else:
        conn.execute(
            "UPDATE receipt_shares SET revoked_at = now() WHERE job_id = %s AND revoked_at IS NULL",
            (job_id,),
        )
    _cache["pages"] = None


def live_supply_models() -> set[str]:
    """GPU models a launch can get right now: live, admitted, placeable hosts."""
    from host_evidence import attach_evidence, only_placeable
    from host_metadata import normalize_gpu_model
    from scheduler import list_hosts

    rows = [
        h
        for h in list_hosts(active_only=False)
        if h.get("status") == "active" and h.get("admitted") is True
    ]
    models = {
        normalize_gpu_model(str(h.get("gpu_model") or ""))
        for h in only_placeable(attach_evidence(rows))
    }
    return models - {""}


def _shared_runs(pool) -> list[dict]:
    """Every active share's receipt that can count, with its source and utilization."""
    from psycopg.rows import dict_row

    from gpu_telemetry import average_utilization
    from host_metadata import normalize_gpu_model
    from run_receipts import build_receipt
    from scheduler import get_job

    runs = []
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        shares = cur.execute("""SELECT job_id, owner_id, source FROM receipt_shares
                WHERE revoked_at IS NULL ORDER BY job_id""").fetchall()
        for share in shares:
            job = get_job(str(share["job_id"]))
            # A share outlives nothing: the job must still belong to whoever shared it.
            if not job or (job.get("owner") or "") != share["owner_id"]:
                continue
            receipt = build_receipt(cur, job)
            if share_blocker(receipt) is not None:
                continue
            runs.append(
                {
                    "template_id": str(receipt["launch_template_id"]),
                    "gpu_model": normalize_gpu_model(str(receipt["gpu_model"])),
                    "source": share["source"],
                    "billed_seconds": float(receipt["billed_seconds"]),
                    "total_micros": int(receipt["total_micros"]),
                    "rate_micros_per_hour": receipt.get("rate_micros_per_hour"),
                    "pricing_mode": str(receipt.get("pricing_mode") or "on_demand"),
                    "ended_at": receipt.get("ended_at"),
                    "utilization": average_utilization(conn, str(receipt["job_id"])),
                }
            )
    return runs


def _median_int(values: list[int]) -> int:
    return int(round(statistics.median(values)))


def build_pages(runs: list[dict], supply: set[str]) -> list[dict]:
    """Group runs by template and GPU model; keep groups of MIN_RUNS on GPUs with supply."""
    catalog = templates()
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        groups.setdefault((run["template_id"], run["gpu_model"]), []).append(run)
    pages = []
    for (template_id, gpu_model), group in sorted(groups.items()):
        if len(group) < MIN_RUNS or gpu_model not in supply or template_id not in catalog:
            continue
        template = catalog[template_id]
        seconds = [r["billed_seconds"] for r in group]
        totals = [r["total_micros"] for r in group]
        rates = [int(r["rate_micros_per_hour"]) for r in group if r["rate_micros_per_hour"]]
        utilization = [r["utilization"] for r in group if r["utilization"] is not None]
        modes: dict[str, int] = {}
        for r in group:
            modes[r["pricing_mode"]] = modes.get(r["pricing_mode"], 0) + 1
        ended = [r["ended_at"] for r in group if r["ended_at"]]
        pages.append(
            {
                "slug": page_slug(template_id, gpu_model),
                "template_id": template_id,
                "template_label": str(template.get("label") or template_id),
                "template_description": str(template.get("description") or ""),
                "category": str(template.get("category") or ""),
                "gpu_model": gpu_model,
                "runs": len(group),
                "customer_runs": sum(1 for r in group if r["source"] == "customer"),
                "operator_runs": sum(1 for r in group if r["source"] == "operator"),
                "median_billed_seconds": round(statistics.median(seconds), 1),
                "min_billed_seconds": round(min(seconds), 1),
                "max_billed_seconds": round(max(seconds), 1),
                "median_total_micros": _median_int(totals),
                "min_total_micros": min(totals),
                "max_total_micros": max(totals),
                "median_rate_micros_per_hour": _median_int(rates) if rates else None,
                "pricing_modes": modes,
                # Shown only when enough of the runs reported per-GPU readings.
                "median_gpu_utilization": (
                    round(statistics.median(utilization), 1)
                    if len(utilization) >= MIN_RUNS
                    else None
                ),
                "utilization_runs": len(utilization),
                "latest_month": (
                    datetime.fromtimestamp(max(ended), tz=timezone.utc).strftime("%Y-%m")
                    if ended
                    else None
                ),
            }
        )
    return pages


def public_pages(*, fresh: bool = False) -> list[dict]:
    """Every page that qualifies now, cached for a few minutes."""
    from db import _get_pg_pool

    now = time.monotonic()
    if not fresh and _cache["pages"] is not None and now - _cache["at"] < _CACHE_SECONDS:
        return list(_cache["pages"])
    pages = build_pages(_shared_runs(_get_pg_pool()), live_supply_models())
    _cache.update(at=now, pages=pages)
    return list(pages)


def page_progress(template_id: str, gpu_model: str) -> dict:
    """How far one template and GPU model are from a page (for the run's owner)."""
    from host_metadata import normalize_gpu_model

    gpu = normalize_gpu_model(gpu_model)
    slug = page_slug(template_id, gpu)
    live = next((p for p in public_pages() if p["slug"] == slug), None)
    return {
        "slug": slug,
        "live": live is not None,
        "runs": live["runs"] if live else None,
        "min_runs": MIN_RUNS,
        "gpu_has_supply": gpu in live_supply_models(),
    }
