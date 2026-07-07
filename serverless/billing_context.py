"""Endpoint billing context for serverless worker charge paths."""

from __future__ import annotations

from typing import Any

# Columns required from serverless_endpoints JOIN for blended/token metering.
ENDPOINT_BILLING_COLUMNS: tuple[str, ...] = (
    "endpoint_id",
    "owner_id",
    "name",
    "mode",
    "model_ref",
    "gpu_tier",
    "region",
    "gpu_count",
)

ENDPOINT_BILLING_SELECT_SQL = ", ".join(f"e.{col}" for col in ENDPOINT_BILLING_COLUMNS)


def endpoint_from_worker_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build the endpoint dict passed to charge_serverless_execution."""
    return {
        "endpoint_id": str(row.get("endpoint_id") or ""),
        "owner_id": str(row.get("owner_id") or ""),
        "name": row.get("name"),
        "mode": str(row.get("mode") or ""),
        "model_ref": str(row.get("model_ref") or ""),
        "gpu_tier": row.get("gpu_tier"),
        "region": row.get("region"),
        "gpu_count": int(row.get("gpu_count") or 1),
    }


def endpoint_billing_context(repo: Any, worker: dict[str, Any]) -> dict[str, Any]:
    """Resolve full billing context for any worker charge path (tick, deprovision, final)."""
    ep_id = str(worker.get("endpoint_id") or "")
    ep = repo.get_endpoint(ep_id) if ep_id and hasattr(repo, "get_endpoint") else None
    merged = {**(ep or {}), **worker}
    return endpoint_from_worker_row(merged)