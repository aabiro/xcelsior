"""Platform metadata: attestation schema, ops plan, PEL guardrails."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from host_attestation import (
    attestation_schema,
    h100_partner_pipeline,
    platform_ops_plan,
    scip_alignment_one_pager,
    scip_partner_loi,
)
from routes._deps import _effective_billing_customer_id, _require_auth
from serverless.capacity_planner import demand_breach_alert, forecast_gpu_demand_14d
from serverless.metering import pricing_for_endpoint, token_pricing_quote
from serverless.repo import ServerlessRepo
from serverless.service import ServerlessService, WalletPreflightError

router = APIRouter()


@router.get("/api/v2/platform/attestation-schema", tags=["Platform"])
def api_attestation_schema():
    """Merged host attestation schema (row 8/10)."""
    return {
        "ok": True,
        "schema": attestation_schema(),
        "scip_partner_loi": scip_partner_loi(),
    }


@router.get("/api/v2/platform/scip-loi", tags=["Platform"])
def api_scip_partner_loi():
    """Partner LOI submitted for SCIP / attested-host pipeline (row 8)."""
    loi = scip_partner_loi()
    return {"ok": True, "loi": loi, "submitted": str(loi.get("status") or "").lower() == "submitted"}


@router.get("/api/v2/platform/scip-alignment", tags=["Platform"])
def api_scip_alignment():
    """SCIP alignment one-pager + application scan (row 9)."""
    return {"ok": True, "alignment": scip_alignment_one_pager()}


@router.get("/api/v2/platform/h100-partner", tags=["Platform"])
def api_h100_partner():
    """H100 attested-host partner pipeline (row 10)."""
    return {"ok": True, "pipeline": h100_partner_pipeline()}


@router.get("/api/v2/platform/ops-plan", tags=["Platform"])
def api_platform_ops_plan():
    """18-month platform ops plan (row 9)."""
    return {"ok": True, "plan": platform_ops_plan()}


class ShouldIRunPelRequest(BaseModel):
    endpoint_id: str | None = Field(default=None, max_length=64)
    model_ref: str | None = Field(default=None, max_length=256)
    estimated_input_tokens: int = Field(default=1000, ge=0, le=10_000_000)
    estimated_output_tokens: int = Field(default=500, ge=0, le=10_000_000)
    duration_hours: float = Field(default=0.1, ge=0, le=168)
    gpu_tier: str = Field(default="RTX 4090", max_length=64)


@router.post("/api/v2/serverless/should-i-run-this", tags=["Serverless", "Platform"])
def api_should_i_run_pel_job(body: ShouldIRunPelRequest, request: Request):
    """PEL/admin guardrail: wallet + token/GPU estimate before serverless spend (row 12)."""
    user = _require_auth(request)
    owner_id = _effective_billing_customer_id(user)
    repo = ServerlessRepo()
    reasons: list[str] = []

    ep = None
    if body.endpoint_id:
        ep = repo.get_endpoint(body.endpoint_id)
        if not ep or str(ep.get("owner_id")) != owner_id:
            raise HTTPException(404, "Endpoint not found")
    elif body.model_ref:
        ep = {
            "mode": "preset",
            "model_ref": body.model_ref,
            "gpu_tier": body.gpu_tier,
            "region": "ca-east",
            "gpu_count": 1,
        }

    try:
        ServerlessService.wallet_preflight(owner_id)
    except WalletPreflightError as e:
        reasons.append(e.message)

    from billing import get_billing_engine

    wallet = get_billing_engine().get_wallet(owner_id) or {}
    balance = float(wallet.get("balance_cad") or 0)
    if balance <= 0:
        reasons.append("Wallet balance is zero — add funds before inference")

    token_cost = 0.0
    gpu_cost = 0.0
    if ep:
        quote = pricing_for_endpoint(ep)
        if quote.get("token_billing"):
            from serverless.metering import token_cost_metadata

            meta = token_cost_metadata(
                body.estimated_input_tokens,
                body.estimated_output_tokens,
                model_ref=str(ep.get("model_ref") or ""),
            )
            token_cost = float(meta.get("total_token_cost_cad") or 0)
        rate_hr = float(quote.get("rate_per_hour_cad") or 0)
        gpu_cost = rate_hr * body.duration_hours
    elif body.model_ref:
        tq = token_pricing_quote(body.model_ref)
        in_p = float(tq.get("input_price_cad_per_m") or 0)
        out_p = float(tq.get("output_price_cad_per_m") or 0)
        token_cost = (
            body.estimated_input_tokens * in_p + body.estimated_output_tokens * out_p
        ) / 1_000_000.0

    est_total = max(token_cost, gpu_cost)
    if est_total > balance:
        reasons.append(
            f"Estimated cost ${est_total:.4f} CAD exceeds wallet ${balance:.2f} CAD"
        )

    return {
        "ok": True,
        "approved": len(reasons) == 0,
        "reasons": reasons,
        "wallet_balance_cad": balance,
        "estimated_token_cost_cad": round(token_cost, 6),
        "estimated_gpu_cost_cad": round(gpu_cost, 6),
        "estimated_total_cad": round(est_total, 6),
    }


@router.get("/api/v2/platform/capacity-forecast/{endpoint_id}", tags=["Platform"])
def api_capacity_forecast(endpoint_id: str, request: Request):
    """14-day demand forecast + breach alert for an endpoint (row 11)."""
    user = _require_auth(request)
    repo = ServerlessRepo()
    ep = repo.get_endpoint(endpoint_id)
    if not ep:
        raise HTTPException(404, "Endpoint not found")
    owner = _effective_billing_customer_id(user)
    if str(ep.get("owner_id")) != owner:
        raise HTTPException(403, "Forbidden")

    from serverless.service import record_queue_depth_sample

    depth = repo.queue_depth(endpoint_id)
    samples = record_queue_depth_sample(endpoint_id, depth)
    forecast = forecast_gpu_demand_14d(samples)
    alert = demand_breach_alert(depth, forecast)
    return {
        "ok": True,
        "endpoint_id": endpoint_id,
        "current_queue_depth": depth,
        "forecast": forecast,
        "breach_alert": alert,
    }