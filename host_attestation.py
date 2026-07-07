"""Host hardware attestation schema (TEE evidence, NRAS verify) — admission without API breaks."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("xcelsior.host_attestation")

_COMPLIANCE_DIR = Path(__file__).resolve().parent / "docs" / "compliance"

ATTESTATION_SCHEMA_VERSION = "1.0"
ATTESTED_PREMIUM_MULTIPLIER = 2.5  # commodity ×2–3 for attested H100-class hosts


def attestation_schema() -> dict[str, Any]:
    """Canonical attestation fields merged into host admission (row 8/10)."""
    return {
        "version": ATTESTATION_SCHEMA_VERSION,
        "fields": {
            "tee_evidence_jwt": {
                "type": "string",
                "max_length": 8192,
                "description": "NVIDIA NRAS / nvtrust TEE evidence JWT",
                "required_for_tier": "attested",
            },
            "nras_verify_status": {
                "type": "string",
                "enum": ["unverified", "pending", "verified", "failed"],
                "default": "unverified",
            },
            "nras_measurement_hash": {
                "type": "string",
                "max_length": 128,
                "description": "SHA256 of attestation measurement",
            },
            "attestation_tier": {
                "type": "string",
                "enum": ["commodity", "attested"],
                "default": "commodity",
            },
            "attested_at": {"type": "number", "description": "Unix timestamp when verified"},
        },
        "premium_rate_multiplier": ATTESTED_PREMIUM_MULTIPLIER,
    }


def normalize_attestation(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw or not isinstance(raw, dict):
        return {
            "tee_evidence_jwt": "",
            "nras_verify_status": "unverified",
            "nras_measurement_hash": "",
            "attestation_tier": "commodity",
            "attested_at": 0.0,
        }
    tier = str(raw.get("attestation_tier") or "commodity").strip().lower()
    if tier not in ("commodity", "attested"):
        tier = "commodity"
    status = str(raw.get("nras_verify_status") or "unverified").strip().lower()
    if status not in ("unverified", "pending", "verified", "failed"):
        status = "unverified"
    return {
        "tee_evidence_jwt": str(raw.get("tee_evidence_jwt") or "")[:8192],
        "nras_verify_status": status,
        "nras_measurement_hash": str(raw.get("nras_measurement_hash") or "")[:128],
        "attestation_tier": tier,
        "attested_at": float(raw.get("attested_at") or 0),
    }


def validate_attestation(att: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate attestation payload shape; does not verify JWT crypto (host ops)."""
    reasons: list[str] = []
    norm = normalize_attestation(att)
    if norm["attestation_tier"] == "attested":
        if not norm["tee_evidence_jwt"]:
            reasons.append("attested tier requires tee_evidence_jwt")
        if norm["nras_verify_status"] not in ("verified", "pending"):
            reasons.append("attested tier requires nras_verify_status verified or pending")
    return len(reasons) == 0, reasons


def merge_attestation_into_admission(
    admission_details: dict[str, Any],
    attestation: dict[str, Any] | None,
) -> dict[str, Any]:
    norm = normalize_attestation(attestation)
    valid, reasons = validate_attestation(norm)
    admission_details["attestation"] = norm
    admission_details["attestation_schema_version"] = ATTESTATION_SCHEMA_VERSION
    admission_details["attestation_valid"] = valid
    if reasons:
        admission_details["attestation_reasons"] = reasons
    return admission_details


def scip_partner_loi() -> dict[str, Any]:
    """Partner LOI on file for SCIP / attested-host pipeline (row 8)."""
    path = _COMPLIANCE_DIR / "scip_partner_loi.json"
    if path.is_file():
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("scip_partner_loi read failed: %s", exc)
    return {
        "document_type": "partner_letter_of_intent",
        "status": "draft",
        "submitted_at": 0,
    }


def scip_loi_submitted() -> bool:
    loi = scip_partner_loi()
    return str(loi.get("status") or "").lower() in ("submitted", "accepted")


def scip_alignment_one_pager() -> dict[str, Any]:
    """SCIP alignment scan for Canadian operator + 18-month ops plan (row 9)."""
    plan = platform_ops_plan()
    loi = scip_partner_loi()
    return {
        "canadian_ownership": plan["ownership"],
        "operational_horizon_months": plan["horizon_months"],
        "infrastructure_plan": plan["infrastructure"],
        "milestones": plan["milestones"],
        "partner_loi": {
            "status": loi.get("status"),
            "partner": loi.get("partner"),
            "submitted_at": loi.get("submitted_at"),
        },
        "attestation_schema_version": ATTESTATION_SCHEMA_VERSION,
        "product_focus": "commodity GPU marketplace + token SKU presets; attested tier optional",
        "generated_at": time.time(),
    }


def h100_partner_pipeline() -> dict[str, Any]:
    """Recruit-1-H100-partner pipeline record (row 10)."""
    loi = scip_partner_loi()
    partner = loi.get("partner") if isinstance(loi.get("partner"), dict) else {}
    return {
        "recruitment_status": "loi_submitted" if scip_loi_submitted() else "seeking",
        "target_gpu": "H100",
        "attestation_tier": "attested",
        "premium_rate_multiplier": ATTESTED_PREMIUM_MULTIPLIER,
        "partner": partner,
        "admission_schema": attestation_schema(),
    }


def platform_ops_plan() -> dict[str, Any]:
    """18-month platform ops plan (row 9) — product/ops focus, no residency positioning."""
    return {
        "horizon_months": 18,
        "ownership": {
            "operator": "Xcelsior Inc.",
            "incorporation": "Canadian corporation",
            "business_number_on_file": True,
        },
        "infrastructure": {
            "mesh_hosts": "consumer GPU mesh + partner DC capacity",
            "llm_stack": "vLLM presets with LMCache prefix reuse",
            "checkpoint_resume": "CRIUgpu on capable hosts (driver ≥570)",
            "token_billing": "parallel GPU-seconds + per-token SKU on presets",
        },
        "milestones": [
            {"month": 1, "goal": "Token SKU GA with published SLOs and cached-token pricing"},
            {"month": 3, "goal": "Multi-host LMCache mesh; KV hit-rate monitoring"},
            {"month": 6, "goal": "Attested H100 partner hosts on premium tier"},
            {"month": 12, "goal": "Batch API + semantic cache savings in product UI"},
            {"month": 18, "goal": "Capacity planner auto-scale from 14-day demand forecast"},
        ],
        "compliance_notes": [
            "Supplier attestation bundle available via /api/billing/attestation",
            "Host admission schema version %s" % ATTESTATION_SCHEMA_VERSION,
        ],
        "generated_at": time.time(),
    }