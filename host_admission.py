"""PostgreSQL-authoritative provider host admission.

The trust boundary is deliberately narrow:

* provider helpers and worker agents can submit **advisory** evidence only;
* only authenticated operator/trusted-verifier evidence is authoritative;
* admission is a row-locked, optimistic-concurrency decision that requires
  fresh authoritative compatibility and hardware evidence; and
* the host row, immutable decision, marketplace visibility, and outbox audit
  intent commit atomically.

Compatibility sessions use an ephemeral Ed25519 key plus a short-lived bearer
token.  That proves one helper possessed the session key and prevents replay or
cross-session substitution.  It does not turn provider-controlled hardware
output into authority.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,
)
from psycopg.types.json import Jsonb

from control_plane.db import run_transaction
from control_plane.outbox import append_event
from host_metadata import normalize_host_region
from security import check_node_versions, recommend_runtime

COMPATIBILITY_SCHEMA_VERSION = "host-compatibility.v1"
SESSION_TTL_SECONDS = 15 * 60
ADVISORY_EVIDENCE_TTL_SECONDS = 24 * 60 * 60
MAX_REPORT_BYTES = 256 * 1024
MAX_SUMMARY_BYTES = 32 * 1024

REQUIRED_AUTHORITATIVE_EVIDENCE = (
    "compatibility",
    "hardware_verification",
)
REQUIRED_WIZARD_CHECKS = frozenset(
    {
        "GPU Identity",
        "CUDA Readiness",
        "PCIe Bandwidth",
        "Thermal Stability",
        "Network Quality",
        "Memory Integrity",
        "Security Posture",
    }
)
REQUIRED_VERSION_COMPONENTS = frozenset(
    {"runc", "docker", "nvidia_driver", "nvidia_toolkit"}
)
_SENSITIVE_KEY = re.compile(
    r"(?:secret|token|password|credential|private|serial|uuid|mac(?:_address)?|"
    r"ip(?:_address)?|tee_evidence|jwt)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(
    r"(?:-----BEGIN [A-Z ]+PRIVATE KEY-----|Bearer\s+[A-Za-z0-9._~+/=-]+)",
    re.IGNORECASE,
)


class AdmissionError(RuntimeError):
    """Base class carrying a stable API code and status."""

    code = "admission_error"
    http_status = 400

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})


class AdmissionNotFound(AdmissionError):
    code = "host_not_found"
    http_status = 404


class AdmissionConflict(AdmissionError):
    code = "admission_conflict"
    http_status = 409


class AdmissionPreconditionFailed(AdmissionError):
    code = "admission_precondition_failed"
    http_status = 422


class AdmissionUnauthorized(AdmissionError):
    code = "compatibility_session_unauthorized"
    http_status = 403


class AdmissionConfigurationError(AdmissionError):
    code = "admission_configuration_error"
    http_status = 503


def canonical_json(value: Any) -> bytes:
    """Canonical UTF-8 JSON used for evidence digests and helper signatures."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(value: bytes | str) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def helper_signature_message(
    session_id: str,
    challenge: str,
    report_digest: str,
) -> bytes:
    return (
        "xcelsior-host-compat-v1\n"
        f"{session_id}\n"
        f"{challenge}\n"
        f"{report_digest}"
    ).encode("utf-8")


def _row_value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    return row[index]


def _actor_id(actor: Mapping[str, Any]) -> str:
    for key in ("user_id", "sub", "email", "client_id"):
        value = str(actor.get(key) or "").strip()
        if value:
            prefix = "client:" if key == "client_id" else ""
            return prefix + value
    return "unknown"


def _tenant_from_host(row: Any) -> str:
    payload = _row_value(row, "payload", 3) or {}
    if not isinstance(payload, Mapping):
        payload = {}
    for value in (
        _row_value(row, "tenant_id", 1),
        _row_value(row, "owner_id", 2),
        payload.get("owner"),
        payload.get("provider_id"),
        _row_value(row, "host_id", 0),
    ):
        normalized = str(value or "").strip()
        if normalized:
            return normalized
    return "legacy"


def _compatibility_secret() -> bytes:
    raw = (os.environ.get("XCELSIOR_COMPAT_SESSION_SECRET") or "").strip()
    if raw:
        return raw.encode("utf-8")
    environment = (os.environ.get("XCELSIOR_ENV") or "dev").strip().lower()
    if environment == "production":
        raise AdmissionConfigurationError(
            "XCELSIOR_COMPAT_SESSION_SECRET is required in production"
        )
    return b"xcelsior-development-only-compatibility-secret"


def _derive_session_value(label: str, session_id: str) -> str:
    digest = hmac.new(
        _compatibility_secret(),
        f"{label}:{session_id}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _load_helper_public_key(encoded: str) -> tuple[bytes, Ed25519PublicKey]:
    try:
        der = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise AdmissionPreconditionFailed(
            "helper_public_key_spki must be base64 DER"
        ) from exc
    if not der or len(der) > 512:
        raise AdmissionPreconditionFailed(
            "helper_public_key_spki has an invalid size"
        )
    try:
        public_key = serialization.load_der_public_key(der)
    except (ValueError, TypeError) as exc:
        raise AdmissionPreconditionFailed(
            "helper_public_key_spki is not a valid public key"
        ) from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise AdmissionPreconditionFailed(
            "compatibility helpers must use an Ed25519 public key"
        )
    return der, public_key


def _version_map(report: Mapping[str, Any]) -> dict[str, str]:
    versions = report.get("versions")
    mapped: dict[str, str] = {}
    if isinstance(versions, Mapping):
        for key, value in versions.items():
            if isinstance(value, str) and value.strip():
                mapped[str(key)] = value.strip()
    elif isinstance(versions, list):
        for item in versions[:32]:
            if not isinstance(item, Mapping):
                continue
            component = str(item.get("component") or "").strip()
            version = str(item.get("version") or "").strip()
            if component and version:
                mapped[component] = version
    return mapped


def _check_map(report: Mapping[str, Any]) -> dict[str, bool]:
    checks = report.get("checks")
    out: dict[str, bool] = {}
    if not isinstance(checks, list):
        return out
    for item in checks[:64]:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or item.get("check_name") or "").strip()
        if name:
            out[name] = bool(item.get("passed") is True)
    return out


def _safe_number(value: Any, *, minimum: float = 0, maximum: float = 1e12) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < minimum or number > maximum:
        return 0.0
    return number


def summarize_provider_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce provider output to bounded, non-secret compatibility evidence."""
    encoded = canonical_json(report)
    if len(encoded) > MAX_REPORT_BYTES:
        raise AdmissionPreconditionFailed(
            f"compatibility report exceeds {MAX_REPORT_BYTES} bytes"
        )

    version_map = _version_map(report)
    compatible, version_reasons = check_node_versions(version_map)
    check_map = _check_map(report)
    all_checks_present = REQUIRED_WIZARD_CHECKS.issubset(check_map)
    hardware_passed = all_checks_present and all(
        check_map[name] for name in REQUIRED_WIZARD_CHECKS
    )

    benchmark = report.get("benchmark")
    if not isinstance(benchmark, Mapping):
        benchmark = {}
    network = report.get("network")
    if not isinstance(network, Mapping):
        network = {}

    versions_summary = [
        {
            "component": component,
            "version": version_map.get(component, ""),
            "present": bool(version_map.get(component)),
        }
        for component in sorted(REQUIRED_VERSION_COMPONENTS)
    ]
    checks_summary = [
        {"name": name, "passed": check_map.get(name, False)}
        for name in sorted(REQUIRED_WIZARD_CHECKS)
    ]

    raw_fingerprint = str(report.get("gpu_fingerprint") or "")
    summary = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "compatibility_passed": bool(
            compatible and REQUIRED_VERSION_COMPONENTS.issubset(version_map)
        ),
        "hardware_verification_passed": bool(hardware_passed),
        "versions": versions_summary,
        "version_reasons": [str(reason)[:240] for reason in version_reasons[:16]],
        "checks": checks_summary,
        "gpu": {
            "model": str(benchmark.get("gpu_model") or "")[:80],
            "total_vram_gb": _safe_number(
                benchmark.get("total_vram_gb"), maximum=1024
            ),
            "compute_capability": str(
                benchmark.get("compute_capability") or ""
            )[:24],
            "driver_version": str(benchmark.get("driver_version") or "")[:32],
        },
        "benchmark": {
            "tflops": _safe_number(benchmark.get("tflops"), maximum=1_000_000),
            "pcie_bandwidth_gbps": _safe_number(
                benchmark.get("pcie_bandwidth_gbps"), maximum=10_000
            ),
            "gpu_temp_celsius": _safe_number(
                benchmark.get("gpu_temp_celsius"), maximum=200
            ),
            "elapsed_s": _safe_number(benchmark.get("elapsed_s"), maximum=86_400),
        },
        "network": {
            "jitter_ms": _safe_number(network.get("jitter_ms"), maximum=60_000),
            "packet_loss_pct": _safe_number(
                network.get("packet_loss_pct"), maximum=100
            ),
            "throughput_mbps": _safe_number(
                network.get("throughput_mbps"), maximum=10_000_000
            ),
        },
        # The raw helper fingerprint may contain a GPU UUID/serial.  Only its
        # one-way hash crosses the control-plane boundary.
        "provider_fingerprint_hash": (
            sha256_hex(raw_fingerprint) if raw_fingerprint else ""
        ),
    }
    if len(canonical_json(summary)) > MAX_SUMMARY_BYTES:
        raise AdmissionPreconditionFailed("sanitized evidence summary is too large")
    return summary


def sanitize_operator_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bound and redact operator notes before durable/audit storage."""

    def clean(item: Any, *, depth: int = 0) -> Any:
        if depth > 5:
            return "<depth-limit>"
        if item is None or isinstance(item, (bool, int)):
            return item
        if isinstance(item, float):
            if item != item or item in (float("inf"), float("-inf")):
                return None
            return item
        if isinstance(item, str):
            text = item[:1000]
            return "<redacted>" if _SENSITIVE_VALUE.search(text) else text
        if isinstance(item, Mapping):
            out: dict[str, Any] = {}
            for key, child in list(item.items())[:100]:
                normalized = str(key)[:80]
                if _SENSITIVE_KEY.search(normalized):
                    out[normalized] = "<redacted>"
                else:
                    out[normalized] = clean(child, depth=depth + 1)
            return out
        if isinstance(item, (list, tuple)):
            return [clean(child, depth=depth + 1) for child in list(item)[:100]]
        return str(item)[:500]

    cleaned = clean(value)
    assert isinstance(cleaned, dict)
    if len(canonical_json(cleaned)) > MAX_SUMMARY_BYTES:
        raise AdmissionPreconditionFailed(
            f"operator evidence summary exceeds {MAX_SUMMARY_BYTES} bytes"
        )
    return cleaned


def create_compatibility_session(
    *,
    host_id: str,
    actor: Mapping[str, Any],
    helper_public_key_spki: str,
    idempotency_key: str,
    ttl_seconds: int = SESSION_TTL_SECONDS,
) -> dict[str, Any]:
    """Create or replay an authenticated, expiring helper session."""
    host_id = str(host_id or "").strip()
    key = str(idempotency_key or "").strip()
    if not host_id:
        raise AdmissionPreconditionFailed("host_id is required")
    if not key or len(key) > 200:
        raise AdmissionPreconditionFailed(
            "idempotency_key is required and must be at most 200 characters"
        )
    ttl = max(60, min(int(ttl_seconds), 3600))
    public_der, _public_key = _load_helper_public_key(helper_public_key_spki)
    key_fingerprint = sha256_hex(public_der)
    actor_id = _actor_id(actor)

    def txn(conn: Any) -> dict[str, Any]:
        host = conn.execute(
            """
            SELECT host_id, tenant_id, owner_id, payload
              FROM hosts
             WHERE host_id = %s
            """,
            (host_id,),
        ).fetchone()
        if host is None:
            raise AdmissionNotFound(f"Host {host_id!r} was not found")
        tenant_id = _tenant_from_host(host)

        existing = conn.execute(
            """
            SELECT session_id, requested_by, helper_key_fingerprint,
                   state, expires_at
              FROM host_compatibility_sessions
             WHERE host_id = %s AND idempotency_key = %s
            """,
            (host_id, key),
        ).fetchone()
        reused = existing is not None
        if existing is not None:
            if (
                str(_row_value(existing, "requested_by", 1)) != actor_id
                or str(_row_value(existing, "helper_key_fingerprint", 2))
                != key_fingerprint
            ):
                raise AdmissionConflict(
                    "idempotency_key was already used with different session inputs"
                )
            session_id = str(_row_value(existing, "session_id", 0))
            state = str(_row_value(existing, "state", 3))
            expires_at = _row_value(existing, "expires_at", 4)
        else:
            session_id = str(uuid.uuid4())
            expires_at = datetime.now(UTC) + timedelta(seconds=ttl)
            state = "created"
            token = _derive_session_value("token", session_id)
            challenge = _derive_session_value("challenge", session_id)
            conn.execute(
                """
                INSERT INTO host_compatibility_sessions (
                    session_id, tenant_id, host_id, requested_by,
                    idempotency_key, helper_public_key_spki,
                    helper_key_fingerprint, token_hash, challenge_hash,
                    expires_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    session_id,
                    tenant_id,
                    host_id,
                    actor_id,
                    key,
                    public_der,
                    key_fingerprint,
                    sha256_hex(token),
                    sha256_hex(challenge),
                    expires_at,
                ),
            )

        return {
            "session_id": session_id,
            "host_id": host_id,
            "state": state,
            "expires_at": expires_at,
            "submit_token": _derive_session_value("token", session_id),
            "challenge": _derive_session_value("challenge", session_id),
            "helper_key_fingerprint": key_fingerprint,
            "reused": reused,
            "authority": "advisory_only",
        }

    return run_transaction(txn, what="host_compatibility_session_create")


def _insert_evidence(
    conn: Any,
    *,
    tenant_id: str,
    host_id: str,
    session_id: str | None,
    evidence_type: str,
    source_type: str,
    trust_level: str,
    verdict: str,
    summary: Mapping[str, Any],
    idempotency_key: str,
    verifier_principal: str,
    observed_at: datetime,
    expires_at: datetime,
) -> str:
    digest = sha256_hex(canonical_json(summary))
    row = conn.execute(
        """
        INSERT INTO host_admission_evidence (
            tenant_id, host_id, session_id, evidence_type, source_type,
            trust_level, verdict, schema_version, evidence_digest,
            idempotency_key, verifier_principal, summary, observed_at,
            expires_at
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (host_id, source_type, idempotency_key) DO NOTHING
        RETURNING evidence_id
        """,
        (
            tenant_id,
            host_id,
            session_id,
            evidence_type,
            source_type,
            trust_level,
            verdict,
            COMPATIBILITY_SCHEMA_VERSION,
            digest,
            idempotency_key,
            verifier_principal,
            Jsonb(dict(summary)),
            observed_at,
            expires_at,
        ),
    ).fetchone()
    if row is not None:
        return str(_row_value(row, "evidence_id", 0))

    existing = conn.execute(
        """
        SELECT evidence_id, evidence_digest, evidence_type, verdict
          FROM host_admission_evidence
         WHERE host_id = %s
           AND source_type = %s
           AND idempotency_key = %s
        """,
        (host_id, source_type, idempotency_key),
    ).fetchone()
    if existing is None:
        raise AdmissionConflict("evidence idempotency conflict could not be resolved")
    if (
        str(_row_value(existing, "evidence_digest", 1)) != digest
        or str(_row_value(existing, "evidence_type", 2)) != evidence_type
        or str(_row_value(existing, "verdict", 3)) != verdict
    ):
        raise AdmissionConflict(
            "idempotency_key was already used with different evidence"
        )
    return str(_row_value(existing, "evidence_id", 0))


def submit_compatibility_evidence(
    *,
    session_id: str,
    actor: Mapping[str, Any],
    submit_token: str,
    report: Mapping[str, Any],
    signature: str,
) -> dict[str, Any]:
    """Verify a helper proof and atomically record advisory evidence."""
    session_id = str(session_id or "").strip()
    actor_id = _actor_id(actor)
    report_digest = sha256_hex(canonical_json(report))
    summary = summarize_provider_report(report)

    try:
        signature_bytes = base64.b64decode(signature, validate=True)
    except (ValueError, TypeError) as exc:
        raise AdmissionUnauthorized("compatibility signature is invalid") from exc

    def txn(conn: Any) -> dict[str, Any]:
        row = conn.execute(
            """
            SELECT s.host_id, s.tenant_id, s.requested_by,
                   s.helper_public_key_spki, s.token_hash, s.challenge_hash,
                   s.state, s.report_digest, s.expires_at,
                   h.admission_state
              FROM host_compatibility_sessions s
              JOIN hosts h ON h.host_id = s.host_id
             WHERE s.session_id = %s
             FOR UPDATE OF s
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise AdmissionNotFound("Compatibility session was not found")

        host_id = str(_row_value(row, "host_id", 0))
        tenant_id = str(_row_value(row, "tenant_id", 1))
        requested_by = str(_row_value(row, "requested_by", 2))
        public_der = bytes(_row_value(row, "helper_public_key_spki", 3))
        token_hash = str(_row_value(row, "token_hash", 4))
        challenge_hash = str(_row_value(row, "challenge_hash", 5))
        state = str(_row_value(row, "state", 6))
        prior_digest = str(_row_value(row, "report_digest", 7) or "")
        expires_at = _row_value(row, "expires_at", 8)
        admission_state = str(_row_value(row, "admission_state", 9))

        if requested_by != actor_id:
            raise AdmissionUnauthorized(
                "Compatibility session belongs to a different principal"
            )
        if not hmac.compare_digest(token_hash, sha256_hex(submit_token)):
            raise AdmissionUnauthorized("Compatibility session token is invalid")
        challenge = _derive_session_value("challenge", session_id)
        if not hmac.compare_digest(challenge_hash, sha256_hex(challenge)):
            raise AdmissionUnauthorized("Compatibility session challenge is invalid")
        now = datetime.now(UTC)
        if expires_at is None or expires_at <= now:
            raise AdmissionConflict("Compatibility session has expired")

        if state == "consumed":
            if prior_digest != report_digest:
                raise AdmissionConflict(
                    "Consumed compatibility session cannot accept different evidence"
                )
            existing = conn.execute(
                """
                SELECT evidence_id, evidence_type, verdict
                  FROM host_admission_evidence
                 WHERE session_id = %s
                 ORDER BY evidence_type
                """,
                (session_id,),
            ).fetchall()
            return {
                "session_id": session_id,
                "host_id": host_id,
                "state": "advisory_evidence_recorded",
                "admission_state": admission_state,
                "admitted": admission_state == "admitted",
                "evidence": [
                    {
                        "evidence_id": str(_row_value(item, "evidence_id", 0)),
                        "evidence_type": str(
                            _row_value(item, "evidence_type", 1)
                        ),
                        "verdict": str(_row_value(item, "verdict", 2)),
                        "trust_level": "advisory",
                    }
                    for item in existing
                ],
                "reused": True,
            }
        if state != "created":
            raise AdmissionConflict(
                f"Compatibility session cannot be consumed from state {state!r}"
            )

        public_key = serialization.load_der_public_key(public_der)
        if not isinstance(public_key, Ed25519PublicKey):
            raise AdmissionUnauthorized("Compatibility helper key is invalid")
        try:
            public_key.verify(
                signature_bytes,
                helper_signature_message(session_id, challenge, report_digest),
            )
        except Exception as exc:
            raise AdmissionUnauthorized(
                "Compatibility helper proof-of-possession failed"
            ) from exc

        evidence_expiry = now + timedelta(seconds=ADVISORY_EVIDENCE_TTL_SECONDS)
        compatibility_verdict = (
            "pass" if summary["compatibility_passed"] else "fail"
        )
        hardware_verdict = (
            "pass" if summary["hardware_verification_passed"] else "fail"
        )
        evidence = []
        for evidence_type, verdict in (
            ("compatibility", compatibility_verdict),
            ("hardware_verification", hardware_verdict),
        ):
            evidence_id = _insert_evidence(
                conn,
                tenant_id=tenant_id,
                host_id=host_id,
                session_id=session_id,
                evidence_type=evidence_type,
                source_type="provider_helper",
                trust_level="advisory",
                verdict=verdict,
                summary=summary,
                idempotency_key=f"session:{session_id}:{evidence_type}",
                verifier_principal=f"helper:{sha256_hex(public_der)[:16]}",
                observed_at=now,
                expires_at=evidence_expiry,
            )
            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "evidence_type": evidence_type,
                    "verdict": verdict,
                    "trust_level": "advisory",
                }
            )

        conn.execute(
            """
            UPDATE host_compatibility_sessions
               SET state = 'consumed',
                   report_digest = %s,
                   consumed_at = clock_timestamp()
             WHERE session_id = %s AND state = 'created'
            """,
            (report_digest, session_id),
        )
        append_event(
            conn,
            aggregate_type="host",
            aggregate_id=host_id,
            event_type="host.v1.condition_changed",
            payload={"host_id": host_id},
            idempotency_key=f"host_compatibility:{session_id}",
        )
        return {
            "session_id": session_id,
            "host_id": host_id,
            "state": "advisory_evidence_recorded",
            "admission_state": admission_state,
            "admitted": admission_state == "admitted",
            "evidence": evidence,
            "reused": False,
            "next_step": (
                "Operator/trusted-verifier review is required before admission"
            ),
            "recommended_runtime": recommend_runtime(
                str(summary.get("gpu", {}).get("model") or "unknown")
            )[0],
        }

    return run_transaction(txn, what="host_compatibility_evidence_submit")


def record_provider_agent_evidence(
    *,
    host_id: str,
    actor: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility bridge for old workers; always advisory."""
    summary = summarize_provider_report(report)
    digest = sha256_hex(canonical_json(report))
    actor_id = _actor_id(actor)

    def txn(conn: Any) -> dict[str, Any]:
        host = conn.execute(
            """
            SELECT host_id, tenant_id, owner_id, payload, admission_state
              FROM hosts
             WHERE host_id = %s
            """,
            (host_id,),
        ).fetchone()
        if host is None:
            raise AdmissionNotFound(f"Host {host_id!r} was not found")
        tenant_id = _tenant_from_host(host)
        now = datetime.now(UTC)
        evidence = []
        for evidence_type, passed in (
            ("compatibility", summary["compatibility_passed"]),
            (
                "hardware_verification",
                summary["hardware_verification_passed"],
            ),
        ):
            evidence_id = _insert_evidence(
                conn,
                tenant_id=tenant_id,
                host_id=host_id,
                session_id=None,
                evidence_type=evidence_type,
                source_type="provider_agent",
                trust_level="advisory",
                verdict="pass" if passed else "fail",
                summary=summary,
                idempotency_key=f"agent:{digest}:{evidence_type}",
                verifier_principal=f"provider-agent:{actor_id}",
                observed_at=now,
                expires_at=now + timedelta(seconds=ADVISORY_EVIDENCE_TTL_SECONDS),
            )
            evidence.append(evidence_id)
        return {
            "host_id": host_id,
            "state": "advisory_evidence_recorded",
            "admission_state": str(_row_value(host, "admission_state", 4)),
            "admitted": False,
            "evidence_ids": evidence,
        }

    return run_transaction(txn, what="host_agent_advisory_evidence")


def record_authoritative_evidence(
    *,
    host_id: str,
    actor: Mapping[str, Any],
    evidence_type: str,
    verdict: str,
    summary: Mapping[str, Any],
    idempotency_key: str,
    validity_seconds: int = 24 * 60 * 60,
    source_type: str = "operator",
) -> dict[str, Any]:
    """Record immutable operator/trusted-verifier evidence."""
    if evidence_type not in {
        "compatibility",
        "hardware_verification",
        "identity",
        "runtime",
        "network",
        "storage",
        "operator_review",
    }:
        raise AdmissionPreconditionFailed("Unsupported evidence_type")
    if verdict not in {"pass", "fail", "inconclusive"}:
        raise AdmissionPreconditionFailed("Unsupported evidence verdict")
    if source_type not in {"operator", "trusted_verifier"}:
        raise AdmissionPreconditionFailed("Authoritative source_type is invalid")
    key = str(idempotency_key or "").strip()
    if not key or len(key) > 200:
        raise AdmissionPreconditionFailed(
            "idempotency_key is required and must be at most 200 characters"
        )
    clean_summary = sanitize_operator_summary(summary)
    validity = max(60, min(int(validity_seconds), 30 * 24 * 60 * 60))
    actor_id = _actor_id(actor)

    def txn(conn: Any) -> dict[str, Any]:
        host = conn.execute(
            """
            SELECT host_id, tenant_id, owner_id, payload
              FROM hosts
             WHERE host_id = %s
             FOR SHARE
            """,
            (host_id,),
        ).fetchone()
        if host is None:
            raise AdmissionNotFound(f"Host {host_id!r} was not found")
        tenant_id = _tenant_from_host(host)
        now = datetime.now(UTC)
        evidence_id = _insert_evidence(
            conn,
            tenant_id=tenant_id,
            host_id=host_id,
            session_id=None,
            evidence_type=evidence_type,
            source_type=source_type,
            trust_level="authoritative",
            verdict=verdict,
            summary=clean_summary,
            idempotency_key=key,
            verifier_principal=f"{source_type}:{actor_id}",
            observed_at=now,
            expires_at=now + timedelta(seconds=validity),
        )
        append_event(
            conn,
            aggregate_type="host",
            aggregate_id=host_id,
            event_type="host.v1.condition_changed",
            payload={"host_id": host_id},
            idempotency_key=f"host_evidence:{host_id}:{source_type}:{key}",
        )
        return {
            "evidence_id": evidence_id,
            "host_id": host_id,
            "evidence_type": evidence_type,
            "source_type": source_type,
            "trust_level": "authoritative",
            "verdict": verdict,
            "observed_at": now,
            "expires_at": now + timedelta(seconds=validity),
        }

    return run_transaction(txn, what="host_authoritative_evidence")


def _latest_authoritative_evidence(
    conn: Any,
    host_id: str,
) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT DISTINCT ON (evidence_type)
               evidence_id, evidence_type, verdict, source_type,
               verifier_principal, observed_at, expires_at, summary
          FROM host_admission_evidence
         WHERE host_id = %s
           AND trust_level = 'authoritative'
           AND superseded_at IS NULL
         ORDER BY evidence_type, observed_at DESC, created_at DESC
        """,
        (host_id,),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        out[str(_row_value(row, "evidence_type", 1))] = {
            "evidence_id": str(_row_value(row, "evidence_id", 0)),
            "evidence_type": str(_row_value(row, "evidence_type", 1)),
            "verdict": str(_row_value(row, "verdict", 2)),
            "source_type": str(_row_value(row, "source_type", 3)),
            "verifier_principal": str(
                _row_value(row, "verifier_principal", 4)
            ),
            "observed_at": _row_value(row, "observed_at", 5),
            "expires_at": _row_value(row, "expires_at", 6),
            "summary": _row_value(row, "summary", 7) or {},
        }
    return out


def evidence_preconditions(
    latest: Mapping[str, Mapping[str, Any]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Pure admission gate over latest authoritative evidence."""
    current = now or datetime.now(UTC)
    missing: list[str] = []
    failed: list[str] = []
    expired: list[str] = []
    evidence_ids: list[str] = []
    for evidence_type in REQUIRED_AUTHORITATIVE_EVIDENCE:
        item = latest.get(evidence_type)
        if not item:
            missing.append(evidence_type)
            continue
        expiry = item.get("expires_at")
        if expiry is None or expiry <= current:
            expired.append(evidence_type)
            continue
        if item.get("verdict") != "pass":
            failed.append(evidence_type)
            continue
        evidence_ids.append(str(item.get("evidence_id") or ""))
    return {
        "ready": not (missing or failed or expired),
        "required": list(REQUIRED_AUTHORITATIVE_EVIDENCE),
        "missing": missing,
        "failed": failed,
        "expired": expired,
        "evidence_ids": [item for item in evidence_ids if item],
    }


def _sync_marketplace(
    conn: Any,
    *,
    host_id: str,
    payload: Mapping[str, Any],
    admitted: bool,
    host_status: str,
) -> str:
    # Disable every stale offer first.  Admission and listing visibility then
    # cannot diverge on a transaction failure.
    conn.execute(
        """
        UPDATE gpu_offers
           SET available = FALSE,
               updated_at = extract(epoch FROM clock_timestamp())
         WHERE host_id = %s AND available = TRUE
        """,
        (host_id,),
    )
    if not admitted or host_status != "active":
        return "disabled"

    gpu_model = str(payload.get("gpu_model") or "").strip()
    provider_id = str(
        payload.get("owner") or payload.get("provider_id") or host_id
    ).strip()
    vram_gb = max(
        0,
        int(
            round(
                _safe_number(
                    payload.get("total_vram_gb") or payload.get("vram_gb"),
                    maximum=1024,
                )
            )
        ),
    )
    ask_cents = max(
        1,
        int(
            round(
                _safe_number(payload.get("cost_per_hour"), maximum=1000)
                * 100
            )
        ),
    )
    gpu_count = max(1, min(int(payload.get("gpu_count") or 1), 64))
    if not gpu_model or vram_gb <= 0:
        raise AdmissionPreconditionFailed(
            "Admitted host requires normalized GPU model and VRAM before listing"
        )

    existing = conn.execute(
        """
        SELECT offer_id
          FROM gpu_offers
         WHERE host_id = %s AND gpu_model = %s
         ORDER BY updated_at DESC
         LIMIT 1
         FOR UPDATE
        """,
        (host_id, gpu_model),
    ).fetchone()
    now_epoch = datetime.now(UTC).timestamp()
    region = normalize_host_region(dict(payload))
    province = str(payload.get("province") or "").upper()
    spot_enabled = bool(payload.get("spot_enabled", True))
    spot_min_cents = max(0, int(payload.get("spot_min_cents") or 0))
    if existing:
        offer_id = str(_row_value(existing, "offer_id", 0))
        conn.execute(
            """
            UPDATE gpu_offers
               SET provider_id = %s,
                   gpu_count_total = %s,
                   gpu_count_available = %s,
                   vram_gb = %s,
                   ask_cents_per_hour = %s,
                   spot_enabled = %s,
                   spot_min_cents = %s,
                   region = %s,
                   province = %s,
                   available = TRUE,
                   updated_at = %s
             WHERE offer_id = %s
            """,
            (
                provider_id,
                gpu_count,
                gpu_count,
                vram_gb,
                ask_cents,
                spot_enabled,
                spot_min_cents,
                region,
                province,
                now_epoch,
                offer_id,
            ),
        )
    else:
        offer_id = f"offer-{uuid.uuid4().hex[:12]}"
        conn.execute(
            """
            INSERT INTO gpu_offers (
                offer_id, provider_id, host_id, gpu_model,
                gpu_count_total, gpu_count_available, vram_gb,
                ask_cents_per_hour, spot_multiplier, spot_enabled,
                spot_min_cents, currency, region, province, available,
                created_at, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, 0.6, %s, %s,
                'CAD', %s, %s, TRUE, %s, %s
            )
            """,
            (
                offer_id,
                provider_id,
                host_id,
                gpu_model,
                gpu_count,
                gpu_count,
                vram_gb,
                ask_cents,
                spot_enabled,
                spot_min_cents,
                region,
                province,
                now_epoch,
                now_epoch,
            ),
        )
    return offer_id


def decide_admission(
    *,
    host_id: str,
    actor: Mapping[str, Any],
    action: str,
    reason: str,
    idempotency_key: str,
    expected_version: int | None,
) -> dict[str, Any]:
    """Apply one idempotent, optimistic-concurrency operator decision."""
    action = str(action or "").strip().lower()
    if action not in {"admit", "reject", "revoke"}:
        raise AdmissionPreconditionFailed("action must be admit, reject, or revoke")
    reason = str(reason or "").strip()
    if len(reason) < 3 or len(reason) > 1000:
        raise AdmissionPreconditionFailed(
            "reason must contain between 3 and 1000 characters"
        )
    key = str(idempotency_key or "").strip()
    if not key or len(key) > 200:
        raise AdmissionPreconditionFailed(
            "idempotency_key is required and must be at most 200 characters"
        )
    actor_id = _actor_id(actor)

    def txn(conn: Any) -> dict[str, Any]:
        host = conn.execute(
            """
            SELECT host_id, tenant_id, owner_id, payload, status,
                   admission_state, admission_version
              FROM hosts
             WHERE host_id = %s
             FOR UPDATE
            """,
            (host_id,),
        ).fetchone()
        if host is None:
            raise AdmissionNotFound(f"Host {host_id!r} was not found")

        duplicate = conn.execute(
            """
            SELECT decision_id, decision_version, action, reason,
                   resulting_state, evidence_ids
              FROM host_admission_decisions
             WHERE host_id = %s AND idempotency_key = %s
            """,
            (host_id, key),
        ).fetchone()
        if duplicate is not None:
            if (
                str(_row_value(duplicate, "action", 2)) != action
                or str(_row_value(duplicate, "reason", 3)) != reason
            ):
                raise AdmissionConflict(
                    "idempotency_key was already used for a different decision"
                )
            resulting = str(_row_value(duplicate, "resulting_state", 4))
            return {
                "decision_id": str(_row_value(duplicate, "decision_id", 0)),
                "host_id": host_id,
                "decision_version": int(
                    _row_value(duplicate, "decision_version", 1)
                ),
                "action": action,
                "admission_state": resulting,
                "admitted": resulting == "admitted",
                "evidence_ids": list(
                    _row_value(duplicate, "evidence_ids", 5) or []
                ),
                "reused": True,
            }

        current_state = str(_row_value(host, "admission_state", 5) or "pending")
        current_version = int(_row_value(host, "admission_version", 6) or 0)
        if expected_version is None:
            raise AdmissionPreconditionFailed(
                "expected_version is required for an admission decision"
            )
        if int(expected_version) != current_version:
            raise AdmissionConflict(
                "Host admission version changed",
                details={
                    "expected_version": int(expected_version),
                    "current_version": current_version,
                    "current_state": current_state,
                },
            )

        latest = _latest_authoritative_evidence(conn, host_id)
        gate = evidence_preconditions(latest)
        evidence_ids: list[str] = []
        if action == "admit":
            if not gate["ready"]:
                raise AdmissionPreconditionFailed(
                    "Fresh authoritative compatibility and hardware evidence "
                    "must pass before admission",
                    details=gate,
                )
            resulting_state = "admitted"
            evidence_ids = list(gate["evidence_ids"])
        elif action == "reject":
            if current_state == "admitted":
                raise AdmissionPreconditionFailed(
                    "Use revoke to remove authority from an admitted host"
                )
            resulting_state = "rejected"
            evidence_ids = [
                str(item["evidence_id"])
                for item in latest.values()
                if item.get("evidence_id")
            ]
        else:
            if current_state != "admitted":
                raise AdmissionPreconditionFailed(
                    "Only an admitted host can be revoked"
                )
            resulting_state = "revoked"
            evidence_ids = [
                str(item["evidence_id"])
                for item in latest.values()
                if item.get("evidence_id")
            ]

        tenant_id = _tenant_from_host(host)
        payload = _row_value(host, "payload", 3) or {}
        if not isinstance(payload, Mapping):
            payload = {}
        payload = dict(payload)
        old_status = str(_row_value(host, "status", 4) or "pending")
        if resulting_state == "admitted":
            new_status = (
                old_status if old_status in {"dead", "draining"} else "active"
            )
        elif resulting_state == "revoked":
            new_status = "disabled"
        else:
            new_status = old_status if old_status == "dead" else "pending"

        new_version = current_version + 1
        decision_id = str(uuid.uuid4())
        payload.update(
            {
                "admitted": resulting_state == "admitted",
                "admission_state": resulting_state,
                "admission_version": new_version,
                "admission_decision_id": decision_id,
                "admission_details": {
                    "decision_id": decision_id,
                    "decision_version": new_version,
                    "action": action,
                    "reason": reason,
                    "evidence_ids": evidence_ids,
                    "decided_at": datetime.now(UTC).isoformat(),
                },
                "status": new_status,
            }
        )
        if action == "admit":
            compatibility = latest.get("compatibility", {})
            runtime = (
                compatibility.get("summary", {}).get("recommended_runtime")
                if isinstance(compatibility.get("summary"), Mapping)
                else None
            )
            if runtime:
                payload["recommended_runtime"] = str(runtime)[:32]

        conn.execute(
            """
            INSERT INTO host_admission_decisions (
                decision_id, tenant_id, host_id, decision_version,
                action, previous_state, resulting_state, actor_principal,
                reason, evidence_ids, idempotency_key
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                decision_id,
                tenant_id,
                host_id,
                new_version,
                action,
                current_state,
                resulting_state,
                f"operator:{actor_id}",
                reason,
                Jsonb(evidence_ids),
                key,
            ),
        )
        conn.execute(
            """
            UPDATE hosts
               SET admission_state = %s,
                   admission_version = %s,
                   admitted_at = CASE
                       WHEN %s = 'admitted' THEN clock_timestamp()
                       ELSE admitted_at
                   END,
                   admission_decision_id = %s,
                   status = %s,
                   payload = %s,
                   version = version + 1
             WHERE host_id = %s
            """,
            (
                resulting_state,
                new_version,
                resulting_state,
                decision_id,
                new_status,
                Jsonb(payload),
                host_id,
            ),
        )
        marketplace = _sync_marketplace(
            conn,
            host_id=host_id,
            payload=payload,
            admitted=resulting_state == "admitted",
            host_status=new_status,
        )
        append_event(
            conn,
            aggregate_type="host",
            aggregate_id=host_id,
            aggregate_version=new_version,
            event_type="host.v1.condition_changed",
            payload={"host_id": host_id},
            idempotency_key=f"host_admission:{host_id}:{key}",
        )
        return {
            "decision_id": decision_id,
            "host_id": host_id,
            "decision_version": new_version,
            "action": action,
            "admission_state": resulting_state,
            "admitted": resulting_state == "admitted",
            "status": new_status,
            "evidence_ids": evidence_ids,
            "marketplace": marketplace,
            "reused": False,
        }

    return run_transaction(txn, what="host_admission_decision")


def admission_status(host_id: str) -> dict[str, Any]:
    """Return the workflow state without exposing session secrets/raw evidence."""

    def txn(conn: Any) -> dict[str, Any]:
        host = conn.execute(
            """
            SELECT host_id, status, admission_state, admission_version,
                   admitted_at, admission_decision_id
              FROM hosts
             WHERE host_id = %s
            """,
            (host_id,),
        ).fetchone()
        if host is None:
            raise AdmissionNotFound(f"Host {host_id!r} was not found")
        latest = _latest_authoritative_evidence(conn, host_id)
        gate = evidence_preconditions(latest)
        advisory_rows = conn.execute(
            """
            SELECT DISTINCT ON (evidence_type)
                   evidence_id, evidence_type, verdict, source_type,
                   observed_at, expires_at
              FROM host_admission_evidence
             WHERE host_id = %s
               AND trust_level = 'advisory'
               AND superseded_at IS NULL
             ORDER BY evidence_type, observed_at DESC, created_at DESC
            """,
            (host_id,),
        ).fetchall()
        return {
            "host_id": host_id,
            "status": str(_row_value(host, "status", 1)),
            "admission_state": str(_row_value(host, "admission_state", 2)),
            "admission_version": int(
                _row_value(host, "admission_version", 3) or 0
            ),
            "admitted_at": _row_value(host, "admitted_at", 4),
            "decision_id": (
                str(_row_value(host, "admission_decision_id", 5))
                if _row_value(host, "admission_decision_id", 5)
                else None
            ),
            "authoritative_gate": gate,
            "authoritative_evidence": {
                key: {
                    "evidence_id": item["evidence_id"],
                    "verdict": item["verdict"],
                    "source_type": item["source_type"],
                    "verifier_principal": item["verifier_principal"],
                    "observed_at": item["observed_at"],
                    "expires_at": item["expires_at"],
                }
                for key, item in latest.items()
            },
            "advisory_evidence": [
                {
                    "evidence_id": str(
                        _row_value(row, "evidence_id", 0)
                    ),
                    "evidence_type": str(
                        _row_value(row, "evidence_type", 1)
                    ),
                    "verdict": str(_row_value(row, "verdict", 2)),
                    "source_type": str(_row_value(row, "source_type", 3)),
                    "observed_at": _row_value(row, "observed_at", 4),
                    "expires_at": _row_value(row, "expires_at", 5),
                }
                for row in advisory_rows
            ],
        }

    return run_transaction(txn, what="host_admission_status")


def admission_queue(limit: int = 100) -> list[dict[str, Any]]:
    """Operator queue of pending/rejected/revoked hosts and evidence readiness."""
    bounded = max(1, min(int(limit), 500))

    def txn(conn: Any) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT host_id
              FROM hosts
             WHERE admission_state <> 'admitted'
             ORDER BY registered_at ASC, host_id ASC
             LIMIT %s
            """,
            (bounded,),
        ).fetchall()
        return [admission_status(str(_row_value(row, "host_id", 0))) for row in rows]

    # Do not nest pooled transactions by calling admission_status from inside
    # one.  Fetch ids first, then take short independent read transactions.
    host_ids = run_transaction(
        lambda conn: [
            str(_row_value(row, "host_id", 0))
            for row in conn.execute(
                """
                SELECT host_id
                  FROM hosts
                 WHERE admission_state <> 'admitted'
                 ORDER BY registered_at ASC, host_id ASC
                 LIMIT %s
                """,
                (bounded,),
            ).fetchall()
        ],
        what="host_admission_queue_ids",
    )
    return [admission_status(host_id) for host_id in host_ids]
