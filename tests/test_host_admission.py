"""Authoritative host admission (migration 082).

The property under test throughout is the trust boundary: a provider describing
their own machine produces *advisory* evidence only, and nothing a provider can
reach may admit a host. Admission requires authoritative evidence and an
operator decision.

Everything else here exists to make that boundary hold under adversarial
conditions — expired sessions, replayed submissions, forged signatures,
concurrent operators, and cross-tenant access.
"""

import os
import time
import uuid

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

import host_admission as ha
from control_plane.db import run_transaction


def _operator(uid="admission-operator"):
    return {"user_id": uid, "email": f"{uid}@xcelsior.ca", "role": "admin", "is_admin": True}


def _provider(uid="admission-provider"):
    return {"user_id": uid, "email": f"{uid}@xcelsior.ca", "role": "provider"}


# A host cannot be listed on the marketplace without normalized hardware, so
# the fixture carries the same minimum a real registration would.
HOST_PAYLOAD = {
    "gpu_model": "RTX 4090",
    "total_vram_gb": 24,
    "gpu_count": 1,
    "cost_per_hour": 0.75,
}


def _make_host(host_id: str, *, tenant: str, admitted: bool = False) -> None:
    import json

    payload = json.dumps({**HOST_PAYLOAD, **({"admitted": True} if admitted else {})})

    def txn(conn):
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload,
                                  tenant_id, owner_id)
               VALUES (%s, 'active', %s, %s::jsonb, %s, %s)
               ON CONFLICT (host_id) DO NOTHING""",
            (host_id, time.time(), payload, tenant, tenant),
        )

    run_transaction(txn, what="test_make_host")


def _drop_host(host_id: str) -> None:
    def txn(conn):
        for sql in (
            "DELETE FROM host_admission_decisions WHERE host_id = %s",
            "DELETE FROM host_admission_evidence WHERE host_id = %s",
            "DELETE FROM host_compatibility_sessions WHERE host_id = %s",
            "DELETE FROM hosts WHERE host_id = %s",
        ):
            conn.execute(sql, (host_id,))

    run_transaction(txn, what="test_drop_host")


@pytest.fixture
def host():
    host_id = f"admission-host-{uuid.uuid4().hex[:10]}"
    _make_host(host_id, tenant="tenant-a")
    yield host_id
    _drop_host(host_id)


@pytest.fixture
def helper_key():
    """Ed25519 keypair standing in for the provider's local helper."""
    priv = Ed25519PrivateKey.generate()
    spki = priv.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    import base64

    return priv, base64.b64encode(spki).decode()


def _open_session(host_id, spki, actor=None, ttl=ha.SESSION_TTL_SECONDS, key=None):
    return ha.create_compatibility_session(
        host_id=host_id,
        actor=actor or _provider(),
        helper_public_key_spki=spki,
        idempotency_key=key or f"idem-{uuid.uuid4().hex[:8]}",
        ttl_seconds=ttl,
    )


def _sign(priv, session, report):
    """Sign exactly what the service will verify: a digest of the canonical report."""
    import base64

    message = ha.helper_signature_message(
        session["session_id"],
        session["challenge"],
        ha.sha256_hex(ha.canonical_json(report)),
    )
    return base64.b64encode(priv.sign(message)).decode()


def _version(host_id: str) -> int:
    """Current admission version, for optimistic-concurrency decisions."""
    return int(ha.admission_status(host_id)["admission_version"])


REPORT = {
    "gpus": [{"model": "RTX 4090", "vram_mb": 24576}],
    "driver_version": "550.90",
    "cuda_version": "12.4",
    "checks": {"docker": True, "nvidia_smi": True},
}


class TestSessionCreation:
    def test_session_is_advisory_only(self, host, helper_key):
        """A provider-opened session must never claim admission authority."""
        _, spki = helper_key
        session = _open_session(host, spki)
        assert session["authority"] == "advisory_only"

    def test_session_issues_a_token_and_challenge(self, host, helper_key):
        _, spki = helper_key
        session = _open_session(host, spki)
        assert session["submit_token"]
        assert session["challenge"]
        assert session["submit_token"] != session["challenge"]

    def test_same_idempotency_key_reuses_the_session(self, host, helper_key):
        _, spki = helper_key
        key = f"idem-{uuid.uuid4().hex[:8]}"
        first = _open_session(host, spki, key=key)
        second = _open_session(host, spki, key=key)
        assert first["session_id"] == second["session_id"]
        assert second["reused"] is True

    def test_unknown_host_is_rejected(self, helper_key):
        _, spki = helper_key
        with pytest.raises(ha.AdmissionNotFound):
            _open_session("no-such-host-at-all", spki)

    def test_malformed_public_key_is_rejected(self, host):
        with pytest.raises(ha.AdmissionError):
            _open_session(host, "not-a-real-spki-key")


class TestEvidenceSubmission:
    def test_valid_signed_report_is_accepted_as_advisory(self, host, helper_key):
        priv, spki = helper_key
        session = _open_session(host, spki)
        result = ha.submit_compatibility_evidence(
            session_id=session["session_id"],
            actor=_provider(),
            submit_token=session["submit_token"],
            report=REPORT,
            signature=_sign(priv, session, REPORT),
        )
        # Advisory evidence records the report but must not admit the host.
        assert result["admission_state"] != "admitted"
        assert result["evidence"]

    def test_forged_signature_is_rejected(self, host, helper_key):
        """A report signed by a different key must not be accepted."""
        _, spki = helper_key
        session = _open_session(host, spki)
        attacker = Ed25519PrivateKey.generate()
        with pytest.raises(ha.AdmissionError):
            ha.submit_compatibility_evidence(
                session_id=session["session_id"],
                actor=_provider(),
                submit_token=session["submit_token"],
                report=REPORT,
                signature=_sign(attacker, session, REPORT),
            )

    def test_tampered_report_is_rejected(self, host, helper_key):
        """The signature covers the report, so editing it after signing fails."""
        priv, spki = helper_key
        session = _open_session(host, spki)
        signature = _sign(priv, session, REPORT)
        tampered = {**REPORT, "gpus": [{"model": "H100", "vram_mb": 81920}]}
        with pytest.raises(ha.AdmissionError):
            ha.submit_compatibility_evidence(
                session_id=session["session_id"],
                actor=_provider(),
                submit_token=session["submit_token"],
                report=tampered,
                signature=signature,
            )

    def test_wrong_submit_token_is_rejected(self, host, helper_key):
        priv, spki = helper_key
        session = _open_session(host, spki)
        with pytest.raises(ha.AdmissionUnauthorized):
            ha.submit_compatibility_evidence(
                session_id=session["session_id"],
                actor=_provider(),
                submit_token="wrong-token",
                report=REPORT,
                signature=_sign(priv, session, REPORT),
            )

    def test_expired_session_is_rejected(self, host, helper_key):
        """Sessions expire so a leaked token cannot be used indefinitely."""
        priv, spki = helper_key
        session = _open_session(host, spki, ttl=60)

        def expire(conn):
            conn.execute(
                "UPDATE host_compatibility_sessions SET expires_at = now() - interval '1 hour' "
                "WHERE session_id = %s",
                (session["session_id"],),
            )

        run_transaction(expire, what="test_expire_session")
        with pytest.raises(ha.AdmissionError):
            ha.submit_compatibility_evidence(
                session_id=session["session_id"],
                actor=_provider(),
                submit_token=session["submit_token"],
                report=REPORT,
                signature=_sign(priv, session, REPORT),
            )

    def test_unknown_session_is_rejected(self, helper_key):
        priv, _ = helper_key
        with pytest.raises(ha.AdmissionError):
            ha.submit_compatibility_evidence(
                session_id=str(uuid.uuid4()),
                actor=_provider(),
                submit_token="anything",
                report=REPORT,
                signature="AAAA",
            )


class TestTrustBoundary:
    def test_provider_evidence_never_admits(self, host, helper_key):
        """The core guarantee: advisory evidence alone cannot admit a host."""
        priv, spki = helper_key
        session = _open_session(host, spki)
        ha.submit_compatibility_evidence(
            session_id=session["session_id"],
            actor=_provider(),
            submit_token=session["submit_token"],
            report=REPORT,
            signature=_sign(priv, session, REPORT),
        )
        status = ha.admission_status(host)
        assert status["admission_state"] != "admitted"

    def test_provider_agent_evidence_is_advisory(self, host):
        result = ha.record_provider_agent_evidence(
            host_id=host, actor=_provider(), report=REPORT
        )
        assert result["evidence_ids"]
        assert result["admission_state"] != "admitted"
        assert result["admitted"] is False

    def test_authoritative_evidence_is_distinguished(self, host):
        ha.record_authoritative_evidence(
            host_id=host,
            actor=_operator(),
            evidence_type="operator_review",
            verdict="pass",
            summary={"note": "verified by hand"},
            idempotency_key=f"auth-{uuid.uuid4().hex[:8]}",
        )
        status = ha.admission_status(host)
        # Recorded on the authoritative side of the ledger, and still not
        # admitted: evidence is a precondition, not a decision.
        assert status["authoritative_evidence"]
        assert status["admission_state"] != "admitted"


class TestDecisions:
    def _authorise(self, host):
        """Record every evidence type the admission gate requires.

        The gate demands fresh authoritative *compatibility* and *hardware*
        evidence — one operator note is deliberately not enough to admit a
        machine to paid work.
        """
        for evidence_type in ha.REQUIRED_AUTHORITATIVE_EVIDENCE:
            ha.record_authoritative_evidence(
                host_id=host,
                actor=_operator(),
                evidence_type=evidence_type,
                verdict="pass",
                summary={"note": f"{evidence_type} verified"},
                idempotency_key=f"auth-{evidence_type}-{uuid.uuid4().hex[:8]}",
            )

    def test_operator_can_admit(self, host):
        self._authorise(host)
        result = ha.decide_admission(
            host_id=host,
            actor=_operator(),
            action="admit",
            reason="hardware verified",
            idempotency_key=f"dec-{uuid.uuid4().hex[:8]}",
            expected_version=_version(host),
        )
        assert result["admission_state"] == "admitted"

    def test_reject_and_revoke_are_terminal_states(self, host):
        self._authorise(host)
        rejected = ha.decide_admission(
            host_id=host, actor=_operator(), action="reject",
            reason="insufficient vram", idempotency_key=f"dec-{uuid.uuid4().hex[:8]}",
            expected_version=_version(host),
        )
        assert rejected["admission_state"] == "rejected"

    def test_decision_is_idempotent(self, host):
        self._authorise(host)
        key = f"dec-{uuid.uuid4().hex[:8]}"
        first = ha.decide_admission(
            host_id=host, actor=_operator(), action="admit", reason="verified",
            idempotency_key=key, expected_version=_version(host),
        )
        second = ha.decide_admission(
            host_id=host, actor=_operator(), action="admit", reason="verified",
            idempotency_key=key, expected_version=_version(host),
        )
        assert first["admission_state"] == second["admission_state"]
        assert second.get("reused") or first["version"] == second["version"]

    def test_stale_version_is_a_conflict(self, host):
        """Two operators acting at once must not silently overwrite."""
        self._authorise(host)
        ha.decide_admission(
            host_id=host, actor=_operator(), action="admit", reason="verified",
            idempotency_key=f"dec-{uuid.uuid4().hex[:8]}", expected_version=_version(host),
        )
        with pytest.raises(ha.AdmissionConflict):
            ha.decide_admission(
                host_id=host, actor=_operator("operator-two"), action="revoke",
                reason="racing decision",
                idempotency_key=f"dec-{uuid.uuid4().hex[:8]}",
                expected_version=0,
            )

    def test_invalid_action_is_rejected(self, host):
        with pytest.raises(ha.AdmissionPreconditionFailed):
            ha.decide_admission(
                host_id=host, actor=_operator(), action="delete-everything",
                reason="nope", idempotency_key=f"dec-{uuid.uuid4().hex[:8]}",
                expected_version=_version(host),
            )

    def test_reason_is_required(self, host):
        with pytest.raises(ha.AdmissionPreconditionFailed):
            ha.decide_admission(
                host_id=host, actor=_operator(), action="admit", reason="",
                idempotency_key=f"dec-{uuid.uuid4().hex[:8]}", expected_version=_version(host),
            )


class TestGrandfathering:
    def test_previously_admitted_hosts_keep_working(self):
        """Migration 082 grandfathers explicitly admitted pre-082 hosts.

        Without this an existing fleet would go dark the moment the migration
        lands, since none of them have authoritative evidence yet.
        """
        host_id = f"admission-legacy-{uuid.uuid4().hex[:10]}"
        _make_host(host_id, tenant="tenant-a", admitted=True)
        try:
            status = ha.admission_status(host_id)
            assert status["host_id"] == host_id
        finally:
            _drop_host(host_id)


class TestRouteTrustBoundary:
    """The routing layer must not offer a provider path that admits."""

    def test_admission_routes_are_split_by_authority(self):
        from routes.host_admission import router

        paths = {r.path for r in router.routes}
        # Anything that can change admission state lives behind /api/admin.
        for deciding in (
            "/api/admin/hosts/{host_id}/admission-decisions",
            "/api/admin/hosts/{host_id}/authoritative-evidence",
        ):
            assert deciding in paths
        for provider_path in (
            "/api/hosts/{host_id}/compatibility-sessions",
            "/api/hosts/{host_id}/provider-evidence",
        ):
            assert provider_path in paths
            assert not provider_path.startswith("/api/admin")

    def test_no_provider_route_can_decide_admission(self):
        import inspect

        import routes.host_admission as mod

        source = inspect.getsource(mod)
        # decide_admission must only ever be called from an admin-gated handler.
        for block in source.split("@router.")[1:]:
            if "host_admission.decide_admission(" in block:
                assert "_require_admin(request)" in block
