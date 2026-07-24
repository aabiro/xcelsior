"""Track B B4.3 — event contract registry (§12.1/§13.4).

Two invariants the registry must enforce (the CI contract test the checklist
names): a schema carrying a `credential_secret`-classified field is rejected, and
a sink mapping with no classification is rejected. Plus: every registered
contract is valid, the Track-A-emitted names are present (not renamed), and
`register_all` round-trips into `event_contracts`.
"""

from __future__ import annotations

import pytest

from analytics.contracts import (
    CONTRACTS,
    ContractViolation,
    EventContract,
    register_all,
    validate_contract,
    validate_sink_mapping,
)

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _has = _c.execute("SELECT to_regclass('event_contracts')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    _pool = None
    _has = False


# ── Validation (no DB) ────────────────────────────────────────────────


def test_rejects_credential_secret_field():
    bad = EventContract("evil.v1.leak", 1, "internal", {"api_key": "credential_secret"})
    with pytest.raises(ContractViolation) as ei:
        validate_contract(bad)
    assert "credential_secret" in str(ei.value)


def test_rejects_unknown_field_classification():
    with pytest.raises(ContractViolation):
        validate_contract(EventContract("x.v1.y", 1, "internal", {"f": "top_secret"}))


def test_rejects_sink_mapping_without_classification():
    with pytest.raises(ContractViolation):
        validate_sink_mapping("job.v1.created", "audit_log", None)
    with pytest.raises(ContractViolation):
        validate_sink_mapping("job.v1.created", "audit_log", "")
    with pytest.raises(ContractViolation):
        validate_sink_mapping("job.v1.created", "audit_log", "bogus")
    # A classified mapping to a known sink is accepted.
    validate_sink_mapping("job.v1.created", "audit_log", "internal")


def test_rejects_unknown_sink():
    with pytest.raises(ContractViolation):
        validate_sink_mapping("job.v1.created", "no_such_sink", "internal")


def test_all_registered_contracts_are_valid_and_secret_free():
    for c in CONTRACTS:
        validate_contract(c)  # raises if any field is credential_secret / unknown


def test_track_a_emitted_names_are_registered_not_renamed():
    registered = {c.event_type for c in CONTRACTS}
    for emitted in (
        "job.v1.submitted", "job.v1.legacy_status_changed", "job.v1.queue_blocked",
        "host.v1.status_changed", "pricing.v1.spot_prices_updated",
        "job.v1.placement_reserved",
    ):
        assert emitted in registered, f"Track A emits {emitted} but it is not registered"


# ── Registration round-trip (DB) ──────────────────────────────────────


@pytest.mark.skipif(not _has, reason="event_contracts missing — upgrade >= 073")
def test_register_all_roundtrips_and_is_idempotent():
    event_types = [c.event_type for c in CONTRACTS]
    try:
        with _pool.connection() as conn:
            n = register_all(conn)
            conn.commit()
        assert n == len(CONTRACTS)
        with _pool.connection() as conn:
            stored = conn.execute(
                "SELECT count(*) FROM event_contracts WHERE event_type = ANY(%s) AND active",
                (event_types,),
            ).fetchone()[0]
            # every contract has a non-empty schema hash
            null_hashes = conn.execute(
                "SELECT count(*) FROM event_contracts WHERE event_type = ANY(%s) AND (schema_sha256 IS NULL OR schema_sha256 = '')",
                (event_types,),
            ).fetchone()[0]
        assert stored == len(CONTRACTS)
        assert null_hashes == 0
        # Idempotent: a second registration upserts, not duplicates.
        with _pool.connection() as conn:
            register_all(conn)
            conn.commit()
            again = conn.execute(
                "SELECT count(*) FROM event_contracts WHERE event_type = ANY(%s)",
                (event_types,),
            ).fetchone()[0]
        assert again == len(CONTRACTS)
    finally:
        with _pool.connection() as conn:
            conn.execute("DELETE FROM event_contracts WHERE event_type = ANY(%s)", (event_types,))
            conn.commit()
