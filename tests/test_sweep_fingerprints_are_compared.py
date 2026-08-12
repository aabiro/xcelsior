"""Gate P7 pieces 3 and 4: the fingerprint, and the boundary it draws.

The fingerprint is produced **by the running container**. The control plane
already knows the digest it sent to all N members; comparing that to itself
establishes the request was consistent, not the containers. `collect()` runs
inside the container and reports what it finds.

## The failure this file exists for

A collector that errors and returns nothing leaves every member `NULL`. A
comparison written as "are all the values equal" then finds one distinct value
— none — and reports a **perfect pass** on a sweep where nothing was measured.
That is the same shape caught earlier today one layer up, in a positive control
that injected nothing and passed. So a missing fingerprint is never agreement,
and `verified` requires at least two members to have actually reported.
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from db import _get_pg_pool

    with _get_pg_pool().connection() as _c:
        _has = (
            _c.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_name = "
                "'image_sweep_members' AND column_name = 'fingerprint_hash'"
            ).fetchone()
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no database: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 114")

import environment_fingerprint as ef  # noqa: E402
from control_plane.image_sweeps import (  # noqa: E402
    SweepRefused,
    compare_fingerprints,
    create_sweep,
    record_fingerprint,
)

DIGEST = "reg.example/o/img@sha256:" + "c" * 64


@pytest.fixture
def sweep():
    image_id = f"img-{uuid.uuid4().hex[:12]}"
    owner = f"owner-{uuid.uuid4().hex[:8]}"
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "INSERT INTO user_images (image_id, owner_id, name, tag, image_ref, "
            "image_digest, status, created_at, deleted_at) "
            "VALUES (%s,%s,'fp','v1','reg.example/o/img:v1',%s,'ready',0,0)",
            (image_id, owner, DIGEST),
        )
        made = create_sweep(
            conn,
            tenant_id=owner,
            owner_id=owner,
            image_id=image_id,
            count=3,
            launch=lambda d, i: {"job_id": f"job-{i}", "host_id": f"h-{i}"},
        )
        conn.commit()
    yield made, owner
    with _get_pg_pool().connection() as conn:
        conn.execute("DELETE FROM image_sweep_members WHERE sweep_id=%s", (made.sweep_id,))
        conn.execute("DELETE FROM image_sweeps WHERE sweep_id=%s", (made.sweep_id,))
        conn.execute("DELETE FROM user_images WHERE image_id=%s", (image_id,))
        conn.commit()


# ── The safety property ───────────────────────────────────────────────


def test_a_sweep_where_nothing_reported_is_not_verified(sweep):
    """The failure the positive control exists for.

    Every member `NULL`, one distinct value (none), and a naive comparison
    calls that unanimous.
    """
    made, owner = sweep
    with _get_pg_pool().connection() as conn:
        result = compare_fingerprints(conn, made.sweep_id, tenant_id=owner)
    assert result["verified"] is False, (
        "a sweep in which no container reported anything was verified — a "
        "broken collector would report a perfect pass"
    )
    assert result["reason"] == "insufficient_reports"
    assert result["missing"] == [0, 1, 2]


def test_one_report_is_not_enough(sweep):
    """Byte-identity across N cannot be read off a single member."""
    made, owner = sweep
    _hash, manifest = ef.fingerprint(DIGEST)
    with _get_pg_pool().connection() as conn:
        record_fingerprint(conn, made.sweep_id, 0, hash_=_hash, manifest=manifest)
        conn.commit()
        result = compare_fingerprints(conn, made.sweep_id, tenant_id=owner)
    assert result["verified"] is False
    assert result["reason"] == "insufficient_reports"
    assert result["reported"] == [0]


def test_agreement_with_a_member_still_missing_is_not_a_full_pass(sweep):
    """Two agreeing and one silent is not "the sweep is byte-identical"."""
    made, owner = sweep
    _hash, manifest = ef.fingerprint(DIGEST)
    with _get_pg_pool().connection() as conn:
        record_fingerprint(conn, made.sweep_id, 0, hash_=_hash, manifest=manifest)
        record_fingerprint(conn, made.sweep_id, 1, hash_=_hash, manifest=manifest)
        conn.commit()
        result = compare_fingerprints(conn, made.sweep_id, tenant_id=owner)
    assert result["verified"] is False
    assert result["reason"] == "identical_but_incomplete"
    assert result["missing"] == [2]


def test_all_members_agreeing_is_verified(sweep):
    """The positive case. Without it the checks above pass by always refusing."""
    made, owner = sweep
    _hash, manifest = ef.fingerprint(DIGEST)
    with _get_pg_pool().connection() as conn:
        for index in range(3):
            record_fingerprint(conn, made.sweep_id, index, hash_=_hash, manifest=manifest)
        conn.commit()
        result = compare_fingerprints(conn, made.sweep_id, tenant_id=owner)
    assert result["verified"] is True, result
    assert result["reason"] == "identical"


# ── A mismatch must be diagnosable ────────────────────────────────────


def test_a_mismatch_names_the_differing_fields(sweep):
    """A bare hash mismatch is undiagnosable; the manifest is why both are stored."""
    made, owner = sweep
    hash_a, manifest_a = ef.fingerprint(DIGEST)
    manifest_b = dict(manifest_a)
    manifest_b["packages"] = sorted(set(manifest_a["packages"]) | {"perturbed==1.0"})
    hash_b = "b" * 64

    with _get_pg_pool().connection() as conn:
        record_fingerprint(conn, made.sweep_id, 0, hash_=hash_a, manifest=manifest_a)
        record_fingerprint(conn, made.sweep_id, 1, hash_=hash_a, manifest=manifest_a)
        record_fingerprint(conn, made.sweep_id, 2, hash_=hash_b, manifest=manifest_b)
        conn.commit()
        result = compare_fingerprints(conn, made.sweep_id, tenant_id=owner)

    assert result["verified"] is False
    assert result["reason"] == "mismatch"
    assert "packages" in result["differing_fields"], f"the differing field was not named: {result}"
    assert 2 in result["differing_fields"]["packages"]


def test_half_a_fingerprint_is_refused(sweep):
    """A hash with no manifest cannot be diagnosed; a manifest with no hash cannot be compared."""
    made, _owner = sweep
    with _get_pg_pool().connection() as conn:
        with pytest.raises(SweepRefused) as refused:
            record_fingerprint(conn, made.sweep_id, 0, hash_="abc", manifest={})
        assert refused.value.code == "incomplete_fingerprint"


# ── The collector, and the boundary it draws ──────────────────────────


def test_the_collector_is_deterministic():
    """Two readings of one environment must hash the same.

    Otherwise the check reports mismatches that are artefacts of dictionary
    ordering, which is worse than no check: it teaches the reader to disregard
    a red result.
    """
    first, _ = ef.fingerprint(DIGEST)
    second, _ = ef.fingerprint(DIGEST)
    assert first == second


def test_the_image_digest_moves_the_fingerprint():
    """Different bytes must not fingerprint the same."""
    a, _ = ef.fingerprint("reg/x@sha256:" + "1" * 64)
    b, _ = ef.fingerprint("reg/x@sha256:" + "2" * 64)
    assert a != b


def test_per_instance_environment_is_excluded(monkeypatch):
    """The exclusions are applied, not merely documented.

    Injected job state differs between members *by design* — two members
    sharing a job id would be the bug — so including it guarantees a mismatch
    and turns the check into a constant `false`.
    """
    monkeypatch.setenv("XCELSIOR_JOB_ID", "job-aaa")
    monkeypatch.setenv("HOSTNAME", "container-aaa")
    first, manifest_a = ef.fingerprint(DIGEST)
    monkeypatch.setenv("XCELSIOR_JOB_ID", "job-bbb")
    monkeypatch.setenv("HOSTNAME", "container-bbb")
    second, _ = ef.fingerprint(DIGEST)

    assert first == second, (
        "per-instance environment changed the fingerprint; every sweep would "
        "mismatch and the check would be a constant false"
    )
    assert not any(k.startswith("XCELSIOR_") for k in manifest_a["env"])
    assert "HOSTNAME" not in manifest_a["env"]


def test_image_derived_environment_is_included(monkeypatch):
    """The other half. Excluding too much makes every sweep pass."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/a")
    first, _ = ef.fingerprint(DIGEST)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/b")
    second, _ = ef.fingerprint(DIGEST)
    assert first != second, (
        "a change to LD_LIBRARY_PATH did not move the fingerprint — it decides "
        "which binary runs, and excluding it would let two different "
        "environments read as identical"
    )


def test_which_inventories_answered_is_recorded():
    """An image with no dpkg and an image whose dpkg failed both return [].

    Without recording which sources answered, that real difference reads as
    agreement.
    """
    _hash, manifest = ef.fingerprint(DIGEST)
    assert "sources" in manifest
    assert set(manifest["sources"]) >= {"python", "system"}
