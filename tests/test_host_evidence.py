"""One read model of a host's evidence, shown on listings and in MCP (Phase 2.3/2.4).

Listings carried ``verified_level``, ``reliability_score`` and ``dlperf_score``
from ``gpu_offers``, which nothing maintained: prod showed the verified RTX 3060
as ``unverified``. ``min_reputation`` filtered on a field no listing had (any
floor above zero returned nothing) and the v2 search model dropped it. Evidence
is now computed from the source tables when it is read.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from host_evidence import (  # noqa: E402
    DEPRECATED_OFFER_FIELDS,
    attach_evidence,
    evidence_badge,
    evidence_for_hosts,
    evidence_summary,
)

DAY = 86400


def _pool():
    from db import _get_pg_pool

    return _get_pg_pool()


def _bench(verdict: str, ran_at: float, tflops: float) -> dict:
    return {
        "schema": "xcelsior-bench/1",
        "bench_version": "1.0.0",
        "gpu_model": "NVIDIA GeForce RTX 3060",
        "fp16_tflops": tflops,
        "memory_bandwidth_gbps": 330.0,
        "allocatable_vram_gb": 11.2,
        "sustained": {"duration_s": 300.0, "temp_peak_c": 71, "thermal_throttle_share": 0.0},
        "verdict": {
            "result": verdict,
            "faults": [] if verdict == "pass" else ["thermal_throttling"],
        },
        "finished_at": ran_at,
    }


@pytest.fixture
def fleet():
    """Two live hosts with different evidence, one overdue, one with none at all."""
    tag = uuid.uuid4().hex[:8]
    now = time.time()
    ids = {name: f"ev-{name}-{tag}" for name in ("good", "weak", "overdue", "bare")}
    jobs: list[str] = []
    with _pool().connection() as conn:
        for name, host_id in ids.items():
            conn.execute(
                """INSERT INTO hosts (host_id, status, registered_at, payload, admission_state,
                       administrative_state)
                   VALUES (%s, 'active', %s, %s::jsonb, 'admitted', 'admitted')""",
                (host_id, now, json.dumps({"host_id": host_id, "gpu_model": "RTX 3060"})),
            )
        stamps = {
            "good": ("verified", now - 3600, now + 20 * 3600),
            "weak": ("deverified", now - 3600, now + 3600),
            "overdue": ("verified", now - 3 * DAY, now - 2 * DAY),
        }
        for name, (state, last, nxt) in stamps.items():
            conn.execute(
                """INSERT INTO host_verifications
                       (verification_id, host_id, state, verified_at, last_check_at,
                        next_check_at, deverify_reason, hold_reason)
                   VALUES (%s, %s, %s, %s, %s, %s, 'secret reason', 'secret hold')""",
                (f"v-{name}-{tag}", ids[name], state, last, last, nxt),
            )
        for host, verdict, ran_at, tflops in (
            ("good", "pass", now - 10 * DAY, 24.0),
            ("good", "pass", now - 3600, 25.4),
            ("weak", "fail", now - 3600, 19.0),
        ):
            conn.execute(
                """INSERT INTO benchmarks (host_id, gpu_model, score, tflops, benchmark_type,
                       details, run_at, gpu_uuid, bench_version, verification_id, verdict)
                   VALUES (%s, 'NVIDIA GeForce RTX 3060', %s, %s, 'xcelsior-bench', %s, %s,
                           'GPU-secret', '1.0.0', 'v', %s)""",
                (
                    ids[host],
                    tflops / 10,
                    tflops,
                    json.dumps(_bench(verdict, ran_at, tflops)),
                    ran_at,
                    verdict,
                ),
            )
        conn.execute(
            """INSERT INTO reputation_scores (entity_id, entity_type, final_score, tier)
               VALUES (%s, 'host', 480, 'bronze'), (%s, 'host', 120, 'bronze')""",
            (ids["good"], ids["weak"]),
        )
        # good: 4 runs ended, one a host fault. weak: 2 ended, both host faults.
        from control_plane.state import initial_desired_state, job_phase

        for host, statuses, faults in (
            ("good", ["succeeded", "succeeded", "cancelled", "failed"], 1),
            ("weak", ["failed", "lost"], 2),
        ):
            for i, status in enumerate(statuses):
                job_id = f"ev-job-{host}-{i}-{tag}"
                jobs.append(job_id)
                conn.execute(
                    """INSERT INTO jobs (job_id, status, priority, submitted_at, host_id, payload,
                           phase, desired_state, effective_priority)
                       VALUES (%s, 'completed', 0, %s, %s, '{}'::jsonb, %s, %s, 0)""",
                    (
                        job_id,
                        now - DAY,
                        ids[host],
                        job_phase("completed"),
                        initial_desired_state("completed"),
                    ),
                )
                conn.execute(
                    """INSERT INTO job_attempts (job_id, attempt_number, status, host_id,
                           fencing_token, started_at, ended_at)
                       VALUES (%s, 1, %s, %s, nextval('placement_fencing_token_seq'),
                               now() - interval '2 hours', now() - interval '1 hour')""",
                    (job_id, status, ids[host]),
                )
            for i in range(faults):
                conn.execute(
                    """INSERT INTO job_failure_log (host_id, job_id, failed_at, reason)
                       VALUES (%s, %s, %s, 'host_disk_full')""",
                    (ids[host], f"ev-job-{host}-{len(statuses) - 1 - i}-{tag}", now - 3600),
                )
        conn.commit()
    yield ids
    with _pool().connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id = ANY(%s)", (jobs,))
        for table, column in (
            ("benchmarks", "host_id"),
            ("host_verifications", "host_id"),
            ("job_failure_log", "host_id"),
            ("reputation_scores", "entity_id"),
            ("hosts", "host_id"),
        ):
            conn.execute(f"DELETE FROM {table} WHERE {column} = ANY(%s)", (list(ids.values()),))
        conn.commit()


pytestmark = pytest.mark.needs_db


def test_each_host_reads_its_own_evidence(fleet):
    ev = evidence_for_hosts(list(fleet.values()), history=5)
    good, weak = ev[fleet["good"]], ev[fleet["weak"]]

    assert good["verification"]["freshness"] == "verified"
    assert good["benchmark"]["fp16_tflops"] == 25.4  # the newest run
    assert good["benchmark"]["verdict"] == "pass"
    assert good["benchmark"]["bench_version"] == "1.0.0"
    assert [r["verdict"] for r in good["benchmark_history"]] == ["pass"]
    # The tier is read from the score with the one ladder (480 is gold), not
    # the stored column, which said bronze.
    assert good["reputation"] == {"tier": "gold", "score": 480.0}
    assert good["jobs"]["runs_ended"] == 4 and good["jobs"]["host_fault_failures"] == 1
    assert good["jobs"]["success_rate"] == 0.75

    assert weak["verification"]["freshness"] == "deverified"
    assert weak["benchmark"]["verdict"] == "fail"
    assert weak["benchmark"]["faults"] == ["thermal_throttling"]
    assert weak["jobs"]["success_rate"] == 0.0


def test_a_stamp_past_its_check_reads_overdue(fleet):
    ev = evidence_for_hosts([fleet["overdue"]])[fleet["overdue"]]
    assert ev["verification"]["state"] == "verified"
    assert ev["verification"]["freshness"] == "overdue" and ev["verification"]["overdue"]


def test_a_host_with_no_rows_reads_unverified_with_no_numbers(fleet):
    ev = evidence_for_hosts([fleet["bare"]])[fleet["bare"]]
    assert ev["verification"]["freshness"] == "unverified"
    assert ev["benchmark"] is None
    assert ev["reputation"] == {"tier": None, "score": None}
    assert ev["jobs"]["success_rate"] is None


def test_nothing_private_to_the_host_is_in_the_evidence(fleet):
    text = json.dumps(evidence_for_hosts(list(fleet.values()), history=5))
    for secret in ("secret reason", "secret hold", "GPU-secret"):
        assert secret not in text


def test_listing_rows_carry_the_badge_and_lose_the_deprecated_fields(fleet):
    rows = [
        {
            "host_id": fleet["good"],
            "verified_level": "unverified",
            "reliability_score": 0,
            "dlperf_score": 0,
        },
        {"host_id": fleet["bare"]},
    ]
    attach_evidence(rows)
    assert not any(name in rows[0] for name in DEPRECATED_OFFER_FIELDS)
    badge = rows[0]["evidence"]
    assert badge["verification"] == "verified"
    assert badge["benchmark"]["fp16_tflops"] == 25.4
    assert badge["reputation_tier"] == "gold" and badge["job_success_rate"] == 0.75
    assert rows[1]["evidence"] == evidence_badge(None)


def test_an_aggregated_row_summarizes_its_hosts(fleet):
    ev = evidence_for_hosts(list(fleet.values()))
    summary = evidence_summary([ev.get(h) for h in fleet.values()])
    assert summary["hosts"] == 4
    assert summary["verified"] == 1 and summary["overdue"] == 1
    assert summary["latest_benchmark_at"] == pytest.approx(time.time() - 3600, abs=120)


def test_an_unreachable_database_badges_nothing_verified(monkeypatch):
    import host_evidence

    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(host_evidence, "host_evidence", boom)
    rows = attach_evidence([{"host_id": "x"}])
    assert rows[0]["evidence"]["verification"] == "unverified"


# ── The routes ──────────────────────────────────────────────────────


@pytest.fixture
def offers(fleet):
    """An available offer on the good and weak hosts."""
    now = time.time()
    with _pool().connection() as conn:
        for name in ("good", "weak"):
            conn.execute(
                """INSERT INTO gpu_offers (offer_id, provider_id, host_id, gpu_model, vram_gb,
                       gpu_count_total, gpu_count_available, ask_cents_per_hour, available,
                       verified_level, reliability_score, dlperf_score, created_at, updated_at)
                   VALUES (%s, 'prov', %s, %s, 12, 1, 1, 35, TRUE, 'unverified', 0, 0, %s, %s)""",
                (f"off-{fleet[name]}", fleet[name], f"RTX 3060 {fleet['good']}", now, now),
            )
        conn.commit()
    yield f"RTX 3060 {fleet['good']}"
    with _pool().connection() as conn:
        conn.execute(
            "DELETE FROM gpu_offers WHERE host_id = ANY(%s)", ([fleet["good"], fleet["weak"]],)
        )
        conn.commit()


def _client():
    from fastapi.testclient import TestClient

    from api import app

    return TestClient(app)


def test_marketplace_search_filters_and_sorts_on_evidence(offers, fleet):
    client = _client()
    body = {"gpu_model": offers, "sort_by": "reputation"}
    found = client.post("/api/v2/marketplace/search", json=body).json()["offers"]
    assert [o["host_id"] for o in found] == [fleet["good"], fleet["weak"]]
    assert all("verified_level" not in o and "evidence" in o for o in found)

    # The MCP tool sent min_reputation from the start; it now filters.
    floor = client.post("/api/v2/marketplace/search", json={**body, "min_reputation": 400}).json()[
        "offers"
    ]
    assert [o["host_id"] for o in floor] == [fleet["good"]]

    verified = client.post(
        "/api/v2/marketplace/search", json={**body, "verified_only": True}
    ).json()["offers"]
    assert [o["host_id"] for o in verified] == [fleet["good"]]


def test_the_evidence_route_needs_a_signed_in_user_and_a_known_host(fleet, monkeypatch):
    import routes._deps as deps
    import routes.marketplace as route

    client = _client()
    monkeypatch.setattr(deps, "AUTH_REQUIRED", True)
    assert client.get(f"/api/v2/hosts/{fleet['good']}/evidence").status_code == 401

    monkeypatch.setattr(route, "_require_auth", lambda request: {"user_id": "u", "role": "user"})
    monkeypatch.setattr(route, "_require_scope", lambda user, scope: None)
    ok = client.get(f"/api/v2/hosts/{fleet['good']}/evidence")
    assert ok.status_code == 200
    data = ok.json()
    assert data["host_id"] == fleet["good"]
    assert data["benchmark"]["fp16_tflops"] == 25.4
    assert len(data["benchmark_history"]) == 1
    assert client.get(f"/api/v2/hosts/no-such-{uuid.uuid4().hex}/evidence").status_code == 404
