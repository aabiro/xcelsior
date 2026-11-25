"""The public verification claims, each held to the code (Phase 0, 2026-11-22).

What was said publicly, and what these tests pin:

* the GPU serial / identity is cross-checked — the agent sends every GPU's UUID
  and serial, the first full pass records a baseline, a different card de-verifies
  and stays de-verified until an admin clears it (0.1);
* checked against what the host claims — "claimed" is the host's listing, not the
  agent's own reading, and VRAM must match within 5% either way (0.2);
* unknown is not a pass — a missing sensor reading, version, UUID or measurement
  blocks `verified` (0.3);
* repeated job failures de-verify — host-fault failures only (0.4);
* re-verified every 24 h — de-verified hosts are asked again, a stamp past its
  check plus the grace reads `overdue` everywhere (0.5).

"Hosts are verified before they take jobs" (0.6) is in
`test_placement_requires_verified.py`.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from verification import (  # noqa: E402
    CHECK_FAIL,
    CHECK_PASS,
    CHECK_UNKNOWN,
    HOLD_IDENTITY_MISMATCH,
    HOLD_JOB_FAILURES,
    OVERDUE,
    VERIFICATION_CHECKS,
    VERIFICATION_THRESHOLDS,
    HostVerification,
    HostVerificationState,
    VerificationEngine,
    VerificationStore,
    check_allocatable_vram,
    check_gpu_identity,
    check_security_posture,
    check_thermal_stability,
    reported_identity,
    verification_check_count,
    verification_freshness,
)

UUID_A = "GPU-f22a551d-2bee-2f17-22af-454481edcdc7"
UUID_B = "GPU-0b1c2d3e-4f50-6172-8394-a5b6c7d8e9f0"
VERSIONS = {
    "runc": "1.4.0",
    "nvidia_toolkit": "1.20.2",
    "nvidia_driver": "580.178.04",
    "docker": "29.1.3",
}
LISTING = {"gpu_model": "RTX 3060", "vram_gb": 12.0, "gpu_count": None, "offers": []}


def _report(**over) -> dict:
    """What the fixed agent sends for a healthy RTX 3060 whose card has no serial."""
    report = {
        "gpu_model": "NVIDIA GeForce RTX 3060",
        "total_vram_gb": 11.76,
        "allocatable_vram_gb": 11.2,
        "cuda_version": "12.4",
        "driver_version": "580.178.04",
        "compute_capability": "8.6",
        "pcie_bandwidth_gbps": 12.5,
        "gpu_temp_celsius": 66.0,
        "gpu_temp_samples": 30,
        "packet_loss_pct": 0.0,
        "jitter_ms": 2.0,
        "throughput_mbps": 400.0,
        "versions": dict(VERSIONS),
        "gpu_identities": [
            {"index": 0, "gpu_uuid": UUID_A, "serial": "", "pci_bus_id": "00000000:01:00.0"}
        ],
        "gpu_uuid": UUID_A,
        "serial": "",
        "claimed_gpu_model": "RTX 3060",
        "claimed_vram_gb": 12.0,
    }
    report.update(over)
    return report


class _ListedStore(VerificationStore):
    """The real PostgreSQL store, with the listing supplied by the test."""

    def __init__(self, listing: dict | None):
        super().__init__()
        self.listing = listing

    def get_advertised_hardware(self, host_id: str):
        return self.listing


def _engine(listing: dict | None = LISTING) -> VerificationEngine:
    return VerificationEngine(store=_ListedStore(listing))


def _host() -> str:
    return f"vclaim-{uuid.uuid4().hex[:10]}"


def _status(v: HostVerification, name: str) -> dict:
    return next(c for c in v.checks if c["check_name"] == name)


def _cleanup(host_id: str) -> None:
    store = VerificationStore()
    with store._conn() as conn:
        for table in ("host_verifications", "verification_history", "job_failure_log"):
            conn.execute(f"DELETE FROM {table} WHERE host_id = %s", (host_id,))


@pytest.fixture
def host_id():
    h = _host()
    yield h
    _cleanup(h)


def _past_cooldown(engine: VerificationEngine, host_id: str) -> None:
    """Move a de-verified record's cooldown into the past, as time would."""
    v = engine.store.get_verification(host_id)
    assert v is not None
    if v.deverified_at:
        v.deverified_at -= VERIFICATION_THRESHOLDS["deverify_cooldown_sec"] + 1
        engine.store.save_verification(v)


# ── 0.1 Real GPU identity ─────────────────────────────────────────────


def test_a_normal_pass_records_the_uuid_baseline_and_says_the_serial_is_unavailable(host_id):
    engine = _engine()
    v = engine.run_verification(host_id, _report())

    assert v.state == HostVerificationState.VERIFIED, v.deverify_reason
    stored = engine.store.get_verification(host_id)
    assert stored is not None
    assert stored.identity_baseline["gpu_uuids"] == [UUID_A]
    assert stored.identity_baseline["serial_status"] == "unavailable"
    assert stored.identity_baseline["pci_bus_ids"] == ["00000000:01:00.0"]
    identity = _status(stored, "gpu_identity")
    assert identity["status"] == CHECK_PASS
    assert "serial: unavailable" in identity["details"]
    assert "serial: ok" not in identity["details"]


def test_a_swapped_card_of_the_same_model_deverifies_and_stays_held(host_id):
    engine = _engine()
    assert engine.run_verification(host_id, _report()).state == HostVerificationState.VERIFIED

    swapped = _report(
        gpu_identities=[{"index": 0, "gpu_uuid": UUID_B, "serial": ""}], gpu_uuid=UUID_B
    )
    v = engine.run_verification(host_id, swapped)
    assert v.state == HostVerificationState.DEVERIFIED
    assert v.hold_reason == HOLD_IDENTITY_MISMATCH and v.hold_until is None
    assert "GPU UUIDs changed" in v.deverify_reason
    assert _status(v, "gpu_identity")["status"] == CHECK_FAIL
    assert v.identity_baseline["gpu_uuids"] == [UUID_A], "the baseline moved to the new card"

    # Every other check passes; the swapped card must still not re-verify itself.
    _past_cooldown(engine, host_id)
    again = engine.run_verification(host_id, swapped)
    assert again.state == HostVerificationState.DEVERIFIED
    assert again.hold_reason == HOLD_IDENTITY_MISMATCH

    # The sweep does not ask a held host.
    assert host_id not in engine.store.list_hosts_needing_reverification(now=time.time() + 10**6)


def test_only_an_admin_clears_a_swap_and_the_next_pass_records_the_new_card(host_id):
    os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
    os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")
    from fastapi.testclient import TestClient

    from api import app

    engine = _engine()
    engine.run_verification(host_id, _report())
    swapped = _report(gpu_identities=[{"index": 0, "gpu_uuid": UUID_B, "serial": ""}])
    assert engine.run_verification(host_id, swapped).hold_reason == HOLD_IDENTITY_MISMATCH

    admin = {
        "Authorization": "Bearer "
        + (os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production")
    }
    client = TestClient(app)
    status = client.get(f"/api/verify/{host_id}/status", headers=admin).json()
    assert status["on_hold"] is True and status["status"] == "deverified"

    r = client.post(f"/api/verify/{host_id}/approve", headers=admin)
    assert r.status_code == 200 and r.json()["cleared_hold"] == HOLD_IDENTITY_MISMATCH
    cleared = engine.store.get_verification(host_id)
    assert cleared.hold_reason == "" and cleared.identity_baseline == {}

    v = engine.run_verification(host_id, swapped)
    assert v.state == HostVerificationState.VERIFIED, v.deverify_reason
    assert engine.store.get_verification(host_id).identity_baseline["gpu_uuids"] == [UUID_B]

    # A rejection holds until an approval, whatever later benchmarks say.
    assert client.post(f"/api/verify/{host_id}/reject", headers=admin).status_code == 200
    again = engine.run_verification(host_id, swapped)
    assert again.state == HostVerificationState.DEVERIFIED
    assert again.hold_reason == "admin_rejected"


def test_a_changed_serial_with_the_same_uuid_is_a_mismatch(host_id):
    engine = _engine()
    with_serial = _report(
        gpu_identities=[{"index": 0, "gpu_uuid": UUID_A, "serial": "1324021034567"}]
    )
    assert engine.run_verification(host_id, with_serial).state == HostVerificationState.VERIFIED
    other = _report(gpu_identities=[{"index": 0, "gpu_uuid": UUID_A, "serial": "1324021099999"}])
    v = engine.run_verification(host_id, other)
    assert v.state == HostVerificationState.DEVERIFIED
    assert "serials changed" in v.deverify_reason


def test_an_agent_that_sends_no_uuid_is_not_verified(host_id):
    """Every agent before this fix: nothing to cross-check, so nothing is claimed."""
    report = _report()
    for key in ("gpu_identities", "gpu_uuid", "serial"):
        report.pop(key)
    v = _engine().run_verification(host_id, report)
    assert v.state == HostVerificationState.DEVERIFIED
    assert _status(v, "gpu_identity")["status"] == CHECK_UNKNOWN
    assert v.identity_baseline == {}


def test_a_uuid_reported_as_the_serial_is_not_a_serial():
    """The old NVML helper fell back to the UUID when the board had no serial."""
    identity = reported_identity({"gpu_identities": [{"gpu_uuid": UUID_A, "serial": UUID_A}]})
    assert identity["serials"] == [] and identity["serial_status"] == "unavailable"


def test_upgrading_to_identity_reporting_is_not_fingerprint_drift(host_id):
    """A stored fingerprint from before the agent sent UUIDs must not read as drift."""
    engine = _engine()
    engine.store.save_verification(
        HostVerification(
            host_id=host_id,
            state=HostVerificationState.VERIFIED,
            gpu_fingerprint="56a1db48dfee46d4",
            last_check_at=time.time(),
        )
    )
    assert engine.run_verification(host_id, _report()).state == HostVerificationState.VERIFIED


# ── 0.2 "Claimed" means the listing ──────────────────────────────────


def test_over_reported_vram_fails_the_two_sided_tolerance():
    report = _report(advertised_gpu_model="RTX 3060", advertised_vram_gb=12.0, total_vram_gb=24.0)
    r = check_gpu_identity(report)
    assert r.status == CHECK_FAIL
    assert "VRAM differs from the listing" in r.details


@pytest.mark.parametrize("measured", [11.5, 11.76, 12.0, 12.5])
def test_vram_within_five_percent_either_way_passes(measured):
    report = _report(advertised_gpu_model="RTX 3060", advertised_vram_gb=12.0)
    report["total_vram_gb"] = measured
    assert check_gpu_identity(report).status == CHECK_PASS


def test_the_listing_not_the_agent_is_the_claim(host_id):
    """The agent says RTX 4090 about itself and measures a 4090; the listing says 3060."""
    report = _report(
        gpu_model="NVIDIA GeForce RTX 4090",
        total_vram_gb=23.6,
        claimed_gpu_model="RTX 4090",
        claimed_vram_gb=24.0,
        # An agent cannot advertise for itself either.
        advertised_gpu_model="RTX 4090",
        advertised_vram_gb=24.0,
    )
    v = _engine().run_verification(host_id, report)
    identity = _status(v, "gpu_identity")
    assert identity["status"] == CHECK_FAIL
    assert "not the listed 'RTX 3060'" in identity["details"]


def test_an_offer_that_disagrees_with_the_card_fails(host_id):
    listing = {
        **LISTING,
        "offers": [{"offer_id": "offer-x", "gpu_model": "RTX 3060", "vram_gb": 24}],
    }
    v = _engine(listing).run_verification(host_id, _report())
    assert _status(v, "gpu_identity")["status"] == CHECK_FAIL
    assert "offer offer-x lists 24GB" in _status(v, "gpu_identity")["details"]


def test_no_listing_is_unknown_not_a_pass(host_id):
    v = _engine(listing=None).run_verification(host_id, _report())
    assert _status(v, "gpu_identity")["status"] == CHECK_UNKNOWN
    assert v.state == HostVerificationState.DEVERIFIED


def test_the_listing_is_read_from_the_hosts_row_and_available_offers():
    from control_plane.db import control_plane_transaction

    host_id = _host()
    now = time.time()
    with control_plane_transaction() as conn:
        conn.execute(
            "INSERT INTO hosts (host_id, status, registered_at, payload) VALUES (%s, 'active', %s, %s)",
            (host_id, now, json.dumps({"gpu_model": "RTX 3060", "total_vram_gb": 12.0})),
        )
        for offer_id, available in ((f"offer-{host_id}-a", True), (f"offer-{host_id}-b", False)):
            conn.execute(
                """INSERT INTO gpu_offers (offer_id, provider_id, host_id, gpu_model, vram_gb,
                       ask_cents_per_hour, available, created_at, updated_at)
                   VALUES (%s, 'p', %s, 'RTX 3060', 12, 35, %s, %s, %s)""",
                (offer_id, host_id, available, now, now),
            )
    try:
        listing = VerificationStore().get_advertised_hardware(host_id)
        assert listing is not None
        assert listing["gpu_model"] == "RTX 3060" and listing["vram_gb"] == 12.0
        assert [o["offer_id"] for o in listing["offers"]] == [f"offer-{host_id}-a"]
        assert VerificationStore().get_advertised_hardware(f"{host_id}-missing") is None
    finally:
        with control_plane_transaction() as conn:
            conn.execute("DELETE FROM gpu_offers WHERE host_id = %s", (host_id,))
            conn.execute("DELETE FROM hosts WHERE host_id = %s", (host_id,))


# ── 0.3 Unknown is not a pass ─────────────────────────────────────────


@pytest.mark.parametrize("temp", [0, 0.0, None])
def test_a_thermal_sensor_that_reads_nothing_is_unknown(temp):
    assert check_thermal_stability(_report(gpu_temp_celsius=temp)).status == CHECK_UNKNOWN


def test_a_missing_component_version_is_unknown():
    versions = dict(VERSIONS)
    versions.pop("nvidia_toolkit")
    r = check_security_posture(_report(versions=versions))
    assert r.status == CHECK_UNKNOWN and "nvidia_toolkit" in r.actual


def test_a_vulnerable_version_still_fails():
    r = check_security_posture(_report(versions={**VERSIONS, "runc": "1.1.0"}))
    assert r.status == CHECK_FAIL


def test_allocatable_vram_below_ninety_percent_of_what_cuda_addresses_fails():
    # CUDA addresses 11.76 of the 12 GB listing; the bar is 90 % of that, 10.58.
    report = _report(advertised_vram_gb=12.0)
    assert check_allocatable_vram({**report, "allocatable_vram_gb": 10.5}).status == CHECK_FAIL
    assert check_allocatable_vram({**report, "allocatable_vram_gb": 10.6}).status == CHECK_PASS
    # Without CUDA's total the bar is the listing, as before.
    no_total = {k: v for k, v in report.items() if k != "total_vram_gb"}
    assert check_allocatable_vram({**no_total, "allocatable_vram_gb": 10.7}).status == CHECK_FAIL
    assert check_allocatable_vram({**no_total, "allocatable_vram_gb": 10.9}).status == CHECK_PASS
    report.pop("allocatable_vram_gb")
    assert check_allocatable_vram(report).status == CHECK_UNKNOWN


#: What a genuine 6 GB laptop RTX 2060 reports: nvidia-smi says 6144 MiB, the
#: driver keeps 407 MiB, CUDA addresses the other 5737 MiB (5.6 GB), and a job
#: can take about 5.3 GB of that in one piece.
RTX_2060 = {
    "gpu_model": "NVIDIA GeForce RTX 2060",
    "board_vram_gb": 6.0,
    "total_vram_gb": 5.6,
    "allocatable_vram_gb": 5.3,
    "advertised_gpu_model": "RTX 2060",
    "advertised_vram_gb": 6.0,
}


def test_a_genuine_6gb_card_is_not_accused_of_a_short_listing():
    """The board's memory is compared with the listing, not CUDA's total.

    Before, CUDA's 5.6 GB was compared with the 6.0 GB listing (6.7 % off), so
    every small card failed identity for the driver's reservation.
    """
    identity = check_gpu_identity(_report(**RTX_2060))
    assert identity.status == CHECK_PASS, identity.details
    assert "CUDA addresses 5.6GB" in identity.details
    allocatable = check_allocatable_vram(_report(**RTX_2060))
    assert allocatable.status == CHECK_PASS, allocatable.actual
    assert "of 5.6GB" in allocatable.actual and "listing 6GB" in allocatable.actual


def test_an_agent_without_the_board_reading_is_judged_as_before():
    old = {k: v for k, v in RTX_2060.items() if k != "board_vram_gb"}
    assert check_gpu_identity(_report(**old)).status == CHECK_FAIL


def test_a_short_board_or_a_memory_squatter_still_fails():
    # A 4 GB card listed as 6 GB.
    short = {**RTX_2060, "board_vram_gb": 4.0, "total_vram_gb": 3.7, "allocatable_vram_gb": 3.4}
    assert check_gpu_identity(_report(**short)).status == CHECK_FAIL
    assert check_allocatable_vram(_report(**short)).status == CHECK_FAIL
    # The right card, with another process holding 2 GB: CUDA's total does not move.
    squatted = {**RTX_2060, "allocatable_vram_gb": 3.3}
    assert check_gpu_identity(_report(**squatted)).status == CHECK_PASS
    assert check_allocatable_vram(_report(**squatted)).status == CHECK_FAIL


@pytest.mark.parametrize(
    "override, check",
    [
        ({"gpu_temp_celsius": 0}, "thermal_stability"),
        ({"versions": {k: v for k, v in VERSIONS.items() if k != "docker"}}, "security_posture"),
        ({"allocatable_vram_gb": None}, "allocatable_vram"),
        ({"pcie_bandwidth_gbps": 0}, "pcie_bandwidth"),
        ({"packet_loss_pct": None}, "network_quality"),
    ],
)
def test_each_unknown_blocks_verified(host_id, override, check):
    v = _engine().run_verification(host_id, _report(**override))
    assert _status(v, check)["status"] == CHECK_UNKNOWN
    assert v.state == HostVerificationState.DEVERIFIED
    assert f"Unknown (not passed): {check}" in v.deverify_reason


def test_memory_fragmentation_is_gone_and_the_count_is_one_source():
    names = [fn.__name__ for fn in VERIFICATION_CHECKS]
    assert "check_memory_fragmentation" not in names
    assert "check_allocatable_vram" in names
    import release_facts

    assert release_facts.verification_check_count() == verification_check_count() == 7
    facts = json.load(open(release_facts.ROOT / "api/frontend/src/lib/release-facts.json"))
    assert facts["verificationCheckCount"] == verification_check_count()


# ── 0.5 Freshness and the sweep ───────────────────────────────────────


def test_overdue_is_next_check_plus_the_grace():
    now = 1_800_000_000.0
    grace = VERIFICATION_THRESHOLDS["reverify_grace_sec"]
    assert verification_freshness("verified", now - 3600, now + 60, now=now) == "verified"
    assert verification_freshness("verified", None, now - grace + 1, now=now) == "verified"
    assert verification_freshness("verified", None, now - grace - 1, now=now) == OVERDUE
    assert verification_freshness("verified", None, None, now=now) == OVERDUE
    assert verification_freshness("deverified", now, now, now=now) == "deverified"
    assert verification_freshness(None, now=now) == "unverified"


def test_a_deverified_host_is_asked_again_after_its_cooldown(host_id):
    engine = _engine()
    # 0.9 GB/s: a GPU on a PCIe 3.0 x1 riser.
    v = engine.run_verification(host_id, _report(pcie_bandwidth_gbps=0.9))
    assert v.state == HostVerificationState.DEVERIFIED
    assert v.next_check_at == pytest.approx(
        v.last_check_at + VERIFICATION_THRESHOLDS["deverify_cooldown_sec"]
    )
    store = engine.store
    assert host_id not in store.list_hosts_needing_reverification(now=time.time())
    later = time.time() + VERIFICATION_THRESHOLDS["deverify_cooldown_sec"] + 1
    assert host_id in store.list_hosts_needing_reverification(now=later)

    # And it can recover by passing.
    _past_cooldown(engine, host_id)
    assert engine.run_verification(host_id, _report()).state == HostVerificationState.VERIFIED


def test_an_unverified_record_is_asked_only_while_its_host_is_active():
    from control_plane.db import control_plane_transaction

    live, gone = _host(), _host()
    store = VerificationStore()
    with control_plane_transaction() as conn:
        conn.execute(
            "INSERT INTO hosts (host_id, status, registered_at, payload) "
            "VALUES (%s, 'active', %s, '{}'), (%s, 'offline', %s, '{}')",
            (live, time.time(), gone, time.time()),
        )
    try:
        for h in (live, gone):
            store.save_verification(HostVerification(host_id=h))
        due = store.list_hosts_needing_reverification()
        assert live in due and gone not in due
    finally:
        _cleanup(live)
        _cleanup(gone)
        with control_plane_transaction() as conn:
            conn.execute("DELETE FROM hosts WHERE host_id = ANY(%s)", ([live, gone],))


# ── 0.4 Host-fault failures de-verify ─────────────────────────────────


def _verified(engine: VerificationEngine, host_id: str) -> None:
    assert engine.run_verification(host_id, _report()).state == HostVerificationState.VERIFIED


def test_failures_under_the_threshold_do_not_deverify(host_id):
    engine = _engine()
    _verified(engine, host_id)
    threshold = VERIFICATION_THRESHOLDS["job_failure_threshold"]
    for n in range(threshold - 1):
        assert engine.check_failure_threshold(host_id, f"job-{n}", "host_disk_full") is False
    assert engine.store.get_verification(host_id).state == HostVerificationState.VERIFIED


def test_failures_at_the_threshold_deverify_and_hold(host_id):
    engine = _engine()
    _verified(engine, host_id)
    threshold = VERIFICATION_THRESHOLDS["job_failure_threshold"]
    results = [
        engine.check_failure_threshold(host_id, f"job-{n}", "host_disk_full")
        for n in range(threshold)
    ]
    assert results[-1] is True and not any(results[:-1])
    v = engine.store.get_verification(host_id)
    assert v.state == HostVerificationState.DEVERIFIED
    assert v.hold_reason == HOLD_JOB_FAILURES
    assert v.hold_until == pytest.approx(
        time.time() + VERIFICATION_THRESHOLDS["job_failure_hold_sec"], abs=30
    )

    # A passing benchmark inside the hold does not undo it ...
    _past_cooldown(engine, host_id)
    assert engine.run_verification(host_id, _report()).state == HostVerificationState.DEVERIFIED
    # ... and once the hold lapses, it can.
    v = engine.store.get_verification(host_id)
    v.hold_until = time.time() - 1
    engine.store.save_verification(v)
    _past_cooldown(engine, host_id)
    assert engine.run_verification(host_id, _report()).state == HostVerificationState.VERIFIED


def test_failures_outside_the_window_do_not_count(host_id):
    engine = _engine()
    _verified(engine, host_id)
    threshold = VERIFICATION_THRESHOLDS["job_failure_threshold"]
    for n in range(threshold - 1):
        engine.check_failure_threshold(host_id, f"old-{n}", "host_disk_full")
    later = time.time() + VERIFICATION_THRESHOLDS["job_failure_window_sec"] + 60
    assert engine.check_failure_threshold(host_id, "new", "host_disk_full", now=later) is False
    assert engine.store.get_verification(host_id).state == HostVerificationState.VERIFIED


def test_one_job_reported_failed_twice_is_one_failure(host_id):
    engine = _engine()
    _verified(engine, host_id)
    for _ in range(VERIFICATION_THRESHOLDS["job_failure_threshold"] + 2):
        engine.check_failure_threshold(host_id, "same-job", "host_disk_full")
    assert engine.store.get_verification(host_id).state == HostVerificationState.VERIFIED


@pytest.mark.parametrize(
    "code, detail, fault",
    [
        ("host_volume_mount_failed", None, "host"),
        ("container_missing", None, "host"),
        ("container_failed", {"error": "required NFS mount failed"}, "host"),
        (
            "container_failed",
            {"error": "Image pull failed: the host is out of disk space (exit 1)"},
            "host",
        ),
        ("container_failed", {"error": "Traceback: ValueError", "exit_code": 1}, "customer"),
        ("image_not_found", None, "customer"),
        ("mining_detected", None, "customer"),
        ("disk_quota_exceeded", None, "customer"),
        ("lease_unconfirmed_before_container_start", None, "unknown"),
        ("agent_exception", None, "unknown"),
        (
            "container_failed",
            {"error": "Image pull failed: the registry could not be reached from this host"},
            "customer",
        ),
    ],
)
def test_failure_classification(code, detail, fault):
    from failure_fault import classify_failure

    assert classify_failure(code, detail) == fault


def test_a_customer_fault_failure_is_ignored(host_id, monkeypatch):
    import failure_fault
    import verification

    engine = _engine()
    _verified(engine, host_id)
    monkeypatch.setattr(verification, "get_verification_engine", lambda: engine)
    for n in range(VERIFICATION_THRESHOLDS["job_failure_threshold"] + 1):
        assert (
            failure_fault.record_failed_attempt(
                host_id, f"job-{n}", "container_failed", {"error": "exit 1"}
            )
            is False
        )
    assert engine.store.get_recent_failures(host_id) == 0
    assert engine.store.get_verification(host_id).state == HostVerificationState.VERIFIED

    for n in range(VERIFICATION_THRESHOLDS["job_failure_threshold"]):
        failure_fault.record_failed_attempt(host_id, f"hjob-{n}", "host_runtime_unavailable")
    assert engine.store.get_verification(host_id).state == HostVerificationState.DEVERIFIED


def test_the_v2_status_route_counts_a_failed_attempt(monkeypatch):
    """The live path: workers report through /agent/v2/attempts/status."""
    os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
    os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")
    from fastapi.testclient import TestClient

    import failure_fault
    import routes.agent_v2 as agent_v2
    from api import app
    from control_plane.attempts import AttemptStatusResult

    calls = []
    monkeypatch.setattr(agent_v2, "_require_agent_auth", lambda *a, **k: None)
    monkeypatch.setattr(
        agent_v2,
        "run_transaction",
        lambda fn, what=None: AttemptStatusResult("j1", "a1", "failed", True, True),
    )
    monkeypatch.setattr(failure_fault, "record_failed_attempt", lambda *a: calls.append(a) or False)
    r = TestClient(app).post(
        "/agent/v2/attempts/status",
        json={
            "job_id": "j1",
            "attempt_id": "a1",
            "host_id": "h1",
            "fencing_token": 1,
            "status": "failed",
            "failure_code": "host_nfs_mount_failed",
            "detail": {"error": "required NFS mount failed"},
        },
    )
    assert r.status_code == 200, r.text
    assert calls == [("h1", "j1", "host_nfs_mount_failed", {"error": "required NFS mount failed"})]


def test_the_legacy_failure_path_no_longer_blames_every_failure_on_the_host():
    import inspect

    import scheduler

    source = inspect.getsource(scheduler.update_job_status)
    assert 'record_job_failure(j["host_id"], is_host_fault=is_host_fault)' in source
    assert "record_failed_attempt" in source


def test_the_pcie_floor_admits_an_x8_link_and_refuses_a_riser():
    """An absolute 8 GB/s failed every PCIe 3.0 x8 card (the prod RTX 2060 laptop
    measured 6.11 GB/s on its Gen3 x8 link); the floor is for riser links."""
    from verification import check_pcie_bandwidth

    assert check_pcie_bandwidth({"pcie_bandwidth_gbps": 6.11}).status == CHECK_PASS
    assert check_pcie_bandwidth({"pcie_bandwidth_gbps": 18.3}).status == CHECK_PASS
    assert check_pcie_bandwidth({"pcie_bandwidth_gbps": 0.85}).status == CHECK_FAIL
    assert check_pcie_bandwidth({"pcie_bandwidth_gbps": 1.6}).status == CHECK_FAIL
