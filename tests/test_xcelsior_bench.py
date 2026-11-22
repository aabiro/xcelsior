"""xcelsior-bench: one versioned benchmark the agent runs and a renter can run (Phase 2.1/2.2).

The agent used to run an inline script that measured fp16 TFLOPS and read the
temperature for 15 seconds, so a card that throttled after a minute passed, and
nothing kept the results. The benchmark is now ``workers/xcelsior_bench.py``:
matmul TFLOPS (fp16, bf16), memory bandwidth, allocatable VRAM, PCIe both ways
and a sustained-load phase sampling clocks, temperature, power and throttle
reasons every second. Each run is kept in ``benchmarks`` (167) with a verdict
judged against the VRAM the listing advertises.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

import xcelsior_bench  # noqa: E402
from xcelsior_bench import (  # noqa: E402
    THROTTLE_BITS,
    summarize_sustained,
    throttle_reasons,
    verdict,
)

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "workers" / "xcelsior_bench.py"

THERMAL = THROTTLE_BITS["hw_thermal"]
POWER_BRAKE = THROTTLE_BITS["hw_power_brake"]
POWER_CAP = THROTTLE_BITS["sw_power_cap"]


def _samples(n: int, throttled: int, bits: int, temp: float = 70.0) -> list[dict]:
    """``n`` one-second samples, the last ``throttled`` of them with ``bits`` set."""
    return [
        {
            "sm_clock_mhz": 1500 if i >= n - throttled else 1900,
            "temp_c": temp,
            "power_w": 170.0,
            "throttle_bits": bits if i >= n - throttled else 0,
        }
        for i in range(n)
    ]


# ── The agent runs exactly the file it was built with ─────────────────


def test_the_agent_pins_the_benchmark_it_was_built_with():
    import worker_agent

    assert worker_agent.BENCH_SHA256 == hashlib.sha256(BENCH.read_bytes()).hexdigest(), (
        "workers/xcelsior_bench.py changed: set worker_agent.BENCH_SHA256 to its sha256 "
        "and re-sign (api/scripts/sign-static-agent.py)"
    )
    assert worker_agent.BENCH_FILE in worker_agent._SIBLING_MODULES


def test_an_agent_without_the_file_fetches_it_and_accepts_only_the_pinned_one(
    tmp_path, monkeypatch
):
    import worker_agent

    monkeypatch.setattr(worker_agent, "_sibling_module_dir", lambda name: tmp_path)
    monkeypatch.setattr(worker_agent, "SCHEDULER_URL", "https://scheduler.invalid")
    fetched = []

    def download(url, sha, max_bytes):
        fetched.append((url, sha))
        return BENCH.read_bytes() if sha == hashlib.sha256(BENCH.read_bytes()).hexdigest() else None

    monkeypatch.setattr(worker_agent, "_download_and_verify", download)
    assert worker_agent._bench_source() == BENCH.read_text()
    assert fetched == [
        ("https://scheduler.invalid/static/xcelsior_bench.py", worker_agent.BENCH_SHA256)
    ]
    assert (tmp_path / "xcelsior_bench.py").read_bytes() == BENCH.read_bytes()

    # Installed and matching: no download.
    fetched.clear()
    assert worker_agent._bench_source() == BENCH.read_text()
    assert fetched == []

    # Installed but altered: it is not run; the pinned one is fetched instead.
    (tmp_path / "xcelsior_bench.py").write_text("print('tampered')\n")
    assert worker_agent._bench_source() == BENCH.read_text()
    assert len(fetched) == 1


def test_a_download_that_fails_verification_runs_nothing(tmp_path, monkeypatch):
    import worker_agent

    monkeypatch.setattr(worker_agent, "_sibling_module_dir", lambda name: tmp_path)
    monkeypatch.setattr(worker_agent, "SCHEDULER_URL", "https://scheduler.invalid")
    monkeypatch.setattr(worker_agent, "_download_and_verify", lambda *a, **k: None)
    assert worker_agent._bench_source() is None
    assert worker_agent.run_compute_benchmark() is None


def test_without_torch_it_prints_an_error_report_and_exits_2(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert xcelsior_bench.main([]) == 2
    import json

    report = json.loads(capsys.readouterr().out.strip())
    assert report["error"] == "no_torch"
    assert report["schema"] == xcelsior_bench.SCHEMA
    assert report["bench_version"] == xcelsior_bench.BENCH_VERSION


# ── The sustained-load summary and the verdict ───────────────────────


def test_throttle_bits_are_named():
    assert throttle_reasons(THERMAL | POWER_CAP) == ["sw_power_cap", "hw_thermal"]
    assert throttle_reasons(0) == []


def test_a_card_that_thermal_throttles_under_sustained_load_fails():
    """Cool for the first minutes, throttled for the last 20 % of the window."""
    summary = summarize_sustained(_samples(300, 60, THERMAL, temp=83.0), 300.0)
    assert summary["thermal_throttle_share"] == 0.2
    assert summary["sm_clock_min_mhz"] == 1500 and summary["sm_clock_p50_mhz"] == 1900
    result = verdict(summary, 11.2, 12.0)
    assert result["result"] == "fail"
    assert result["faults"] == ["thermal_throttling"]


def test_a_brief_throttle_under_the_threshold_passes():
    summary = summarize_sustained(_samples(300, 15, THERMAL), 300.0)
    assert summary["thermal_throttle_share"] == 0.05
    assert verdict(summary, 11.2, 12.0)["result"] == "pass"


def test_the_board_power_limit_is_reported_but_is_not_a_fault():
    """A card at its configured power cap is working as designed; a power brake is not."""
    capped = summarize_sustained(_samples(300, 300, POWER_CAP), 300.0)
    assert capped["power_cap_share"] == 1.0 and capped["power_fault_share"] == 0.0
    assert verdict(capped, 11.2, 12.0)["result"] == "pass"

    braked = summarize_sustained(_samples(300, 90, POWER_BRAKE), 300.0)
    assert verdict(braked, 11.2, 12.0)["faults"] == ["power_throttling"]


def test_allocatable_vram_short_of_the_listing_fails():
    """A "12 GB" listing whose card lets a job allocate 9 GB."""
    summary = summarize_sustained(_samples(300, 0, 0), 300.0)
    result = verdict(summary, 9.0, 12.0)
    assert result["result"] == "fail"
    assert result["faults"] == ["allocatable_vram_below_listing"]
    # 11.2 of 12 is 93 %: above the 90 % floor.
    assert verdict(summary, 11.2, 12.0)["result"] == "pass"


def test_a_missing_measurement_is_unknown_not_a_pass():
    no_bits = summarize_sustained(
        [{"sm_clock_mhz": 1900, "temp_c": 70, "power_w": 170, "throttle_bits": None}] * 300,
        300.0,
    )
    assert no_bits["thermal_throttle_share"] is None
    assert verdict(no_bits, 11.2, 12.0)["result"] == "unknown"
    assert verdict(None, 11.2, 12.0)["unknown"] == ["sustained_load"]
    full = summarize_sustained(_samples(300, 0, 0), 300.0)
    no_probe = verdict(full, None, 12.0)
    assert no_probe["result"] == "unknown" and no_probe["unknown"] == ["allocatable_vram"]


def test_a_short_sustained_phase_cannot_pass():
    """XCELSIOR_BENCH_SUSTAINED_SEC=5 must not turn the throttle check into a formality."""
    short = summarize_sustained(_samples(5, 0, 0), 5.0)
    result = verdict(short, 11.2, 12.0)
    assert result["result"] == "unknown"
    assert result["unknown"] == ["sustained_load_too_short"]
    # A fault seen in a short run is still a fault.
    assert verdict(summarize_sustained(_samples(5, 5, THERMAL), 5.0), 11.2, 12.0)["result"] == (
        "fail"
    )
    # Long enough in time but mostly unsampled is not enough either.
    sparse = summarize_sustained(_samples(100, 0, 0), 300.0)
    assert verdict(sparse, 11.2, 12.0)["unknown"] == ["sustained_load_too_short"]
    # Longer than the minimum is fine.
    longer = summarize_sustained(_samples(600, 0, 0), 600.0)
    assert verdict(longer, 11.2, 12.0)["result"] == "pass"
    assert xcelsior_bench.MIN_SUSTAINED_SECONDS == xcelsior_bench.DEFAULT_SUSTAINED_SECONDS


def test_a_dead_temperature_sensor_reads_as_none():
    summary = summarize_sustained(_samples(10, 0, 0, temp=0), 10.0)
    assert summary["temp_peak_c"] is None


# ── The server reads the same thresholds and the sustained phase ─────


def test_the_server_thresholds_are_the_benchmark_constants():
    from verification import VERIFICATION_THRESHOLDS

    assert VERIFICATION_THRESHOLDS["max_throttle_share_pct"] == pytest.approx(
        xcelsior_bench.THROTTLE_FAULT_SHARE * 100
    )
    assert VERIFICATION_THRESHOLDS["min_allocatable_vram_pct"] == pytest.approx(
        xcelsior_bench.ALLOCATABLE_FAULT_RATIO * 100
    )


def _report_with(sustained: dict | None) -> dict:
    report = {"gpu_temp_celsius": 72.0, "gpu_temp_samples": 300}
    if sustained is not None:
        report["bench"] = {"bench_version": "1.0.0", "sustained": sustained}
    return report


def test_the_server_thermal_check_fails_a_throttling_card_under_the_temperature_limit():
    from verification import check_thermal_stability

    throttled = summarize_sustained(_samples(300, 60, THERMAL, temp=72.0), 300.0)
    result = check_thermal_stability(_report_with(throttled))
    assert result.passed is False
    assert "thermal throttling 20.0%" in result.actual

    clean = summarize_sustained(_samples(300, 0, 0, temp=72.0), 300.0)
    assert check_thermal_stability(_report_with(clean)).passed is True


def test_the_server_thermal_check_is_unknown_without_throttle_samples():
    from verification import CHECK_UNKNOWN, check_thermal_stability

    result = check_thermal_stability(_report_with({"duration_s": 300, "temp_peak_c": 72.0}))
    assert result.status == CHECK_UNKNOWN


def test_the_server_thermal_check_refuses_a_short_clean_run_but_fails_a_short_faulty_one():
    from verification import CHECK_UNKNOWN, check_thermal_stability

    short = check_thermal_stability(_report_with(summarize_sustained(_samples(5, 0, 0), 5.0)))
    assert short.status == CHECK_UNKNOWN
    assert "needs 300s" in short.details
    faulty = summarize_sustained(_samples(5, 5, THERMAL, temp=72.0), 5.0)
    result = check_thermal_stability(_report_with(faulty))
    assert result.passed is False and result.status != CHECK_UNKNOWN


def test_a_report_from_an_older_agent_is_judged_on_peak_temperature():
    from verification import check_thermal_stability

    assert check_thermal_stability(_report_with(None)).passed is True


def test_the_server_flags_allocatable_vram_short_of_the_listing():
    from verification import check_allocatable_vram

    result = check_allocatable_vram({"allocatable_vram_gb": 9.0, "advertised_vram_gb": 12.0})
    assert result.passed is False and "75.0%" in result.actual


# ── Each run is kept, judged against the listing ─────────────────────


@pytest.mark.needs_db
def test_a_verification_keeps_the_run_with_a_verdict_against_the_listing():
    """The agent's own verdict cannot see the listing; the stored one is judged with it."""
    from verification import VerificationStore

    class Listed(VerificationStore):
        def get_advertised_hardware(self, host_id):
            return {"gpu_model": "RTX 3060", "vram_gb": 12.0, "gpu_count": 1, "offers": []}

    host_id = f"bench-{uuid.uuid4().hex[:10]}"
    store = Listed()
    clean = summarize_sustained(_samples(300, 0, 0), 300.0)
    bench = {
        "schema": xcelsior_bench.SCHEMA,
        "bench_version": xcelsior_bench.BENCH_VERSION,
        "gpu_model": "NVIDIA GeForce RTX 3060",
        "gpu_uuid": "GPU-aaaa",
        "allocatable_vram_gb": 9.0,
        "fp16_tflops": 25.4,
        "memory_bandwidth_gbps": 330.0,
        "sustained": clean,
        "verdict": verdict(clean, 9.0, None),  # the agent knew no listing: a pass
        "finished_at": time.time(),
    }
    assert bench["verdict"]["result"] == "pass"
    from verification import VerificationEngine

    engine = VerificationEngine(store=store)
    report = {"bench": bench, "allocatable_vram_gb": 9.0, "gpu_temp_celsius": 70.0}
    try:
        v = engine.run_verification(host_id, report)
        with store._conn() as conn:
            row = conn.execute(
                """SELECT benchmark_type, verdict, bench_version, gpu_uuid, verification_id,
                          tflops, details FROM benchmarks WHERE host_id = %s""",
                (host_id,),
            ).fetchone()
        assert row is not None
        assert row["benchmark_type"] == "xcelsior-bench"
        assert row["verdict"] == "fail"
        assert row["details"]["verdict"]["faults"] == ["allocatable_vram_below_listing"]
        assert row["details"]["advertised_vram_gb"] == 12.0
        assert row["bench_version"] == xcelsior_bench.BENCH_VERSION
        assert row["gpu_uuid"] == "GPU-aaaa"
        assert row["verification_id"] == v.verification_id
        assert row["tflops"] == 25.4
    finally:
        with store._conn() as conn:
            for table in ("benchmarks", "host_verifications", "verification_history"):
                conn.execute(f"DELETE FROM {table} WHERE host_id = %s", (host_id,))


@pytest.mark.needs_db
def test_a_report_without_the_benchmark_writes_no_history():
    from verification import VerificationStore

    host_id = f"bench-{uuid.uuid4().hex[:10]}"
    store = VerificationStore()
    assert store.record_benchmark(host_id, "v-1", {"tflops": 10.0}) is False
    with store._conn() as conn:
        n = conn.execute("SELECT count(*) AS n FROM benchmarks WHERE host_id = %s", (host_id,))
        assert n.fetchone()["n"] == 0
