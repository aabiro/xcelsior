import pytest

from host_profiles import (
    build_host_add_command,
    build_market_list_command,
    get_host_profile,
    validate_job_for_profile,
)
from reputation import GPU_REFERENCE_PRICING_CAD, get_reference_rate


def test_rtx3060_profile_defaults():
    profile = get_host_profile("rtx3060-local")

    assert profile["gpu_model"] == "RTX 3060"
    assert profile["total_vram_gb"] == 12.0
    assert profile["usable_vram_gb"] == 11.0
    assert profile["default_rate_cad_per_hour"] == 0.16
    assert "worker telemetry" in " ".join(profile["recommended_workloads"])


def test_rtx3060_aliases_resolve():
    assert get_host_profile("3060")["profile_id"] == "rtx3060-local"
    assert get_host_profile("local-dev")["profile_id"] == "rtx3060-local"


def test_unknown_profile_raises_clear_error():
    with pytest.raises(KeyError, match="Unknown host profile"):
        get_host_profile("not-a-gpu")


def test_host_add_command_uses_profile_defaults():
    cmd = build_host_add_command("3060", "tower-server", "100.64.0.4", province="ON")

    assert "--gpu 'RTX 3060'" in cmd
    assert "--vram 12.0" in cmd
    assert "--free-vram 11.0" in cmd
    assert "--rate 0.16" in cmd
    assert "--province ON" in cmd


def test_market_command_uses_profile_rate():
    cmd = build_market_list_command("3060", "tower-server", owner="internal")

    assert "market-list tower-server" in cmd
    assert "--gpu 'RTX 3060'" in cmd
    assert "--price 0.16" in cmd
    assert "--owner internal" in cmd


def test_validate_job_for_profile_accepts_small_job():
    result = validate_job_for_profile("3060", 8, gpu_model="NVIDIA GeForce RTX 3060")

    assert result["fits"] is True
    assert result["reasons"] == []


def test_validate_job_for_profile_rejects_oversized_job():
    result = validate_job_for_profile("3060", 16)

    assert result["fits"] is False
    assert "exceeds usable profile VRAM" in result["reasons"][0]


def test_rtx3060_reference_pricing_available():
    assert "RTX 3060" in GPU_REFERENCE_PRICING_CAD
    assert get_reference_rate("NVIDIA GeForce RTX 3060") == 0.16
