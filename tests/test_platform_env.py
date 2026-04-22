"""Tests for platform env-var injection (P1.4)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import worker_agent  # noqa: E402


def _gpu():
    return {"gpu_model": "RTX 4090", "total_vram_gb": 24.0, "free_vram_gb": 23.5}


def test_all_reserved_keys_present():
    env = worker_agent.build_platform_env(
        {"job_id": "abcd1234", "owner": "alice@example.com", "name": "myjob"}, _gpu()
    )
    for key in worker_agent.PLATFORM_ENV_KEYS:
        assert key in env, f"missing reserved key: {key}"
        assert isinstance(env[key], str), f"{key} must be str, got {type(env[key])}"


def test_user_cannot_override_reserved_keys(monkeypatch):
    """Simulates the start_job merge order: user env first, platform env wins."""
    monkeypatch.setattr(worker_agent, "HOST_ID", "real-host-id")
    monkeypatch.setattr(worker_agent, "get_gpu_info", _gpu)

    user_env = {
        "XCELSIOR_JOB_ID": "HACKED",
        "XCELSIOR_OWNER": "attacker",
        "XCELSIOR_API_URL": "http://evil.example.com",
        "MY_OWN_VAR": "fine",
    }
    platform_env = worker_agent.build_platform_env({"job_id": "deadbeef", "owner": "alice"}, _gpu())
    merged = {**user_env, **platform_env}

    assert merged["XCELSIOR_JOB_ID"] == "deadbeef"
    assert merged["XCELSIOR_OWNER"] == "alice"
    assert merged["XCELSIOR_API_URL"] != "http://evil.example.com"
    assert merged["MY_OWN_VAR"] == "fine"  # non-reserved user vars pass through


def test_missing_gpu_info_is_graceful():
    env = worker_agent.build_platform_env({"job_id": "abcd1234"}, None)
    assert env["XCELSIOR_GPU_MODEL"] == ""
    assert env["XCELSIOR_GPU_VRAM_GB"] == ""
    # Still includes all other keys
    for key in worker_agent.PLATFORM_ENV_KEYS:
        assert key in env


def test_public_ssh_port_deterministic():
    p1 = worker_agent._compute_public_ssh_port("abcd1234")
    p2 = worker_agent._compute_public_ssh_port("abcd1234")
    assert p1 == p2
    assert 10000 <= p1 < 65001


def test_public_ssh_port_non_hex_falls_back_safely():
    # Non-hex prefix must not raise; returns 0 sentinel.
    assert worker_agent._compute_public_ssh_port("zzzz") == 0
    assert worker_agent._compute_public_ssh_port("") == 0


def test_instance_name_falls_back_to_job_id():
    env = worker_agent.build_platform_env({"job_id": "abcd1234"}, _gpu())
    assert env["XCELSIOR_INSTANCE_NAME"] == "abcd1234"
    env2 = worker_agent.build_platform_env({"job_id": "abcd1234", "name": "my-gpu"}, _gpu())
    assert env2["XCELSIOR_INSTANCE_NAME"] == "my-gpu"


def test_public_ssh_host_is_connect_xcelsior_ca():
    env = worker_agent.build_platform_env({"job_id": "abcd1234"}, _gpu())
    assert env["XCELSIOR_PUBLIC_SSH_HOST"] == "connect.xcelsior.ca"
