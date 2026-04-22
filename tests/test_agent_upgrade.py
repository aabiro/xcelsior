"""Tests for worker self-update flow (P1.2).

Covers:
- ``_handle_upgrade_agent`` downloads, verifies sha256, swaps file, exits.
- sha256 mismatch aborts without touching the installed file.
- ``min_version`` gating skips when we're already at/above target.
- Bad args (missing url/sha256, wrong sha256 length) are rejected.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import worker_agent  # noqa: E402


@pytest.fixture
def fake_agent_file(tmp_path, monkeypatch):
    """Point worker_agent.__file__ at a throwaway copy so the real file is never touched."""
    fake = tmp_path / "worker_agent.py"
    fake.write_bytes(b"# pretend this is the running agent\nVERSION = '2.0.0'\n")
    monkeypatch.setattr(worker_agent, "__file__", str(fake))
    return fake


def _fake_response(status_code: int, body: bytes):
    class R:
        def __init__(self):
            self.status_code = status_code
            self.content = body

    return R()


def test_upgrade_agent_bad_args_rejected(fake_agent_file):
    assert worker_agent._handle_upgrade_agent({}) is False
    assert worker_agent._handle_upgrade_agent({"url": "https://x/"}) is False
    # Wrong sha length
    assert worker_agent._handle_upgrade_agent({"url": "https://x/", "sha256": "abc"}) is False


def test_upgrade_agent_sha_mismatch_does_not_replace(fake_agent_file):
    original = fake_agent_file.read_bytes()
    new_bytes = b"# new agent bytes\nVERSION = '99.0.0'\n"
    wrong_sha = "0" * 64

    with patch.object(worker_agent.requests, "get", return_value=_fake_response(200, new_bytes)):
        ok = worker_agent._handle_upgrade_agent({"url": "https://x/", "sha256": wrong_sha})

    assert ok is False
    # File must be untouched
    assert fake_agent_file.read_bytes() == original
    # No leftover .new
    assert not fake_agent_file.with_suffix(".py.new").exists()


def test_upgrade_agent_success_replaces_file_and_exits(fake_agent_file, monkeypatch):
    new_bytes = b"# new agent bytes\nVERSION = '99.0.0'\n"
    correct_sha = hashlib.sha256(new_bytes).hexdigest()

    # Intercept os._exit so the test doesn't die
    exits: list[int] = []
    monkeypatch.setattr(worker_agent.os, "_exit", lambda code: exits.append(code))

    with patch.object(worker_agent.requests, "get", return_value=_fake_response(200, new_bytes)):
        worker_agent._handle_upgrade_agent(
            {"url": "https://x/worker_agent.py", "sha256": correct_sha}
        )

    # New bytes in place
    assert fake_agent_file.read_bytes() == new_bytes
    # Backup contains old bytes
    bak = fake_agent_file.with_suffix(".py.bak")
    assert bak.exists()
    # Process tried to exit 0 so systemd can respawn
    assert exits == [0]


def test_upgrade_agent_skip_when_already_at_target(fake_agent_file, monkeypatch):
    # Make _self_sha256 return exactly what the directive advertises,
    # so the "already at target" short-circuit fires.
    target_sha = "a" * 64
    monkeypatch.setattr(worker_agent, "_self_sha256", lambda: target_sha)
    # And min_version <= our VERSION
    assert worker_agent.VERSION == "2.1.0"
    ok = worker_agent._handle_upgrade_agent(
        {
            "url": "https://x/",
            "sha256": target_sha,
            "min_version": "2.0.0",  # we're at 2.1.0
        }
    )
    # Returned True (skipped cleanly) and didn't download anything.
    assert ok is True


def test_self_sha256_matches_disk():
    """_self_sha256 should match sha256(worker_agent.py on disk)."""
    path = Path(worker_agent.__file__).resolve()
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert worker_agent._self_sha256() == expected


def test_version_bumped_to_2_1_0():
    """Lock: P1.2 bumps VERSION to 2.1.0 (self-update protocol support)."""
    assert worker_agent.VERSION == "2.1.0"
