"""P3/B5-B9 — worker snapshot/resume hardening + bidi strip."""

from pathlib import Path

SRC_INSTANCES = Path(__file__).resolve().parent.parent / "api" / "routes" / "instances.py"
SRC_WORKER = Path(__file__).resolve().parent.parent / "workers" / "worker_agent.py"


# ---------- B5 ----------


def test_b5_owner_slug_uses_16_char_digest():
    """hash prefix must be 16 hex chars (64 bits entropy), not 8."""
    from routes.instances import _owner_slug

    slug = _owner_slug("alice@example.com")
    digest = slug.rsplit("-", 1)[-1]
    assert len(digest) == 16, f"expected 16-char digest, got {len(digest)}: {slug!r}"
    assert all(c in "0123456789abcdef" for c in digest)


def test_b5_owner_slug_deterministic():
    from routes.instances import _owner_slug

    assert _owner_slug("alice@example.com") == _owner_slug("alice@example.com")


def test_b5_source_no_longer_uses_short_digest():
    src = SRC_INSTANCES.read_text()
    # Find _owner_slug function body
    idx = src.find("def _owner_slug(")
    end = src.find("\ndef ", idx + 1)
    body = src[idx:end]
    assert "hexdigest()[:8]" not in body, "_owner_slug still truncates to 8 chars"
    assert "hexdigest()[:16]" in body


# ---------- B6 + B7 ----------


def test_b6_snapshot_rmi_on_push_failure():
    """Push-failure path must `docker rmi` the local tag to prevent disk leak."""
    src = SRC_WORKER.read_text()
    # Find the snapshot_container branch
    idx = src.find('elif name == "snapshot_container"')
    assert idx >= 0
    body = src[idx : idx + 6000]
    assert "push failed" in body.lower(), "B7 distinct push-failure message missing"
    assert '"docker", "rmi"' in body, "B6 docker rmi on push-fail missing"


def test_b7_snapshot_distinguishes_commit_vs_push_errors():
    src = SRC_WORKER.read_text()
    idx = src.find('elif name == "snapshot_container"')
    body = src[idx : idx + 6000]
    assert "commit failed" in body.lower(), "B7 distinct commit-failure message missing"
    assert "push failed" in body.lower()


# ---------- B8 ----------
# B8 hardened the unfenced host `start_container` directive (report `stopped`
# when `docker start` failed or timed out). That directive was retired with
# the rest of the unfenced host lifecycle (40855f8a): a stopped container is
# now resumed only by a fresh fenced `start_attempt`, so there is no unfenced
# start left whose failure could strand the UI. These assert the retirement.


def test_b8_start_container_is_no_longer_a_worker_directive():
    import worker_agent as worker

    assert "start_container" not in worker._AGENT_COMMAND_ALLOWED
    assert 'name == "start_container"' not in SRC_WORKER.read_text()


def test_b8_delivered_start_container_is_refused_without_touching_docker(monkeypatch):
    from unittest import mock

    import worker_agent as worker

    docker = mock.Mock(side_effect=AssertionError("retired directive reached Docker"))
    monkeypatch.setattr(worker.subprocess, "run", docker)
    monkeypatch.setattr(worker, "_v2_completed_command_result", lambda _: None)
    nack = mock.Mock()
    monkeypatch.setattr(worker, "v2_nack_command", nack)
    cmd = {"command_id": "old-start", "command": "start_container", "args": {"job_id": "j"}}
    assert worker._dispatch_v2_command(cmd) == 0
    nack.assert_called_once_with(
        "old-start", "unsupported_command", {"command": "start_container"}, retryable=False
    )
    docker.assert_not_called()


# ---------- B9 ----------


def test_b9_description_strips_bidi_override_chars():
    from routes.instances import SnapshotIn

    # Trojan Source-style payload
    raw = "Hello\u202eevil\u202c world"
    m = SnapshotIn(name="img", tag="v1", description=raw)
    assert "\u202e" not in m.description
    assert "\u202c" not in m.description
    assert "Helloevil world" == m.description


def test_b9_description_strips_all_bidi_chars():
    from routes.instances import SnapshotIn

    bidi = "".join(
        [
            "\u202a",
            "\u202b",
            "\u202c",
            "\u202d",
            "\u202e",
            "\u2066",
            "\u2067",
            "\u2068",
            "\u2069",
        ]
    )
    m = SnapshotIn(name="img", tag="v1", description=f"safe{bidi}text")
    assert m.description == "safetext"


def test_b9_description_preserves_normal_unicode():
    from routes.instances import SnapshotIn

    m = SnapshotIn(name="img", tag="v1", description="Hello 你好 émigré 🚀")
    assert m.description == "Hello 你好 émigré 🚀"
