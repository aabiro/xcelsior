"""P3/B5-B9 — worker snapshot/resume hardening + bidi strip."""
from pathlib import Path


SRC_INSTANCES = Path(__file__).resolve().parent.parent / "routes" / "instances.py"
SRC_WORKER = Path(__file__).resolve().parent.parent / "worker_agent.py"


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
    idx = src.find('"snapshot_container"')
    assert idx >= 0
    # Look at the next ~4000 chars (the handler body)
    body = src[idx:idx + 6000]
    assert 'push failed' in body.lower(), "B7 distinct push-failure message missing"
    assert '"docker", "rmi"' in body, "B6 docker rmi on push-fail missing"


def test_b7_snapshot_distinguishes_commit_vs_push_errors():
    src = SRC_WORKER.read_text()
    idx = src.find('"snapshot_container"')
    body = src[idx:idx + 6000]
    assert 'commit failed' in body.lower(), "B7 distinct commit-failure message missing"
    assert 'push failed' in body.lower()


# ---------- B8 ----------

def test_b8_start_container_reports_failure_to_api():
    """start_container failure must call report_job_status to unstick UI."""
    src = SRC_WORKER.read_text()
    idx = src.find('"start_container"')
    assert idx >= 0
    end = src.find('elif name ==', idx + 1)
    body = src[idx: end if end > 0 else idx + 4000]
    # Must revert to user_paused on failure (not leave at 'running')
    assert 'report_job_status' in body, "B8 missing report_job_status callback"
    assert '"user_paused"' in body, "B8 must revert to user_paused on failure"
    assert 'resume failed' in body.lower()


def test_b8_start_container_handles_timeout():
    src = SRC_WORKER.read_text()
    idx = src.find('"start_container"')
    end = src.find('elif name ==', idx + 1)
    body = src[idx:end]
    # Both non-zero exit AND timeout should callback
    assert body.count("report_job_status") >= 2, (
        "B8 must handle both non-zero rc and TimeoutExpired with callback"
    )


# ---------- B9 ----------

def test_b9_description_strips_bidi_override_chars():
    from routes.instances import SnapshotIn

    # Trojan Source-style payload
    raw = "Hello\u202Eevil\u202C world"
    m = SnapshotIn(name="img", tag="v1", description=raw)
    assert "\u202E" not in m.description
    assert "\u202C" not in m.description
    assert "Helloevil world" == m.description


def test_b9_description_strips_all_bidi_chars():
    from routes.instances import SnapshotIn

    bidi = "".join([
        "\u202A", "\u202B", "\u202C", "\u202D", "\u202E",
        "\u2066", "\u2067", "\u2068", "\u2069",
    ])
    m = SnapshotIn(name="img", tag="v1", description=f"safe{bidi}text")
    assert m.description == "safetext"


def test_b9_description_preserves_normal_unicode():
    from routes.instances import SnapshotIn

    m = SnapshotIn(name="img", tag="v1", description="Hello 你好 émigré 🚀")
    assert m.description == "Hello 你好 émigré 🚀"
