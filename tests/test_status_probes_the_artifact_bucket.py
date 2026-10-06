"""The status page has to notice when the artifact bucket disappears.

Every bucket in the Backblaze account was deleted at some point after
2026-08-12, and it was found on 2026-10-06 by hand, from a deploy that would
not start. Nothing had noticed in between: `storage_healthcheck` only queries
Postgres, so no probe anywhere touched the bucket. `/api/status` now lists one
object from it, and reports what it finds.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402

import routes.health as health  # noqa: E402


class _Client:
    def __init__(self, behaviour):  # noqa: ANN001
        self.behaviour = behaviour

    def list_objects(self, prefix="", max_keys=1000):  # noqa: ANN001
        return self.behaviour()


@pytest.fixture
def storage(monkeypatch):  # noqa: ANN001
    import artifacts

    def install(behaviour, backend="s3"):  # noqa: ANN001
        monkeypatch.setattr(artifacts, "StorageClient", lambda cfg: _Client(behaviour))
        monkeypatch.setattr(
            artifacts.StorageConfig, "from_env",
            classmethod(lambda cls, prefix="XCELSIOR_STORAGE": cls(backend=backend, bucket="xcelsior-artifacts")),
        )

    return install


def test_a_deleted_bucket_reads_as_down(storage) -> None:  # noqa: ANN001
    from artifacts import StorageUnavailable

    def missing():
        raise StorageUnavailable("list_objects failed: NoSuchBucket: The specified bucket does not exist")

    storage(missing)
    name, state, detail, required = health._probe_artifact_storage()
    assert (name, state) == ("Artifact storage", "down")
    assert "NoSuchBucket" in detail
    assert required is False, "a dead bucket degrades the platform; it does not block sign-in"


def test_a_hanging_provider_does_not_hang_the_status_page(storage, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(health, "_ARTIFACT_PROBE_TIMEOUT_SEC", 0.3)
    storage(lambda: time.sleep(5))
    started = time.monotonic()
    _, state, detail, _ = health._probe_artifact_storage()
    assert time.monotonic() - started < 2, "the probe waited for the provider"
    assert state == "degraded" and "no answer" in detail


def test_a_reachable_bucket_reads_as_operational(storage) -> None:  # noqa: ANN001
    storage(lambda: [])
    _, state, detail, _ = health._probe_artifact_storage()
    assert state == "operational" and "xcelsior-artifacts" in detail


def test_the_probe_is_part_of_the_status_response(storage) -> None:  # noqa: ANN001
    """Wired in, not merely defined."""
    from artifacts import StorageUnavailable

    def missing():
        raise StorageUnavailable("NoSuchBucket")

    storage(missing)
    body = health._compute_service_status()
    entry = next((s for s in body["services"] if s["name"] == "Artifact storage"), None)
    assert entry is not None and entry["state"] == "down"
    assert body["verdict"] != "operational", "a missing bucket must not leave the verdict green"
