"""P3/C6 — end-to-end snapshot integration test.

Exercises the full pipeline against a real Docker daemon:
    1. docker run a throwaway alpine container
    2. build image_ref via _build_image_ref
    3. docker commit → docker image inspect
    4. verify image exists
    5. docker rmi cleanup

Skipped by default. Set ``XCELSIOR_RUN_INTEGRATION=1`` to execute.
Requires ``docker`` on PATH and permission to run containers.
"""
from __future__ import annotations

import os
import subprocess
import time
import uuid

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("XCELSIOR_RUN_INTEGRATION") != "1",
    reason="integration test — set XCELSIOR_RUN_INTEGRATION=1 to run",
)


def _docker_available() -> bool:
    try:
        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def throwaway_container():
    if not _docker_available():
        pytest.skip("docker daemon not reachable")
    name = f"xcelsior-it-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name,
         "alpine", "sleep", "3600"],
        check=True, capture_output=True, timeout=60,
    )
    # Give daemon a moment.
    time.sleep(0.5)
    yield name
    # Cleanup.
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


def test_c6_owner_slug_produces_valid_docker_tag(throwaway_container):
    """Integration: the slug helper output is accepted by `docker tag`."""
    from routes.instances import _owner_slug

    slug = _owner_slug("integration-test@example.com")
    image_ref = f"xcelsior-test/{slug}:probe"

    # `docker commit` the running container to verify slug is tag-safe.
    commit = subprocess.run(
        ["docker", "commit", throwaway_container, image_ref],
        capture_output=True, text=True, timeout=60,
    )
    assert commit.returncode == 0, f"docker commit rejected slug: {commit.stderr}"

    # Verify inspect returns non-zero size.
    insp = subprocess.run(
        ["docker", "image", "inspect", "-f", "{{.Size}}", image_ref],
        capture_output=True, text=True, timeout=10,
    )
    assert insp.returncode == 0
    assert int(insp.stdout.strip()) > 0

    # Cleanup.
    subprocess.run(["docker", "rmi", image_ref], capture_output=True, timeout=30)


def test_c6_snapshot_pipeline_commit_then_rmi(throwaway_container):
    """Integration: simulate B6 path — commit succeeds, then rmi cleans up."""
    image_ref = f"xcelsior-test/it-{uuid.uuid4().hex[:8]}:v1"

    commit = subprocess.run(
        ["docker", "commit", throwaway_container, image_ref],
        capture_output=True, text=True, timeout=60,
    )
    assert commit.returncode == 0

    # B6 — rmi after simulated push failure.
    rmi = subprocess.run(
        ["docker", "rmi", image_ref],
        capture_output=True, text=True, timeout=30,
    )
    assert rmi.returncode == 0

    # Verify the image is truly gone.
    insp = subprocess.run(
        ["docker", "image", "inspect", image_ref],
        capture_output=True, text=True, timeout=10,
    )
    assert insp.returncode != 0, "image still exists after rmi"
