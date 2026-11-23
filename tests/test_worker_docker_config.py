"""Every docker call the worker makes reads the worker's own config directory.

Prod, 2026-11-22: the worker unit runs with ProtectHome=true, so each plain
`docker` call (run, exec, inspect) failed to read ~/.docker/config.json and
printed "WARNING: Error loading config file ... permission denied" into the
renter's instance log, right under "Interactive instance ready".
"""

from __future__ import annotations

import json
import os

import worker_agent


def test_the_worker_points_docker_at_its_own_config(monkeypatch, tmp_path):
    monkeypatch.delenv("DOCKER_CONFIG", raising=False)
    monkeypatch.setattr(worker_agent, "_WORKER_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(worker_agent, "_registry_config_dir_cached", None)

    worker_agent._use_worker_docker_config()

    target = os.environ["DOCKER_CONFIG"]
    assert target == str(tmp_path / "docker-config")
    config = json.loads((tmp_path / "docker-config" / "config.json").read_text())
    # Readable, and nothing inherited: no credential store, no helpers.
    assert set(config) == {"auths"}


def test_an_operator_set_docker_config_is_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCKER_CONFIG", "/etc/xcelsior/docker")
    monkeypatch.setattr(worker_agent, "_registry_config_dir_cached", None)
    worker_agent._use_worker_docker_config()
    assert os.environ["DOCKER_CONFIG"] == "/etc/xcelsior/docker"
