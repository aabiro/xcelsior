"""An unavailable inventory or renamed container cannot prove lost state."""

import json
from unittest.mock import Mock

import pytest

from scripts import audit_legacy_containers as audit


def _job():
    return {"job_id": "job-1", "host_id": "host-1", "container_name": "xcl-job-1", "volume_ids": []}


@pytest.mark.parametrize(
    "status,body",
    [
        (None, None),
        (500, {"message": "daemon unavailable"}),
        (404, {"message": "not found"}),
        (200, {}),
        (200, {"Id": "abc", "State": []}),
    ],
)
def test_inspect_error_is_unknown(monkeypatch, status, body):
    monkeypatch.setattr(audit, "docker_json", lambda *args: (status, body))
    assert audit.inspect_container("xcl-job-1")["presence"] == "unknown"


def test_renamed_labeled_container_is_not_reported_missing(monkeypatch):
    def get(path, socket):
        if path == "/containers/json?all=1":
            return 200, [
                {
                    "Id": "old-state",
                    "Names": ["/renamed"],
                    "Labels": {"ai.xcelsior.job_id": "job-1"},
                }
            ]
        if path == "/info":
            return 200, {"Name": "host-1"}
        return 404, {"message": "No such container: xcl-job-1"}

    monkeypatch.setattr(audit, "docker_json", get)
    result = audit.inventory([_job()], "host-1", expected_daemon_name="host-1")
    assert result["observation_complete"] is True
    assert result["jobs"][0]["next_gate"] == "identity_conflict"
    assert result["jobs"][0]["observed"]["presence"] == "present_elsewhere"


def test_failed_census_cannot_prove_absence(monkeypatch):
    monkeypatch.setattr(
        audit,
        "docker_json",
        lambda path, *args: (
            (200, {"Name": "host-1"})
            if path == "/info"
            else (404, {"message": "No such container: xcl-job-1"})
        ),
    )
    report = audit.inventory([_job()], "host-1", expected_daemon_name="host-1")
    assert report["observation_complete"] is False
    assert report["jobs"][0]["next_gate"] == "unverified"
    assert report["jobs"][0]["observed"]["presence"] == "unknown"


def test_wrong_host_cannot_prove_missing_state(monkeypatch):
    get = Mock(return_value=(200, {"Name": "wrong-host"}))
    monkeypatch.setattr(audit, "docker_json", get)
    report = audit.inventory([_job()], "host-1", expected_daemon_name="expected-host")
    assert report["observation_complete"] is False and report["host_verified"] is False
    assert report["jobs"][0]["observed"] == {
        "presence": "unknown",
        "reason": "host_identity_unverified",
    }
    get.assert_called_once_with("/info", "/var/run/docker.sock")


def test_metadata_inventory_excludes_environment_and_command(monkeypatch):
    body = {
        "Id": "abc",
        "Name": "/xcl-job-1",
        "State": {"Status": "exited"},
        "Config": {
            "Labels": {"ai.xcelsior.job_id": "job-1"},
            "Env": ["TOKEN=secret"],
            "Cmd": ["secret"],
        },
        "Mounts": [
            {
                "Type": "volume",
                "Name": "data",
                "Destination": "/workspace",
                "RW": True,
                "DriverOptions": {"password": "secret"},
            }
        ],
    }
    monkeypatch.setattr(audit, "docker_json", lambda *args: (200, body))
    result = audit.inspect_container("xcl-job-1")
    assert result["presence"] == "present" and result["state"] == "exited"
    assert result["mounts"][0]["Name"] == "data"
    assert "secret" not in json.dumps(result)


def test_inspection_uses_only_get_on_explicit_socket(monkeypatch):
    calls = []

    class Connection:
        def __init__(self, path):
            calls.append(("socket", path))

        def request(self, method, path):
            calls.append((method, path))

        def getresponse(self):
            return self

        status = 404

        def read(self, limit):
            return b'{"message":"No such container: job"}'

        def close(self):
            calls.append(("closed", True))

    monkeypatch.setattr(audit, "DockerSocket", Connection)
    assert audit.inspect_container("name/with?characters", "/test/docker.sock") == {
        "presence": "missing"
    }
    assert calls == [
        ("socket", "/test/docker.sock"),
        ("GET", "/containers/name%2Fwith%3Fcharacters/json"),
        ("closed", True),
    ]
