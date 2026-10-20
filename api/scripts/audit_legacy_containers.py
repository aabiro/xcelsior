#!/usr/bin/env python3
"""Read-only container metadata inventory for legacy-job retirement.

Uses only GET requests on an explicit local Docker Unix socket. It never starts,
stops, exports or removes a container, and excludes environment/command content.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import http.client
import json
from pathlib import Path
import socket
import sys
from urllib.parse import quote


class DockerSocket(http.client.HTTPConnection):
    def __init__(self, path: str):
        super().__init__("localhost", timeout=10)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def docker_json(path: str, socket_path: str):
    connection = DockerSocket(socket_path)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            return None, None
        body = json.loads(raw)
        return response.status, body
    except (OSError, http.client.HTTPException, ValueError, TypeError):
        return None, None
    finally:
        connection.close()


def inspect_container(name: str, socket_path: str = "/var/run/docker.sock") -> dict:
    status, body = docker_json(f"/containers/{quote(name, safe='')}/json", socket_path)
    try:
        if (
            status == 404
            and isinstance(body, dict)
            and str(body.get("message", "")).lower().startswith("no such container")
        ):
            return {"presence": "missing"}
        if status != 200:
            return {"presence": "unknown", "reason": f"docker_http_{status}"}
        if (
            not isinstance(body, dict)
            or not body.get("Id")
            or not isinstance(body.get("State"), dict)
        ):
            return {"presence": "unknown", "reason": "malformed_inspect"}
        labels = (body.get("Config") or {}).get("Labels") or {}
        mounts = body.get("Mounts") or []
        return {
            "presence": "present",
            "container_id": body["Id"],
            "container_name": str(body.get("Name") or "").lstrip("/"),
            "state": body["State"].get("Status", "unknown"),
            "job_label": labels.get("ai.xcelsior.job_id") or labels.get("xcelsior.job_id"),
            "attempt_label": labels.get("ai.xcelsior.attempt_id")
            or labels.get("xcelsior.attempt_id"),
            "mounts": [
                {k: mount.get(k) for k in ("Type", "Name", "Source", "Destination", "RW")}
                for mount in mounts
            ],
        }
    except (ValueError, TypeError, AttributeError):
        return {"presence": "unknown", "reason": "inspect_unavailable_or_invalid"}


def container_census(socket_path: str) -> list[dict] | None:
    status, body = docker_json("/containers/json?all=1", socket_path)
    if status != 200 or not isinstance(body, list):
        return None
    records = []
    for item in body:
        if (
            not isinstance(item, dict)
            or not item.get("Id")
            or not isinstance(item.get("Names"), list)
        ):
            return None
        labels = item.get("Labels") or {}
        if not isinstance(labels, dict):
            return None
        records.append(
            {
                "container_id": item["Id"],
                "names": [str(name).lstrip("/") for name in item["Names"]],
                "job_label": labels.get("ai.xcelsior.job_id") or labels.get("xcelsior.job_id"),
                "attempt_label": labels.get("ai.xcelsior.attempt_id")
                or labels.get("xcelsior.attempt_id"),
            }
        )
    return records


def inventory(
    jobs: list[dict],
    host_id: str,
    socket_path: str = "/var/run/docker.sock",
    *,
    expected_daemon_name: str | None = None,
) -> dict:
    records = []
    info_status, info = docker_json("/info", socket_path)
    daemon_name = info.get("Name") if info_status == 200 and isinstance(info, dict) else None
    host_verified = bool(expected_daemon_name) and daemon_name == expected_daemon_name
    census = container_census(socket_path) if host_verified else None
    for job in jobs:
        if job.get("host_id") != host_id:
            continue
        job_id = str(job["job_id"])
        name = str(job.get("container_name") or f"xcl-{job_id}")
        observed = (
            inspect_container(name, socket_path)
            if host_verified
            else {"presence": "unknown", "reason": "host_identity_unverified"}
        )
        matches = [
            item
            for item in census or []
            if item.get("job_label") == job_id
            or any(
                candidate == name or candidate.startswith(f"xcl-{job_id}-")
                for candidate in item["names"]
            )
        ]
        alternatives = [
            item for item in matches if item["container_id"] != observed.get("container_id")
        ]
        if alternatives and observed["presence"] == "present":
            observed["matching_containers"] = alternatives
        if observed["presence"] == "missing" and census is None:
            observed = {"presence": "unknown", "reason": "container_census_unavailable"}
        elif observed["presence"] == "missing" and matches:
            observed = {"presence": "present_elsewhere", "matching_containers": matches}
        if observed["presence"] == "unknown":
            disposition = "unverified"
        elif observed["presence"] == "missing":
            disposition = "recover_storage" if job.get("volume_ids") else "assess_reprovision"
        elif (
            observed["presence"] == "present_elsewhere"
            or alternatives
            or observed.get("attempt_label")
            or observed.get("job_label") not in (None, "", job_id)
            or observed.get("container_name") != name
        ):
            disposition = "identity_conflict"
        elif observed.get("state") in ("exited", "created"):
            disposition = "preserve_and_export"
        else:
            disposition = "investigate_runtime_state"
        records.append(
            {
                "job_id": job_id,
                "host_id": host_id,
                "container_name": name,
                "volume_ids": job.get("volume_ids") or [],
                "observed": observed,
                "next_gate": disposition,
            }
        )
    return {
        "host_id": host_id,
        "read_only": True,
        "socket": socket_path,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "daemon_name": daemon_name,
        "expected_daemon_name": expected_daemon_name,
        "host_verified": host_verified,
        "census_complete": census is not None,
        "observation_complete": census is not None
        and all(j["observed"]["presence"] != "unknown" for j in records),
        "jobs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-id", required=True)
    parser.add_argument(
        "--expected-daemon-name",
        required=True,
        help="Exact Docker daemon name independently verified for this host",
    )
    parser.add_argument("--jobs", required=True, help="JSON inventory file or - for stdin")
    parser.add_argument("--socket", default="/var/run/docker.sock")
    parser.add_argument("--output", default="-", help="JSON output file or - for stdout")
    args = parser.parse_args()
    payload = json.loads(sys.stdin.read() if args.jobs == "-" else Path(args.jobs).read_text())
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not all(
        isinstance(job, dict) and job.get("job_id") for job in jobs
    ):
        parser.error("jobs must be a list of job objects with job_id")
    report = inventory(
        jobs, args.host_id, args.socket, expected_daemon_name=args.expected_daemon_name
    )
    result = json.dumps(report, indent=2) + "\n"
    if args.output == "-":
        print(result, end="")
    else:
        Path(args.output).write_text(result)
    if not report["observation_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
