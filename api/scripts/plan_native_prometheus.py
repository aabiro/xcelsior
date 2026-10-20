#!/usr/bin/env python3
"""Prepare and validate additive native Prometheus changes without applying them.

Reads the existing configuration locally. The JSON report contains only new
operations, hashes and validation results, never the original configuration or
credentials. Validation uses stdin; no service files are written or reloaded.
An explicit --output writes a private candidate file without overwriting files.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

import yaml

GROUP_NAME = "xcelsior.legacy-retirement"


def parse_target(value: str) -> tuple[str, str]:
    label, separator, target = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", label):
        raise ValueError("API targets require a bounded replica=loopback-address:port")
    parsed = urlsplit("http://" + target)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError as exc:
        raise ValueError("API target address or port is invalid") from exc
    if (
        not address.is_loopback
        or not port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Native API scrape targets must be explicit loopback sockets")
    host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    return label, f"{host}:{port}"


def prepare_config(config: dict, targets: list[tuple[str, str]], rules_path: str):
    path = PurePosixPath(rules_path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Rules destination must be an absolute path without traversal")
    if not targets or len(set(label for label, _ in targets)) != len(targets):
        raise ValueError("An explicit unique API replica inventory is required")
    if len(set(target for _, target in targets)) != len(targets):
        raise ValueError("Duplicate API sockets are not an independent replica inventory")
    result = deepcopy(config)
    operations = []
    if "scrape_configs" not in result:
        operations.append({"op": "add", "path": "/scrape_configs", "value": []})
    jobs = result.setdefault("scrape_configs", [])
    matches = [index for index, job in enumerate(jobs) if job.get("job_name") == "xcelsior-api"]
    if len(matches) > 1:
        raise ValueError("Existing API scrape job is ambiguous")
    if not matches:
        job = {
            "job_name": "xcelsior-api",
            "metrics_path": "/metrics/prometheus",
            "static_configs": [
                {"targets": [target], "labels": {"replica": label}} for label, target in targets
            ],
        }
        jobs.append(job)
        operations.append({"op": "add", "path": "/scrape_configs/-", "value": job})
    else:
        index = matches[0]
        job = jobs[index]
        if job.get("metrics_path", "/metrics") != "/metrics/prometheus":
            raise ValueError("Existing API scrape path requires manual review")
        if "static_configs" not in job:
            operations.append(
                {"op": "add", "path": f"/scrape_configs/{index}/static_configs", "value": []}
            )
        groups = job.setdefault("static_configs", [])
        for label, target in targets:
            matching = [group for group in groups if target in group.get("targets", [])]
            if matching:
                if len(matching) != 1 or matching[0].get("labels", {}).get("replica") != label:
                    raise ValueError("Existing API target identity requires manual review")
                continue
            if any(group.get("labels", {}).get("replica") == label for group in groups):
                raise ValueError("Existing API replica points at another socket")
            group = {"targets": [target], "labels": {"replica": label}}
            groups.append(group)
            operations.append(
                {"op": "add", "path": f"/scrape_configs/{index}/static_configs/-", "value": group}
            )
    if "rule_files" not in result:
        operations.append({"op": "add", "path": "/rule_files", "value": []})
    rules = result.setdefault("rule_files", [])
    if rules_path not in rules:
        rules.append(rules_path)
        operations.append({"op": "add", "path": "/rule_files/-", "value": rules_path})
    return result, operations


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def validate(tool: str, arguments: list[str], document: str) -> dict:
    result = subprocess.run(
        [tool, *arguments, "/dev/stdin"], input=document, text=True, capture_output=True, timeout=30
    )
    # Native validation errors can quote configuration values. Report status
    # and digests, never raw output containing credentials from the source file.
    return {
        "exit_code": result.returncode,
        "stdout_sha256": digest(result.stdout),
        "stderr_sha256": digest(result.stderr),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--api-target", action="append", required=True)
    parser.add_argument("--rules-path", required=True)
    parser.add_argument(
        "--rule-source", required=True, help="Alert-rule YAML, or - for one JSON group on stdin"
    )
    parser.add_argument("--promtool", default=shutil.which("promtool"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.promtool:
        parser.error("An installed native promtool is required for validation")
    try:
        source = args.config.read_text()
        original = yaml.safe_load(source)
        if not isinstance(original, dict):
            raise ValueError("Prometheus source must be a mapping")
        if args.rule_source == "-":
            group = json.load(sys.stdin)
        else:
            document = yaml.safe_load(Path(args.rule_source).read_text())
            groups = [item for item in document["groups"] if item.get("name") == GROUP_NAME]
            if len(groups) != 1:
                raise ValueError("Exactly one legacy retirement group is required")
            group = groups[0]
        if not isinstance(group, dict) or group.get("name") != GROUP_NAME:
            raise ValueError("Only the legacy retirement group may be prepared")
        if args.config.resolve() == Path(args.rules_path).resolve():
            raise ValueError("Rules destination must not replace the main configuration")
        targets = [parse_target(value) for value in args.api_target]
        candidate, operations = prepare_config(original, targets, args.rules_path)
        candidate_text = yaml.safe_dump(candidate, sort_keys=False)
        rules_text = yaml.safe_dump({"groups": [group]}, sort_keys=False)
        validations = {
            "config": validate(args.promtool, ["check", "config", "--syntax-only"], candidate_text),
            "rules": validate(args.promtool, ["check", "rules"], rules_text),
        }
        version = subprocess.run(
            [args.promtool, "--version"], check=True, capture_output=True, text=True, timeout=10
        ).stdout.splitlines()[0]
        passed = all(item["exit_code"] == 0 for item in validations.values())
        report = {
            "read_only": args.output is None,
            "applied": False,
            "source_sha256": digest(source),
            "candidate_sha256": digest(candidate_text),
            "rules_sha256": digest(rules_text),
            "api_targets": dict(targets),
            "native_tool_version": version,
            "operations": operations,
            "rules_document": {"groups": [group]},
            "validation": validations,
            "passed": passed,
        }
        if passed and args.output:
            fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as output:
                output.write(candidate_text)
            report["candidate_written"] = True
        print(json.dumps(report, indent=2))
        return 0 if passed else 2
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        yaml.YAMLError,
        subprocess.SubprocessError,
    ) as exc:
        print(
            json.dumps(
                {
                    "passed": False,
                    "applied": False,
                    "error_type": type(exc).__name__,
                    "message": "Cannot prepare a validated native monitoring plan",
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
