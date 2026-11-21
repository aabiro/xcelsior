#!/usr/bin/env python3
"""Validate and optionally apply scoped fixes to the existing native alert stack.

Receiver credentials and unrelated rules stay host-owned. No configuration or
validator output is printed, since either can contain notification credentials.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import glob
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.request

import yaml


def merge_rules(
    installed: dict, source: dict, group_name: str, *, allow_new_group: bool = False
) -> dict:
    result = deepcopy(installed)
    groups = [g for g in result["groups"] if g["name"] == group_name]
    incoming = [g for g in source["groups"] if g["name"] == group_name]
    if len(incoming) == 1 and not groups and allow_new_group:
        result["groups"].append(deepcopy(incoming[0]))
        return result
    if len(groups) != 1 or len(incoming) != 1:
        raise ValueError("Expected exactly one installed and source rule group")
    replacements = {r["alert"]: r for r in incoming[0]["rules"]}
    rules = groups[0]["rules"]
    existing = {r.get("alert") for r in rules}
    groups[0]["rules"] = [deepcopy(replacements.get(r.get("alert"), r)) for r in rules]
    groups[0]["rules"].extend(
        deepcopy(r) for name, r in replacements.items() if name not in existing
    )
    return result


def prepare_prometheus(config: dict) -> dict:
    result = deepcopy(config)
    jobs = [j for j in result["scrape_configs"] if j["job_name"] == "xcelsior-api"]
    if len(jobs) != 1 or jobs[0].get("metrics_path") != "/metrics/prometheus":
        raise ValueError("Expected the existing native API scrape job")
    groups = jobs[0].setdefault("static_configs", [])
    for replica, port in (("green", 9500), ("blue", 9501)):
        target = f"127.0.0.1:{port}"
        matches = [g for g in groups if target in g.get("targets", [])]
        if matches:
            if len(matches) != 1 or matches[0].get("labels", {}).get("replica") != replica:
                raise ValueError("Existing API target identity needs review")
        else:
            if any(g.get("labels", {}).get("replica") == replica for g in groups):
                raise ValueError("Existing replica points at a different target")
            groups.append({"targets": [target], "labels": {"replica": replica}})
    return result


def prepare_alertmanager(config: dict) -> dict:
    result = deepcopy(config)
    routes = [
        r
        for r in result["route"].get("routes", [])
        if {'platform="xcelsior"', 'component="control-plane"'} <= set(r.get("matchers", []))
    ]
    if len(routes) != 1:
        raise ValueError("Expected the existing control-plane receiver route")
    route = routes[0]
    route["group_by"] = ["alertname", "job", "service"]
    route["group_wait"] = "30s"
    route["group_interval"] = "5m"
    route["repeat_interval"] = "12h"
    # Preserve more-specific existing child routes and every receiver secret.
    critical = {
        "receiver": route["receiver"],
        "matchers": ['severity="critical"'],
        "repeat_interval": "4h",
    }
    children = route.setdefault("routes", [])
    if critical not in children:
        children.append(critical)
    policies = [
        {
            "source_matchers": ['alertname="XcelsiorApiUnavailable"'],
            "target_matchers": ['alertname=~"XcelsiorControlPlaneMetrics.*"'],
            "equal": ["job", "platform"],
        },
        {
            "source_matchers": ['alertname="XcelsiorControlPlaneMetricsUnavailable"'],
            "target_matchers": ['alertname="XcelsiorControlPlaneMetricsStale"'],
            "equal": ["job", "platform"],
        },
        {
            "source_matchers": ['alertname="TowerHostDown"'],
            "target_matchers": [
                'alertname=~"TowerOllamaDown|TowerStorageExporterDown|TowerGPUExporterDown|TowerHealthCheckStale|TowerPkgscanStale"'
            ],
            "equal": ["host"],
        },
        {
            "source_matchers": ['alertname="TowerDiskCritical"'],
            "target_matchers": [
                'alertname=~"TowerDiskHigh|TowerSrvDiskHigh|TowerDiskFillsWithin24h"'
            ],
            "equal": ["instance", "mountpoint"],
        },
        {
            "source_matchers": ['alertname="XcelsiorProdDiskCritical"'],
            "target_matchers": ['alertname=~"XcelsiorProdDiskHigh|XcelsiorProdDiskFillsWithin24h"'],
            "equal": ["instance", "mountpoint"],
        },
        {
            "source_matchers": ['alertname="XcelsiorProdVolumeFull"'],
            "target_matchers": ['alertname="XcelsiorProdVolumeHigh"'],
            "equal": ["instance", "mountpoint"],
        },
        {
            "source_matchers": ['alertname="TowerGPUTempCritical"'],
            "target_matchers": ['alertname="TowerGPUTempHigh"'],
            "equal": ["instance", "uuid"],
        },
    ]
    inhibitors = result.setdefault("inhibit_rules", [])
    for policy in policies:
        if policy not in inhibitors:
            inhibitors.append(policy)
    return result


def prepare_integrity_rules(config: dict) -> dict:
    result = deepcopy(config)
    for group in result["groups"]:
        # A successful diagnostic completion is status, not an incident.
        group["rules"] = [
            rule for rule in group["rules"] if rule.get("alert") != "TowerCoreCycleComplete"
        ]
    return result


def validate(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} {command[1]} validation failed")


def replace_file(path: Path, data: bytes) -> None:
    """Atomically replace a host-owned file while retaining its access policy."""
    stat = path.stat()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), stat.st_mode & 0o777)
            os.fchown(output.fileno(), stat.st_uid, stat.st_gid)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_reloads(started_at: float) -> None:
    """HUP is asynchronous; require each process to confirm the new load."""
    pending = {9091: "prometheus", 9093: "alertmanager"}
    deadline = time.monotonic() + 20
    while pending and time.monotonic() < deadline:
        for port, prefix in list(pending.items()):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/metrics", timeout=2
                ) as response:
                    metrics = {}
                    for line in response.read().decode().splitlines():
                        fields = line.split()
                        if len(fields) == 2 and fields[0].startswith(
                            prefix + "_config_last_reload_"
                        ):
                            metrics[fields[0]] = float(fields[1])
                if (
                    metrics.get(prefix + "_config_last_reload_successful") == 1
                    and metrics.get(prefix + "_config_last_reload_success_timestamp_seconds", 0)
                    >= started_at
                ):
                    del pending[port]
            except (OSError, ValueError):
                pass
        if pending:
            time.sleep(0.25)
    if pending:
        raise RuntimeError("Native monitoring did not confirm its configuration reload")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("/etc/prometheus"))
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "infra/observability/prometheus",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    base = args.config_dir
    sources = args.source_dir
    paths = [
        base / "prometheus.yml",
        base / "alertmanager.yml",
        base / "xcelsior-rules/control-plane.yml",
        base / "rules/tower.yml",
    ]
    integrity_path = base / "rules/tower-integrity.yml"
    if integrity_path.exists():
        paths.append(integrity_path)
    originals = {p: p.read_bytes() for p in paths}
    docs = {p: yaml.safe_load(data) for p, data in originals.items()}
    control_plane_source = yaml.safe_load((sources / "native-control-plane-rules.yml").read_text())
    control_plane = merge_rules(docs[paths[2]], control_plane_source, "xcelsior.control-plane")
    updated = {
        paths[0]: prepare_prometheus(docs[paths[0]]),
        paths[1]: prepare_alertmanager(docs[paths[1]]),
        paths[2]: merge_rules(
            control_plane, control_plane_source, "xcelsior.prod-host-disk", allow_new_group=True
        ),
        paths[3]: merge_rules(
            docs[paths[3]],
            yaml.safe_load((sources / "native-tower-capacity-rules.yml").read_text()),
            "tower",
        ),
    }
    if integrity_path in docs:
        updated[integrity_path] = prepare_integrity_rules(docs[integrity_path])
    # This old alert name refers to the inference router, not the Ollama daemon.
    for group in updated[paths[3]]["groups"]:
        for rule in group["rules"]:
            if rule.get("alert") == "TowerOllamaDown":
                rule["annotations"] = {
                    "summary": "Tower inference router on port 8001 is unavailable",
                    "description": "Check tower-serverless.service and its Python environment; Ollama on port 11434 is a separate service.",
                }
    changed = {p: d for p, d in updated.items() if d != docs[p]}
    if not changed:
        print("Native monitoring configuration already matches; no reload needed.")
        return 0
    with tempfile.TemporaryDirectory(prefix="native-alerts-") as folder:
        candidates = {}
        for number, (path, doc) in enumerate(changed.items()):
            candidate = Path(folder) / f"{number}.yml"
            candidate.write_text(yaml.safe_dump(doc, sort_keys=False, width=110))
            candidate.chmod(0o600)
            candidates[path] = candidate
        # Validate the complete candidate inventory, including unrelated rules.
        candidate_config = deepcopy(updated[paths[0]])
        inventory = []
        for pattern in candidate_config.get("rule_files", []):
            for name in sorted(glob.glob(pattern)):
                path = Path(name)
                inventory.append(str(candidates.get(path, path)))
        if not all(str(candidates.get(p, p)) in inventory for p in paths[2:]):
            raise ValueError("Managed rule files are not loaded by the installed configuration")
        candidate_config["rule_files"] = inventory
        validation_config = Path(folder) / "complete-config.yml"
        validation_config.write_text(yaml.safe_dump(candidate_config, sort_keys=False))
        validation_config.chmod(0o600)
        validate(["promtool", "check", "config", str(validation_config)])
        validate(["amtool", "check-config", str(candidates.get(paths[1], paths[1]))])
        print(f"Complete native configuration validated; {len(changed)} file(s) changed.")
        if not args.apply:
            return 0
        # Refuse to overwrite a deployment/operator change made during validation.
        if any(p.read_bytes() != original for p, original in originals.items()):
            raise RuntimeError("Installed configuration changed during validation")
        backup = base / ".alert-backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup.mkdir(parents=True, mode=0o700)
        for path in changed:
            destination = backup / path.relative_to(base)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        try:
            for path, candidate in candidates.items():
                replace_file(path, candidate.read_bytes())
            validate(["promtool", "check", "config", str(paths[0])])
            validate(["amtool", "check-config", str(paths[1])])
            reload_started_at = time.time()
            for service in ["prometheus", "prometheus-alertmanager"]:
                subprocess.run(
                    ["systemctl", "kill", "--kill-whom=main", "--signal=HUP", service],
                    check=True,
                    capture_output=True,
                )
            verify_reloads(reload_started_at)
        except Exception:
            for path in changed:
                replace_file(path, originals[path])
            for service in ["prometheus", "prometheus-alertmanager"]:
                subprocess.run(
                    ["systemctl", "kill", "--kill-whom=main", "--signal=HUP", service],
                    capture_output=True,
                )
            raise
        print(f"Applied and reloaded native monitoring. Rollback files: {backup}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        KeyError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        yaml.YAMLError,
    ) as exc:
        raise SystemExit(
            f"Native alert configuration failed: {type(exc).__name__}; configuration and credentials withheld"
        )
