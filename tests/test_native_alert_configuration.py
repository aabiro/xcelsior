"""Scoped native-monitoring edits preserve unrelated rules and receiver secrets."""

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "native_alert_configuration", ROOT / "api/scripts/configure_native_alerts.py"
)
configuration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(configuration)


def test_scraping_both_replicas_preserves_other_jobs_and_is_idempotent():
    original = {
        "scrape_configs": [
            {"job_name": "tower-host", "static_configs": [{"targets": ["100.64.0.4:9100"]}]},
            {
                "job_name": "xcelsior-api",
                "metrics_path": "/metrics/prometheus",
                "authorization": {"credentials": "synthetic-test-token"},
                "static_configs": [{"targets": ["127.0.0.1:9500"], "labels": {"replica": "green"}}],
            },
        ]
    }
    before = deepcopy(original)
    result = configuration.prepare_prometheus(original)
    assert original == before
    assert result["scrape_configs"][0] == original["scrape_configs"][0]
    assert result["scrape_configs"][1]["authorization"] == {"credentials": "synthetic-test-token"}
    assert result["scrape_configs"][1]["static_configs"][-1] == {
        "targets": ["127.0.0.1:9501"],
        "labels": {"replica": "blue"},
    }
    assert configuration.prepare_prometheus(result) == result


def test_conflicting_replica_identity_requires_review():
    config = {
        "scrape_configs": [
            {
                "job_name": "xcelsior-api",
                "metrics_path": "/metrics/prometheus",
                "static_configs": [
                    {"targets": ["127.0.0.1:9500"], "labels": {"replica": "blue"}},
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="identity"):
        configuration.prepare_prometheus(config)


def test_rule_merge_preserves_unrelated_groups_and_host_safety_rules():
    original = {
        "groups": [
            {
                "name": "tower",
                "interval": "30s",
                "rules": [
                    {"alert": "TowerHostDown", "expr": "up == 0"},
                    {"alert": "TowerDiskHigh", "expr": "old"},
                ],
            },
            {"name": "backups", "rules": [{"alert": "BackupFailed", "expr": "failed > 0"}]},
        ]
    }
    source = {
        "groups": [
            {
                "name": "tower",
                "rules": [
                    {"alert": "TowerDiskHigh", "expr": "new"},
                    {"alert": "TowerDiskCritical", "expr": "critical"},
                ],
            }
        ]
    }
    result = configuration.merge_rules(original, source, "tower")
    assert result["groups"][1] == original["groups"][1]
    assert result["groups"][0]["interval"] == "30s"
    assert result["groups"][0]["rules"][0] == original["groups"][0]["rules"][0]
    assert result["groups"][0]["rules"][1]["expr"] == "new"
    assert configuration.merge_rules(result, source, "tower") == result


def test_disk_group_can_be_installed_then_updated():
    old = {"alert": "XcelsiorProdVolumeFull", "expr": "old"}
    new = {**old, "expr": "new"}
    group = "xcelsior.prod-host-disk"
    installed = {"groups": [{"name": "other", "rules": []}]}
    source = {"groups": [{"name": group, "rules": [old]}]}
    result = configuration.merge_rules(installed, source, group, allow_new_group=True)
    assert result["groups"] == installed["groups"] + source["groups"]
    updated = {"groups": [{"name": group, "rules": [new]}]}
    assert configuration.merge_rules(result, updated, group, allow_new_group=True)["groups"] == (
        installed["groups"] + updated["groups"]
    )


def test_routing_retains_every_destination_and_secret_and_deduplicates_replicas():
    original = {
        "route": {
            "receiver": "tower",
            "repeat_interval": "12h",
            "routes": [
                {
                    "receiver": "control-plane",
                    "matchers": ['platform="xcelsior"', 'component="control-plane"'],
                    "group_by": ["alertname", "instance"],
                    "routes": [{"receiver": "billing", "matchers": ['owner="billing"']}],
                },
                {"receiver": "other", "matchers": ['owner="other"']},
            ],
        },
        "receivers": [
            {
                "name": "control-plane",
                "telegram_configs": [{"bot_token": "synthetic-test-token", "chat_id": 123}],
            }
        ],
        "inhibit_rules": [
            {"source_matchers": ['alertname="Existing"'], "target_matchers": ['alertname="Other"']}
        ],
    }
    before = deepcopy(original)
    result = configuration.prepare_alertmanager(original)
    assert original == before
    assert result["receivers"] == original["receivers"]
    assert result["route"]["routes"][1] == original["route"]["routes"][1]
    route = result["route"]["routes"][0]
    assert route["group_by"] == ["alertname", "job", "service"]
    assert route["routes"][0] == original["route"]["routes"][0]["routes"][0]
    assert result["inhibit_rules"][0] == original["inhibit_rules"][0]
    assert {
        "source_matchers": ['alertname="XcelsiorProdVolumeFull"'],
        "target_matchers": ['alertname="XcelsiorProdVolumeHigh"'],
        "equal": ["instance", "mountpoint"],
    } in result["inhibit_rules"]
    assert configuration.prepare_alertmanager(result) == result


def test_failed_validation_does_not_disclose_validator_output(monkeypatch):
    class Failure:
        returncode = 1
        stdout = b"synthetic-secret-from-config"
        stderr = b"synthetic-secret-from-config"

    monkeypatch.setattr(configuration.subprocess, "run", lambda *args, **kwargs: Failure())
    with pytest.raises(RuntimeError) as failure:
        configuration.validate(["amtool", "check-config", "/private/config"])
    assert "synthetic-secret" not in str(failure.value)


def test_successful_diagnostic_completion_is_not_an_incident():
    original = {
        "groups": [
            {
                "name": "tower-integrity",
                "rules": [
                    {"alert": "TowerCoreCycleComplete", "expr": "done == 1"},
                    {"alert": "TowerCoreCycleFailure", "expr": "failures > 0"},
                    {"alert": "TowerRasErrors", "expr": "errors > 0"},
                ],
            }
        ]
    }
    result = configuration.prepare_integrity_rules(original)
    assert [r["alert"] for r in result["groups"][0]["rules"]] == [
        "TowerCoreCycleFailure",
        "TowerRasErrors",
    ]
    assert len(original["groups"][0]["rules"]) == 3


def test_atomic_replacement_preserves_file_mode_and_leaves_no_candidate(tmp_path):
    path = tmp_path / "config.yml"
    path.write_bytes(b"before")
    path.chmod(0o640)
    configuration.replace_file(path, b"after")
    assert path.read_bytes() == b"after"
    assert path.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [path]


def test_reload_requires_a_successful_recent_confirmation_from_each_service(monkeypatch):
    from io import BytesIO

    requested = []

    def response(url, timeout):
        prefix = "prometheus" if ":9091/" in url else "alertmanager"
        requested.append(prefix)
        return BytesIO(
            (
                f"{prefix}_config_last_reload_successful 1\n"
                f"{prefix}_config_last_reload_success_timestamp_seconds 101\n"
            ).encode()
        )

    monkeypatch.setattr(configuration.urllib.request, "urlopen", response)
    configuration.verify_reloads(100)
    assert requested == ["prometheus", "alertmanager"]


@pytest.mark.parametrize("success,timestamp", [(0, 101), (1, 99)])
def test_failed_or_old_reload_confirmation_is_not_accepted(monkeypatch, success, timestamp):
    from io import BytesIO

    ticks = iter([0, 0, 21])
    monkeypatch.setattr(configuration.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(configuration.time, "sleep", lambda _: None)

    def response(url, timeout):
        prefix = "prometheus" if ":9091/" in url else "alertmanager"
        return BytesIO(
            (
                f"{prefix}_config_last_reload_successful {success}\n"
                f"{prefix}_config_last_reload_success_timestamp_seconds {timestamp}\n"
            ).encode()
        )

    monkeypatch.setattr(configuration.urllib.request, "urlopen", response)
    with pytest.raises(RuntimeError, match="did not confirm"):
        configuration.verify_reloads(100)


@pytest.mark.parametrize("fixture", ["native_control_plane_alerts.yml", "native_alert_noise.yml"])
def test_real_prometheus_alert_scenarios(fixture):
    promtool = os.environ.get("PROMTOOL") or shutil.which("promtool")
    if not promtool:
        pytest.skip("needs native promtool; set PROMTOOL to its executable path")
    result = subprocess.run(
        [promtool, "test", "rules", str(ROOT / "tests/fixtures" / fixture)],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
