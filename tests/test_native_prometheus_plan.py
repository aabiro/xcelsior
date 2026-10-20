"""A native monitoring plan preserves other tenants and never reports secrets."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "native_prometheus_plan", ROOT / "api/scripts/plan_native_prometheus.py"
)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)
RULES = "/etc/prometheus/xcelsior-rules/legacy-retirement.yml"
SYNTHETIC_CREDENTIAL = uuid.uuid4().hex


def existing():
    return {
        "global": {"scrape_interval": "30s"},
        "scrape_configs": [
            {
                "job_name": "other-service",
                "basic_auth": {"username": "synthetic-user", "password": SYNTHETIC_CREDENTIAL},
                "static_configs": [{"targets": ["127.0.0.1:9100"]}],
            }
        ],
        "rule_files": ["/etc/prometheus/other-rules.yml"],
        "alerting": {"alertmanagers": [{"static_configs": [{"targets": ["127.0.0.1:9093"]}]}]},
    }


def test_additive_plan_preserves_all_existing_jobs_rules_and_receivers():
    source = existing()
    result, operations = planner.prepare_config(source, [("green", "127.0.0.1:9500")], RULES)
    assert result["global"] == source["global"]
    assert result["alerting"] == source["alerting"]
    assert result["scrape_configs"][:-1] == source["scrape_configs"]
    assert result["rule_files"][:-1] == source["rule_files"]
    assert SYNTHETIC_CREDENTIAL not in json.dumps(operations)
    again, additional = planner.prepare_config(result, [("green", "127.0.0.1:9500")], RULES)
    assert again == result
    assert additional == []
    assert len(source["scrape_configs"]) == 1


def test_existing_api_authentication_and_custom_settings_are_preserved():
    source = existing()
    source["scrape_configs"].append(
        {
            "job_name": "xcelsior-api",
            "metrics_path": "/metrics/prometheus",
            "authorization": {"credentials": SYNTHETIC_CREDENTIAL},
            "scrape_interval": "15s",
        }
    )
    result, operations = planner.prepare_config(source, [("green", "127.0.0.1:9500")], RULES)
    assert result["scrape_configs"][1]["authorization"] == {"credentials": SYNTHETIC_CREDENTIAL}
    assert result["scrape_configs"][1]["scrape_interval"] == "15s"
    assert SYNTHETIC_CREDENTIAL not in json.dumps(operations)


@pytest.mark.parametrize(
    "target",
    [
        "green=192.168.1.1:9500",
        "green=localhost:9500",
        "green=user:password@127.0.0.1:9500",
        "green=127.0.0.1:0",
        "green=127.0.0.1:99999",
        "green=127.0.0.1:9500/metrics",
        "green=127.0.0.1:9500?token=secret",
        "bad label=127.0.0.1:9500",
    ],
)
def test_only_explicit_loopback_inventory_is_accepted(target):
    with pytest.raises(ValueError):
        planner.parse_target(target)


def test_ipv6_loopback_is_supported():
    assert planner.parse_target("blue=[::1]:9501") == ("blue", "[::1]:9501")


@pytest.mark.parametrize(
    "targets",
    [
        [],
        [("green", "127.0.0.1:9500"), ("green", "127.0.0.1:9501")],
        [("green", "127.0.0.1:9500"), ("blue", "127.0.0.1:9500")],
    ],
)
def test_ambiguous_replica_inventories_are_rejected(targets):
    with pytest.raises(ValueError):
        planner.prepare_config(existing(), targets, RULES)


@pytest.mark.parametrize(
    "label,address,path",
    [
        ("blue", "127.0.0.1:9500", "/metrics/prometheus"),
        ("green", "127.0.0.1:9501", "/metrics/prometheus"),
        ("green", "127.0.0.1:9500", "/private-metrics"),
    ],
)
def test_existing_identity_or_path_conflicts_refuse_silent_replacement(label, address, path):
    source = existing()
    source["scrape_configs"].append(
        {
            "job_name": "xcelsior-api",
            "metrics_path": path,
            "static_configs": [{"targets": [address], "labels": {"replica": label}}],
        }
    )
    with pytest.raises(ValueError):
        planner.prepare_config(source, [("green", "127.0.0.1:9500")], RULES)


@pytest.mark.parametrize("validation_exit", [0, 1])
def test_cli_validates_before_writing_and_never_reports_native_errors_or_source_secrets(
    tmp_path, monkeypatch, capsys, validation_exit
):
    config = tmp_path / "prometheus.yml"
    config.write_text(yaml.safe_dump(existing()))
    before = config.read_bytes()
    output = tmp_path / "candidate.yml"
    args = [
        "plan",
        "--config",
        str(config),
        "--api-target",
        "green=127.0.0.1:9500",
        "--rules-path",
        RULES,
        "--rule-source",
        "-",
        "--promtool",
        "/synthetic/promtool",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(planner.sys, "argv", args)
    monkeypatch.setattr(
        planner.sys, "stdin", io.StringIO(json.dumps({"name": planner.GROUP_NAME, "rules": []}))
    )

    def native(argv, **kwargs):
        return SimpleNamespace(
            returncode=0 if "--version" in argv else validation_exit,
            stdout="promtool, version test\n",
            stderr="native-error-with-private-secret",
        )

    monkeypatch.setattr(planner.subprocess, "run", native)
    assert planner.main() == (0 if validation_exit == 0 else 2)
    captured = capsys.readouterr()
    assert SYNTHETIC_CREDENTIAL not in captured.out + captured.err
    assert "native-error-with-private-secret" not in captured.out + captured.err
    assert config.read_bytes() == before
    assert output.exists() == (validation_exit == 0)
    if output.exists():
        assert output.stat().st_mode & 0o777 == 0o600
        assert yaml.safe_load(output.read_text())["alerting"] == existing()["alerting"]
