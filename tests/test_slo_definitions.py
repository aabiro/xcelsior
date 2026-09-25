from __future__ import annotations
import pytest
import re
from pathlib import Path
import yaml

from slo_definitions import ALL_SLOS, HARD_INVARIANTS

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "infra" / "observability"


def _prometheus_rules() -> list[dict]:
    documents = (
        yaml.safe_load((OBS / "prometheus" / "recording-rules.yml").read_text()),
        yaml.safe_load((OBS / "prometheus" / "alert-rules.yml").read_text()),
    )
    return [
        rule for document in documents for group in document["groups"] for rule in group["rules"]
    ]


def test_every_slo_has_unique_name():
    names = [s.name for s in ALL_SLOS]
    assert len(names) == len(set(names)), "SLO names must be unique"


def test_hard_invariant_target_is_zero():
    for slo in HARD_INVARIANTS:
        assert slo.target == 0.0, f"{slo.name} target must be 0.0"


def test_hard_invariant_has_burn_rate_windows():
    for slo in HARD_INVARIANTS:
        assert len(slo.burn_rate_windows) > 0, f"{slo.name} must have burn rate windows"


def test_every_slo_has_alert_name_and_metric():
    for slo in ALL_SLOS:
        assert slo.alert_name, f"{slo.name} must have alert_name"
        assert slo.metric, f"{slo.name} must have metric"


def test_slo_alert_exists_in_alert_rules():
    rules = _prometheus_rules()
    alert_names = {rule.get("alert") for rule in rules if "alert" in rule}
    
    for slo in ALL_SLOS:
        assert slo.alert_name in alert_names, f"Alert {slo.alert_name} for SLO {slo.name} not found in alert-rules.yml"


def test_hard_invariant_metric_exists_in_catalog():
    catalog = (ROOT / "metrics_catalog.py").read_text()
    for slo in HARD_INVARIANTS:
        assert f'"{slo.metric}"' in catalog or f"'{slo.metric}'" in catalog, f"Metric {slo.metric} not in metrics_catalog.py"


def test_slo_has_description():
    for slo in ALL_SLOS:
        assert slo.description, f"{slo.name} must have a description"
