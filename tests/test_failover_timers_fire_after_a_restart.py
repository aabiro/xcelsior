"""The standby's timers must schedule a run whenever they are started.

The ops runbook pauses both timers around a VPS reboot. On 2026-11-18 they were
found stopped since 2026-11-01, with the replica 17 days old. Even after
`systemctl enable --now`, `systemctl list-timers` showed NEXT "-": OnBootSec had
long passed, OnUnitActiveSec counts from the service's last run (none since this
boot), and Persistent= only applies to OnCalendar. So systemd scheduled nothing.

OnActiveSec counts from the timer's own start, so a stop/start always schedules
a run. These pin it, along with the intervals the runbook promises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

UNIT_DIR = Path(__file__).resolve().parents[1] / "api" / "infra" / "systemd"


def _timer_section(name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    section = ""
    for raw in (UNIT_DIR / name).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section == "Timer":
            key, _, value = line.partition("=")
            assert key not in values, f"{name}: {key} set twice"
            values[key] = value
    return values


@pytest.mark.parametrize(
    ("timer", "interval", "service"),
    [
        ("headscale-failover-watchdog.timer", "2min", "headscale-failover-watchdog.service"),
        ("headscale-replicate.timer", "15min", "headscale-replicate.service"),
    ],
)
def test_a_started_timer_always_has_a_next_run(timer: str, interval: str, service: str) -> None:
    t = _timer_section(timer)
    assert t.get("OnActiveSec"), f"{timer} needs OnActiveSec, or a stop/start schedules nothing"
    assert t.get("OnUnitActiveSec") == interval
    assert t.get("Unit") == service
    assert (UNIT_DIR / service).is_file()
