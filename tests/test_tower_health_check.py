"""Exercise the installed collector with hardware and process-signal records."""

import os
from pathlib import Path
import subprocess

import pytest

COLLECTOR = (
    Path(__file__).resolve().parents[1] / "api/infra/observability/native/tower-health-check"
)


def run_collector(tmp_path, records, *, ras_exit=0):
    commands = tmp_path / "bin"
    commands.mkdir()
    ras_fixture = tmp_path / "ras.txt"
    ras_fixture.write_text(records)
    kernel_fixture = tmp_path / "kernel.txt"
    kernel_fixture.write_text("mce: In-kernel MCE decoding enabled.\n")
    for name, body in {
        "ras-mc-ctl": 'cat "$RAS_FIXTURE"\nexit "$RAS_EXIT_CODE"\n',
        "journalctl": 'cat "$KERNEL_FIXTURE"\n',
    }.items():
        command = commands / name
        command.write_text("#!/bin/sh\n" + body)
        command.chmod(0o755)
    output = tmp_path / "tower_health.prom"
    result = subprocess.run(
        ["bash", str(COLLECTOR)],
        env={
            **os.environ,
            "PATH": str(commands) + os.pathsep + os.environ["PATH"],
            "RAS_FIXTURE": str(ras_fixture),
            "RAS_EXIT_CODE": str(ras_exit),
            "KERNEL_FIXTURE": str(kernel_fixture),
            "TOWER_HEALTH_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, output


def test_process_signals_do_not_report_hardware_failure(tmp_path):
    records = "No Memory errors.\nNo MCE errors.\n\nSIGNAL events:\n"
    records += "".join(
        f"{i} 2026-11-01 11:27:50 -0500 error: signal=17, comm=systemd\n" for i in range(1, 381)
    )
    result, output = run_collector(tmp_path, records)
    assert result.returncode == 0, result.stderr
    assert "tower_health_ras_errors_total 0\n" in output.read_text()
    assert "tower_health_kernel_errors_boot 0\n" in output.read_text()


@pytest.mark.parametrize(
    "section",
    [
        "Memory errors",
        "PCIe AER errors",
        "MCE errors",
        "CXL AER uncorrectable errors",
        "disk errors",
        "Memory failure errors",
    ],
)
def test_hardware_records_survive_signals_before_and_after(tmp_path, section):
    result, output = run_collector(
        tmp_path,
        "SIGNAL events:\n1 process signal\n\n"
        f"{section}:\n1 hardware fault\n2 another hardware fault\n\n"
        "SIGNAL events:\n2 another process signal\n",
    )
    assert result.returncode == 0, result.stderr
    assert "tower_health_ras_errors_total 2\n" in output.read_text()


def test_no_records_reports_zero(tmp_path):
    result, output = run_collector(tmp_path, "No Memory errors.\nNo MCE errors.\n")
    assert result.returncode == 0, result.stderr
    assert "tower_health_ras_errors_total 0\n" in output.read_text()


def test_failed_ras_query_preserves_previous_metrics(tmp_path):
    output = tmp_path / "tower_health.prom"
    previous = "tower_health_ras_errors_total 7\n"
    output.write_text(previous)
    result, output = run_collector(tmp_path, "No MCE errors.\n", ras_exit=1)
    assert result.returncode != 0
    assert output.read_text() == previous
    assert list(tmp_path.glob("tower_health.prom.*")) == []
