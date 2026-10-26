"""Real transactional billing and backup failure cases from the data audit."""

import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from unittest import mock

import pytest
from psycopg.types.json import Jsonb

from billing import get_billing_engine
from db import _get_pg_pool
from tests.fixtures.native_runtime import running_instance
from tests.test_gpu_billing_writes_its_cycle_row import (
    unclaimed_running_job as running_job,
)  # noqa: F401
from tests.test_volume_db import engine, cleanup_vids  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def compute():
    owner = f"data-audit-{uuid.uuid4().hex}"
    billing = get_billing_engine()
    billing.deposit(owner, 100)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=120) as instance:
        yield instance, owner, billing


def test_admitted_but_unclaimed_job_cannot_gain_wallet_billing_authority(running_job):
    job_id, owner, started = running_job
    billing = get_billing_engine()
    before = billing.get_wallet(owner)["balance_micros"]
    result = billing.bill_running_period(job_id, period_end=started + 60)
    assert not result.get("billed")
    assert billing.get_wallet(owner)["balance_micros"] == before


def test_gpu_cycle_failure_rolls_back_debit_and_notice_with_cycle(compute, monkeypatch):
    instance, owner, billing = compute
    original = billing._bill_gpu_period

    def fail_after_cycle(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("after cycle insertion before commit")

    monkeypatch.setattr(billing, "_bill_gpu_period", fail_after_cycle)
    before = billing.get_wallet(owner)["balance_micros"]
    with pytest.raises(RuntimeError, match="before commit"):
        billing.bill_running_period(instance["job_id"], period_end=instance["started_at"] + 60)
    assert billing.get_wallet(owner)["balance_micros"] == before
    with _get_pg_pool().connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM billing_cycles WHERE job_id=%s", (instance["job_id"],)
        ).fetchone() == (0,)


def test_corrupt_execution_spec_cannot_authorize_periodic_billing(compute):
    instance, owner, billing = compute
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET spec_hash='sha256:invalid' WHERE job_id=%s", (instance["job_id"],)
        )
    before = billing.get_wallet(owner)["balance_micros"]
    with pytest.raises(ValueError):
        billing.bill_running_period(instance["job_id"], period_end=instance["started_at"] + 60)
    assert billing.get_wallet(owner)["balance_micros"] == before


def test_gpu_tick_uses_native_start_even_if_observation_is_missing(compute):
    instance, owner, billing = compute
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload=payload-'started_at' WHERE job_id=%s", (instance["job_id"],)
        )
    before = billing.get_wallet(owner)["balance_micros"]
    result = billing.auto_billing_cycle()
    assert result["billed"] >= 1
    assert billing.get_wallet(owner)["balance_micros"] < before


def test_storage_debit_cannot_commit_when_cycle_transaction_fails(
    engine, cleanup_vids, monkeypatch
):
    owner = f"storage-audit-{uuid.uuid4().hex}"
    billing = get_billing_engine()
    billing.deposit(owner, 100)
    volume = engine.create_volume(owner, "atomic-storage", 100)
    cleanup_vids.append(volume["volume_id"])
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE volumes SET created_at=EXTRACT(EPOCH FROM clock_timestamp())-3600 WHERE volume_id=%s",
            (volume["volume_id"],),
        )
    original = billing.charge

    def lose_response(customer, *args, **kwargs):
        result = original(customer, *args, **kwargs)
        if customer == owner:
            raise RuntimeError("lost cycle transaction")
        return result

    monkeypatch.setattr(billing, "charge", lose_response)
    before = billing.get_wallet(owner)["balance_micros"]
    result = billing.auto_billing_cycle()
    assert result["errors"] >= 1
    assert billing.get_wallet(owner)["balance_micros"] == before


def shell_env(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    backups = tmp_path / "backups"
    backups.mkdir()
    bins = tmp_path / "bin"
    bins.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XCELSIOR_", "PG"))}
    env.update(
        PATH=str(bins) + ":" + env["PATH"],
        XCELSIOR_PROJECT_DIR=str(project),
        XCELSIOR_BACKUP_DIR=str(backups),
        XCELSIOR_BACKUP_LOCK_FILE=str(backups / ".backup.lock"),
        XCELSIOR_NODE_EXPORTER_TEXTFILE_DIR=str(tmp_path / "metrics"),
    )
    return env, bins, backups


def private_backup_script(tmp_path, lock_path):
    # Substitute only the old hard-coded global lock pathname in the test copy.
    # The reproduction must never touch another backup's real /tmp lock.
    script = tmp_path / "backup-db.sh"
    script.write_text(
        (ROOT / "api/scripts/backup-db.sh")
        .read_text()
        .replace('LOCK_FILE="/tmp/xcelsior-backup.lock"', f'LOCK_FILE="{lock_path}"')
    )
    return script


def test_backup_cannot_remove_a_competing_run_lock(tmp_path):
    env, bins, backups = shell_env(tmp_path)
    env["XCELSIOR_POSTGRES_PORT"] = "5434"
    lock_path = backups / ".backup.lock"
    script = private_backup_script(tmp_path, lock_path)
    with lock_path.open("w") as held:
        held.write(str(os.getpid()))
        held.flush()
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = lock_path.stat().st_ino
        result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
        assert result.returncode != 0
        assert lock_path.exists() and lock_path.stat().st_ino == before


@pytest.mark.parametrize("name", ["backup-db.sh", "restore-db.sh", "restore-db-drill.sh"])
def test_database_tools_refuse_an_unspecified_cluster_before_admin_commands(tmp_path, name):
    env, bins, backups = shell_env(tmp_path)
    marker = tmp_path / "admin-ran"
    for command in ("pg_dump", "pg_restore", "createdb", "dropdb", "docker"):
        target = bins / command
        target.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 42\n')
        target.chmod(0o755)
    dump = backups / "xcelsior_fixture.dump"
    dump.write_bytes(b"fixture")
    dump.with_suffix(".dump.sha256").write_text(
        hashlib.sha256(dump.read_bytes()).hexdigest() + "  " + dump.name + "\n"
    )
    script = (
        private_backup_script(tmp_path, backups / ".backup.lock")
        if name == "backup-db.sh"
        else ROOT / "api/scripts" / name
    )
    result = subprocess.run(
        ["bash", str(script), str(dump)], input="yes\n", env=env, text=True, capture_output=True
    )
    assert result.returncode != 0
    assert not marker.exists(), result.stdout + result.stderr


def test_manual_restore_rejects_invalid_checksum_before_stopping_writers(tmp_path):
    env, bins, backups = shell_env(tmp_path)
    env["XCELSIOR_POSTGRES_PORT"] = "5434"
    marker = tmp_path / "mutated"
    for command in ("docker", "dropdb", "createdb", "pg_restore"):
        target = bins / command
        target.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 42\n')
        target.chmod(0o755)
    dump = backups / "xcelsior_fixture.dump"
    dump.write_bytes(b"fixture")
    dump.with_suffix(".dump.sha256").write_text("0" * 64 + "  " + dump.name + "\n")
    result = subprocess.run(
        ["bash", str(ROOT / "api/scripts/restore-db.sh"), str(dump)],
        input="yes\n",
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert not marker.exists(), result.stdout + result.stderr


@pytest.mark.parametrize("stop_fails", [False, True])
def test_restore_quiesces_every_writer_and_never_claims_restart(tmp_path, stop_fails):
    import json
    import sys

    env, bins, backups = shell_env(tmp_path)
    env["XCELSIOR_POSTGRES_PORT"] = "5434"
    project = Path(env["XCELSIOR_PROJECT_DIR"])
    (project / "docker-compose.yml").write_text("services: {}\n")
    events = tmp_path / "events.jsonl"
    program = f"""#!{sys.executable}
import json,sys
with open({str(events)!r},'a') as log:log.write(json.dumps(sys.argv)+'\\n')
if sys.argv[1:]==['compose','config','--services']:
 print('api\\napi-blue\\nscheduler-worker\\nbg-worker\\nunrelated-service')
if sys.argv[1:3]==['compose','stop'] and {stop_fails!r}:raise SystemExit(1)
"""
    for command in ("docker", "dropdb", "createdb", "pg_restore"):
        target = bins / command
        target.write_text(program)
        target.chmod(0o755)
    # The migrator owns the schema under per-service roles; the restoring login
    # can act as it.
    psql = bins / "psql"
    psql.write_text(
        program + "if any('pg_roles' in a for a in sys.argv):print('1')\n",
    )
    psql.chmod(0o755)
    dump = backups / "xcelsior_fixture.dump"
    dump.write_bytes(b"fixture")
    dump.with_suffix(".dump.sha256").write_text(
        hashlib.sha256(dump.read_bytes()).hexdigest() + "  " + dump.name + "\n"
    )
    result = subprocess.run(
        ["bash", str(ROOT / "api/scripts/restore-db.sh"), str(dump)],
        input="yes\n",
        env=env,
        text=True,
        capture_output=True,
    )
    commands = [json.loads(line) for line in events.read_text().splitlines()]
    stops = [cmd for cmd in commands if cmd[1:3] == ["compose", "stop"]]
    assert len(stops) == 1
    assert set(stops[0][3:]) == {"api", "api-blue", "scheduler-worker", "bg-worker"}
    assert not any(cmd[1:3] == ["compose", "up"] for cmd in commands)
    dropped = any(Path(cmd[0]).name == "dropdb" for cmd in commands)
    if stop_fails:
        assert result.returncode != 0 and not dropped
    else:
        assert result.returncode == 0 and dropped, result.stdout + result.stderr
        assert "writers remain stopped" in result.stdout
        created = next(cmd for cmd in commands if Path(cmd[0]).name == "createdb")
        restored = next(
            cmd for cmd in commands if Path(cmd[0]).name == "pg_restore" and "--list" not in cmd
        )
        assert created[created.index("-O") + 1] == "xcelsior_migrator"
        assert "--role=xcelsior_migrator" in restored
        assert "provision_db_roles.py" in result.stdout + result.stderr


def test_restore_refuses_before_any_change_when_it_cannot_act_as_the_migrator(tmp_path):
    import json
    import sys

    env, bins, backups = shell_env(tmp_path)
    env["XCELSIOR_POSTGRES_PORT"] = "5434"
    project = Path(env["XCELSIOR_PROJECT_DIR"])
    (project / "docker-compose.yml").write_text("services: {}\n")
    events = tmp_path / "events.jsonl"
    program = f"""#!{sys.executable}
import json,sys
with open({str(events)!r},'a') as log:log.write(json.dumps(sys.argv)+'\\n')
if sys.argv[1:]==['compose','config','--services']:print('api')
"""
    for command in ("docker", "dropdb", "createdb", "pg_restore"):
        (bins / command).write_text(program)
        (bins / command).chmod(0o755)
    psql = bins / "psql"
    psql.write_text(
        program + "if any('pg_roles' in a for a in sys.argv):print('1')\n"
        "if any(a.startswith('SET ROLE') for a in sys.argv):raise SystemExit(1)\n"
    )
    psql.chmod(0o755)
    dump = backups / "xcelsior_fixture.dump"
    dump.write_bytes(b"fixture")
    dump.with_suffix(".dump.sha256").write_text(
        hashlib.sha256(dump.read_bytes()).hexdigest() + "  " + dump.name + "\n"
    )
    result = subprocess.run(
        ["bash", str(ROOT / "api/scripts/restore-db.sh"), str(dump)],
        input="yes\n",
        env=env,
        text=True,
        capture_output=True,
    )
    commands = [json.loads(line) for line in events.read_text().splitlines()]
    assert result.returncode != 0
    assert "cannot act as xcelsior_migrator" in result.stdout + result.stderr
    assert not any(Path(cmd[0]).name in {"dropdb", "createdb"} for cmd in commands)
    assert not any(cmd[1:3] == ["compose", "stop"] for cmd in commands)


def test_terminal_compute_slice_is_settled_once_without_a_user_stop_request(compute):
    from control_plane.attempts import report_attempt_status
    from control_plane.db import run_transaction

    instance, owner, billing = compute
    before = billing.get_wallet(owner)["balance_micros"]
    run_transaction(
        lambda conn: report_attempt_status(
            conn,
            job_id=instance["job_id"],
            host_id=instance["host_id"],
            **instance["authority"],
            status="succeeded",
        )
    )
    with _get_pg_pool().connection() as conn:
        expected = conn.execute(
            "SELECT total_cost_micros FROM usage_meters WHERE job_id=%s", (instance["job_id"],)
        ).fetchone()[0]
    assert expected > 0
    first = billing.auto_billing_cycle()
    assert first["billed"] >= 1
    assert before - billing.get_wallet(owner)["balance_micros"] == expected
    billing.auto_billing_cycle()
    assert before - billing.get_wallet(owner)["balance_micros"] == expected


def test_insufficient_native_slice_stops_work_without_advancing_the_paid_cursor(compute):
    instance, owner, billing = compute
    with _get_pg_pool().connection() as conn:
        conn.execute("UPDATE wallets SET balance_micros=10000 WHERE customer_id=%s", (owner,))
    billing.auto_billing_cycle()
    with _get_pg_pool().connection() as conn:
        assert conn.execute(
            "SELECT status FROM jobs WHERE job_id=%s", (instance["job_id"],)
        ).fetchone() == ("stopping",)
        assert conn.execute(
            "SELECT count(*) FROM billing_cycles WHERE job_id=%s AND status='charged'",
            (instance["job_id"],),
        ).fetchone() == (0,)
    assert billing.get_wallet(owner)["balance_micros"] == 10000


def finish_compute(instance):
    from control_plane.attempts import report_attempt_status
    from control_plane.db import run_transaction

    run_transaction(
        lambda conn: report_attempt_status(
            conn,
            job_id=instance["job_id"],
            host_id=instance["host_id"],
            **instance["authority"],
            status="succeeded",
        )
    )


def test_overlapping_terminal_billing_ticks_charge_one_native_slice(compute):
    from concurrent.futures import ThreadPoolExecutor

    instance, owner, billing = compute
    before = billing.get_wallet(owner)["balance_micros"]
    finish_compute(instance)
    with _get_pg_pool().connection() as conn:
        expected = conn.execute(
            "SELECT total_cost_micros FROM usage_meters WHERE job_id=%s", (instance["job_id"],)
        ).fetchone()[0]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: billing.auto_billing_cycle(), range(2)))
    assert sum(row["billed"] for row in results) == 1
    assert before - billing.get_wallet(owner)["balance_micros"] == expected


def test_failed_final_slice_retries_after_funding_without_repeating_failed_rows(compute):
    instance, owner, billing = compute
    billing.charge(owner, 99.99, description="fixture balance reduction")
    finish_compute(instance)
    billing.auto_billing_cycle()
    with _get_pg_pool().connection() as conn:
        expected = conn.execute(
            "SELECT total_cost_micros FROM usage_meters WHERE job_id=%s", (instance["job_id"],)
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT count(*) FROM billing_cycles WHERE job_id=%s AND status='failed'",
            (instance["job_id"],),
        ).fetchone()[0]
    assert failed == 1
    billing.auto_billing_cycle()
    with _get_pg_pool().connection() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM billing_cycles WHERE job_id=%s AND status='failed'",
                (instance["job_id"],),
            ).fetchone()[0]
            == failed
        )
    billing.deposit(owner, 10)
    before = billing.get_wallet(owner)["balance_micros"]
    billing.auto_billing_cycle()
    assert before - billing.get_wallet(owner)["balance_micros"] == expected


def test_future_period_cannot_bill_past_server_observed_execution(compute):
    import time

    instance, owner, billing = compute
    result = billing.bill_running_period(instance["job_id"], period_end=time.time() + 3600)
    assert result["billed"]
    assert 0.11 < result["amount_cad"] < 0.14
