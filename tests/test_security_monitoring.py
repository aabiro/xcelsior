"""Security alarms must distinguish a verified zero from unavailable evidence."""

import os
from pathlib import Path
import uuid

import psycopg
import pytest

from control_plane import security_metrics
from scripts.check_ingress_policy import public_servers_protected
from scripts.collect_host_security_status import ingress_ready

INCLUDE = "include /etc/nginx/snippets/xcelsior-abuse-deny*.conf;"


@pytest.mark.parametrize(
    "text,protected",
    [
        (f"server {{listen 80; {INCLUDE}}} server {{listen 443 ssl; {INCLUDE}}}", True),
        (f"server {{listen 80; {INCLUDE}}} server {{listen 443 ssl;}}", False),
        (f"server {{listen 80; location / {{ {INCLUDE} }} }}", False),
        (f"server {{listen 80; # {INCLUDE}\n}}", False),
        (f"server {{listen 80; {INCLUDE}}} server {{listen 127.0.0.1:9555;}}", True),
        (f"server {{listen 80; {INCLUDE}", False),
        ("", False),
    ],
)
def test_ingress_scope_and_malformed_configs(text, protected):
    assert public_servers_protected(text) is protected


def test_every_shipped_public_vhost_is_protected():
    root = Path(__file__).resolve().parents[1]
    for name in ("xcelsior", "mcp-xcelsior", "agent-xcelsior"):
        assert public_servers_protected((root / "nginx" / f"{name}.conf").read_text()), name


def test_host_collector_detects_missing_or_disabled_policy(tmp_path):
    for folder in ("sites-enabled", "snippets", "conf.d"):
        (tmp_path / folder).mkdir()
    snippet = tmp_path / "snippets/xcelsior-abuse-deny.conf"
    snippet.write_text("if ($xcelsior_abuse_ip) { return 403; }")
    (tmp_path / "conf.d/xcelsior-abuse-ips.conf").write_text(
        "geo $xcelsior_abuse_ip { default 0; }"
    )
    for name in ("xcelsior", "mcp-xcelsior", "agent-xcelsior"):
        (tmp_path / "sites-enabled" / name).write_text(f"server {{listen 443; {INCLUDE}}}")
    assert ingress_ready(tmp_path)
    snippet.write_text("")
    assert not ingress_ready(tmp_path)
    snippet.write_text("if ($xcelsior_abuse_ip) { return 403; }")
    (tmp_path / "sites-enabled/mcp-xcelsior").unlink()
    assert not ingress_ready(tmp_path)


def test_security_metrics_detect_revived_job_without_exposing_identifiers():
    job = "security-signal-" + uuid.uuid4().hex
    with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"]) as conn:
        conn.execute(
            "INSERT INTO jobs(job_id,status,payload,submitted_at,priority) VALUES (%s,'queued','{}',0,0)",
            (job,),
        )
        conn.execute(
            "INSERT INTO state(namespace,payload) VALUES (%s,'{}')", ("security.job." + job,)
        )
        snapshot = security_metrics.collect(conn)
        assert snapshot["abuse_jobs_runnable"] >= 1
        conn.execute("UPDATE jobs SET status='terminated' WHERE job_id=%s", (job,))
        after = security_metrics.collect(conn)
        assert after["abuse_jobs_runnable"] == snapshot["abuse_jobs_runnable"] - 1
        assert job not in "\n".join(security_metrics.render(snapshot))
        conn.rollback()


def test_unavailable_security_metrics_never_emit_false_zeroes():
    rendered = "\n".join(security_metrics.render(None))
    assert "xcelsior_security_available 0" in rendered
    assert "abuse_jobs_runnable" not in rendered
    assert "database_identity_mismatches" not in rendered


def test_background_task_contract_covers_every_registered_task():
    import ast
    from control_plane.task_identities import TASK_SERVICES, DDL_TASKS

    source = (Path(__file__).resolve().parents[1] / "bg_worker.py").read_text()
    registered = {
        node.args[0].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "register_task"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert registered == set(TASK_SERVICES) | DDL_TASKS


def test_strict_background_worker_requires_external_ddl(monkeypatch):
    import bg_worker

    monkeypatch.setenv("XCELSIOR_DB_ROLES_REQUIRED", "1")
    monkeypatch.delenv("XCELSIOR_EXTERNAL_PARTITION_MAINTENANCE", raising=False)
    with pytest.raises(RuntimeError, match="external partition"):
        bg_worker.main()


def test_strict_task_cannot_be_registered_without_identity(monkeypatch):
    from control_plane import scheduled_tasks

    monkeypatch.setenv("XCELSIOR_DB_ROLES_REQUIRED", "1")
    with pytest.raises(RuntimeError, match="no qualified database identity"):
        scheduled_tasks.register_task("unqualified-test", lambda: None, 10)
    assert "unqualified-test" not in scheduled_tasks._registry
