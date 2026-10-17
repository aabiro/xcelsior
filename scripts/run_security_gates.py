#!/usr/bin/env python3
"""Run security regressions on a fresh local PostgreSQL cluster, never tower.

By default no Docker, sudo, existing database, .env, or real credentials are used.
The explicit --worker-billing-acceptance mode uses a local cached CUDA image
and GPU for a bounded, network-disabled container against the disposable DB.
Logs, JUnit results, and a provenance manifest remain in the output directory;
the disposable database is stopped and removed even when a gate fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTS = [
    "tests/test_public_http.py",
    "tests/test_serverless_limits_webhooks.py",
    "tests/test_security_monitoring.py",
    "tests/test_bg_worker.py",
    "tests/test_db_service_roles.py",
    "tests/test_account_security_hold.py",
    "tests/test_oauth_operator_scope_refusal.py",
    "tests/test_mcp_connector_auth.py",
    "tests/test_nginx_vhosts_install_safely.py",
    "tests/test_hardened_runtime_and_ingress.py",
    "tests/test_worker_privileged.py",
    "tests/test_egress_isolation.py",
    "tests/test_identity_admission_fails_closed.py",
    "tests/test_host_agent_tokens.py",
    "tests/test_job_reconcile_queue.py",
    "tests/test_control_plane_leases.py",
    "tests/test_billing_security_sweep.py",
    "tests/test_serverless_chaos_billing.py",
    "tests/test_artifact_download_is_owner_scoped.py",
    "tests/test_artifact_finalize_verification.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--postgres-bin", type=Path)
    parser.add_argument("--worker-billing-acceptance", action="store_true")
    parser.add_argument("--cuda-image", default="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime")
    parser.add_argument("--gpu-device", default="0")
    parser.add_argument("tests", nargs="*")
    args = parser.parse_args()
    if args.worker_billing_acceptance and args.tests:
        parser.error("choose worker acceptance or pytest suites, not both")
    if not args.tests and not args.worker_billing_acceptance:
        args.tests = DEFAULT_TESTS
    pg_bin = args.postgres_bin
    if pg_bin is None:
        candidates = sorted(
            Path("/usr/lib/postgresql").glob("*/bin/initdb"),
            key=lambda p: int(p.parent.parent.name),
            reverse=True,
        )
        if not candidates:
            parser.error("local PostgreSQL tools are required; pass --postgres-bin")
        pg_bin = candidates[0].parent
    output = args.output or Path(tempfile.mkdtemp(prefix="xcelsior-security-evidence-"))
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output = output.resolve()
    scratch = Path(tempfile.mkdtemp(prefix="xcl-sec-pg-"))
    data = scratch / "data"
    sock = scratch / "socket"
    sock.mkdir(mode=0o700)
    state = scratch / "state"
    state.mkdir(mode=0o700)
    dsn = f"postgresql:///xcelsior_security_test?host={sock}&user=security_test"
    # Explicit allowlist: never inherit production credentials or test dotenv.
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL") if k in os.environ}
    env.update(
        {
            "CI": "1",
            "PYTHON_DOTENV_DISABLED": "1",
            "XCELSIOR_SKIP_DOTENV": "1",
            "XCELSIOR_ENV": "test",
            "XCELSIOR_DB_BACKEND": "postgres",
            "XCELSIOR_POSTGRES_DSN": dsn,
            "XCELSIOR_TEST_POSTGRES_DSN": dsn,
            "XCELSIOR_PG_DSN": dsn,
            "DATABASE_URL": dsn,
            "XCELSIOR_BG_TASKS": "false",
            "XCELSIOR_SMTP_HOST": "",
            "XCELSIOR_API_TOKEN": "isolated-security-test-token",
            "XCELSIOR_AUTH_RATE_LIMIT_REQUESTS": "5000",
            "XCELSIOR_RATE_LIMIT_REQUESTS": "5000",
        }
    )
    for name in (
        "HOSTS",
        "JOBS",
        "BILLING",
        "MARKETPLACE",
        "AUTOSCALE",
        "SPOT_PRICES",
        "COMPUTE_SCORES",
    ):
        env[f"XCELSIOR_{name}_FILE"] = str(state / f"{name.lower()}.json")
    env["XCELSIOR_LOG_FILE"] = str(state / "xcelsior.log")
    env["STATE_DIRECTORY"] = str(state)
    env["XCELSIOR_AGENT_TOKEN_FILE"] = str(state / "agent-token.json")
    env["XCELSIOR_V2_JOURNAL_PATH"] = str(state / "attempts.json")
    manifest = {
        "gate_exit_code": 2,
        "python": sys.version,
        "started_at_unix": time.time(),
        "tests": args.tests,
        "mode": "worker_billing_acceptance" if args.worker_billing_acceptance else "pytest",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines(),
        "isolation": "fresh local Unix-socket-only PostgreSQL; no dotenv or inherited credentials",
        "tower_access": "none",
        "phases": {},
    }
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    manifest["source_sha256"] = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in files
        if (ROOT / p).is_file()
        and (
            p.endswith(".py")
            or p
            in {
                "docker-compose.yml",
                "scripts/install_nginx_vhosts.sh",
                "pyproject.toml",
                "uv.lock",
                "requirements.txt",
            }
        )
    }

    def run(phase, command):
        started = time.monotonic()
        with (output / f"{phase}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        manifest["phases"][phase] = {
            "exit_code": result.returncode,
            "seconds": round(time.monotonic() - started, 2),
        }
        print(
            f"{phase}: exit {result.returncode}; evidence {output / (phase + '.log')}", flush=True
        )
        return result.returncode

    try:
        if run(
            "initdb",
            [
                str(pg_bin / "initdb"),
                "-D",
                str(data),
                "-U",
                "security_test",
                "-A",
                "trust",
                "--no-locale",
                "--encoding=UTF8",
            ],
        ):
            return 2
        with (data / "postgresql.conf").open("a") as config:
            config.write(
                f"\nlisten_addresses = ''\nunix_socket_directories = '{sock}'\nmax_connections = 150\n"
            )
        if run(
            "start",
            [
                str(pg_bin / "pg_ctl"),
                "-D",
                str(data),
                "-l",
                str(output / "postgres.log"),
                "-w",
                "start",
            ],
        ):
            return 2
        if run(
            "createdb",
            [
                str(pg_bin / "createdb"),
                "-h",
                str(sock),
                "-U",
                "security_test",
                "xcelsior_security_test",
            ],
        ):
            return 2
        if run("migrate", [sys.executable, "-m", "alembic", "upgrade", "head"]):
            return 2
        if args.worker_billing_acceptance:
            env["XCELSIOR_ISOLATED_ACCEPTANCE"] = "1"
            env["XCELSIOR_ACCEPTANCE_STATE"] = str(state)
            code = run(
                "worker_billing",
                [
                    sys.executable,
                    "scripts/worker_billing_acceptance.py",
                    "--output",
                    str(output / "worker_billing.json"),
                    "--image",
                    args.cuda_image,
                    "--gpu-device",
                    args.gpu_device,
                ],
            )
            manifest["gate_exit_code"] = code
            return code
        code = run(
            "pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "--tb=short",
                f"--junitxml={output / 'junit.xml'}",
                *args.tests,
            ],
        )
        if (output / "junit.xml").exists():
            suites = ET.parse(output / "junit.xml").getroot()
            manifest["results"] = {
                key: sum(int(s.get(key, 0)) for s in suites.iter("testsuite"))
                for key in ("tests", "failures", "errors", "skipped")
            }
            print(json.dumps(manifest["results"]), flush=True)
            if not code and manifest["results"]["skipped"]:
                code = 3  # incomplete evidence cannot be a passing release gate
        manifest["gate_exit_code"] = code
        return code
    finally:
        if (data / "postmaster.pid").exists():
            run("stop", [str(pg_bin / "pg_ctl"), "-D", str(data), "-m", "fast", "-w", "stop"])
        if not (data / "postmaster.pid").exists():
            shutil.rmtree(scratch)
        else:
            manifest["cleanup_required"] = str(scratch)
        manifest["finished_at_unix"] = time.time()
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Manifest: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
