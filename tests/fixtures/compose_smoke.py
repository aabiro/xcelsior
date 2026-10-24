"""Build the current tracked source and boot Compose without deployment access."""

import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import uuid

from psycopg.conninfo import conninfo_to_dict, make_conninfo


def run_compose_smoke(root: Path, scratch: Path, dsn: str) -> None:
    config = conninfo_to_dict(dsn)
    socket = Path(config["host"])
    # Only the security runner's disposable socket is mountable. No TCP DSN,
    # deployment environment, project volume or host Docker socket enters it.
    assert socket.is_absolute() and socket.parent.name.startswith("xcl-sec-pg-")
    assert config["dbname"].startswith("bootstrap_empty_")
    original_mode = socket.stat().st_mode & 0o777
    project = "xcelsior-smoke-" + uuid.uuid4().hex[:12]
    image = project + ":test"
    context = scratch / "context"
    context.mkdir()
    archive = scratch / "source.tar"
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    with tarfile.open(archive, "w") as stream:
        for name in filter(None, names):
            path = Path(name)
            if path.name.startswith(".env") or path.name == "agent-signing.key":
                continue
            source = root / path
            assert source.resolve().is_relative_to(root.resolve())
            stream.add(source, arcname=name, recursive=False)
    with tarfile.open(archive) as stream:
        stream.extractall(context, filter="data")
    config["host"] = "/acceptance-db"
    isolated_dsn = make_conninfo(**config)
    environment = {
        "CI": "1",
        "PYTHON_DOTENV_DISABLED": "1",
        "XCELSIOR_SKIP_DOTENV": "1",
        "XCELSIOR_ENV": "test",
        "XCELSIOR_DB_BACKEND": "postgres",
        "XCELSIOR_POSTGRES_DSN": isolated_dsn,
        "XCELSIOR_PG_DSN": isolated_dsn,
        "DATABASE_URL": isolated_dsn,
        "XCELSIOR_BG_TASKS": "false",
        "GUNICORN_WORKERS": "2",
        "XCELSIOR_API_TOKEN": "isolated-compose-token",
        "XCELSIOR_AUTH_CACHE_BACKEND": "memory",
        "XCELSIOR_NFS_REQUIRED": "false",
        "XCELSIOR_ALLOW_UNAUTH_AGENT": "0",
    }
    manifest = {
        "services": {
            "api": {
                "build": {"context": str(context)},
                "image": image,
                "network_mode": "none",
                "read_only": True,
                "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges:true"],
                "pids_limit": 256,
                "mem_limit": "1g",
                "cpus": 2,
                "environment": environment,
                "tmpfs": [
                    "/tmp:rw,nosuid,noexec,size=64m",
                    "/data:rw,nosuid,noexec,size=16m,uid=10001,gid=10001",
                ],
                "volumes": [
                    {
                        "type": "bind",
                        "source": str(socket),
                        "target": "/acceptance-db",
                        "read_only": True,
                    }
                ],
            }
        }
    }
    compose_file = scratch / "compose.json"
    compose_file.write_text(json.dumps(manifest))
    compose_file.chmod(0o600)
    # A fresh process environment also prevents COMPOSE_FILE/DOCKER_HOST and
    # interpolation from redirecting this smoke test to a deployment stack.
    child_env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG") if k in os.environ}
    base = ["docker", "compose", "--project-name", project, "-f", str(compose_file)]

    def command(*args, check=True, timeout=60):
        result = subprocess.run(
            list(args), cwd=scratch, env=child_env, capture_output=True, text=True, timeout=timeout
        )
        if check:
            assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
        return result

    try:
        command(*base, "build", timeout=480)
        socket.chmod(0o711)
        command(*base, "up", "-d", "--no-build", "--no-deps")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            result = command(
                *base,
                "exec",
                "-T",
                "api",
                "curl",
                "-sf",
                "--max-time",
                "2",
                "http://127.0.0.1:9500/readyz",
                check=False,
            )
            if result.returncode == 0:
                break
            time.sleep(0.5)
        else:
            raise AssertionError(command(*base, "logs", check=False).stdout[-6000:])
        assert json.loads(result.stdout)["ok"]
        command(
            *base,
            "exec",
            "-T",
            "api",
            "python",
            "-c",
            "from billing import get_billing_engine; "
            "get_billing_engine().deposit('anonymous', 10, description='Isolated Compose smoke')",
        )
        response = command(
            *base,
            "exec",
            "-T",
            "api",
            "curl",
            "-sf",
            "--max-time",
            "15",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps({"name": project, "vram_needed_gb": 4, "gpu_model": "RTX 4090"}),
            "http://127.0.0.1:9500/instance",
        )
        job_id = json.loads(response.stdout)["instance"]["job_id"]
        import psycopg
        from control_plane.job_specs import validate_execution_spec

        with psycopg.connect(dsn) as conn:
            row = conn.execute(
                "SELECT phase,active_attempt_id,spec,spec_hash FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
        assert row and row[0] == "pending" and row[1] is None
        validate_execution_spec(row[2], row[3], require_admission=True)
    finally:
        try:
            command(*base, "down", "--volumes", "--remove-orphans")
            assert not command(*base, "ps", "--all", "--quiet").stdout.strip()
        finally:
            socket.chmod(original_mode)
            command("docker", "image", "rm", image, check=False)
