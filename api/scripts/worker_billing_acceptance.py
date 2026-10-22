#!/usr/bin/env python3
"""Real CUDA/worker HTTP/ledger acceptance, only inside run_security_gates.py.

Uses the actual worker protocol, observation collector and exit monitor. The
harness launches one bounded Docker container after lease claim; it does not
exercise public launch, run_job provisioning, wallet debit or production ingress.
No image pulls, host mounts, network access, privileged helper or real credentials.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
for _pkg in ("archive", "billing", "workers", "auth", "hosts", "privacy", "api", "core", ""):
    sys.path.insert(0, str(ROOT / _pkg if _pkg else ROOT))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def command(*args: str, timeout: int = 30, check: bool = True):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=check)


def validate_isolation() -> Path:
    from psycopg.conninfo import conninfo_to_dict

    require(os.environ.get("XCELSIOR_ISOLATED_ACCEPTANCE") == "1", "use run_security_gates.py")
    require(os.environ.get("XCELSIOR_ENV") == "test", "test environment required")
    require(os.environ.get("XCELSIOR_SKIP_DOTENV") == "1", "dotenv must be disabled")
    dsn = os.environ.get("XCELSIOR_POSTGRES_DSN", "")
    config = conninfo_to_dict(dsn)
    require(set(config) <= {"host", "dbname", "user"}, "unexpected database connection override")
    host = Path(config.get("host", ""))
    require(
        host.is_absolute()
        and host.parent.name.startswith("xcl-sec-pg-")
        and host.name == "socket"
        and host.is_dir(),
        "disposable Unix socket required",
    )
    require(
        config.get("dbname") == "xcelsior_security_test" and config.get("user") == "security_test",
        "disposable database identity required",
    )
    for key in ("DATABASE_URL", "XCELSIOR_PG_DSN", "XCELSIOR_TEST_POSTGRES_DSN"):
        require(os.environ.get(key) == dsn, f"{key} must use the same isolated database")
    state = Path(os.environ.get("XCELSIOR_ACCEPTANCE_STATE", "")).resolve()
    require(state.parent == host.parent and state.is_dir(), "isolated worker state required")
    return state


def wait_for(predicate, seconds: float, description: str):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.2)
    raise RuntimeError(f"timed out: {description}")


def run(args, evidence: dict) -> None:
    state = validate_isolation()
    # Worker helpers invoke the Docker CLI themselves. An empty CLI config plus
    # an explicit Unix socket prevents a user's saved remote Docker context
    # from turning this local acceptance into a remote workload.
    docker_config = state / "docker"
    docker_config.mkdir(mode=0o700)
    os.environ.update(DOCKER_CONFIG=str(docker_config), DOCKER_HOST="unix:///var/run/docker.sock")
    image = json.loads(command("docker", "image", "inspect", args.image).stdout)[0]
    image_id = image["Id"]  # launch these exact cached bytes, never pull a tag
    gpu = (
        command(
            "nvidia-smi",
            f"--id={args.gpu_device}",
            "--query-gpu=uuid,name,memory.free",
            "--format=csv,noheader,nounits",
        )
        .stdout.strip()
        .split(",")
    )
    require(len(gpu) == 3 and float(gpu[2]) >= 1024, "one GPU with at least 1 GiB free is required")
    gpu_uuid, gpu_model = gpu[0].strip(), gpu[1].strip()
    evidence.update(image_id=image_id, gpu_uuid=gpu_uuid, gpu_model=gpu_model, journeys=[])
    marker = uuid.uuid4().hex[:12]
    host_id, owner = f"acceptance-host-{marker}", f"acceptance-owner-{marker}"
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(64)
    base = f"http://127.0.0.1:{listener.getsockname()[1]}"
    os.environ.update(
        {
            "XCELSIOR_BASE_URL": base,
            "XCELSIOR_SCHEDULER_URL": base,
            "XCELSIOR_HOST_ID": host_id,
            "XCELSIOR_AGENT_V2_HOSTS": host_id,
            "XCELSIOR_AGENT_HOST_TOKENS": "require",
            "XCELSIOR_AGENT_PUBLIC_INGRESS": "allow",
            "XCELSIOR_ALLOW_UNAUTH_AGENT": "0",
            "XCELSIOR_BG_TASKS": "false",
            "XCELSIOR_AUTH_CACHE_BACKEND": "memory",
            "XCELSIOR_AGENT_TOKEN_FILE": str(state / "agent-token.json"),
            "XCELSIOR_V2_JOURNAL_PATH": str(state / "attempts.json"),
            "STATE_DIRECTORY": str(state),
            "XCELSIOR_RECONCILE_ACTION_BILLING_ORPHANED_METER": "enforce",
        }
    )
    from psycopg.types.json import Jsonb
    from db import _get_pg_pool
    from control_plane.agent_tokens import issue_token
    from control_plane.billing_controller import reconcile_billing_meters
    from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
    from control_plane.scheduler.service import SchedulerService
    from billing import BillingEngine
    import uvicorn
    from api import app
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    faults: dict[str, dict] = {}

    class TerminalFaults(BaseHTTPMiddleware):
        """Inject transport-visible failures around the real API transaction."""

        async def dispatch(self, request, call_next):
            if request.url.path != "/agent/v2/attempts/status":
                return await call_next(request)
            body = await request.json()
            fault = faults.get(body.get("job_id"))
            if fault is None or body.get("status") not in ("succeeded", "failed"):
                return await call_next(request)
            fault["requests"] += 1
            if fault["requests"] == 2:
                journal = json.loads((state / "attempts.json").read_text())
                require(body["job_id"] in journal, "retry lost durable authority record")
                require(
                    command("docker", "inspect", fault["container"], check=False).returncode == 0,
                    "container was removed before terminal retry",
                )
                fault["evidence_retained"] = True
            if fault["requests"] == 1:
                if fault["mode"] == "after_commit":
                    response = await call_next(request)
                    require(response.status_code == 200, "terminal transaction did not commit")
                    async for _ in response.body_iterator:
                        pass
                    fault["committed_before_response_loss"] = True
                fault["injected"].set()
                return JSONResponse({"error": "isolated terminal-report fault"}, status_code=503)
            return await call_next(request)

    app.add_middleware(TerminalFaults)

    pool = _get_pg_pool()
    with pool.connection() as conn:
        require(
            conn.execute("SELECT count(*) FROM jobs").fetchone() == (0,),
            "acceptance needs an empty job ledger",
        )
        conn.execute(
            "INSERT INTO hosts (host_id,status,registered_at,payload,admission_state) "
            "VALUES (%s,'active',%s,%s,'admitted')",
            (
                host_id,
                time.time(),
                Jsonb(
                    {
                        "host_id": host_id,
                        "gpu_model": gpu_model,
                        "gpu_count": 1,
                        "free_vram_gb": float(gpu[2]) / 1024,
                        "total_vram_gb": float(gpu[2]) / 1024,
                        "cost_per_hour": 1,
                        "admitted": True,
                        "last_seen": time.time(),
                        "capabilities": {"egress_isolation": True},
                    }
                ),
            ),
        )
        token = issue_token(conn, host_id, issued_by="isolated-acceptance")
    token_file = state / "agent-token.json"
    token_file.write_text(json.dumps({"token": token.secret, "host_id": host_id}))
    token_file.chmod(0o600)
    import worker_agent as worker

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    server_thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    server_thread.start()
    container = None
    try:
        wait_for(lambda: server.started, 30, "loopback API startup")
        require(worker.negotiate_protocol_v2(), "worker did not negotiate v2 over HTTP")
        for exit_code, fault_mode in (
            (0, "none"),
            (7, "none"),
            (0, "before_commit"),
            (7, "after_commit"),
            (0, "restart"),
        ):
            job_id = f"acceptance-{marker}-{exit_code}-{fault_mode}"
            journey = {
                "job_id": job_id,
                "expected_exit_code": exit_code,
                "terminal_fault": fault_mode,
            }
            evidence["journeys"].append(journey)
            program = (
                "import json,time,torch; from pathlib import Path; "
                "a=torch.ones((32,32),device='cuda'); result=float((a@a).sum().item()); "
                "assert result==32768.0; "
                "print(json.dumps({'cuda_result':result,'device':torch.cuda.get_device_name(0)}),flush=True); "
                "deadline=time.monotonic()+90\n"
                "while not Path('/tmp/release').exists():\n"
                " if time.monotonic()>deadline: raise RuntimeError('acceptance release timeout')\n"
                " time.sleep(.1)\n"
                f"raise SystemExit({exit_code})\n"
            )
            spec = {
                "job_id": job_id,
                "name": job_id,
                "owner": owner,
                "gpu_model": gpu_model,
                "num_gpus": 1,
                "vram_needed_gb": 0.125,
                "status": "queued",
                "image": image_id,
                "command": ["python", "-u", "-c", program],
            }
            with pool.connection() as conn:
                conn.execute(
                    "INSERT INTO jobs (job_id,status,priority,submitted_at,payload) "
                    "VALUES (%s,'queued',0,%s,%s)",
                    (job_id, time.time(), Jsonb(spec)),
                )
            report = SchedulerService(
                SchedulerConfig(
                    mode=SchedulerMode.CANARY,
                    canary_gpu_models=frozenset({gpu_model.lower()}),
                    canary_host_ids=frozenset({host_id}),
                    lease_claim_ttl_sec=300,
                    lease_renewal_ttl_sec=300,
                )
            ).tick()
            require(len(report.placements) == 1, "scheduler did not place exactly one job")
            claims = worker.v2_claim_commands(limit=1)
            require(
                len(claims) == 1 and claims[0]["job_id"] == job_id, "wrong/missing worker command"
            )
            claim = claims[0]
            journey.update(
                command_id=claim["command_id"],
                spec_hash=claim["spec_hash"],
                fencing_token=claim["fencing_token"],
                lease_id=claim["args"]["lease_id"],
            )
            require(
                worker.compute_spec_hash(claim["args"]["spec"]) == claim["spec_hash"],
                "spec hash mismatch",
            )
            auth = {
                "job_id": job_id,
                "attempt_id": claim["attempt_id"],
                "fencing_token": claim["fencing_token"],
                "lease_id": claim["args"]["lease_id"],
                "spec_hash": claim["spec_hash"],
                "stop": threading.Event(),
            }
            require(worker.v2_claim_lease_fenced(auth) is not None, "lease claim rejected")
            require(
                worker.v2_report_attempt_status(auth, "lease_claimed") == "ok",
                "claim status rejected",
            )
            require(
                worker.v2_ack_command(claim["command_id"], {"lease_claimed": True}),
                "command ACK rejected",
            )
            container = worker._v2_container_name(job_id, auth["attempt_id"])
            auth["container_name"] = container
            with worker._v2_attempts_lock:
                worker._v2_attempts[job_id] = auth
            worker._v2_journal_save()
            docker_args = [
                "docker",
                "run",
                "-d",
                "--pull=never",
                "--name",
                container,
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--memory=1g",
                "--cpus=1",
                "--pids-limit=128",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
                "--gpus",
                f"device={gpu_uuid}",
            ]
            for key in ("job_id", "attempt_id", "fencing_token", "spec_hash"):
                docker_args += ["--label", f"ai.xcelsior.{key}={auth[key]}"]
            docker_args += [image_id, "python", "-u", "-c", program]
            journey["container_id"] = command(*docker_args).stdout.strip()

            def cuda_ready():
                lines = command("docker", "logs", container).stdout.splitlines()
                for line in lines:
                    if line.startswith('{"cuda_result":'):
                        return json.loads(line)
                return None

            journey["cuda"] = wait_for(cuda_ready, 45, "real CUDA result")
            require(
                worker.report_job_status(
                    job_id, "running", host_id=host_id, container_name=container
                ),
                "worker running report rejected",
            )
            require(worker.report_observations_v2(force=True), "real Docker observation rejected")
            os.environ["XCELSIOR_RECONCILE_ACTION_BILLING_ACTIVE_METER_MISSING"] = "report_only"
            with pool.connection() as conn:
                reconcile_billing_meters(conn)
                require(
                    conn.execute(
                        "SELECT count(*) FROM usage_meters WHERE attempt_id=%s",
                        (auth["attempt_id"],),
                    ).fetchone()
                    == (0,),
                    "report-only created a meter",
                )
            os.environ["XCELSIOR_RECONCILE_ACTION_BILLING_ACTIVE_METER_MISSING"] = "enforce"
            with pool.connection() as conn:
                result = reconcile_billing_meters(conn)
                require(len(result.meters_created) == 1, "active meter repair did not execute")
                journey["meter_id"] = result.meters_created[0]
                action = conn.execute(
                    "SELECT action_result FROM reconciliation_findings WHERE resource_id=%s "
                    "AND finding_type='billing_active_meter_missing' AND action_taken='open_active_meter'",
                    (auth["attempt_id"],),
                ).fetchone()
                require(
                    action is not None and bool(action[0].get("observation_id")),
                    "repair observation evidence missing",
                )
                journey["observation_id"] = action[0]["observation_id"]
                require(
                    conn.execute(
                        "SELECT completed_at,total_cost_micros FROM usage_meters WHERE meter_id=%s",
                        (journey["meter_id"],),
                    ).fetchone()
                    == (None, 0),
                    "active usage was charged",
                )
            engine = BillingEngine()
            before = engine.generate_invoice(
                owner, "isolated acceptance", 0, time.time() + 1, persist=False
            )
            command(
                "docker",
                "exec",
                container,
                "python",
                "-c",
                "from pathlib import Path; Path('/tmp/release').touch()",
            )
            actual_exit = int(command("docker", "wait", container, timeout=30).stdout.strip())
            require(actual_exit == exit_code, "unexpected workload exit")
            journey["actual_exit_code"] = actual_exit
            if fault_mode != "none":
                faults[job_id] = {
                    "mode": fault_mode,
                    "container": container,
                    "requests": 0,
                    "injected": threading.Event(),
                    "evidence_retained": False,
                }
            if fault_mode == "restart":
                # Two genuinely separate worker processes share only their
                # on-disk journal and exited Docker container. SIGTERM ends the
                # first monitor after the failed HTTP request, before retry.
                child_code = """
import json, signal, sys, threading
from pathlib import Path
root = Path(sys.argv[1])
for package in ("archive", "billing", "workers", "auth", "hosts", "privacy", "api", "core", ""):
    sys.path.insert(0, str(root / package))
import worker_agent as worker
signal.signal(signal.SIGTERM, lambda *_: worker._shutdown.set())
worker._v2_enabled = True
if sys.argv[2] == "monitor":
    records = worker._v2_journal_load()
    auth = records[sys.argv[3]]
    auth["stop"] = threading.Event()
    worker._v2_attempts[auth["job_id"]] = auth
    worker._monitor_container(auth["job_id"], auth["container_name"], is_v2_attempt=True)
else:
    worker.v2_adopt_attempts()
    assert sys.argv[3] not in worker._v2_journal_load(), "restart did not complete terminal reporting"
"""
                child = subprocess.Popen(
                    [sys.executable, "-c", child_code, str(ROOT), "monitor", job_id]
                )
                try:
                    require(
                        faults[job_id]["injected"].wait(20), "worker did not reach injected fault"
                    )
                    child.terminate()
                    require(child.wait(timeout=20) == 0, "worker monitor did not stop cleanly")
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=10)
                require(
                    job_id in json.loads((state / "attempts.json").read_text()),
                    "worker restart lost terminal journal",
                )
                command(sys.executable, "-c", child_code, str(ROOT), "adopt", job_id, timeout=30)
                worker._v2_forget_attempt(job_id, auth)
                journey["separate_worker_processes"] = True
            else:
                worker._monitor_container(job_id, container, is_v2_attempt=True)
            if fault_mode != "none":
                fault = faults[job_id]
                require(
                    fault["requests"] == 2 and fault["evidence_retained"],
                    "terminal recovery did not retry with durable evidence",
                )
                journey["terminal_report_requests"] = fault["requests"]
                journey["retained_until_retry"] = fault["evidence_retained"]
                journey["committed_before_response_loss"] = fault.get(
                    "committed_before_response_loss", False
                )
            require(
                worker.v2_report_attempt_status(auth, "running") == "fenced",
                "terminal authority accepted a stale report",
            )
            journey["stale_report_refused"] = True
            require(
                command("docker", "inspect", container, check=False).returncode != 0,
                "worker did not remove its container",
            )
            container = None
            with pool.connection() as conn:
                reconcile_billing_meters(conn)
                row = conn.execute(
                    "SELECT a.status,l.status,m.meter_id,m.started_at,m.completed_at,m.total_cost_micros,"
                    "EXTRACT(EPOCH FROM a.started_at)::float8,EXTRACT(EPOCH FROM a.ended_at)::float8 "
                    "FROM job_attempts a JOIN placement_leases l ON l.attempt_id=a.attempt_id "
                    "JOIN usage_meters m ON m.attempt_id=a.attempt_id::text WHERE a.attempt_id=%s",
                    (auth["attempt_id"],),
                ).fetchone()
                require(row is not None, "terminal attempt/meter evidence missing")
                require(
                    row[0] == ("succeeded" if exit_code == 0 else "failed")
                    and row[1] == "released",
                    "actual exit did not settle attempt and lease",
                )
                require(
                    row[2] == journey["meter_id"]
                    and row[3] == row[6]
                    and row[4] == row[7]
                    and row[5] > 0,
                    "terminal meter differs from authoritative usage",
                )
                require(
                    conn.execute(
                        "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id=%s AND status='active'",
                        (auth["attempt_id"],),
                    ).fetchone()
                    == (0,),
                    "allocation was not released",
                )
                repeated = reconcile_billing_meters(conn)
                require(
                    not repeated.meters_created and not repeated.meters_closed,
                    "repair replay changed usage",
                )
                require(
                    conn.execute(
                        "SELECT count(*) FROM usage_meters WHERE attempt_id=%s",
                        (auth["attempt_id"],),
                    ).fetchone()
                    == (1,),
                    "duplicate meter",
                )
                require(
                    conn.execute(
                        "SELECT count(*) FROM wallet_transactions WHERE job_id=%s", (job_id,)
                    ).fetchone()
                    == (0,),
                    "meter repair must not create wallet transactions",
                )
            after = engine.generate_invoice(
                owner, "isolated acceptance", 0, time.time() + 1, persist=False
            )
            # Line items retain micro-CAD; the displayed invoice subtotal rounds
            # once to cents and cannot be compared to a sub-cent GPU run.
            from money import cad_to_micros, cad_to_minor, minor_to_cad

            before_micros = sum(cad_to_micros(li["subtotal_cad"]) for li in before.line_items)
            after_micros = sum(cad_to_micros(li["subtotal_cad"]) for li in after.line_items)
            require(
                after_micros - before_micros == row[5], "invoice lines and exact meter disagree"
            )
            require(
                after.subtotal_cad == minor_to_cad(cad_to_minor(Decimal(after_micros) / 1_000_000)),
                "invoice display rounding disagrees with exact usage",
            )
            journey.update(
                attempt_id=auth["attempt_id"],
                attempt_status=row[0],
                lease_status=row[1],
                started_at=row[3],
                completed_at=row[4],
                total_cost_micros=row[5],
                invoice_delta_micros=after_micros - before_micros,
                invoice_display_subtotal_cad=after.subtotal_cad,
            )

    finally:
        if container:
            command("docker", "rm", "-f", container, check=False)
            if command("docker", "inspect", container, check=False).returncode == 0:
                evidence["cleanup_required"] = container
        server.should_exit = True
        server_thread.join(timeout=15)
        listener.close()
        require(not server_thread.is_alive(), "API did not shut down")
        pool.close()
        require("cleanup_required" not in evidence, "acceptance container cleanup failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu-device", default="0")
    args = parser.parse_args()
    evidence = {
        "passed": False,
        "started_at": time.time(),
        "scope": "isolated CUDA worker protocol to invoice",
    }
    try:
        run(args, evidence)
        evidence["passed"] = True
        return 0
    except Exception as exc:
        # Never dump process arguments/environment or authentication material.
        evidence["error"] = (
            f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, subprocess.CalledProcessError)
            else f"command failed with exit {exc.returncode}: {exc.stderr[-1000:]}"
        )
        print(evidence["error"], file=sys.stderr)
        return 1
    finally:
        evidence["finished_at"] = time.time()
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
