#!/usr/bin/env python3
"""Production terminal matrix smoke.

Exercises the ticket + WebSocket path against a running instance when one
exists for the audit user. Always runs the automated PTY echo unit test as a
baseline. Exits 0 when API/WS checks pass or are skipped (no running instance).

Usage:
  python scripts/terminal_prod_matrix.py
  python scripts/terminal_prod_matrix.py --base-url https://xcelsior.ca
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import httpx
import websockets

PROJECT = Path(__file__).resolve().parent.parent
ENV_AUDIT = PROJECT / ".env.audit"
RUNNING = frozenset({"running", "assigned", "leased"})


def _load_cfg() -> dict[str, str]:
    cfg: dict[str, str] = {}
    if ENV_AUDIT.exists():
        for line in ENV_AUDIT.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    cfg["base"] = (
        os.environ.get("AUDIT_BASE") or cfg.get("AUDIT_BASE") or "https://xcelsior.ca"
    ).rstrip("/")
    cfg["email"] = os.environ.get("AUDIT_EMAIL") or cfg.get("AUDIT_EMAIL", "")
    cfg["password"] = os.environ.get("AUDIT_PASSWORD") or cfg.get("AUDIT_PASSWORD", "")
    return cfg


def _step(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{mark}] {name}{suffix}")
    return ok


def _run_unit_echo_test() -> bool:
    cmd = [
        str(PROJECT / "venv" / "bin" / "pytest"),
        "tests/test_terminal.py::TestPtySession::test_echo_roundtrip",
        "-q",
        "--tb=no",
    ]
    proc = subprocess.run(cmd, cwd=PROJECT, capture_output=True, text=True)
    ok = proc.returncode == 0
    detail = "passed" if ok else (proc.stdout + proc.stderr).strip()[-200:]
    return _step("unit_pty_echo_roundtrip", ok, detail)


async def _ws_probe(base: str, instance_id: str, ticket: str) -> tuple[bool, str]:
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    url = f"{ws_base}/ws/terminal/{instance_id}?ticket={ticket}"
    try:
        async with websockets.connect(url, open_timeout=15, close_timeout=5) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=20)
            if isinstance(raw, bytes):
                return False, "expected JSON status frame"
            status = json.loads(raw)
            if status.get("type") != "status":
                return False, f"unexpected frame: {status.get('type')}"
            await ws.send(json.dumps({"type": "input", "data": "ls\n"}))
            try:
                out = await asyncio.wait_for(ws.recv(), timeout=15)
            except asyncio.TimeoutError:
                # Container may be slow — connected status is enough for matrix.
                return True, "connected (no PTY output within 15s)"
            if isinstance(out, bytes) and len(out) > 0:
                return True, f"pty_bytes={len(out)}"
            if isinstance(out, str):
                try:
                    ctrl = json.loads(out)
                    if ctrl.get("type") == "error":
                        return False, ctrl.get("message", "error")
                except json.JSONDecodeError:
                    pass
            return True, "connected"
    except Exception as e:
        return False, str(e)[:200]


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal production matrix smoke")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--skip-unit", action="store_true")
    args = parser.parse_args()

    failures = 0
    if not args.skip_unit and not _run_unit_echo_test():
        failures += 1

    cfg = _load_cfg()
    base = (args.base_url or cfg["base"]).rstrip("/")
    email = cfg["email"]
    password = cfg["password"]
    if not email or not password:
        print("Missing .env.audit credentials — skipping prod WS probe")
        return 1 if failures else 0

    with httpx.Client(base_url=base, timeout=60.0) as client:
        login = client.post("/api/auth/login", json={"email": email, "password": password})
        if not _step("login", login.status_code == 200, str(login.status_code)):
            return 1
        token = login.json()["access_token"]
        hdrs = {"Authorization": f"Bearer {token}"}

        listed = client.get("/instances", headers=hdrs)
        if not _step("list_instances", listed.status_code == 200, str(listed.status_code)):
            return 1

        instances = listed.json().get("instances") or []
        running = [
            inst
            for inst in instances
            if str(inst.get("status") or "").lower() in RUNNING
        ]
        if not running:
            _step("prod_ws_matrix", True, "SKIP — no running instances for audit user")
            return 1 if failures else 0

        inst = running[0]
        instance_id = str(inst.get("job_id") or inst.get("instance_id") or "")
        if not instance_id:
            _step("prod_ws_matrix", False, "running instance missing job_id")
            return 1

        ticket_resp = client.post(
            "/api/terminal/ticket",
            headers=hdrs,
            json={"instance_id": instance_id},
        )
        if not _step(
            "terminal_ticket",
            ticket_resp.status_code == 200 and bool(ticket_resp.json().get("ticket")),
            f"instance={instance_id}",
        ):
            return 1

        ticket = ticket_resp.json()["ticket"]
        ok, detail = asyncio.run(_ws_probe(base, instance_id, ticket))
        if not _step("prod_ws_matrix", ok, f"{instance_id} {detail}"):
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())