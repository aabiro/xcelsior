#!/usr/bin/env python3
"""Post-deploy smoke for serverless inference endpoints.

Usage:
  python scripts/serverless_e2e_smoke.py --email you@example.com --password '...'
  python scripts/serverless_e2e_smoke.py --token 'eyJ...' --base-url https://xcelsior.ca
  python scripts/serverless_e2e_smoke.py --skip-stream   # skip SSE probe

Flow: enabled probe → create → run → status → stream → cancel → scale-to-zero → delete.
Exits 0 on success, 1 on any failure. Does not require a live GPU worker for API checks
(jobs may remain IN_QUEUE).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parent.parent
ENV_AUDIT = PROJECT / ".env.audit"

POLL_INTERVAL_SEC = 2
POLL_MAX_WAIT_SEC = 30
SMOKE_IMAGE = "xcelsior/serverless-base:cuda12.4-py3.12"


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


def _headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    if token.lower().startswith("bearer "):
        return {"Authorization": token}
    return {"Authorization": f"Bearer {token}"}


def _step(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{mark}] {name}{suffix}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Serverless post-deploy smoke")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--email", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--token", default="")
    parser.add_argument("--skip-stream", action="store_true")
    args = parser.parse_args()

    cfg = _load_cfg()
    base = (args.base_url or cfg["base"]).rstrip("/")
    token = args.token.strip()

    if not token:
        email = args.email or cfg["email"]
        password = args.password or cfg["password"]
        if not email or not password:
            print("Missing credentials — pass --token or --email/--password (or .env.audit)")
            return 1
    else:
        email = password = ""

    results: dict[str, object] = {}
    failures = 0

    with httpx.Client(base_url=base, timeout=60.0) as client:
        if not token:
            login = client.post(
                "/api/auth/login",
                json={"email": email, "password": password},
            )
            if not _step("login", login.status_code == 200, str(login.status_code)):
                return 1
            token = login.json()["access_token"]

        hdrs = _headers(token)

        enabled = client.get("/api/v2/serverless/enabled", headers=hdrs)
        results["enabled"] = enabled.json() if enabled.status_code == 200 else enabled.text[:200]
        if not _step(
            "feature_enabled",
            enabled.status_code == 200 and enabled.json().get("enabled") is True,
            json.dumps(results["enabled"]),
        ):
            return 1

        name = f"smoke-{uuid.uuid4().hex[:8]}"
        created = client.post(
            "/api/v2/serverless/endpoints",
            headers=hdrs,
            json={
                "name": name,
                "mode": "custom",
                "docker_image": SMOKE_IMAGE,
                "min_workers": 0,
                "max_workers": 1,
                "idle_timeout_sec": 60,
            },
        )
        if not _step("create_endpoint", created.status_code == 200, created.text[:200]):
            return 1
        endpoint_id = created.json()["endpoint"]["endpoint_id"]
        results["endpoint_id"] = endpoint_id

        run = client.post(
            f"/v1/serverless/{endpoint_id}/run",
            headers=hdrs,
            json={"input": {"smoke": True, "ts": time.time()}},
        )
        if not _step("run_job", run.status_code == 200, run.text[:200]):
            failures += 1
            job_id = ""
        else:
            job_id = str(run.json().get("id") or "")
            results["job_id"] = job_id

        if job_id:
            deadline = time.time() + POLL_MAX_WAIT_SEC
            status_body: dict = {}
            while time.time() < deadline:
                status = client.get(
                    f"/v1/serverless/{endpoint_id}/status/{job_id}",
                    headers=hdrs,
                )
                if status.status_code == 200:
                    status_body = status.json()
                    st = str(status_body.get("status") or "")
                    if st not in ("", "IN_QUEUE", "IN_PROGRESS"):
                        break
                time.sleep(POLL_INTERVAL_SEC)
            if not _step(
                "job_status",
                status.status_code == 200 and bool(status_body.get("status")),
                str(status_body.get("status")),
            ):
                failures += 1

            if not args.skip_stream:
                try:
                    with client.stream(
                        "GET",
                        f"/v1/serverless/{endpoint_id}/stream/{job_id}",
                        headers=hdrs,
                        timeout=15.0,
                    ) as stream_resp:
                        chunks = []
                        for chunk in stream_resp.iter_bytes():
                            chunks.append(chunk)
                            if sum(len(c) for c in chunks) > 0:
                                break
                        stream_ok = stream_resp.status_code == 200
                except Exception as e:
                    stream_ok = False
                    chunks = [str(e).encode()]
                if not _step("job_stream", stream_ok, f"bytes={sum(len(c) for c in chunks)}"):
                    failures += 1

            cancel = client.post(
                f"/v1/serverless/{endpoint_id}/cancel/{job_id}",
                headers=hdrs,
            )
            if not _step(
                "cancel_job",
                cancel.status_code == 200,
                str(cancel.json().get("status") if cancel.status_code == 200 else cancel.text[:120]),
            ):
                failures += 1

        scaled = client.patch(
            f"/api/v2/serverless/endpoints/{endpoint_id}",
            headers=hdrs,
            json={"min_workers": 0, "max_workers": 1, "idle_timeout_sec": 60},
        )
        if not _step("scale_to_zero", scaled.status_code == 200, scaled.text[:120]):
            failures += 1

        deleted = client.delete(
            f"/api/v2/serverless/endpoints/{endpoint_id}",
            headers=hdrs,
        )
        if not _step("delete_endpoint", deleted.status_code == 200, deleted.text[:120]):
            failures += 1

    print(json.dumps(results, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())