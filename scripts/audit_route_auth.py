#!/usr/bin/env python3
"""Scan route modules for handlers missing auth guards.

Heuristic: flag @router.(get|post|put|patch|delete) defs whose body does not
reference a known guard helper within the first 40 lines of the handler.

Usage:
  python scripts/audit_route_auth.py
  python scripts/audit_route_auth.py --strict        # exit 1 if any findings
  python scripts/audit_route_auth.py --critical      # admin-only infra handlers
  python scripts/audit_route_auth.py --critical --strict
  python scripts/audit_route_auth.py --guarded       # CI-tracked sensitive routes
  python scripts/audit_route_auth.py --guarded --strict
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
ROUTES_DIR = PROJECT / "routes"

GUARD_NAMES = frozenset(
    {
        "_require_auth",
        "_require_admin",
        "_require_platform_worker",
        "_require_worker_status_update",
        "_require_customer_access",
        "_require_provider_access",
        "_require_user_grant",
        "_require_agent_auth",
        "_require_scope",
        "_require_write_access",
        "_require_entity_event_access",
        "_require_inference_job_access",
        "_get_current_user",
        "_check_job_access",
        "_check_terminal_access",
        "_authorize_instance_mutation",
        "_is_platform_admin",
    }
)

# Public-by-design (health probes, OAuth metadata, marketing data)
ALLOWLIST = frozenset(
    {
        "api_health",
        "api_healthz",
        "api_ready",
        "api_liveness",
        "oauth_authorization_server_metadata",
        "api_gpu_available",
        "api_list_tiers",
        "api_image_templates",
        "api_trust_tiers",
        "api_sla_targets",
        "api_get_pubkey",
        "oauth_verify_page",
        "oauth_verify_device",
        "api_auth_register",
        "api_auth_login",
        "api_auth_oauth_initiate",
        "api_auth_oauth_callback",
        "api_auth_password_reset",
        "api_auth_password_reset_confirm",
        "api_auth_verify_email",
        "api_auth_resend_verification",
        "api_accept_team_invite",
    }
)

# Handlers that must never be reachable without admin auth.
CRITICAL_ADMIN_HANDLERS = frozenset(
    {
        ("routes/health.py", "api_generate_token"),
        ("routes/health.py", "api_nfs_config"),
        ("routes/health.py", "api_build_image"),
        ("routes/health.py", "api_list_builds"),
        ("routes/health.py", "api_generate_dockerfile"),
    }
)

ADMIN_GUARD_NAMES = frozenset({"_require_admin"})

# Customer/host routes audited in CI — add new sensitive handlers here.
GUARDED_ROUTE_HANDLERS = frozenset(
    {
        ("routes/billing.py", "api_reserve_commitment"),
        ("routes/jurisdiction.py", "api_list_canadian_hosts"),
        ("routes/jurisdiction.py", "api_jurisdiction_hosts"),
    }
)


def _handler_body_text(lines: list[str], node: ast.FunctionDef, max_lines: int = 40) -> str:
    start = node.lineno - 1
    end = min(len(lines), start + max_lines)
    return "\n".join(lines[start:end])


def audit_file(
    path: Path,
    *,
    critical_only: bool,
    guarded_only: bool,
) -> list[dict]:
    source = path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    rel = str(path.relative_to(PROJECT))
    findings: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in ALLOWLIST:
            continue
        if critical_only and (rel, node.name) not in CRITICAL_ADMIN_HANDLERS:
            continue
        if guarded_only and (rel, node.name) not in GUARDED_ROUTE_HANDLERS:
            continue
        if not any(
            isinstance(d, ast.Call)
            and isinstance(getattr(d.func, "attr", None), str)
            and d.func.attr in ("get", "post", "put", "patch", "delete")
            for d in node.decorator_list
            if isinstance(d, ast.Call)
        ):
            continue
        body_text = _handler_body_text(lines, node)
        if critical_only:
            if any(g in body_text for g in ADMIN_GUARD_NAMES):
                continue
        elif guarded_only:
            if any(g in body_text for g in GUARD_NAMES):
                continue
        elif any(g in body_text for g in GUARD_NAMES):
            continue
        findings.append(
            {
                "file": rel,
                "handler": node.name,
                "line": node.lineno,
            }
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--critical",
        action="store_true",
        help="Only check high-risk infrastructure handlers that require admin",
    )
    parser.add_argument(
        "--guarded",
        action="store_true",
        help="Only check CI-tracked sensitive routes (billing, host inventory, …)",
    )
    args = parser.parse_args()

    all_findings: list[dict] = []
    for path in sorted(ROUTES_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        all_findings.extend(
            audit_file(path, critical_only=args.critical, guarded_only=args.guarded)
        )

    if not all_findings:
        print('{"ok": true, "findings": []}')
        return 0

    print(json.dumps({"ok": False, "findings": all_findings}, indent=2))
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())