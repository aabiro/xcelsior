#!/usr/bin/env python3
"""Split monolithic api.py into domain-based APIRouter modules under routes/.

Usage: python scripts/split_api.py          # dry-run (print summary)
       python scripts/split_api.py --write  # generate route files
"""
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
API_PY = ROOT / "api.py"
ROUTES_DIR = ROOT / "routes"

# ── Tag → module mapping ──────────────────────────────────────────────
TAG_TO_MODULE = {
    "Hosts": "hosts", "Instances": "instances", "Billing": "billing",
    "Infrastructure": "infra", "SSH Keys": "ssh", "Auth": "auth",
    "Auth – MFA": "mfa", "Teams": "teams", "Marketplace": "marketplace",
    "Marketplace v2": "marketplace", "Jurisdiction": "jurisdiction",
    "Autoscale": "autoscale", "Agent": "agent", "Spot Pricing": "spot",
    "Events": "events", "Verification": "verification",
    "Reputation": "reputation", "Compliance": "compliance",
    "Privacy": "privacy", "Transparency": "transparency", "SLA": "sla",
    "Providers": "providers", "Notifications": "notifications",
    "Admin": "admin", "Chat": "chat", "AI Assistant": "chat",
    "Inference": "inference", "Inference v2": "inference",
    "Telemetry": "agent", "Volumes": "volumes", "Cloud Burst": "cloudburst",
    "GPU": "gpu", "Jobs": "scheduler_routes",
}

PATH_OVERRIDES = {
    "/api/slurm/": "slurm", "/api/nfs/": "slurm",
    "/api/pricing/": "pricing", "/api/analytics/": "billing",
    "/api/billing/gst-threshold": "compliance",
    "/token/generate": "auth", "/api/auth/device": "auth",
    "/api/auth/token": "auth", "/api/auth/verify": "auth",
    "/api/users/me/preferences": "auth", "/api/auth/me/data-export": "auth",
    "/api/artifacts": "artifacts",
    "/api/queue/process-sovereign": "jurisdiction",
    "/api/audit/": "events",
    "/build": "infra", "/builds": "infra",
    "/alerts/": "infra", "/api/alerts/": "infra",
    "/healthz": "health", "/readyz": "health", "/metrics": "health",
    "/dashboard": "health", "/legacy": "health", "/llms.txt": "health",
    "/api/stream": "health",
    "/compute-score": "hosts", "/compute-scores": "hosts",
    "/v1/chat/completions": "inference", "/v1/inference": "inference",
    "/api/v2/inference/complete/": "inference",
    "/api/v2/scheduler/": "scheduler_routes",
    "/api/v2/billing/": "billing", "/api/v2/burst/": "cloudburst",
    "/api/v2/privacy/": "privacy", "/api/v2/marketplace/": "marketplace",
    "/api/v2/gpu/": "gpu", "/api/v2/volumes/": "volumes",
    "/ssh/": "ssh", "/api/ssh/": "ssh",
    "/tiers": "instances", "/ws/": "instances",
}


def classify_route(path: str, tag: str) -> str:
    best_match, best_module = "", ""
    for prefix, module in PATH_OVERRIDES.items():
        if path.startswith(prefix) and len(prefix) > len(best_match):
            best_match, best_module = prefix, module
    if best_module:
        return best_module
    if path == "/":
        return "health"
    return TAG_TO_MODULE.get(tag, "misc")


# ── Per-module helper patterns (state/functions to move) ──────────────
MODULE_HELPERS = {
    "teams": [r'^(?:async\s+)?def _send_team_email\('],
    "instances": [
        r'^_TERMINAL_STATES\s*=', r'^_job_log_buffers\s*[=:{]',
        r'^_ws_connections\s*[=:{]', r'^(?:async\s+)?def _validate_ws_auth\(',
    ],
    "agent": [
        r'^_host_telemetry\s*[=:{]',
    ],
    "auth": [
        r'^_OAUTH_PROVIDERS\s*=', r'^_pending_device_codes\s*[=:{]',
        r'^_pending_verifications\s*[=:{]',
    ],
    "chat": [
        r'^(?:async\s+)?def _require_ai_enabled\(',
    ],
}

# ── Per-module import mapping ─────────────────────────────────────────
# Maps module name → list of import lines needed
# Common imports added to every module automatically

COMMON_IMPORTS = """\
import json
import os
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from routes._deps import (
    _require_auth, _require_admin, _is_platform_admin,
    broadcast_sse, log,
)
"""

MODULE_IMPORTS = {
    "hosts": """\
from pydantic import BaseModel, Field
from scheduler import (
    register_host, remove_host, list_hosts, get_host,
    API_TOKEN, estimate_compute_score, register_compute_score, get_compute_score,
)
from security import admit_node
from events import get_event_store
""",
    "instances": """\
import asyncio
import secrets
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from scheduler import (
    submit_job, list_jobs, update_job_status, process_queue, cancel_job,
    failover_and_reassign, requeue_job, list_tiers, run_job, kill_job,
    PRIORITY_TIERS, API_TOKEN, list_hosts,
)
from events import get_event_store, get_state_machine, JobState, EventType
from routes._deps import (
    _get_current_user, _require_provider_or_admin, AUTH_REQUIRED,
    _user_lock, _sessions, _api_keys, _USE_PERSISTENT_AUTH,
)
from db import UserStore
""",
    "billing": """\
from pydantic import BaseModel, Field
from scheduler import (
    bill_job, bill_all_completed, get_total_revenue, load_billing,
    list_jobs, API_TOKEN,
)
from billing import get_billing_engine, get_tax_rate_for_province
from stripe_connect import get_stripe_manager
from events import get_event_store
from routes._deps import _require_provider_or_admin, _get_current_user
from db import UserStore
""",
    "health": """\
import asyncio
from pathlib import Path
from collections import defaultdict

from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, PlainTextResponse
from scheduler import (
    list_jobs, list_hosts, get_metrics_snapshot, storage_healthcheck,
    get_total_revenue, load_billing, API_TOKEN, get_current_spot_prices,
)
from billing import get_billing_engine
from stripe_connect import get_stripe_manager
from routes._deps import (
    _sse_subscribers, _sse_lock, _get_current_user,
)
""",
    "ssh": """\
import secrets
from scheduler import generate_ssh_keypair, get_public_key, API_TOKEN
from routes._deps import _get_current_user
from db import UserStore
""",
    "auth": """\
import asyncio
import secrets
import hashlib as _hashlib
import re
import uuid
from collections import defaultdict
from pathlib import Path

from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from scheduler import list_jobs, API_TOKEN
from routes._deps import (
    _get_current_user, _require_provider_or_admin,
    _check_auth_rate_limit, _hash_password, _admin_flag,
    _create_session, _set_auth_cookie, _clear_auth_cookie,
    _merge_auth_user, AUTH_REQUIRED, VALID_ACCOUNT_ROLES,
    SESSION_EXPIRY, _AUTH_COOKIE_NAME,
    _users_db, _sessions, _api_keys, _user_lock, _USE_PERSISTENT_AUTH,
)
from db import UserStore
from security import sanitize_input
""",
    "mfa": """\
import secrets
import hashlib as _hashlib
import uuid

from pydantic import BaseModel, Field
from routes._deps import (
    _require_auth, _get_current_user, _check_auth_rate_limit,
    _create_session, _set_auth_cookie,
    _users_db, _user_lock, _USE_PERSISTENT_AUTH,
)
from db import UserStore, MfaStore
""",
    "teams": """\
from pydantic import BaseModel, Field
from routes._deps import _require_auth, _get_current_user
from db import UserStore
""",
    "marketplace": """\
from pydantic import BaseModel, Field
from scheduler import (
    list_rig, unlist_rig, get_marketplace, marketplace_bill,
    marketplace_stats, list_hosts, get_current_spot_prices,
)
from routes._deps import _require_provider_or_admin, _get_current_user
""",
    "slurm": """\
from pydantic import BaseModel, Field
from scheduler import list_jobs, list_hosts
from routes._deps import _get_current_user
""",
    "jurisdiction": """\
from pydantic import BaseModel
from scheduler import (
    register_host_ca, list_hosts_filtered, process_queue_filtered,
    set_canada_only, list_hosts, process_queue_sovereign,
    allocate_jurisdiction_aware,
)
from jurisdiction import (
    TrustTier, generate_residency_trace, TRUST_TIER_REQUIREMENTS,
)
from routes._deps import _get_current_user, _require_provider_or_admin
""",
    "autoscale": """\
from pydantic import BaseModel
from scheduler import (
    add_to_pool, remove_from_pool, load_autoscale_pool,
    autoscale_cycle, autoscale_up, autoscale_down,
)
""",
    "agent": """\
import threading as _threading
from collections import defaultdict

from pydantic import BaseModel, Field
from scheduler import list_hosts, list_jobs, API_TOKEN
from security import check_node_versions
from routes._deps import _get_current_user, _require_provider_or_admin
""",
    "spot": """\
from pydantic import BaseModel, Field
from scheduler import (
    get_current_spot_prices, update_spot_prices,
    submit_spot_job, preemption_cycle,
)
""",
    "events": """\
from scheduler import list_jobs
from events import get_event_store, get_state_machine
from routes._deps import _get_current_user
""",
    "verification": """\
from pydantic import BaseModel
from scheduler import list_hosts, get_host
from verification import get_verification_engine
from routes._deps import _get_current_user, _require_provider_or_admin
""",
    "reputation": """\
from pydantic import BaseModel
from scheduler import list_hosts
from reputation import (
    get_reputation_engine, ReputationTier, VerificationType,
    estimate_job_cost, GPU_REFERENCE_PRICING_CAD,
)
from routes._deps import _get_current_user
""",
    "compliance": """\
from pydantic import BaseModel
from jurisdiction import (
    PROVINCE_COMPLIANCE, TRUST_TIER_REQUIREMENTS, compute_fund_eligible_amount,
)
from billing import get_billing_engine, get_tax_rate_for_province, PROVINCE_TAX_RATES
from privacy import requires_quebec_pia
from routes._deps import _require_provider_or_admin, _get_current_user
""",
    "privacy": """\
from pydantic import BaseModel
from privacy import (
    get_lifecycle_manager, PrivacyConfig, RETENTION_POLICIES,
    redact_job_record, DataCategory,
)
from routes._deps import _get_current_user
""",
    "transparency": """\
import uuid

from pydantic import BaseModel, Field
from routes._deps import _get_current_user
""",
    "sla": """\
from pydantic import BaseModel, Field
from scheduler import list_hosts
from sla import get_sla_engine, SLATier, SLA_TARGETS
from routes._deps import _get_current_user
""",
    "providers": """\
from pydantic import BaseModel, Field
from stripe_connect import get_stripe_manager
from routes._deps import _get_current_user, _require_provider_or_admin
from db import UserStore
""",
    "notifications": """\
from db import NotificationStore
from routes._deps import _require_auth
""",
    "admin": """\
from scheduler import list_jobs, list_hosts, get_total_revenue
from billing import get_billing_engine
from routes._deps import (
    _require_admin, _get_current_user, _admin_flag,
    _users_db, _user_lock, _USE_PERSISTENT_AUTH,
)
from db import UserStore
from verification import get_verification_engine
""",
    "chat": """\
import asyncio

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from chat import (
    CHAT_API_KEY, check_chat_rate_limit, build_system_prompt,
    stream_chat_response, get_or_create_conversation,
    get_conversation_messages, get_user_conversations,
    record_feedback, append_message,
)
from routes._deps import _get_current_user
""",
    "inference": """\
import asyncio
import uuid

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from scheduler import list_hosts, list_jobs
from inference_store import (
    store_inference_job, get_inference_job,
    store_inference_result, get_inference_result,
)
from routes._deps import _get_current_user, _require_provider_or_admin
""",
    "volumes": """\
from pydantic import BaseModel, Field
from routes._deps import _get_current_user
""",
    "cloudburst": """\
from routes._deps import _require_auth
""",
    "gpu": """\
from scheduler import list_hosts
""",
    "scheduler_routes": """\
from scheduler import process_queue_binpack
""",
    "artifacts": """\
from pydantic import BaseModel, Field
from scheduler import list_jobs
from artifacts import get_artifact_manager
from routes._deps import _get_current_user
""",
    "pricing": """\
from pydantic import BaseModel, Field
from reputation import estimate_job_cost, GPU_REFERENCE_PRICING_CAD
from routes._deps import _get_current_user
""",
    "infra": """\
from pydantic import BaseModel, Field
from scheduler import (
    configure_alerts, ALERT_CONFIG,
    build_and_push, list_builds, generate_dockerfile,
)
from routes._deps import _get_current_user
""",
}

# Tags for each module's APIRouter
MODULE_TAGS = {
    "hosts": ["Hosts"], "instances": ["Instances"], "billing": ["Billing"],
    "health": ["Infrastructure"], "ssh": ["SSH Keys"], "auth": ["Auth"],
    "mfa": ["Auth – MFA"], "teams": ["Teams"], "marketplace": ["Marketplace"],
    "slurm": ["Agent"], "jurisdiction": ["Jurisdiction"],
    "autoscale": ["Autoscale"], "agent": ["Agent"], "spot": ["Spot Pricing"],
    "events": ["Events"], "verification": ["Verification"],
    "reputation": ["Reputation"], "compliance": ["Compliance"],
    "privacy": ["Privacy"], "transparency": ["Transparency"],
    "sla": ["SLA"], "providers": ["Providers"],
    "notifications": ["Notifications"], "admin": ["Admin"],
    "chat": ["Chat"], "inference": ["Inference"], "volumes": ["Volumes"],
    "cloudburst": ["Cloud Burst"], "gpu": ["GPU"],
    "scheduler_routes": ["Jobs"], "artifacts": ["Artifacts"],
    "pricing": ["Billing"], "infra": ["Infrastructure"],
}


# ── Parsing ───────────────────────────────────────────────────────────

def extract_routes(lines):
    """Parse api.py and extract route function line ranges + metadata."""
    route_pat = re.compile(
        r'^@app\.(get|post|put|patch|delete|websocket)\(\s*["\']([^"\']+)["\']'
    )
    tag_pat = re.compile(r'tags=\["([^"]+)"\]')
    routes = []
    i = 0
    while i < len(lines):
        m = route_pat.match(lines[i])
        if not m:
            i += 1
            continue
        method, path = m.group(1), m.group(2)
        tm = tag_pat.search(lines[i])
        tag = tm.group(1) if tm else ""
        dec_start = i

        # Find function def
        func_start = i + 1
        while func_start < len(lines) and not lines[func_start].startswith(("def ", "async def ")):
            if route_pat.match(lines[func_start]):
                break
            func_start += 1
        if func_start >= len(lines):
            break

        func_name_m = re.match(r'(?:async\s+)?def\s+(\w+)', lines[func_start])
        func_name = func_name_m.group(1) if func_name_m else "unknown"

        # Find end of function
        func_end = func_start + 1
        while func_end < len(lines):
            l = lines[func_end]
            if l.strip() == "" or l.startswith("#"):
                func_end += 1
                continue
            if l[0] in (' ', '\t'):
                func_end += 1
                continue
            break
        while func_end > func_start and lines[func_end - 1].strip() == "":
            func_end -= 1

        routes.append({
            "start": dec_start, "end": func_end,
            "method": method, "path": path, "tag": tag,
            "func_name": func_name,
            "module": classify_route(path, tag),
        })
        i = func_end
    return routes


def find_pydantic_models(lines):
    """Find all Pydantic model class definitions and their line ranges."""
    models = []
    pat = re.compile(r'^class\s+(\w+)\(BaseModel\):')
    i = 0
    while i < len(lines):
        m = pat.match(lines[i])
        if not m:
            i += 1
            continue
        cs = i
        ce = i + 1
        while ce < len(lines):
            l = lines[ce]
            if l.strip() == "" or l.startswith("#"):
                ce += 1; continue
            if l[0] in (' ', '\t'):
                ce += 1; continue
            break
        while ce > cs and lines[ce - 1].strip() == "":
            ce -= 1
        models.append({"name": m.group(1), "start": cs, "end": ce})
        i = ce
    return models


def find_helpers(lines, module):
    """Find module-specific helper functions/state in api.py."""
    patterns = MODULE_HELPERS.get(module, [])
    helpers = []
    for pat_str in patterns:
        pat = re.compile(pat_str)
        for i, line in enumerate(lines):
            if not pat.match(line):
                continue
            end = i + 1
            while end < len(lines):
                l = lines[end]
                if l.strip() == "" or l.startswith("#"):
                    end += 1; continue
                if l[0] in (' ', '\t'):
                    end += 1; continue
                break
            while end > i and lines[end - 1].strip() == "":
                end -= 1
            helpers.append({"start": i, "end": end})
            break
    return helpers


def associate_models(models, routes, lines):
    """Map each Pydantic model to the module that uses it."""
    result = {}
    for model in models:
        name = model["name"]
        using = set()
        for route in routes:
            code = "\n".join(lines[route["start"]:route["end"]])
            if name in code:
                using.add(route["module"])
        if len(using) == 1:
            result[name] = list(using)[0]
        elif len(using) > 1:
            result[name] = sorted(using)[0]
        else:
            for route in routes:
                if route["start"] > model["end"]:
                    result[name] = route["module"]
                    break
    return result


def generate_module_file(module, mod_routes, mod_models, mod_helpers, lines):
    """Generate the code for a route module file."""
    tags = MODULE_TAGS.get(module, [module.title()])
    tag_str = json.dumps(tags) if len(tags) > 1 else f'["{tags[0]}"]'

    parts = []

    # Header
    parts.append(f'"""Routes: {module}."""')
    parts.append("")

    # Imports
    parts.append(COMMON_IMPORTS.rstrip())
    extra = MODULE_IMPORTS.get(module, "")
    if extra:
        parts.append(extra.rstrip())
    parts.append("")

    # Router
    parts.append(f"router = APIRouter()")
    parts.append("")

    # Models
    for model in sorted(mod_models, key=lambda m: m["start"]):
        parts.append("")
        parts.append("\n".join(lines[model["start"]:model["end"]]))

    # Helpers
    for helper in sorted(mod_helpers, key=lambda h: h["start"]):
        parts.append("")
        parts.append("")
        parts.append("\n".join(lines[helper["start"]:helper["end"]]))

    # Routes
    for route in sorted(mod_routes, key=lambda r: r["start"]):
        parts.append("")
        parts.append("")
        code = "\n".join(lines[route["start"]:route["end"]])
        # Replace @app.X with @router.X
        code = re.sub(r'^@app\.(get|post|put|patch|delete|websocket)\(', r'@router.\1(', code, flags=re.MULTILINE)
        parts.append(code)

    parts.append("")
    return "\n".join(parts)


import json as json  # for tag_str


def main():
    write = "--write" in sys.argv

    print("Parsing api.py...")
    lines = API_PY.read_text().splitlines()
    routes = extract_routes(lines)
    models = find_pydantic_models(lines)
    model_to_module = associate_models(models, routes, lines)

    # Group by module
    modules = defaultdict(list)
    for route in routes:
        modules[route["module"]].append(route)
    module_models = defaultdict(list)
    for model in models:
        mod = model_to_module.get(model["name"], "misc")
        module_models[mod].append(model)

    print(f"\nFound {len(routes)} routes across {len(modules)} modules:")
    for mod, mod_routes in sorted(modules.items()):
        mm = module_models.get(mod, [])
        total = sum(r["end"] - r["start"] for r in mod_routes) + sum(m["end"] - m["start"] for m in mm)
        print(f"  {mod:20s}: {len(mod_routes):3d} routes, {len(mm):2d} models, ~{total:4d} lines")

    if not write:
        # Print model mapping and exit
        print(f"\nPydantic models ({len(models)}):")
        for model in models:
            mod = model_to_module.get(model["name"], "?")
            print(f"  {model['name']:40s} → {mod} (L{model['start']+1}-{model['end']})")
        print("\nRun with --write to generate route files.")
        return

    # ── Generate route module files ──────────────────────────────────
    ROUTES_DIR.mkdir(exist_ok=True)

    for mod, mod_routes in sorted(modules.items()):
        mm = module_models.get(mod, [])
        helpers = find_helpers(lines, mod)
        code = generate_module_file(mod, mod_routes, mm, helpers, lines)

        out_path = ROUTES_DIR / f"{mod}.py"
        out_path.write_text(code)
        print(f"  wrote {out_path.relative_to(ROOT)} ({len(mod_routes)} routes, {len(mm)} models)")

    # ── Generate __init__.py with all router imports ─────────────────
    init_lines = ['"""Route modules for Xcelsior API."""', ""]
    for mod in sorted(modules.keys()):
        init_lines.append(f"from routes.{mod} import router as {mod}_router")
    init_lines.append("")
    init_lines.append("ALL_ROUTERS = [")
    for mod in sorted(modules.keys()):
        init_lines.append(f"    {mod}_router,")
    init_lines.append("]")
    init_lines.append("")

    init_path = ROUTES_DIR / "__init__.py"
    init_path.write_text("\n".join(init_lines))
    print(f"  wrote {init_path.relative_to(ROOT)}")

    print("\nDone! Next: create new api.py skeleton that includes all routers.")


if __name__ == "__main__":
    main()
