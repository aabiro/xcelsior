#!/usr/bin/env python3
"""Split api.py into domain route modules using AST-based extraction.

This script:
1. Parses api.py with AST to find all route functions with exact line ranges
2. Groups them by tag → module mapping
3. Extracts complete function bodies (decorators + function + all code)
4. Also extracts non-route helpers and Pydantic models that belong to each domain
5. Generates route files with correct imports
"""

import ast
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

API_FILE = "api.py"
ROUTES_DIR = Path("routes")

# ── Tag → Module mapping ─────────────────────────────────────────────

TAG_TO_MODULE = {
    "Hosts": "hosts",
    "Instances": "instances",
    "Jobs": "instances",  # single job-related route
    "Billing": "billing",
    "Marketplace": "marketplace",
    "Marketplace v2": "marketplace",
    "Spot Pricing": "spot",
    "Reputation": "reputation",
    "Verification": "verification",
    "SLA": "sla",
    "Providers": "providers",
    "Artifacts": "artifacts",
    "Jurisdiction": "jurisdiction",
    "Compliance": "compliance",
    "Privacy": "privacy",
    "Transparency": "transparency",
    "Telemetry": "agent",  # telemetry is agent-facing
    "Agent": "agent",
    "Autoscale": "autoscale",
    "Events": "events",
    "Infrastructure": "health",
    "Auth": "auth",
    "Auth – MFA": "mfa",  # note the em-dash
    "Auth \u2013 MFA": "mfa",  # en-dash variant
    "Teams": "teams",
    "Notifications": "notifications",
    "Chat": "chat",
    "AI Assistant": "chat",  # combine with chat
    "Inference": "inference",
    "Inference v2": "inference",
    "SSH Keys": "ssh",
    "GPU": "gpu",
    "Cloud Burst": "cloudburst",
    "Volumes": "volumes",
    "Admin": "admin",
}

# Routes to skip (internal endpoints that stay in api.py as middleware/handlers)
SKIP_FUNCTIONS = {
    "csrf_origin_check",
    "http_exception_handler",
    "request_validation_exception_handler",
    "lifespan",
}


def parse_api():
    """Parse api.py and return source + AST."""
    with open(API_FILE) as f:
        source = f.read()
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    return source, lines, tree


def find_route_functions(source, tree):
    """Find all @app.xxx decorated route functions with their tags and line ranges."""
    routes = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            dec_src = ast.get_source_segment(source, dec) or ""
            if "app." not in dec_src:
                continue
            
            # Extract tag
            tag_match = re.search(r'tags=\["([^"]+)"\]', dec_src)
            tag = tag_match.group(1) if tag_match else "Unknown"
            
            # Extract path
            path_match = re.search(r'"(/[^"]*)"', dec_src)
            path = path_match.group(1) if path_match else "?"
            
            # Extract HTTP method
            method_match = re.search(r'app\.(get|post|put|patch|delete|websocket)\(', dec_src)
            method = method_match.group(1).upper() if method_match else "?"
            
            # Line range: from first decorator to end of function
            start = min(d.lineno for d in node.decorator_list)
            end = node.end_lineno
            
            routes.append({
                "name": node.name,
                "tag": tag,
                "path": path,
                "method": method,
                "start_line": start,
                "end_line": end,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
            })
            break
    
    return sorted(routes, key=lambda r: r["start_line"])


def find_pydantic_models(source, tree):
    """Find all Pydantic model classes in api.py."""
    models = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == "BaseModel":
                    models.append({
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                    })
                    break
    return sorted(models, key=lambda m: m["start_line"])


def find_helper_functions(source, lines, tree, route_ranges):
    """Find non-route helper functions that are between route functions.
    
    These are functions NOT decorated with @app.xxx that are defined
    at module level and used by route handlers.
    """
    # Build a set of lines occupied by routes
    route_lines = set()
    for r in route_ranges:
        for ln in range(r["start_line"], r["end_line"] + 1):
            route_lines.add(ln)
    
    helpers = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.lineno in route_lines:
            continue
        # Only module-level functions (col_offset == 0)
        if node.col_offset != 0:
            continue
        # Skip known framework functions
        if node.name in SKIP_FUNCTIONS:
            continue
        if node.name.startswith("_bg_") or node.name in ("_start_background_tasks", "_stop_background_tasks"):
            continue
        
        start = min(d.lineno for d in node.decorator_list) if node.decorator_list else node.lineno
        helpers.append({
            "name": node.name,
            "start_line": start,
            "end_line": node.end_lineno,
        })
    
    return sorted(helpers, key=lambda h: h["start_line"])


def assign_models_to_modules(models, routes, lines):
    """Assign Pydantic models to modules based on proximity to routes."""
    # For each model, find the nearest following route and assign to the same module
    assignments = {}
    for model in models:
        best_module = "auth"  # default
        best_dist = float("inf")
        for route in routes:
            if route["name"] in SKIP_FUNCTIONS:
                continue
            module = TAG_TO_MODULE.get(route["tag"])
            if not module:
                continue
            dist = abs(route["start_line"] - model["end_line"])
            if dist < best_dist:
                best_dist = dist
                best_module = module
        assignments[model["name"]] = best_module
    return assignments


def assign_helpers_to_modules(helpers, routes, source, lines):
    """Assign helper functions to modules based on who calls them."""
    # Build a map of helper name → which route functions reference it
    assignments = {}
    for helper in helpers:
        caller_modules = set()
        helper_name = helper["name"]
        
        for route in routes:
            if route["name"] in SKIP_FUNCTIONS:
                continue
            module = TAG_TO_MODULE.get(route["tag"])
            if not module:
                continue
            # Check if the route's code references this helper
            route_code = "".join(lines[route["start_line"]-1:route["end_line"]])
            if helper_name in route_code:
                caller_modules.add(module)
        
        if len(caller_modules) == 1:
            assignments[helper_name] = caller_modules.pop()
        elif len(caller_modules) > 1:
            # Used by multiple modules — keep in _deps or assign to the nearest route's module
            # For now, assign to nearest
            best_module = None
            best_dist = float("inf")
            for route in routes:
                if route["name"] in SKIP_FUNCTIONS:
                    continue
                module = TAG_TO_MODULE.get(route["tag"])
                if not module:
                    continue
                dist = abs(route["start_line"] - helper["end_line"])
                if dist < best_dist:
                    best_dist = dist
                    best_module = module
            assignments[helper_name] = best_module
        else:
            # No callers found among routes — assign to nearest
            best_module = None
            best_dist = float("inf")
            for route in routes:
                if route["name"] in SKIP_FUNCTIONS:
                    continue
                module = TAG_TO_MODULE.get(route["tag"])
                if not module:
                    continue
                dist = abs(route["start_line"] - helper["end_line"])
                if dist < best_dist:
                    best_dist = dist
                    best_module = module
            assignments[helper_name] = best_module or "health"
    
    return assignments


def extract_code_block(lines, start_line, end_line):
    """Extract lines from the source (1-indexed)."""
    return "".join(lines[start_line - 1:end_line])


def collect_module_items(routes, models, helpers, model_assignments, helper_assignments, lines):
    """Collect all code items grouped by target module."""
    modules = defaultdict(lambda: {"routes": [], "models": [], "helpers": [], "code_blocks": []})
    
    for route in routes:
        if route["name"] in SKIP_FUNCTIONS:
            continue
        module = TAG_TO_MODULE.get(route["tag"])
        if not module:
            print(f"  WARNING: No module for tag '{route['tag']}' (func: {route['name']})")
            continue
        code = extract_code_block(lines, route["start_line"], route["end_line"])
        modules[module]["routes"].append(route)
        modules[module]["code_blocks"].append({
            "type": "route",
            "name": route["name"],
            "start_line": route["start_line"],
            "code": code,
        })
    
    for model in models:
        module = model_assignments.get(model["name"], "auth")
        code = extract_code_block(lines, model["start_line"], model["end_line"])
        modules[module]["models"].append(model)
        modules[module]["code_blocks"].append({
            "type": "model",
            "name": model["name"],
            "start_line": model["start_line"],
            "code": code,
        })
    
    for helper in helpers:
        module = helper_assignments.get(helper["name"])
        if not module:
            continue
        code = extract_code_block(lines, helper["start_line"], helper["end_line"])
        modules[module]["helpers"].append(helper)
        modules[module]["code_blocks"].append({
            "type": "helper",
            "name": helper["name"],
            "start_line": helper["start_line"],
            "code": code,
        })
    
    return modules


def scan_imports_needed(code_text):
    """Scan code text to determine which imports are needed."""
    needs = set()
    
    # FastAPI / Starlette
    if "Request" in code_text:
        needs.add("Request")
    if "HTTPException" in code_text:
        needs.add("HTTPException")
    if "WebSocket" in code_text:
        needs.add("WebSocket")
    if "WebSocketDisconnect" in code_text:
        needs.add("WebSocketDisconnect")
    if "JSONResponse" in code_text:
        needs.add("JSONResponse")
    if "HTMLResponse" in code_text:
        needs.add("HTMLResponse")
    if "RedirectResponse" in code_text:
        needs.add("RedirectResponse")
    if "StreamingResponse" in code_text:
        needs.add("StreamingResponse")
    if "PlainTextResponse" in code_text:
        needs.add("PlainTextResponse")
    
    # Pydantic
    if "BaseModel" in code_text:
        needs.add("BaseModel")
    if re.search(r'\bField\b', code_text):
        needs.add("Field")
    
    # _deps helpers
    deps_funcs = [
        "_get_current_user", "_require_auth", "_require_admin",
        "_require_provider_or_admin", "_require_write_access",
        "_is_platform_admin", "_merge_auth_user",
        "broadcast_sse", "_sse_subscribers", "_sse_lock",
        "_check_auth_rate_limit", "_hash_password",
        "_create_session", "_set_auth_cookie", "_clear_auth_cookie",
        "_users_db", "_sessions", "_api_keys", "_user_lock",
        "_USE_PERSISTENT_AUTH", "AUTH_REQUIRED", "XCELSIOR_ENV",
        "_AUTH_COOKIE_NAME", "SESSION_EXPIRY", "MAX_SESSION_LIFETIME",
        "VALID_ACCOUNT_ROLES", "_RATE_BUCKETS", "_AUTH_RATE_BUCKETS",
        "_get_real_client_ip", "_admin_flag",
        "log",
    ]
    found_deps = set()
    for fn in deps_funcs:
        if fn in code_text:
            found_deps.add(fn)
    needs.add(("deps", frozenset(found_deps)))
    
    # Standard library
    stdlib = {
        "asyncio": r'\basyncio\.',
        "json": r'\bjson\.',
        "os": r'\bos\.',
        "time": r'\btime\.',
        "secrets": r'\bsecrets\.',
        "uuid": r'\buuid\.',
        "hashlib": r'\bhashlib\.',
        "hmac": r'\bhmac\.',
        "re": r'\bre\.',
        "base64": r'\bbase64\.',
        "urllib.parse": r'\burllib\.parse\.',
        "threading": r'\bthreading\.',
        "subprocess": r'\bsubprocess\.',
        "shutil": r'\bshutil\.',
        "io": r'\bio\.',
    }
    found_stdlib = set()
    for mod, pattern in stdlib.items():
        if re.search(pattern, code_text):
            found_stdlib.add(mod)
    needs.add(("stdlib", frozenset(found_stdlib)))
    
    # collections
    if "defaultdict" in code_text:
        needs.add("defaultdict")
    if "deque" in code_text:
        needs.add("deque")
    
    # pathlib
    if "Path(" in code_text:
        needs.add("Path")
    
    # scheduler imports
    sched_funcs = [
        "register_host", "update_host", "remove_host", "get_host",
        "list_hosts", "list_jobs", "submit_job", "get_job", "update_job",
        "cancel_job", "requeue_job", "process_queue", "failover_job",
        "get_metrics_snapshot", "storage_healthcheck", "list_builds",
        "build_image", "generate_dockerfile", "get_total_revenue",
        "load_billing", "bill_job", "bill_all_running", "get_compute_score",
        "list_compute_scores", "API_TOKEN", "ALERT_CONFIG", "configure_alerts",
        "get_current_spot_prices", "update_spot_prices", "spot_submit",
        "preemption_cycle", "add_to_pool", "remove_from_pool", "get_pool",
        "autoscale_up", "autoscale_down", "autoscale_cycle",
        "AUTOSCALE_ENABLED", "save_hosts", "check_hosts",
        "list_popular_images", "allocate_binpack", "process_queue_binpack",
        "log",
    ]
    found_sched = set()
    for fn in sched_funcs:
        if re.search(rf'\b{fn}\b', code_text):
            found_sched.add(fn)
    if found_sched:
        needs.add(("scheduler", frozenset(found_sched)))
    
    # db imports
    db_funcs = [
        "UserStore", "NotificationStore", "MfaStore",
        "emit_event", "start_pg_listen",
    ]
    found_db = set()
    for fn in db_funcs:
        if fn in code_text:
            found_db.add(fn)
    if found_db:
        needs.add(("db", frozenset(found_db)))
    
    # Domain engine imports
    engine_patterns = {
        "events": [
            ("events", ["get_event_store", "get_state_machine", "JobState", "EventType", "Event"]),
        ],
        "verification": [
            ("verification", ["get_verification_engine"]),
        ],
        "jurisdiction": [
            ("jurisdiction", ["TrustTier", "JurisdictionConstraint", "generate_residency_trace",
                             "get_jurisdiction_engine", "TRUST_TIERS", "RESIDENCY_REQUIREMENTS",
                             "PROVINCE_DATA_RESIDENCY"]),
        ],
        "billing": [
            ("billing", ["get_billing_engine", "get_tax_rate_for_province", "PROVINCE_TAX_RATES"]),
        ],
        "reputation": [
            ("reputation", ["get_reputation_engine", "ReputationTier", "VerificationType",
                           "TIER_THRESHOLDS", "VERIFICATION_TYPE_MAP", "calculate_composite_score",
                           "get_reputation_tiers"]),
        ],
        "artifacts": [
            ("artifacts", ["get_artifact_manager"]),
        ],
        "privacy": [
            ("privacy", ["get_lifecycle_manager", "PrivacyConfig", "RETENTION_POLICIES",
                        "redact_pii", "get_privacy_engine", "get_crypto_shredder",
                        "get_consent_manager", "execute_right_to_erasure"]),
        ],
        "sla": [
            ("sla", ["get_sla_engine", "SLATier", "SLA_TARGETS"]),
        ],
        "stripe_connect": [
            ("stripe_connect", ["get_stripe_manager"]),
        ],
        "chat_mod": [
            ("chat", ["build_system_prompt", "stream_chat_response", "CHAT_API_KEY",
                      "CHAT_PROVIDER", "CHAT_MODEL", "CHAT_MAX_TOKENS",
                      "get_chat_engine", "list_suggestions"]),
        ],
        "inference_store": [
            ("inference_store", ["store_inference_job", "get_inference_job",
                                "list_inference_jobs", "update_inference_job"]),
        ],
        "marketplace_mod": [
            ("marketplace", ["get_marketplace_engine"]),
        ],
        "inference_mod": [
            ("inference", ["get_inference_engine"]),
        ],
        "volumes_mod": [
            ("volumes", ["get_volume_engine"]),
        ],
        "cloudburst_mod": [
            ("cloudburst", ["get_burst_engine"]),
        ],
        "security_mod": [
            ("security", ["admit_node", "check_node_versions"]),
        ],
        "slurm_mod": [
            ("slurm_adapter", ["submit_slurm_job", "poll_slurm_status", "cancel_slurm_job",
                              "list_slurm_profiles", "get_slurm_instance_map", "SlurmProfile"]),
        ],
        "ai_assistant_mod": [
            ("ai_assistant", ["build_ai_system_prompt", "parse_tool_calls_from_text",
                             "execute_tool_call", "get_ai_provider", "AI_TOOLS",
                             "build_ai_context", "format_ai_response",
                             "get_ai_conversation_store"]),
        ],
        "bitcoin_mod": [
            ("bitcoin", ["get_btc_manager"]),
        ],
        "nvml_mod": [
            ("nvml_telemetry", ["parse_telemetry_payload"]),
        ],
    }
    
    for key, entries in engine_patterns.items():
        for mod_name, symbols in entries:
            found = set()
            for sym in symbols:
                if re.search(rf'\b{sym}\b', code_text):
                    found.add(sym)
            if found:
                needs.add(("engine", mod_name, frozenset(found)))
    
    return needs


def generate_imports(needs):
    """Convert a needs set into import lines."""
    import_lines = []
    
    # Standard library
    for item in needs:
        if isinstance(item, tuple) and item[0] == "stdlib":
            for mod in sorted(item[1]):
                if mod == "urllib.parse":
                    import_lines.append("import urllib.parse")
                elif mod == "threading":
                    import_lines.append("import threading as _threading")
                else:
                    import_lines.append(f"import {mod}")
    
    # collections
    collections_items = []
    if "defaultdict" in needs:
        collections_items.append("defaultdict")
    if "deque" in needs:
        collections_items.append("deque")
    if collections_items:
        import_lines.append(f"from collections import {', '.join(sorted(collections_items))}")
    
    if "Path" in needs:
        import_lines.append("from pathlib import Path")
    
    if import_lines:
        import_lines.append("")
    
    # FastAPI
    fastapi_items = []
    for item in ["Request", "HTTPException", "WebSocket", "WebSocketDisconnect"]:
        if item in needs:
            fastapi_items.append(item)
    if fastapi_items:
        import_lines.append(f"from fastapi import APIRouter, {', '.join(sorted(fastapi_items))}")
    else:
        import_lines.append("from fastapi import APIRouter")
    
    # Response types
    response_types = []
    for rt in ["HTMLResponse", "JSONResponse", "RedirectResponse", "StreamingResponse", "PlainTextResponse"]:
        if rt in needs:
            response_types.append(rt)
    if response_types:
        import_lines.append(f"from fastapi.responses import {', '.join(sorted(response_types))}")
    
    # Pydantic
    pydantic_items = []
    if "BaseModel" in needs:
        pydantic_items.append("BaseModel")
    if "Field" in needs:
        pydantic_items.append("Field")
    if pydantic_items:
        import_lines.append(f"from pydantic import {', '.join(sorted(pydantic_items))}")
    
    import_lines.append("")
    
    # _deps
    for item in needs:
        if isinstance(item, tuple) and item[0] == "deps" and item[1]:
            deps_list = sorted(item[1])
            # Group into lines of ~80 chars
            import_lines.append(f"from routes._deps import (")
            for dep in deps_list:
                import_lines.append(f"    {dep},")
            import_lines.append(")")
    
    # scheduler
    for item in needs:
        if isinstance(item, tuple) and item[0] == "scheduler":
            sched_list = sorted(item[1])
            import_lines.append(f"from scheduler import (")
            for s in sched_list:
                import_lines.append(f"    {s},")
            import_lines.append(")")
    
    # db
    for item in needs:
        if isinstance(item, tuple) and item[0] == "db":
            db_list = sorted(item[1])
            import_lines.append(f"from db import {', '.join(db_list)}")
    
    # Engine imports
    for item in needs:
        if isinstance(item, tuple) and item[0] == "engine":
            mod_name = item[1]
            symbols = sorted(item[2])
            import_lines.append(f"from {mod_name} import {', '.join(symbols)}")
    
    return "\n".join(import_lines)


def generate_route_file(module_name, module_data, lines):
    """Generate a complete route module file."""
    # Sort code blocks by original line number
    blocks = sorted(module_data["code_blocks"], key=lambda b: b["start_line"])
    
    # Combine all code
    all_code = "\n\n".join(b["code"] for b in blocks)
    
    # Replace @app.xxx with @router.xxx
    all_code = re.sub(r'@app\.(get|post|put|patch|delete|websocket)\(', r'@router.\1(', all_code)
    
    # Scan for needed imports
    needs = scan_imports_needed(all_code)
    imports = generate_imports(needs)
    
    # Build the file
    content = f'"""Routes: {module_name}."""\n\n'
    content += imports
    content += "\n\nrouter = APIRouter()\n\n"
    
    # Add separator comments for readability
    for block in blocks:
        if block["type"] == "model":
            content += f"\n# ── Model: {block['name']} ──\n\n"
        elif block["type"] == "helper":
            content += f"\n# ── Helper: {block['name']} ──\n\n"
        content += block["code"]
        if not block["code"].endswith("\n"):
            content += "\n"
        content += "\n"
    
    return content


def main():
    print("=" * 60)
    print("Xcelsior api.py → routes/ splitter")
    print("=" * 60)
    
    source, lines, tree = parse_api()
    print(f"\nParsed api.py: {len(lines)} lines")
    
    # 1. Find routes
    routes = find_route_functions(source, tree)
    print(f"Found {len(routes)} route functions")
    
    # 2. Find Pydantic models
    models = find_pydantic_models(source, tree)
    print(f"Found {len(models)} Pydantic models")
    
    # 3. Find helpers
    helpers = find_helper_functions(source, lines, tree, routes)
    print(f"Found {len(helpers)} helper functions")
    
    # 4. Assign models and helpers to modules
    model_assignments = assign_models_to_modules(models, routes, lines)
    helper_assignments = assign_helpers_to_modules(helpers, routes, source, lines)
    
    # 5. Collect items by module
    modules = collect_module_items(routes, models, helpers, model_assignments, helper_assignments, lines)
    
    print(f"\nModules to generate: {len(modules)}")
    for mod_name in sorted(modules.keys()):
        data = modules[mod_name]
        r_count = len(data["routes"])
        m_count = len(data["models"])
        h_count = len(data["helpers"])
        print(f"  {mod_name}: {r_count} routes, {m_count} models, {h_count} helpers")
    
    # 6. Generate route files
    ROUTES_DIR.mkdir(exist_ok=True)
    
    for mod_name, mod_data in sorted(modules.items()):
        filepath = ROUTES_DIR / f"{mod_name}.py"
        content = generate_route_file(mod_name, mod_data, lines)
        filepath.write_text(content)
        line_count = content.count("\n")
        print(f"  Wrote {filepath} ({line_count} lines)")
    
    # 7. Generate __init__.py
    init_content = '"""Route modules for Xcelsior API."""\n\n'
    all_mod_names = sorted(modules.keys())
    for mod_name in all_mod_names:
        init_content += f"from routes.{mod_name} import router as {mod_name}_router\n"
    init_content += "\nALL_ROUTERS = [\n"
    for mod_name in all_mod_names:
        init_content += f"    {mod_name}_router,\n"
    init_content += "]\n"
    
    (ROUTES_DIR / "__init__.py").write_text(init_content)
    print(f"\n  Wrote routes/__init__.py")
    
    print("\n✅ Done!")
    print(f"\nNext steps:")
    print(f"  1. Review generated files")
    print(f"  2. Create new slim api.py that includes all routers")
    print(f"  3. Update test imports")
    print(f"  4. Run tests")


if __name__ == "__main__":
    main()
