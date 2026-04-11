#!/usr/bin/env python3
"""Generate the public/client-facing OpenAPI spec.

The FastAPI app exposes many internal worker, admin, maintenance, and callback
endpoints that should not appear in customer-facing docs or the public SDK.
This generator keeps a small, curated operation allowlist and intersects it
with the Fern override file so only explicitly published routes remain.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FERN_DIR = ROOT / "fern"
OVERRIDES_PATH = FERN_DIR / "openapi-overrides.yml"
OUTPUT_PATH = FERN_DIR / "openapi.json"
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
CLIENT_OPERATION_ALLOWLIST = {
    ("/.well-known/oauth-authorization-server", "get"),
    ("/oauth/authorize", "get"),
    ("/oauth/token", "post"),
    ("/oauth/device/authorize", "post"),
    ("/api/auth/register", "post"),
    ("/api/auth/login", "post"),
    ("/api/auth/me", "get"),
    ("/api/auth/logout", "post"),
    ("/api/artifacts/upload", "post"),
    ("/api/artifacts/download", "post"),
    ("/api/artifacts/{job_id}", "get"),
    ("/api/artifacts/{job_id}/expiry", "get"),
    ("/api/billing/payment-intent", "post"),
    ("/api/billing/paypal/enabled", "get"),
    ("/api/billing/paypal/create-order", "post"),
    ("/api/billing/paypal/capture-order", "post"),
    ("/api/billing/wallet/{customer_id}", "get"),
    ("/api/billing/wallet/{customer_id}/deposit", "post"),
    ("/api/billing/wallet/{customer_id}/history", "get"),
    ("/api/billing/wallet/{customer_id}/depletion", "get"),
    ("/api/billing/usage/{customer_id}", "get"),
    ("/api/billing/invoices/{customer_id}", "get"),
    ("/api/billing/invoice/{customer_id}", "get"),
    ("/api/billing/invoice/{customer_id}/download", "get"),
    ("/api/pricing/estimate", "post"),
    ("/api/pricing/reference", "get"),
    ("/api/v2/billing/auto-topup", "get"),
    ("/api/v2/billing/auto-topup", "post"),
    ("/instance", "post"),
    ("/instances", "get"),
    ("/instance/{job_id}", "get"),
    ("/instances/{job_id}/cancel", "post"),
    ("/instances/{job_id}/logs", "get"),
    ("/tiers", "get"),
    ("/marketplace/search", "get"),
    ("/marketplace", "get"),
    ("/marketplace/stats", "get"),
    ("/api/v2/marketplace/search", "post"),
    ("/api/v2/marketplace/spot-prices", "get"),
    ("/api/v2/marketplace/spot-prices/{gpu_model}/history", "get"),
    ("/api/v2/marketplace/stats", "get"),
    ("/api/v2/marketplace/reservations", "post"),
    ("/api/v2/marketplace/reservations/{reservation_id}", "delete"),
    ("/api/v2/volumes", "get"),
    ("/api/v2/volumes", "post"),
    ("/api/v2/volumes/{volume_id}", "get"),
    ("/api/v2/volumes/{volume_id}", "delete"),
    ("/api/v2/volumes/{volume_id}/attach", "post"),
    ("/api/v2/volumes/{volume_id}/detach", "post"),
    ("/v1/inference", "post"),
    ("/v1/inference/async", "post"),
    ("/v1/inference/{job_id}", "get"),
    ("/v1/chat/completions", "post"),
    ("/api/v2/inference/endpoints", "get"),
    ("/api/v2/inference/endpoints", "post"),
    ("/api/v2/inference/endpoints/{endpoint_id}", "get"),
    ("/api/v2/inference/endpoints/{endpoint_id}", "delete"),
    ("/api/v2/inference/endpoints/{endpoint_id}/health", "get"),
    ("/api/v2/inference/endpoints/{endpoint_id}/usage", "get"),
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import app


def _load_allowlist() -> set[tuple[str, str]]:
    overrides = yaml.safe_load(OVERRIDES_PATH.read_text(encoding="utf-8")) or {}
    allowlist: set[tuple[str, str]] = set()
    for path, methods in (overrides.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for method, meta in methods.items():
            if method.lower() not in HTTP_METHODS or not isinstance(meta, dict):
                continue
            if meta.get("x-fern-ignore"):
                continue
            allowlist.add((path, method.lower()))
    return allowlist


def _walk_refs(node, refs: set[tuple[str, str]]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/"):
            _, _, section, name = ref.split("/", 3)
            refs.add((section, name))
        for value in node.values():
            _walk_refs(value, refs)
    elif isinstance(node, list):
        for item in node:
            _walk_refs(item, refs)


def _prune_components(spec: dict) -> None:
    components = copy.deepcopy(spec.get("components") or {})
    if not components:
        return

    refs: set[tuple[str, str]] = set()
    _walk_refs(spec.get("paths", {}), refs)

    kept: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()
    while refs:
        section, name = refs.pop()
        if (section, name) in seen:
            continue
        seen.add((section, name))
        section_items = components.get(section) or {}
        if name not in section_items:
            continue
        kept.setdefault(section, {})[name] = section_items[name]
        _walk_refs(section_items[name], refs)

    # Keep security schemes referenced from global or operation-level security blocks.
    security_names: set[str] = set()
    for security_block in spec.get("security") or []:
        if isinstance(security_block, dict):
            security_names.update(security_block.keys())
    for path_item in (spec.get("paths") or {}).values():
        if not isinstance(path_item, dict):
            continue
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            for security_block in operation.get("security") or []:
                if isinstance(security_block, dict):
                    security_names.update(security_block.keys())
    if security_names and (components.get("securitySchemes") or {}):
        kept.setdefault("securitySchemes", {})
        for name in security_names:
            scheme = components["securitySchemes"].get(name)
            if scheme is not None:
                kept["securitySchemes"][name] = scheme

    if kept:
        spec["components"] = kept
    else:
        spec.pop("components", None)


def build_public_spec() -> dict:
    full_spec = copy.deepcopy(app.openapi())
    allowlist = _load_allowlist()

    filtered_paths: dict[str, dict] = {}
    for path, path_item in (full_spec.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        kept_ops: dict[str, dict] = {}
        for method, operation in path_item.items():
            method_lc = method.lower()
            if (
                method_lc not in HTTP_METHODS
                or (path, method_lc) not in allowlist
                or (path, method_lc) not in CLIENT_OPERATION_ALLOWLIST
            ):
                continue
            kept_ops[method] = operation
        if kept_ops:
            filtered_paths[path] = kept_ops

    full_spec["paths"] = filtered_paths
    full_spec.pop("webhooks", None)

    used_tags = {
        tag
        for path_item in filtered_paths.values()
        for operation in path_item.values()
        for tag in (operation.get("tags") or [])
    }
    if used_tags:
        declared = {
            tag.get("name"): tag
            for tag in (full_spec.get("tags") or [])
            if isinstance(tag, dict) and tag.get("name")
        }
        full_spec["tags"] = [
            declared.get(tag_name, {"name": tag_name})
            for tag_name in sorted(used_tags)
        ]

    info = full_spec.setdefault("info", {})
    description = str(info.get("description", "")).strip()
    if description:
        info["description"] = (
            f"{description}\n\n"
            "This published OpenAPI document contains the public developer surface only. "
            "Account settings, internal worker, admin, maintenance, and callback endpoints are intentionally omitted."
        )

    _prune_components(full_spec)
    return full_spec


def main() -> None:
    spec = build_public_spec()
    OUTPUT_PATH.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    operation_count = sum(len(path_item) for path_item in spec.get("paths", {}).values())
    print(f"Wrote {OUTPUT_PATH} with {len(spec.get('paths', {}))} paths / {operation_count} operations")


if __name__ == "__main__":
    main()
