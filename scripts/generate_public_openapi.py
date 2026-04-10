#!/usr/bin/env python3
"""Generate the public/client-facing OpenAPI spec consumed by Fern.

The FastAPI app exposes internal worker, admin, maintenance, and callback
endpoints that should not appear in public docs or the client SDK.  Fern's
override file already acts as the curated surface, so this script treats
non-ignored override entries as the allowlist for the generated public spec.
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
        kept_ops = {
            method: operation
            for method, operation in path_item.items()
            if method.lower() in HTTP_METHODS and (path, method.lower()) in allowlist
        }
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
            "This published OpenAPI document contains the client-facing/public API surface only. "
            "Internal worker, admin, maintenance, and callback endpoints are intentionally omitted."
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
