#!/usr/bin/env python3
"""Wire SPIRE enrollment into both API slots of the production compose override.

`api` and `api-blue` are separate Compose services. Nginx sends traffic to
whichever one is live. Environment keys and volume mounts under `api` do not
apply to `api-blue`, so a patch that stops at the first match leaves the live
slot without the SPIRE socket or the `spire-server` binary.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ENV_LINES = (
    "XCELSIOR_SPIRE_SERVER_CMD: ${XCELSIOR_SPIRE_SERVER_CMD:-}",
    "XCELSIOR_SPIRE_SERVER_SOCKET: ${XCELSIOR_SPIRE_SERVER_SOCKET:-}",
    "XCELSIOR_SPIRE_SERVER_ADDRESS: ${XCELSIOR_SPIRE_SERVER_ADDRESS:-}",
    "XCELSIOR_SPIRE_SERVER_PORT: ${XCELSIOR_SPIRE_SERVER_PORT:-8095}",
    "XCELSIOR_HEADSCALE_URL: ${XCELSIOR_HEADSCALE_URL:-}",
    "XCELSIOR_HEADSCALE_API_KEY: ${XCELSIOR_HEADSCALE_API_KEY:-}",
    "XCELSIOR_HEADSCALE_USER_ID: ${XCELSIOR_HEADSCALE_USER_ID:-}",
)
VOL_LINES = (
    "/var/lib/xcelsior-spire/sock:/run/xcelsior-spire",
    "/usr/local/lib/xcelsior/spire-server:/usr/local/bin/spire-server:ro",
)
TARGETS = ("api", "api-blue")

_SERVICE_HEADER = re.compile(r"^  ([A-Za-z0-9_.-]+):\s*(?:#.*)?$")


def merge_env(enrollment_text: str, env_text: str) -> tuple[str, list[str]]:
    """Append enrollment keys that the env file does not already name.

    An existing key is left alone, including when the enrollment file has a
    different value. Callers keep production values that were set by hand.
    """
    wanted: list[str] = []
    for line in enrollment_text.splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        wanted.append(line)
    have: set[str] = set()
    for line in env_text.splitlines():
        if line and not line.startswith("#") and "=" in line:
            have.add(line.split("=", 1)[0])
    extra: list[str] = []
    added: list[str] = []
    for line in wanted:
        key = line.split("=", 1)[0]
        if key not in have:
            extra.append(line)
            added.append(key)
    if not extra:
        return env_text, added
    if env_text and not env_text.endswith("\n"):
        env_text += "\n"
    env_text += "\n# Host enrollment (issue #67)\n" + "\n".join(extra) + "\n"
    return env_text, added


def _is_top_level(line: str) -> bool:
    return bool(line.strip()) and line[:1] not in (" ", "\t", "#")


def split_services(text: str) -> tuple[list[str], list[tuple[str, list[str]]], list[str]]:
    lines = text.splitlines(keepends=True)
    start = None
    for index, line in enumerate(lines):
        if line.strip() == "services:" and not line.startswith((" ", "\t")):
            start = index
            break
    if start is None:
        raise SystemExit("override has no services: block")
    preamble = lines[: start + 1]
    services: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current: list[str] = []
    tail: list[str] = []
    in_services = True
    for line in lines[start + 1 :]:
        if in_services and _is_top_level(line):
            in_services = False
        if not in_services:
            tail.append(line)
            continue
        header = _SERVICE_HEADER.fullmatch(line.rstrip("\n"))
        if header:
            if current_name is not None:
                services.append((current_name, current))
            elif current:
                preamble.extend(current)
            current_name = header.group(1)
            current = [line]
        else:
            current.append(line)
    if current_name is not None:
        services.append((current_name, current))
    elif current:
        preamble.extend(current)
    return preamble, services, tail


def _ensure_block(block: str) -> tuple[str, bool]:
    changed = False
    if "XCELSIOR_SPIRE_SERVER_ADDRESS:" not in block:
        env_body = "".join(f"      {line}\n" for line in ENV_LINES)
        marker = "    environment:\n"
        if marker in block:
            block = block.replace(marker, marker + env_body, 1)
        else:
            header, separator, rest = block.partition("\n")
            block = header + separator + "    environment:\n" + env_body + rest
        changed = True
    missing = [item for item in VOL_LINES if item not in block]
    if missing:
        vol_body = "".join(f"      - {item}\n" for item in missing)
        marker = "    volumes:\n"
        if marker in block:
            block = block.replace(marker, marker + vol_body, 1)
        else:
            if not block.endswith("\n"):
                block += "\n"
            block += "    volumes:\n" + vol_body
        changed = True
    return block, changed


def _new_service(name: str) -> str:
    env_body = "".join(f"      {line}\n" for line in ENV_LINES)
    vol_body = "".join(f"      - {item}\n" for item in VOL_LINES)
    return f"  {name}:\n    environment:\n{env_body}    volumes:\n{vol_body}"


def _patch_json(text: str) -> tuple[str, list[str]]:
    """Production keeps this override as JSON. Compose accepts that form.

    A YAML-only scan raises "no services: block" on `{ "services": ... }`
    and stops the deploy before migration. Existing keys and mounts stay.
    """
    data = json.loads(text)
    services = data.get("services") if isinstance(data, dict) else None
    if not isinstance(services, dict):
        raise SystemExit("override has no services: block")
    touched: list[str] = []
    for name in TARGETS:
        service = services.get(name)
        if not isinstance(service, dict):
            service = {}
            services[name] = service
        changed = False
        environment = service.get("environment")
        if environment is None:
            environment = {}
            service["environment"] = environment
        if isinstance(environment, dict):
            for line in ENV_LINES:
                key, value = line.split(":", 1)
                key = key.strip()
                if key not in environment:
                    environment[key] = value.strip()
                    changed = True
        elif isinstance(environment, list):
            present = {
                item.split("=", 1)[0]
                for item in environment
                if isinstance(item, str) and "=" in item
            }
            for line in ENV_LINES:
                key, value = line.split(":", 1)
                key = key.strip()
                if key not in present:
                    environment.append(f"{key}={value.strip()}")
                    changed = True
        else:
            raise SystemExit(f"override service {name} environment is not a map or list")
        volumes = service.get("volumes")
        if volumes is None:
            volumes = []
            service["volumes"] = volumes
        if not isinstance(volumes, list):
            raise SystemExit(f"override service {name} volumes is not a list")
        for mount in VOL_LINES:
            if mount not in volumes:
                volumes.append(mount)
                changed = True
        if changed:
            touched.append(name)
    if not touched:
        return text, []
    return json.dumps(data, indent=2) + "\n", touched


def patch_override(text: str) -> tuple[str, list[str]]:
    """Return the override text and the service names whose blocks changed."""
    if text.lstrip().startswith("{"):
        return _patch_json(text)
    preamble, services, tail = split_services(text)
    present = {name for name, _body in services}
    pieces: list[str] = []
    touched: list[str] = []
    for name, body in services:
        block = "".join(body)
        if name in TARGETS:
            block, changed = _ensure_block(block)
            if changed:
                touched.append(name)
        pieces.append(block)
    for name in TARGETS:
        if name not in present:
            pieces.append(_new_service(name))
            touched.append(name)
    rebuilt = "".join(preamble)
    for piece in pieces:
        if rebuilt and not rebuilt.endswith("\n"):
            rebuilt += "\n"
        rebuilt += piece
    if tail:
        if rebuilt and not rebuilt.endswith("\n"):
            rebuilt += "\n"
        rebuilt += "".join(tail)
    if rebuilt and not rebuilt.endswith("\n"):
        rebuilt += "\n"
    return rebuilt, touched


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print(
            "usage: spire_enrollment_override.py ENROLLMENT OVERRIDE ENVFILE",
            file=sys.stderr,
        )
        return 2
    enrollment, override_path, env_path = map(Path, args)
    if not enrollment.is_file() or not override_path.is_file() or not env_path.is_file():
        print("SPIRE enrollment files are not on this host; leaving compose unchanged")
        return 0
    env_text, added = merge_env(enrollment.read_text(), env_path.read_text())
    if added:
        env_path.write_text(env_text)
        env_path.chmod(0o600)
    original = override_path.read_text()
    updated, touched = patch_override(original)
    changed = updated != original
    if changed:
        override_path.write_text(updated)
    print("spire_env_keys_added", ",".join(added) if added else "none")
    print("spire_override", "updated" if changed else "already_present")
    print("spire_services", ",".join(touched) if touched else "unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
