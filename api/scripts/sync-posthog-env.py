#!/usr/bin/env python3
"""Copy the local browser PostHog settings into the deploy environment.

Values are deliberately never printed. The project token is public/write-only,
but keeping one ignored source of truth prevents the browser and MCP analytics
projects from drifting apart.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "frontend" / ".env.local"
TARGET = ROOT / ".env"
REPORT = ROOT / "frontend" / "posthog-setup-report.md"
KEY_LINE = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = KEY_LINE.match(raw_line.strip())
        if not match:
            continue
        value = match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[match.group(1)] = value
    return values


def render_env(path: Path, updates: dict[str, str]) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    last_indexes: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = KEY_LINE.match(line)
        if match and match.group(1) in updates:
            last_indexes[match.group(1)] = index

    for key, value in updates.items():
        rendered = f"{key}={value}"
        if key in last_indexes:
            lines[last_indexes[key]] = rendered
        else:
            lines.append(rendered)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--personal-api-key-stdin",
        action="store_true",
        help="read a phx_ personal API key from a non-echoing prompt",
    )
    parser.add_argument(
        "--verify-private-api",
        action="store_true",
        help="make a read-only persons request using the configured credential",
    )
    return parser.parse_args()


def verify_private_api(values: dict[str, str]) -> None:
    api_key = values.get("XCELSIOR_POSTHOG_PERSONAL_API_KEY", "").strip()
    project_id = values.get("XCELSIOR_POSTHOG_PROJECT_ID", "").strip()
    api_host = values.get("XCELSIOR_POSTHOG_API_HOST", "").strip()
    if not api_key.startswith("phx_") or not project_id.isdigit() or not api_host:
        raise SystemExit("PostHog private API credential, project ID, or host is missing")
    request = urllib.request.Request(
        f"{api_host.rstrip('/')}/api/projects/{project_id}/persons/?limit=1",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"PostHog private API verification failed with HTTP {exc.code}") from exc
    print("PostHog private API credential verified (response body not printed).")


def main() -> None:
    args = parse_args()
    if not SOURCE.is_file() or not TARGET.is_file():
        raise SystemExit("frontend/.env.local and .env must both exist")

    source = read_env(SOURCE)
    token = source.get("NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN", "").strip()
    host = source.get("NEXT_PUBLIC_POSTHOG_HOST", "").strip()
    if not token.startswith("phc_"):
        raise SystemExit("NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN is missing or invalid")
    if not host.startswith(("https://", "http://")):
        raise SystemExit("NEXT_PUBLIC_POSTHOG_HOST is missing or invalid")

    private_api_host = {
        "https://us.i.posthog.com": "https://us.posthog.com",
        "https://eu.i.posthog.com": "https://eu.posthog.com",
    }.get(host.rstrip("/"), host.rstrip("/"))

    updates = {
        "NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN": token,
        "NEXT_PUBLIC_POSTHOG_HOST": host,
        "XCELSIOR_MCP_POSTHOG_PROJECT_API_KEY": token,
        "XCELSIOR_MCP_POSTHOG_HOST": host,
        "XCELSIOR_POSTHOG_API_HOST": private_api_host,
    }
    if REPORT.is_file():
        project_match = re.search(
            r"https://(?:us|eu)\.posthog\.com/project/(\d+)",
            REPORT.read_text(encoding="utf-8"),
        )
        if project_match:
            updates["XCELSIOR_POSTHOG_PROJECT_ID"] = project_match.group(1)
    if args.personal_api_key_stdin:
        personal_api_key = getpass.getpass("PostHog personal API key: ").strip()
        if not personal_api_key.startswith("phx_"):
            raise SystemExit("PostHog personal API key is missing or invalid")
        updates["XCELSIOR_POSTHOG_PERSONAL_API_KEY"] = personal_api_key
    rendered = render_env(TARGET, updates)
    target_mode = TARGET.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=TARGET.parent, delete=False
    ) as temporary:
        temporary.write(rendered)
        temporary_path = Path(temporary.name)
    os.chmod(temporary_path, target_mode)
    os.replace(temporary_path, TARGET)
    print(f"Updated {len(updates)} PostHog settings in .env (values redacted).")
    if args.verify_private_api:
        verify_private_api(read_env(TARGET))


if __name__ == "__main__":
    main()
