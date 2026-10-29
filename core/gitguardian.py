"""GitGuardian /v1/multiscan integration for pre-deploy and job config secret scanning."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.request
from typing import Any, Mapping, Sequence

DEFAULT_API_URL = "https://api.gitguardian.com/v1"
MULTISCAN_BATCH_SIZE = 20
HTTP_TIMEOUT_SECONDS = 10.0


def _resolve_api_url() -> str:
    raw = (os.environ.get("GITGUARDIAN_API_URL") or DEFAULT_API_URL).strip().rstrip("/")
    if not raw.endswith("/v1"):
        raw = f"{raw}/v1"
    return raw


def _resolve_api_key() -> str:
    for key in (
        "GITGUARDIAN_API_KEY",
        "GITGUARDIAN_PAT",
        "GITGAURDIAN_PAT",
        "GITGUARDIAN_SAT",
        "GITGAURDIAN_SAT",
    ):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    raise RuntimeError(
        "Missing GitGuardian token: set GITGUARDIAN_API_KEY or GITGUARDIAN_PAT in environment/.env"
    )


def scan_documents(
    documents: Sequence[Mapping[str, str]],
    *,
    api_key: str | None = None,
    api_url: str | None = None,
) -> list[dict[str, Any]]:
    """Scan up to N text documents via GitGuardian POST /v1/multiscan.

    Each item in ``documents`` must contain ``{"document": str, "filename": str}``.
    Returns a list of scan result dicts (one per document) containing
    ``policy_break_count`` and ``policy_breaks``.
    """
    if not documents:
        return []

    token = api_key or _resolve_api_key()
    base_url = (api_url or _resolve_api_url()).rstrip("/")
    results: list[dict[str, Any]] = []

    for idx in range(0, len(documents), MULTISCAN_BATCH_SIZE):
        batch = [
            {"document": str(d["document"]), "filename": str(d.get("filename", "inline"))}
            for d in documents[idx : idx + MULTISCAN_BATCH_SIZE]
        ]
        req = urllib.request.Request(
            f"{base_url}/multiscan",
            data=json.dumps(batch).encode("utf-8"),
            headers={
                "Authorization": f"Token {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            batch_results = json.loads(resp.read().decode("utf-8"))
            results.extend(batch_results)

    return results


def scan_env_vars(
    env_vars: Mapping[str, str],
    *,
    filename: str = "job-config.env",
    api_key: str | None = None,
    api_url: str | None = None,
) -> list[dict[str, Any]]:
    """Scan a dictionary of job/container environment variables for leaked secrets.

    Returns the list of ``policy_breaks`` detected (empty list if clean).
    """
    if not env_vars:
        return []
    rendered = "\n".join(f"{k}={v}" for k, v in sorted(env_vars.items()))
    res = scan_documents(
        [{"document": rendered, "filename": filename}],
        api_key=api_key,
        api_url=api_url,
    )
    if not res:
        return []
    return list(res[0].get("policy_breaks", []))


def scan_files(
    paths: Sequence[str | pathlib.Path],
    *,
    api_key: str | None = None,
    api_url: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Scan local files via /v1/multiscan and return a mapping of filename -> policy_breaks."""
    docs: list[dict[str, str]] = []
    for p in paths:
        path_obj = pathlib.Path(p)
        docs.append(
            {
                "filename": str(path_obj),
                "document": path_obj.read_text(encoding="utf-8"),
            }
        )
    scan_res = scan_documents(docs, api_key=api_key, api_url=api_url)
    return {
        doc["filename"]: list(res.get("policy_breaks", []))
        for doc, res in zip(docs, scan_res)
    }


def _load_dotenv_if_present() -> None:
    env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, v = stripped.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv: Sequence[str] | None = None) -> int:
    _load_dotenv_if_present()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("Usage: python3 -m core.gitguardian {status|scan <file>...}")
        return 0

    cmd = args[0]
    if cmd == "status":
        token = _resolve_api_key()
        base_url = _resolve_api_url()
        req = urllib.request.Request(
            f"{base_url}/api_tokens/self",
            headers={"Authorization": f"Token {token}"},
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        print(
            json.dumps(
                {
                    "name": info.get("name"),
                    "type": info.get("type"),
                    "status": info.get("status"),
                    "workspace_id": info.get("workspace_id"),
                    "scopes": info.get("scopes"),
                },
                indent=2,
            )
        )
        return 0

    if cmd == "scan":
        files = args[1:]
        if not files:
            print("Usage: python3 -m core.gitguardian scan <file>...", file=sys.stderr)
            return 2
        findings = scan_files(files)
        total_breaks = 0
        for fname, breaks in findings.items():
            if breaks:
                total_breaks += len(breaks)
                detectors = [b.get("type", "unknown") for b in breaks]
                print(f"✗ {fname}: {len(breaks)} policy break(s) -> {detectors}")
            else:
                print(f"✓ {fname}: clean (0 policy breaks)")
        return 1 if total_breaks > 0 else 0

    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
