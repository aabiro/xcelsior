# Anchor workload payload builders from repos under project_search_roots().
# pixelenhance-labs/cli/llm_bridge.py — GPU Strategist chat completions.
# phantom-trades-mvp/financial_beast.py — LLMTradeAnalyzer trade journal prompts.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_DEFAULT_ROOTS = (
    "/mnt/storage/projects",
    "/Users/aaryn/Projects",
    str(Path.home() / "Projects"),
    "/home/aaryn/Projects",
)

# Marker files used to discover repos with runnable LLM anchor payloads.
_ANCHOR_MARKERS: dict[str, tuple[str, ...]] = {
    "pixelenhance-labs": ("cli/llm_bridge.py", "llm_bridge.py"),
    "phantom-trades-mvp": ("financial_beast.py",),
    "ara-code": ("llm_client.py", "api_agent.py"),
}

_ARA_DEFAULT_SYSTEM_PROMPT = """You are Ara, a helpful AI assistant. You provide clear, concise, and accurate responses.
Be direct and helpful. If you don't know something, say so."""


class AnchorWorkloadError(RuntimeError):
    """Raised when a required anchor repo or module is unavailable."""


_GENERIC_MARKERS: tuple[str, ...] = (
    "financial_beast.py",
    "llm_client.py",
    "cli/llm_bridge.py",
    "llm_bridge.py",
    "api_agent.py",
)


def _configured_root_paths() -> list[str]:
    """Ordered unique configured root path strings (includes unreachable Mac paths)."""
    seen: set[str] = set()
    paths: list[str] = []
    extra = os.environ.get("XCELSIOR_PROJECTS_ROOTS", "")
    for part in (*extra.split(","), *_DEFAULT_ROOTS):
        part = part.strip()
        if not part or part in seen:
            continue
        seen.add(part)
        paths.append(part)
    return paths


def _mac_projects_reachable() -> bool:
    host = os.environ.get("XCELSIOR_MAC_HOST", "aaryn@100.64.0.3").strip()
    if not host:
        return False
    try:
        subprocess.check_output(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                host,
                "test -d ~/Projects && echo ok",
            ],
            text=True,
            timeout=12,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def configured_project_roots() -> list[dict[str, Any]]:
    """Every configured search root (Mac ~/Projects probed via SSH when not mounted)."""
    rows: list[dict[str, Any]] = []
    mac_ok = _mac_projects_reachable()
    for part in _configured_root_paths():
        p = Path(part)
        exists = p.is_dir()
        reachable = "local"
        if part in ("/Users/aaryn/Projects", str(Path.home() / "Projects")) and not exists:
            exists = mac_ok
            reachable = "ssh" if mac_ok else "unreachable"
        row: dict[str, Any] = {
            "path": part,
            "exists": exists,
            "reachable": reachable,
            "repo_count": 0,
        }
        if p.is_dir():
            try:
                row["repo_count"] = sum(1 for c in p.iterdir() if c.is_dir())
            except OSError:
                row["repo_count"] = 0
        elif mac_ok and part == "/Users/aaryn/Projects":
            row["repo_count"] = len(_discover_mac_anchor_repos())
        rows.append(row)
    return rows


_MAC_BUILDER_SCRIPTS: dict[str, str] = {
    "pel-chat": """
import json, os, sys
root = os.path.expanduser("~/Projects")
sys.path.insert(0, os.path.join(root, "pixelenhance-labs/cli"))
from llm_bridge import ConversationalLLMClient
c = ConversationalLLMClient()
msgs = list(c._build_messages(channel="GPU Strategist", context="Fleet warm.", history=[{"role":"user","content":"status?"}]))
print(json.dumps({"model":"Qwen/Qwen3-8B","messages":msgs,"temperature":c.temperature}))
""",
    "pt-signal": """
import json, os, sys
root = os.path.expanduser("~/Projects")
sys.path.insert(0, os.path.join(root, "phantom-trades-mvp"))
from financial_beast import LLMTradeAnalyzer
a = LLMTradeAnalyzer(provider="openai", api_key="")
trades=[{"pnl_pct":1.2,"r_multiple":0.8,"market_regime":"trend","strategy_used":"vwap","emotional_state":"calm","conviction_level":"high"}]
stats={"trades":1,"win_rate":0.55,"profit_factor":1.3,"avg_win":1.1,"avg_loss":-0.7,"expectancy":0.15,"avg_r_multiple":0.4}
p = a._format_trades_for_prompt(trades, stats)
print(json.dumps({"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":p}],"max_tokens":512}))
""",
    "ara-agent": """
import json
msgs=[{"role":"system","content":"You are Ara."},{"role":"user","content":"refactor safely?"}]
print(json.dumps({"model":"Qwen/Qwen3-8B","messages":msgs,"temperature":0.7,"max_tokens":1024}))
""",
}


def execute_anchor_on_mac(workload: str) -> dict[str, Any] | None:
    """Run anchor builder on Mac ~/Projects via SSH when local import unavailable."""
    if workload not in _MAC_BUILDER_SCRIPTS:
        return None
    host = os.environ.get("XCELSIOR_MAC_HOST", "aaryn@100.64.0.3").strip()
    if not host or not _mac_projects_reachable():
        return None
    remote = f"""set -euo pipefail
python3 - <<'PY'
{_MAC_BUILDER_SCRIPTS[workload]}
PY
"""
    try:
        out = subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "bash", "-s"],
            input=remote,
            text=True,
            timeout=45,
            stderr=subprocess.DEVNULL,
        ).strip()
        payload = json.loads(out.splitlines()[-1])
        return {
            "workload": workload,
            "executed_on": "mac_ssh",
            "host": host,
            "payload": payload,
            "executed": True,
        }
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
        return None


def post_openai_from_mac(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    workload: str = "",
    host: str | None = None,
) -> dict[str, Any]:
    """POST an OpenAI-compatible request from Mac via SSH (row 30 Mac-executed inference)."""
    mac_host = (host or os.environ.get("XCELSIOR_MAC_HOST", "aaryn@100.64.0.3")).strip()
    if not mac_host or not _mac_projects_reachable():
        return {"ok": False, "reason": "mac_unreachable", "workload": workload}
    merged_headers = {"Content-Type": "application/json", **{str(k): str(v) for k, v in headers.items()}}
    hdr_json = json.dumps(merged_headers)
    body_json = json.dumps(payload)
    remote = f"""set -euo pipefail
python3 - <<'PY'
import json, urllib.request, sys
url = {json.dumps(url)}
headers = json.loads({json.dumps(hdr_json)})
payload = json.loads({json.dumps(body_json)})
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers=headers,
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode()
        body = json.loads(raw) if raw else {{}}
        print(json.dumps({{
            "ok": True,
            "status": resp.status,
            "usage": body.get("usage"),
            "model": body.get("model"),
        }}))
except Exception as exc:
    print(json.dumps({{"ok": False, "error": str(exc)}}))
    sys.exit(1)
PY
"""
    try:
        out = subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", mac_host, "bash", "-s"],
            input=remote,
            text=True,
            timeout=150,
            stderr=subprocess.STDOUT,
        ).strip()
        result = json.loads(out.splitlines()[-1])
        result["executed_on"] = "mac_ssh"
        result["host"] = mac_host
        result["workload"] = workload
        result["url"] = url
        return result
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "reason": str(exc),
            "executed_on": "mac_ssh",
            "host": mac_host,
            "workload": workload,
        }


def execute_anchor_builder(workload: str) -> dict[str, Any]:
    """Import and run a real anchor repo builder (local disk, else Mac SSH)."""
    builders = {
        "pel-chat": pixelenhance_chat_payload,
        "pel-embed": pixelenhance_embed_payload,
        "pt-signal": phantom_trades_chat_payload,
        "ara-agent": ara_code_chat_payload,
    }
    fn = builders.get(workload)
    if not fn:
        raise AnchorWorkloadError(f"unknown workload {workload}")
    try:
        payload = fn()
        origin = "local_import"
    except AnchorWorkloadError:
        mac = execute_anchor_on_mac(workload)
        if mac:
            return mac
        raise
    return {
        "workload": workload,
        "builder": fn.__name__,
        "module": fn.__module__,
        "payload": payload,
        "executed": True,
        "executed_on": origin,
    }


def _discover_mac_anchor_repos() -> list[dict[str, str]]:
    """SSH to Mac ~/Projects when reachable; returns metadata only (paths not mounted locally)."""
    host = os.environ.get("XCELSIOR_MAC_HOST", "aaryn@100.64.0.3").strip()
    if not host:
        return []
    script = r"""
set -euo pipefail
ROOT=~/Projects
for repo in pixelenhance-labs phantom-trades-mvp ara-code deal-ghost stellar-subs; do
  for marker in financial_beast.py llm_client.py cli/llm_bridge.py llm_bridge.py api_agent.py; do
    if test -f "$ROOT/$repo/$marker"; then
      echo "$repo|$ROOT/$repo|$marker|mac"
      break
    fi
  done
done
"""
    try:
        out = subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", host, "bash", "-s"],
            input=script,
            text=True,
            timeout=20,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []
    rows: list[dict[str, str]] = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 4:
            continue
        repo, path, marker, origin = parts
        rows.append(
            {
                "repo": repo,
                "root": origin,
                "path": path,
                "marker": marker,
                "dynamic": "false",
                "reachable": "ssh_only",
            }
        )
    return rows


def discover_anchor_repos(*, include_remote: bool = False) -> list[dict[str, str]]:
    """Scan project roots for repos that expose real LLM workload builders."""
    found: list[dict[str, str]] = []
    seen: set[str] = set()

    def _append(repo_name: str, root: Path, repo_path: Path, marker: str, *, dynamic: bool) -> None:
        key = f"{repo_name}@{root}"
        if key in seen:
            return
        found.append(
            {
                "repo": repo_name,
                "root": str(root),
                "path": str(repo_path),
                "marker": marker,
                "dynamic": str(dynamic).lower(),
            }
        )
        seen.add(key)

    for root in project_search_roots():
        for repo_name, markers in _ANCHOR_MARKERS.items():
            repo_path = root / repo_name
            if not repo_path.is_dir():
                continue
            for marker in markers:
                if (repo_path / marker).exists():
                    _append(repo_name, root, repo_path, marker, dynamic=False)
                    break

        try:
            children = sorted(root.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir() or child.name.startswith("."):
                continue
            for marker in _GENERIC_MARKERS:
                if (child / marker).exists():
                    _append(child.name, root, child, marker, dynamic=True)
                    break

    if include_remote:
        for row in _discover_mac_anchor_repos():
            key = row["repo"]
            if key in seen:
                continue
            found.append(row)
            seen.add(key)
    return found


def project_search_roots() -> list[Path]:
    """Ordered list of existing directories to search for anchor workload repos."""
    return [Path(p) for p in _configured_root_paths() if Path(p).is_dir()]


def _require_repo(repo_name: str, subpath: str = "") -> Path:
    """Locate repo on disk or raise with every path that was checked."""
    checked: list[str] = []
    for root in project_search_roots():
        candidates = [root / repo_name]
        if subpath:
            nested = root / repo_name / subpath
            if nested.is_dir():
                candidates.insert(0, nested)
        for p in candidates:
            checked.append(str(p))
            if p.is_dir():
                s = str(p)
                if s not in sys.path:
                    sys.path.insert(0, s)
                return p
    raise AnchorWorkloadError(
        f"Anchor repo '{repo_name}' not found. Searched: {', '.join(checked)}"
    )


def pixelenhance_chat_payload(
    *,
    channel: str = "GPU Strategist",
    user_message: str = "Status of the enhancement fleet?",
) -> dict[str, Any]:
    """Build OpenAI chat payload the same way pixelenhance-labs CLI does."""
    _require_repo("pixelenhance-labs", "cli")
    from llm_bridge import ConversationalLLMClient  # type: ignore

    client = ConversationalLLMClient()
    messages = list(
        client._build_messages(
            channel=channel,
            context="Fleet telemetry: 3 GPUs warm, queue depth 2.",
            history=[{"role": "user", "content": user_message}],
        )
    )
    return {
        "model": "Qwen/Qwen3-8B",
        "messages": messages,
        "temperature": client.temperature,
    }


def phantom_trades_chat_payload(
    *,
    trades: list[dict[str, Any]] | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build chat payload using phantom-trades-mvp LLMTradeAnalyzer prompt formatting."""
    repo = _require_repo("phantom-trades-mvp")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from financial_beast import LLMTradeAnalyzer  # type: ignore

    sample_trades = trades or [
        {
            "pnl_pct": 1.2,
            "r_multiple": 0.8,
            "market_regime": "trend",
            "strategy_used": "vwap",
            "emotional_state": "calm",
            "conviction_level": "high",
        },
        {
            "pnl_pct": -0.6,
            "r_multiple": -0.4,
            "market_regime": "range",
            "strategy_used": "breakout",
            "emotional_state": "anxious",
            "conviction_level": "low",
        },
    ]
    sample_stats = stats or {
        "trades": len(sample_trades),
        "win_rate": 0.55,
        "profit_factor": 1.3,
        "avg_win": 1.1,
        "avg_loss": -0.7,
        "expectancy": 0.15,
        "avg_r_multiple": 0.4,
    }
    analyzer = LLMTradeAnalyzer(provider="openai", api_key="")
    prompt = analyzer._format_trades_for_prompt(sample_trades, sample_stats)
    return {
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }


def ara_code_chat_payload(
    *,
    user_message: str = "What is the safest way to refactor this module without breaking tests?",
    system_prompt: str | None = None,
    history: list[dict[str, str]] | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Build OpenAI chat payload the same way ara-code APIAgent formats messages."""
    _require_repo("ara-code")
    messages: list[dict[str, str]] = []
    prompt = _ARA_DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
    if prompt:
        messages.append({"role": "system", "content": prompt})
    for entry in history or []:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})
    return {
        "model": "Qwen/Qwen3-8B",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
    }


def pixelenhance_embed_payload(
    *,
    texts: list[str] | None = None,
) -> dict[str, Any]:
    """Caption/embed batch shape for BGE-M3 style endpoints."""
    _require_repo("pixelenhance-labs")
    inputs = texts or [
        "Professional video enhancement preset: film grain preservation.",
        "GPU queue depth elevated on ca-east region.",
    ]
    return {
        "model": "BAAI/bge-m3",
        "input": inputs,
        "encoding_format": "float",
    }