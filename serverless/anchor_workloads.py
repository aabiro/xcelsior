# Anchor workload payload builders from repos under project_search_roots().
# pixelenhance-labs/cli/llm_bridge.py — GPU Strategist chat completions.
# phantom-trades-mvp/financial_beast.py — LLMTradeAnalyzer trade journal prompts.

from __future__ import annotations

import os
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


def configured_project_roots() -> list[dict[str, Any]]:
    """Every configured search root (including unreachable Mac paths) with exists flag."""
    rows: list[dict[str, Any]] = []
    for part in _configured_root_paths():
        p = Path(part)
        row: dict[str, Any] = {"path": part, "exists": p.is_dir(), "repo_count": 0}
        if p.is_dir():
            try:
                row["repo_count"] = sum(1 for c in p.iterdir() if c.is_dir())
            except OSError:
                row["repo_count"] = 0
        rows.append(row)
    return rows


def discover_anchor_repos() -> list[dict[str, str]]:
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