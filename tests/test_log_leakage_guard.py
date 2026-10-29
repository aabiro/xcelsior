"""Static guard: no log call may pass a raw secret/token variable.

Walks all production *.py files (excluding tests/ and venv) and fails if any
log.info/debug/warning/error/critical format string, or its arguments, directly
reference a sensitive variable name that would expose secret material.

Patterns detected:
  log.X("...", some_token_var)           — positional arg named like a secret
  log.X(f"... {secret} ...")             — f-string embedding a secret name
  log.X("... %s ...", full_response_dict) — full dict passed whose name signals it

The allowlist is used for lines that are deliberately safe (e.g., logging that
a key *is configured* without logging its value).
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]

# Directories to skip entirely
SKIP_DIRS = {"tests", "venv", ".venv", "env", "__pycache__", "site-packages", "node_modules"}

# Regex to match log-style call names
_LOG_METHODS = re.compile(r"^(info|debug|warning|warn|error|critical|exception)$")

# Sensitive variable-name fragments (lower-case match)
_SENSITIVE_FRAGMENTS = (
    "secret",
    "password",
    "passwd",
    "private_key",
    "client_secret",
    "access_token",
    "refresh_token",
    "id_token",
    "api_key",
    "auth_token",
    "webhook_secret",
    "signing_key",
    "jwt_secret",
    "stripe_secret",
)

# Allowlist: (relative_path, line_number) pairs that are known-safe.
# Add new entries here when a false-positive is reviewed and confirmed safe.
_ALLOWLIST: set[tuple[str, int]] = {
    # stripe_connect.py logs only the public key-type prefix (sk_live / sk_test),
    # not any secret material — STRIPE_SECRET_KEY[:7] == "sk_live" or "sk_test".
    ("stripe_connect.py", 50),
    # security.py warns that the dev key is in use — no value is logged.
    ("security.py", 41),
    # security.py logs failure to decrypt by *name*, not by value.
    ("security.py", 1140),
    # worker_agent.py logs an exception object, not the credential value.
    ("worker_agent.py", 593),
}


# ── AST helpers ──────────────────────────────────────────────────────────────


def _is_log_call(node: ast.Call) -> bool:
    """Return True if `node` is a log.<method>(...) call."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return _LOG_METHODS.match(func.attr) is not None
    return False


def _names_in_node(node: ast.expr) -> list[str]:
    """Return names that represent a *whole* secret value passed to a log call.

    Skips names that are merely the receiver object in a subscript
    (``api_key["email"]``) or attribute access (``api_key.attr``) — those pass
    safe fields, not the raw secret.  Does flag subscript *slice* constants
    (e.g. ``data["access_token"]``) because that pulls a sensitive field out.
    """
    # Collect ids of Name nodes used as the receiver of obj[k] or obj.attr
    subscript_receivers: set[int] = set()
    attr_receivers: set[int] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            subscript_receivers.add(id(child.value))
        elif isinstance(child, ast.Attribute):
            attr_receivers.add(id(child.value))

    names: list[str] = []
    for child in ast.walk(node):
        # Whole-variable Name (not a dict/object receiver)
        if isinstance(child, ast.Name):
            if id(child) not in subscript_receivers and id(child) not in attr_receivers:
                names.append(child.id.lower())
        # Subscript slice: data["access_token"] — the string key is sensitive
        elif isinstance(child, ast.Subscript):
            slc = child.slice
            if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                names.append(slc.value.lower())
        # f-string embedded name: f"...{access_token}..."
        elif isinstance(child, ast.FormattedValue):
            if isinstance(child.value, ast.Name):
                if (
                    id(child.value) not in subscript_receivers
                    and id(child.value) not in attr_receivers
                ):
                    names.append(child.value.id.lower())
    return names


def _is_sensitive_name(name: str) -> bool:
    return any(frag in name for frag in _SENSITIVE_FRAGMENTS)


def _check_file(path: Path) -> list[str]:
    """Return a list of violation strings for the given file."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    rel = str(path.relative_to(ROOT))
    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_log_call(node):
            continue

        lineno = node.lineno
        if (rel, lineno) in _ALLOWLIST or (path.name, lineno) in _ALLOWLIST:
            continue

        # Check each argument (skip the format string itself — index 0)
        for arg in node.args[1:]:
            for name in _names_in_node(arg):
                if _is_sensitive_name(name):
                    violations.append(f"{rel}:{lineno}  argument name '{name}' looks like a secret")
                    break

        # Also check f-strings in the first positional arg
        if node.args:
            fmt_arg = node.args[0]
            if isinstance(fmt_arg, ast.JoinedStr):  # f-string
                for name in _names_in_node(fmt_arg):
                    if _is_sensitive_name(name):
                        violations.append(
                            f"{rel}:{lineno}  f-string embeds sensitive name '{name}'"
                        )
                        break

    return violations


# ── Test ─────────────────────────────────────────────────────────────────────


def test_no_log_leakage():
    """No production log call may pass a raw secret/token variable."""
    all_violations: list[str] = []

    for dirpath, dirnames, filenames in os.walk(ROOT):
        # Prune ignored directories in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            full = Path(dirpath) / fname
            # Skip test files
            if full.parent.name == "tests" or fname.startswith("test_"):
                continue
            all_violations.extend(_check_file(full))

    if all_violations:
        report = "\n".join(all_violations)
        raise AssertionError(
            f"Log leakage guard: {len(all_violations)} violation(s) found:\n{report}\n\n"
            "To suppress a false positive, add (relative_path, line_number) to _ALLOWLIST."
        )
