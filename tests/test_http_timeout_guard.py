# Static test: ensure all requests/httpx calls have timeout param
import ast
import os
import re
import pytest

# Allowlist for legacy/test/known-safe files (edit as needed)
KNOWN_SAFE = {
    "tests/test_e2e_live.py",  # test code, not prod
}

REQ_PATTERN = re.compile(r"\b(requests|httpx)\.(get|post|put|delete|patch|head)\b")


def find_calls_missing_timeout(pyfile):
    with open(pyfile, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=pyfile)
    missing = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.value and isinstance(node.func.value, ast.Name):
                lib = node.func.value.id
                meth = node.func.attr
                if lib in ("requests", "httpx") and meth in (
                    "get",
                    "post",
                    "put",
                    "delete",
                    "patch",
                    "head",
                ):
                    # Check for timeout kwarg
                    if not any(
                        (isinstance(kw, ast.keyword) and kw.arg == "timeout")
                        for kw in node.keywords
                    ):
                        # Allow test/legacy files
                        if os.path.relpath(pyfile, os.getcwd()) in KNOWN_SAFE:
                            continue
                        missing.append((pyfile, node.lineno, f"{lib}.{meth}"))
    return missing


def test_requests_calls_have_timeout():
    root = os.path.dirname(os.path.dirname(__file__))
    offenders = []
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(dirpath, fname)
                # Only check files in main repo, not venv/ or .tox/
                if any(skip in fpath for skip in ("/venv/", "/.tox/", "/site-packages/")):
                    continue
                offenders.extend(find_calls_missing_timeout(fpath))
    if offenders:
        msg = "\n".join(f"{f}:{l}: {call} missing timeout=" for f, l, call in offenders)
        pytest.fail(f"requests/httpx calls missing timeout= param:\n{msg}")
