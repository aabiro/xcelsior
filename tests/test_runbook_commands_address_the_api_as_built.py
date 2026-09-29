"""Every `$API/...` command in a runbook has to be one the API answers.

`docs/snapshot-registry.md` carried three that it did not, all three written
against an earlier shape of the endpoints and never re-run:

    curl -s "$API/admin/user-images?status=pending" ...
      → 404. `/admin/user-images` is not a route; the listing is `/user-images`.
        And there is no `status` query parameter on it, so even against the
        right path the filter is dropped in silence and you get an unfiltered
        page that reads like an answer.

    curl -s "$API/instances" ... | jq '.[] | select(.status == "running")'
      → jq: Cannot index array with string "status". `/instances` returns
        `{"concurrency": ..., "instances": [...]}`, so `.[]` yields the two
        *values*, not the rows.

    curl -s "$API/user-images" ... | jq '.[] | select(.name == ...)'
      → the same, against `{"images": [...], "limit": ..., "offset": ...}`.

The first is a diagnostic step for "rows stuck at status=pending" that cannot
observe rows stuck at pending. The other two are the end-to-end verification
for the snapshot registry.

Docs drift because nothing runs them. This runs the parts that can be run: the
paths are matched against the mounted routes, and the `jq` root key is checked
against the envelope the endpoint actually returns. Neither needs the doc to be
executable — only to be addressing the API that exists.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

client = TestClient(app)
ROOT = Path(__file__).resolve().parents[1]

#: Only docs that address *this* API. Runbooks also carry `curl` against
#: Headscale, Ollama and the tier-1 inference server on the tower, which are
#: different services on different ports and are none of this test's business.
_API_VAR = re.compile(r'\$\{?(?:API|API_URL|XCELSIOR_API_URL)\}?(/[^\s"\'|\\]*)')

#: `| jq '.images[] ...'`, `| jq '[.images[] ...]'`, or the bare `| jq '.[]'`.
#: The captured group is the key the filter expects to iterate, and it is
#: deliberately allowed to be empty: `.[]` iterates the top level, which is
#: correct against a list and wrong against an object. Writing this as
#: `[A-Za-z_][A-Za-z0-9_]*` — one or more — is how the first version of this
#: test passed against the very commands it was written for, since all three
#: of them used the bare form.
_JQ_ROOT = re.compile(r"\|\s*jq\s+'\[?\.([A-Za-z_][A-Za-z0-9_]*)?\[\]")

#: A shell variable standing in for a path segment (`$HOST_ID`, `${JOB_ID}`).
_SHELL_VAR = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


def _docs_with_api_calls() -> list[Path]:
    found = []
    for path in sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]:
        if path.exists() and _API_VAR.search(path.read_text(encoding="utf-8")):
            found.append(path)
    return found


def _mounted() -> list[re.Pattern]:
    patterns = []
    for route in app.routes:
        path = getattr(route, "path", "")
        if path:
            patterns.append(re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", path) + "$"))
    return patterns


def _commands() -> list[tuple[str, str, str | None]]:
    """(doc, path, jq root key) for every `$API/...` call in the runbooks."""
    out: list[tuple[str, str, str | None]] = []
    for doc in _docs_with_api_calls():
        text = doc.read_text(encoding="utf-8")
        # Join backslash continuations so a `curl \` + `| jq ...` pipeline
        # reads as the single command it is.
        text = re.sub(r"\\\n\s*", " ", text)
        for line in text.splitlines():
            match = _API_VAR.search(line)
            if not match:
                continue
            jq = _JQ_ROOT.search(line)
            # `None` = no jq in the pipeline at all; `""` = a bare `.[]`, which
            # is a claim about the shape just as much as a named key is.
            root = None if jq is None else (jq.group(1) or "")
            out.append((doc.name, match.group(1), root))
    return out


COMMANDS = _commands()


def test_the_runbooks_still_contain_api_commands() -> None:
    """A parser that silently matches nothing would pass every test below."""
    assert COMMANDS, (
        "no `$API/...` commands were found in docs/. Either the runbooks lost "
        "them or the pattern in this test stopped matching — check before "
        "assuming the docs are clean."
    )


@pytest.mark.parametrize("doc,path,_jq", COMMANDS, ids=[f"{d}:{p}" for d, p, _ in COMMANDS])
def test_the_documented_path_is_a_route_the_api_serves(doc: str, path: str, _jq) -> None:
    probe = _SHELL_VAR.sub("x", path.split("?", 1)[0]).rstrip("/") or "/"
    assert any(rx.match(probe) for rx in _mounted()), (
        f"{doc} tells an operator to call `{path}`, which is not a mounted "
        f"route — it answers 404. `curl -s` prints the 404 body and exits 0, "
        "so pasting the command looks like it worked."
    )


_JQ_COMMANDS = [(d, p, j) for d, p, j in COMMANDS if j is not None]


@pytest.mark.parametrize(
    "doc,path,root", _JQ_COMMANDS, ids=[f"{d}:{p}:.{j}[]" for d, p, j in _JQ_COMMANDS]
)
def test_the_jq_filter_matches_the_shape_the_endpoint_returns(
    doc: str, path: str, root: str
) -> None:
    """`jq '.images[]'` against `{"images": [...]}` — the key has to be real.

    A wrong root key is `jq: Cannot index array with string ...` and a non-zero
    exit, so the operator is stopped rather than misled — but stopped in the
    middle of a runbook step, with no hint that the doc is what is wrong.
    """
    probe = _SHELL_VAR.sub("x", path.split("?", 1)[0]).rstrip("/") or "/"
    response = client.get(probe)
    if response.status_code != 200:
        pytest.skip(f"GET {probe} → {response.status_code}; shape not observable here")
    body = response.json()
    if root == "":
        assert isinstance(body, list), (
            f"{doc} pipes `{path}` into `jq '.[]'`, which iterates the top "
            f"level. The endpoint returns an object keyed {sorted(body)}, so "
            "`.[]` yields those *values* — not the rows — and the `select(...)` "
            "after it fails with `Cannot index array with string ...`."
        )
        return
    assert isinstance(body, dict), (
        f"{doc} pipes `{path}` into `jq '.{root}[]'`, which needs an object, "
        f"but the endpoint returns a {type(body).__name__}"
    )
    assert root in body, (
        f"{doc} pipes `{path}` into `jq '.{root}[]'`, but the response is keyed "
        f"{sorted(body)}. jq exits non-zero and the runbook step stops."
    )
