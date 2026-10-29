"""Two `def`s with one name in a route module hide a route from the ledger.

`UNTESTED_ENDPOINTS.md` scores a route as covered when either its path or its
**handler function name** appears anywhere under `tests/`. That heuristic is
fine until two handlers share a name, at which point one test ticks both boxes
and the untested route is marked tested and stops being looked at.

Both instances found were real:

* `routes/marketplace.py` defined `api_marketplace_search` twice, for
  `GET /marketplace/search` and `POST /api/v2/marketplace/search`. Only the v2
  route had a test; the v1 route had been reported covered since the ledger was
  first generated.
* `routes/auth.py` defined `api_auth_update_profile` twice, for
  `PATCH /api/auth/me` and `PUT /api/auth/me/profile`. Those two do *not*
  behave the same way — one honours `role` and the other silently discarded it
  — so the shared name concealed a behavioural difference as well as a coverage
  gap.

Both routes keep working in either case, because the decorator captures the
function object before the name is rebound. Nothing fails; the module attribute
simply resolves to whichever `def` came last. That is precisely why this needs
a test rather than a convention.
"""

from __future__ import annotations

import ast
import collections

from tests._source_tree import iter_source_files, read_source


def test_no_module_defines_two_handlers_with_the_same_name() -> None:
    offenders: list[str] = []
    for path, rel in sorted(iter_source_files(), key=lambda pair: pair[1]):
        if not rel.startswith("routes/"):
            continue
        tree = ast.parse(read_source(path))
        seen: dict[str, list[int]] = collections.defaultdict(list)
        for node in tree.body:  # module scope only
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seen[node.name].append(node.lineno)
        for name, lines in seen.items():
            if len(lines) > 1:
                offenders.append(f"{rel}: {name} defined at lines {lines}")

    assert not offenders, (
        "a later def shadows an earlier one at module scope; both routes still "
        "serve, but the coverage ledger counts them as one and the module "
        "attribute resolves to only the last:\n  " + "\n  ".join(offenders)
    )
