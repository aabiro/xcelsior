"""Money is stored and mutated as integer micros, never as float.

Binary floating point cannot represent most decimal amounts exactly, so every
arithmetic step on a float money column accumulates error. The schema still
carries `_cad` columns, but they are *projections*: `wallets_project_money`
derives them from the integer column on write. Application code must never
write one, or the projection direction inverts and the float becomes the
source of truth for that row.

This guards the invariant rather than a single call site, because the failure
is silent — a float write produces a plausible-looking number.
"""

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent

# The float half of a dual-representation pair, keyed by the table it lives on.
# Scoped deliberately: a `_cad` column is only a projection where a trigger
# derives it. Columns with no `_micros` twin (serverless_endpoints.total_cost_cad)
# are ordinary float columns and out of scope here.
#
# payout_splits is excluded on purpose: it carries all four pairs but has NO
# projection trigger, so its `_cad` and `_micros` columns are independently
# written and can silently diverge. Writing only micros there would leave the
# float stale, so stripe_connect correctly writes both. Giving payout_splits a
# trigger (or dropping its float columns outright) is tracked in HANDOFF.
PROJECTED_FLOAT_COLUMNS = {
    "wallets": (
        "balance_cad",
        "total_deposited_cad",
        "total_spent_cad",
        "total_refunded_cad",
    ),
    "wallet_transactions": ("amount_cad", "balance_after_cad"),
    "wallet_holds": ("amount_cad",),
    "usage_meters": ("total_cost_cad",),
}
FLOAT_MONEY_COLUMNS = tuple(
    sorted({c for cols in PROJECTED_FLOAT_COLUMNS.values() for c in cols})
)

SKIP_DIRS = {".venv", "venv", "node_modules", ".next", "migrations", "tests", "__pycache__"}


def _application_sources():
    for path in REPO.rglob("*.py"):
        if SKIP_DIRS & set(path.parts):
            continue
        yield path


def test_no_application_code_writes_a_float_money_column():
    """A SQL assignment to a float money column inverts the projection."""
    offenders = []
    for path in _application_sources():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # Only look inside SQL string literals. A Python keyword argument
        # named amount_cad is fine — the API contract speaks CAD; what must
        # not happen is a float column being assigned in an UPDATE or INSERT.
        for block in re.finditer(r'"""(.*?)"""', text, re.S):
            sql = block.group(1)
            if not re.search(r"\b(UPDATE|INSERT)\b", sql, re.I):
                continue
            tables = [t for t in PROJECTED_FLOAT_COLUMNS if re.search(rf"\b{t}\b", sql)]
            if not tables:
                continue
            columns = {c for t in tables for c in PROJECTED_FLOAT_COLUMNS[t]}
            for column in sorted(columns):
                if re.search(rf"(?:SET\s+|,\s*){re.escape(column)}\s*=(?!=)", sql):
                    line = text[: block.start()].count("\n") + 1
                    offenders.append(
                        f"{path.relative_to(REPO)}:~{line} writes {column} in SQL"
                    )
    assert not offenders, (
        "Money must be written as integer micros; the _cad columns are "
        "projections maintained by wallets_project_money. Offenders:\n  "
        + "\n  ".join(offenders)
    )


def test_the_money_helpers_round_rather_than_truncate():
    """int(x * 1_000_000) truncates; 10.07 CAD would land a micro short."""
    from money import cad_to_micros, micros_to_cad

    assert cad_to_micros("10.07") == 10_070_000
    assert cad_to_micros(10.07) == 10_070_000
    assert cad_to_micros("0.1") + cad_to_micros("0.2") == cad_to_micros("0.3")
    assert micros_to_cad(10_070_000) == 10.07


def test_float_arithmetic_would_have_drifted():
    """Documents why this matters, and fails if the helpers regress to floats."""
    from money import cad_to_micros

    # The classic float result is 0.30000000000000004.
    assert 0.1 + 0.2 != 0.3
    assert cad_to_micros(0.1) + cad_to_micros(0.2) == 300_000
