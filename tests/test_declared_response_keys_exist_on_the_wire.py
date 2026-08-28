"""A key the frontend declares but the route never sends reads as "no data".

Four bugs this week shared one shape: `api.ts` declares a response field, the
route sends a different name, and the reader's `?? 0` / `|| {}` / `|| []` turns
`undefined` into a plausible-looking nothing.

* `/api/trust-tiers` — `tiers` typed as a Record, sent as a list. Cards rendered
  their array **index** as the tier name and "Min Score: 0" on every one.
* `/api/sla/targets` — read `d.tiers`, route sends `targets`. The compliance
  page showed "No SLA tier data available" permanently, which reads as *the
  platform has no SLA tiers* rather than *this screen is broken*.
* `/api/v2/marketplace/stats` — read `avg_price`, route sends
  `avg_cad_per_hour`. Always exactly 0.
* `/api/pricing/reserved-plans` — read `plans` as an array of a six-field
  object; route sends `reserved_tiers`, a dict, sharing one field.

None threw. None logged. Two were on screens nobody had reported as broken.

## What this checks, and what it cannot

It calls every **parameterless GET** the frontend declares an `apiFetch<…>` type
for, and asserts each declared **top-level** key exists in the response. That is
the axis all four bugs sat on.

It does not cover paths with parameters, POST responses, nested field names, or
value types — a key present with the wrong *type* passes here. The nested case is
excluded deliberately rather than forgotten: this file's earlier draft flattened
nested names and reported `gst` and `partitions` as missing top-level keys, which
is noise a reader learns to skim past.

Routes that reach a processor or the network are skipped by prefix, with the
list visible below rather than buried — a silent skip is how a guard quietly
stops covering half its surface.
"""

from __future__ import annotations

import os
import pathlib
import re

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
API_TS = ROOT / "frontend/src/lib/api.ts"

client = TestClient(app)

#: Reaches Stripe, a Bitcoin node, or PayPal. Not a correctness exemption —
#: these simply cannot be probed from a test process without network.
NETWORK_PREFIXES = (
    "/api/billing/crypto",
    "/api/billing/lightning",
    "/api/billing/paypal",
    "/crypto/",
    "/lightning/",
    "/paypal",
    "/stripe",
)

#: Server-sent events: it does not return, by design.
STREAMING = ("/api/stream",)

#: Declared keys that are legitimately absent, with the reason.
ALLOWED_ABSENT = {
    # These routes return their payload without an `ok` envelope, and no caller
    # reads `ok` from them — the compliance page goes straight to `d.rates` /
    # `d.policies`. Typed loosely rather than wrongly.
    ("/api/compliance/tax-rates", "ok"),
    ("/api/privacy/retention-policies", "ok"),
}


def _declarations() -> dict[str, set[str]]:
    """`path -> declared top-level keys`, for calls with no path parameters."""
    text = API_TS.read_text(encoding="utf-8")
    found: dict[str, set[str]] = {}
    for m in re.finditer(r"apiFetch<", text):
        i = m.end()
        depth, j = 1, i
        while j < len(text) and depth:
            if text[j] == "<":
                depth += 1
            elif text[j] == ">":
                depth -= 1
            j += 1
        typ = text[i : j - 1]
        # Bounded to *this* call, not a fixed character window. A 300-char tail
        # runs past a short function into the next one and attributes its path
        # to this type — which produced four confident, wrong failures
        # (`/api/audit/verify-chain` "declaring" an analytics shape). The next
        # `export` or `apiFetch<` is the end of this call, whichever is nearer.
        stop = len(text)
        for marker in ("\nexport ", "apiFetch<"):
            k = text.find(marker, j)
            if k != -1:
                stop = min(stop, k)
        tail = text[j:stop]
        # A literal with no `${}` interpolation takes no path parameters.
        pm = re.search(r'[`"](/[^`"${]+)[`"]', tail)
        if not pm:
            continue
        path = pm.group(1).split("?")[0]
        # GET only. Several paths carry both a GET and a POST with different
        # response shapes — `/api/ssh/keys` lists `keys` and creates
        # `{id, name, fingerprint}` — and attributing the POST's type to the
        # GET probe reports a mismatch that does not exist. The first draft did
        # exactly that for six of twelve failures, which is the ratio at which
        # a guard gets switched off rather than read.
        if re.search(r'method:\s*"(POST|PUT|PATCH|DELETE)"', tail):
            continue
        # Top level only: nested names are a different question and reporting
        # them here is noise. Depth is tracked across the braces of the type.
        # Required keys only. `ok?: boolean` *declares* that the field may be
        # absent, so reporting it as missing is reporting a correct
        # declaration — the difference between a type that is wrong and one
        # that is deliberately loose.
        keys, depth = set(), 0
        for token in re.finditer(r"[{}]|([a-z_][a-z_0-9]*)(\??)\s*:", typ):
            if token.group(0) == "{":
                depth += 1
            elif token.group(0) == "}":
                depth -= 1
            elif depth == 1 and token.group(1) and token.group(2) != "?":
                keys.add(token.group(1))
        if keys:
            found.setdefault(path, set()).update(keys)
    return found


DECLARED = _declarations()
PROBEABLE = sorted(
    p for p in DECLARED if not p.startswith(NETWORK_PREFIXES) and not p.startswith(STREAMING)
)


def test_the_scan_finds_paths_to_probe():
    """Calibration — an empty list would satisfy every assertion below."""
    # 46 at the time of writing, from 51 declared parameterless GETs. A floor
    # well under that catches a parser that broke, without failing every time
    # someone adds or removes an endpoint.
    assert len(PROBEABLE) > 35, f"only {len(PROBEABLE)} probeable paths; the parser is wrong"
    assert "/api/trust-tiers" in DECLARED


@pytest.mark.parametrize("path", PROBEABLE)
def test_every_declared_top_level_key_exists_in_the_response(path):
    response = client.get(path)
    if response.status_code != 200:
        pytest.skip(f"{path} answered {response.status_code} for an unauthenticated probe")
    try:
        body = response.json()
    except ValueError:
        pytest.skip(f"{path} is not JSON")
    if not isinstance(body, dict):
        pytest.skip(f"{path} returns a {type(body).__name__}, not an object")

    missing = {
        key for key in DECLARED[path] if key not in body and (path, key) not in ALLOWED_ABSENT
    }
    assert not missing, (
        f"{path} does not send {sorted(missing)}, which frontend/src/lib/api.ts "
        f"declares. It sends {sorted(body)}. A reader's `?? 0` or `|| []` turns "
        "the missing value into a plausible nothing rather than an error."
    )
