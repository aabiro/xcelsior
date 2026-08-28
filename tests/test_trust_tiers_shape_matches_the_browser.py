"""The trust-tier ladder the browser renders is not the one the API sends.

`GET /api/trust-tiers` returns `tiers` as a **list** of objects keyed
`tier / threshold / search_boost / pricing_premium_pct / platform_commission /
description / unlock_requirements`.

`frontend/src/lib/api.ts` declares it as a **dict keyed by tier name**, whose
values carry `min_score` and `requirements` — two fields the API has never sent.

The dashboard then does `Object.entries(tiers).map(([name, tier]) => …)`. On an
array that yields `["0", {…}], ["1", {…}]`, so:

* `name` is `"0"`, `"1"`, `"2"`… — `TIER_LABELS[name]` misses and the card falls
  back to rendering the **index** as the tier's name.
* `TIER_COLORS[name]` and `TIER_ICONS[name]` miss too, so every tier gets the
  same generic icon and no colour.
* `tier.min_score` does not exist — the API sends `threshold` — and the `?? 0`
  renders **"Min Score: 0"** on every card.
* `tier.requirements` does not exist — the API sends `unlock_requirements` — and
  the `|| []` renders **no requirements at all**, silently.

The defensive defaults are what hide it. Nothing throws, nothing logs, and the
section looks plausible while being entirely wrong.

`get_trust_tiers` is a published MCP tool over the same route, so an agent reads
the real ladder — thresholds, commission, premium — while the person looking at
the dashboard sees six cards numbered 0 to 5. That is the third instance this
week of a field reaching the API and the tool and not the browser, and the first
where the *shape* disagreed rather than a field being dropped.

## Why the test that existed did not catch it

`TestTrustTierEndpoints.test_trust_tiers` asserts `status_code == 200` and
nothing else. A status-only test answers "is it wired?" and never "does it say
what the caller thinks it says" — the same gap `test_public_openapi.py` records,
where comparing only the operation set let five schemas drift.

These assertions are about the **contract**, so they are written against the
response rather than against the constants: renaming `threshold` in
`reputation.py` should fail here, because the browser and the tool both read it.
"""

from __future__ import annotations

import os
import pathlib
import re

os.environ.setdefault("XCELSIOR_ENV", "test")

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

client = TestClient(app)

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: What the route documents itself as returning, per entry.
EXPECTED_ENTRY_KEYS = {
    "tier",
    "threshold",
    "search_boost",
    "pricing_premium_pct",
    "platform_commission",
    "description",
    "unlock_requirements",
}


def _tiers():
    response = client.get("/api/trust-tiers")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("ok") is True
    return body["tiers"]


def test_tiers_is_a_list_and_the_browser_must_treat_it_as_one():
    tiers = _tiers()
    assert isinstance(tiers, list), (
        f"tiers is a {type(tiers).__name__}. The frontend types it as a Record "
        "keyed by tier name; if this ever becomes a dict, fix the type rather "
        "than leaving both shapes in play."
    )
    assert len(tiers) >= 4, f"only {len(tiers)} tiers; the ladder is incomplete"


def test_every_entry_carries_the_keys_both_surfaces_read():
    for entry in _tiers():
        missing = EXPECTED_ENTRY_KEYS - set(entry)
        assert not missing, f"tier {entry.get('tier')!r} is missing {sorted(missing)}"


def test_the_fields_the_dashboard_reads_are_not_invented():
    """`min_score` and `requirements` were read for real and never sent."""
    for entry in _tiers():
        assert "min_score" not in entry, (
            "the API now sends `min_score`. If that is deliberate the frontend "
            "type is finally correct — but `threshold` must then be retired "
            "rather than both existing."
        )
        assert "requirements" not in entry, (
            "the API now sends `requirements`; see the note above for `min_score`."
        )


def test_the_tier_name_is_a_name_and_not_an_index():
    """The bug's visible symptom: cards labelled 0, 1, 2 instead of tier names."""
    names = [entry["tier"] for entry in _tiers()]
    for name in names:
        assert not str(name).isdigit(), f"tier name {name!r} is a number"
        assert str(name).strip(), "a tier has a blank name"
    assert len(set(names)) == len(names), f"duplicate tier names: {names}"


def test_thresholds_are_ordered_so_a_ladder_reads_as_one():
    thresholds = [entry["threshold"] for entry in _tiers()]
    assert thresholds == sorted(thresholds), (
        f"thresholds are not ascending: {thresholds}. A ladder rendered in "
        "response order would read out of sequence."
    )


REPUTATION_PAGE = ROOT / "frontend/src/app/(dashboard)/dashboard/reputation/page.tsx"


def _hardcoded_ladder() -> dict[str, dict[str, str]]:
    """The `TIERS` array the reputation page renders from, parsed."""
    text = REPUTATION_PAGE.read_text(encoding="utf-8")
    start = text.index("const TIERS = [")
    block = text[start : start + 6000]
    found = {}
    for m in re.finditer(
        r'key: "(\w+)".*?threshold: (\d+).*?searchBoost: "([^"]+)"'
        r'.*?pricingPremium: "([^"]+)".*?platformFee: "([^"]+)"',
        block,
        re.DOTALL,
    ):
        found[m.group(1)] = {
            "threshold": m.group(2),
            "search_boost": m.group(3),
            "pricing_premium_pct": m.group(4),
            "platform_commission": m.group(5),
        }
    return found


def test_the_hardcoded_reputation_ladder_still_matches_the_server():
    """A second copy of the ladder, and this one decides what providers earn.

    `/dashboard/reputation` renders a local `TIERS` array carrying thresholds,
    search boost, pricing premium and **platform fee**. Those numbers are the
    ones a provider reads to decide whether chasing a tier is worth it, and the
    server enforces its own copy in `reputation.py`.

    They agree today. The reason to pin them is that the *other* copy of this
    ladder — `TIER_ICONS` / `TIER_COLORS` / `TIER_LABELS` on the trust page —
    was keyed on `sla.SLATier` entirely, a different ladder, and nobody noticed.
    Two hand-maintained copies of a number that determines payouts is a drift
    that costs money rather than a cosmetic one.

    The deeper fix is for the page to read `/api/trust-tiers` and keep only its
    presentation locally. This guard is the cheap half: it fails the day the
    numbers part company, rather than the day a provider queries their invoice.
    """
    served = {str(entry["tier"]): entry for entry in _tiers()}
    hardcoded = _hardcoded_ladder()

    assert hardcoded, "could not parse `TIERS` from the reputation page; re-point this guard"
    assert set(hardcoded) == set(served), (
        f"the page's ladder lists {sorted(hardcoded)} and the server serves {sorted(served)}"
    )

    for tier, local in sorted(hardcoded.items()):
        remote = served[tier]
        assert int(local["threshold"]) == int(remote["threshold"]), (
            f"{tier}: page says {local['threshold']} points, server says {remote['threshold']}"
        )
        # Fractions on the wire, formatted strings on the page: compare numbers.
        assert float(local["pricing_premium_pct"].rstrip("%")) == round(
            float(remote["pricing_premium_pct"]) * 100
        ), (
            f"{tier}: page advertises a {local['pricing_premium_pct']} price "
            f"premium, server applies {float(remote['pricing_premium_pct']) * 100:.0f}%"
        )
        assert float(local["platform_commission"].rstrip("%")) == round(
            float(remote["platform_commission"]) * 100
        ), (
            f"{tier}: page advertises a {local['platform_commission']} platform "
            f"fee, server charges "
            f"{float(remote['platform_commission']) * 100:.0f}%. A provider "
            "decides whether to chase this tier on that number."
        )
        assert float(local["search_boost"].rstrip("\u00d7x")) == float(remote["search_boost"]), (
            f"{tier}: page says {local['search_boost']} search boost, server "
            f"applies {remote['search_boost']}"
        )


def test_the_dashboards_tier_maps_key_on_the_ladder_it_renders():
    """The page styled a different ladder than the one it displays.

    `TIER_ICONS` / `TIER_COLORS` / `TIER_LABELS` keyed on
    `community / secure / dedicated / regulated` — that is `sla.SLATier`, host
    hardware class, a different ladder from provider standing. Nothing on the
    page ever supplied those keys, so every lookup missed: no colour, one
    fallback icon on every card, and the raw name where a label belonged.

    Two vocabularies that both spell themselves "tier" is exactly the confusion
    worth a guard. Checked as **coverage, not equality** — a map may carry an
    extra key harmlessly, but a tier the API sends and the map does not know
    renders unstyled, and that is the failure being prevented.
    """
    import re

    page = (ROOT / "frontend/src/app/(dashboard)/dashboard/trust/page.tsx").read_text(
        encoding="utf-8"
    )
    sent = {str(entry["tier"]) for entry in _tiers()}

    for map_name in ("TIER_ICONS", "TIER_COLORS", "TIER_LABELS"):
        block = re.search(rf"const {map_name}[^=]*= \{{(.*?)\n\}};", page, re.DOTALL)
        assert block, f"{map_name} is gone from the trust page; re-point this guard"
        keys = set(re.findall(r"^\s{2}([a-z_][a-z_0-9]*):", block.group(1), re.M))
        missing = sent - keys
        assert not missing, (
            f"{map_name} does not cover {sorted(missing)}, which "
            "`/api/trust-tiers` sends. Those tiers render unstyled. If the map "
            "is meant for `sla.SLATier` instead, it is on the wrong page."
        )


def test_the_economics_a_provider_decides_on_are_present():
    """P6 parity: the agent quotes these, so the dashboard must be able to.

    A provider choosing whether to chase a tier is comparing commission against
    premium. Both are on the wire; only the browser could not see them.
    """
    for entry in _tiers():
        assert isinstance(entry["platform_commission"], (int, float))
        assert isinstance(entry["pricing_premium_pct"], (int, float))
        # A **string**, not a list. Worth pinning: the dashboard's dead
        # `requirements` field was typed `string[]` and mapped over, so simply
        # renaming it to `unlock_requirements` would have replaced a silently
        # empty list with a `.map is not a function` crash.
        assert isinstance(entry["unlock_requirements"], str)
        assert entry["unlock_requirements"].strip()
