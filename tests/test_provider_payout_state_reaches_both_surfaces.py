"""The payout state the API returns must reach the browser, not just the agent.

Gate P6: *"Provider dashboard parity: the same earnings and payout state the
tools return."*

Earnings matched already — every field `get_provider_earnings` returns was
rendered on the earnings page. Payout state did not. `stripe_connect.get_provider`
builds a `payouts` dict — `charges_enabled`, `payouts_enabled`, `currently_due`,
`past_due`, `disabled_reason`, `checked_live` — `GET /api/providers/{id}` serves
it, and `get_provider_account` hands the whole object to the model. The
dashboard's own hand-copied `ProviderInfo` interface stopped at `paypal`.

TypeScript then *enforced* the omission: `provider.payouts` was a compile error,
so nobody could render it by accident. A provider asking their agent got a list
of what Stripe still needs; the same provider on the dashboard got the word
"Restricted" — the exact insufficiency the backend was changed to fix:

    `status` alone collapses "we need your bank account" and "we need a photo of
    your ID" into `restricted`, which tells a provider nothing they can act on.

It was fixed for the agent and left broken for the person.

## Why this reads the producer, not a list

The obvious guard is a list of expected keys in this file. That is a third copy
of the same vocabulary, and it would go stale the same way the interface did —
silently, and in the direction of saying less. So the keys come from
`stripe_connect.py` itself, and the TypeScript interface must cover whatever it
finds. A field added to the producer fails this test until the browser can see
it too.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

PRODUCER = ROOT / "stripe_connect.py"
TS_TYPES = ROOT / "frontend/src/lib/api.ts"
EARNINGS_PAGE = ROOT / "frontend/src/app/(dashboard)/dashboard/earnings/page.tsx"


def _producer_keys() -> set[str]:
    """Keys of the `payouts` dict `get_provider` builds."""
    text = PRODUCER.read_text(encoding="utf-8")
    # The zero-value literal, which is the complete shape by construction — the
    # live branch overwrites the same keys.
    block = re.search(r'provider\["payouts"\]\s*=\s*\{(.*?)\n\s*\}', text, re.DOTALL)
    assert block, (
        "could not find the `payouts` dict in stripe_connect.get_provider. If it "
        "moved, re-point this guard; if it was removed, the API stopped telling "
        "anyone why they are unpaid."
    )
    return set(re.findall(r'"([a-z_]+)"\s*:', block.group(1)))


def _ts_interface_fields() -> set[str]:
    text = TS_TYPES.read_text(encoding="utf-8")
    block = re.search(r"export interface ProviderPayoutState \{(.*?)\n\}", text, re.DOTALL)
    assert block, "ProviderPayoutState is gone from frontend/src/lib/api.ts"
    # Field names only — skip doc comments, which quote field names in prose.
    fields = set()
    for line in block.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith(("*", "/*", "//")):
            continue
        match = re.match(r"([a-z_]+)\??\s*:", stripped)
        if match:
            fields.add(match.group(1))
    return fields


def test_the_extraction_finds_a_real_shape():
    """Calibration — two empty sets compare equal and prove nothing."""
    produced = _producer_keys()
    assert len(produced) >= 5, f"only found {produced}; the pattern is wrong"
    assert "checked_live" in produced
    assert len(_ts_interface_fields()) >= 5


def test_every_payout_field_the_api_returns_is_typed_for_the_browser():
    produced = _producer_keys()
    typed = _ts_interface_fields()
    missing = produced - typed
    assert not missing, (
        "the API returns payout fields the dashboard's types do not declare, so "
        "TypeScript will reject any attempt to render them: "
        f"{sorted(missing)}. This is how the gap arose the first time."
    )


def test_the_browser_type_does_not_invent_fields_the_api_never_sends():
    """A field typed but never sent renders as `undefined` — silently blank."""
    extra = _ts_interface_fields() - _producer_keys()
    assert not extra, (
        f"the dashboard types payout fields the API does not send: {sorted(extra)}. "
        "These render as undefined, which on this surface means a blank space "
        "where an explanation should be."
    )


def test_the_earnings_page_actually_renders_the_payout_state():
    """Typed is not rendered. The P5 launch control was typed and wired to nothing."""
    page = EARNINGS_PAGE.read_text(encoding="utf-8")
    assert "PayoutRequirements" in page, (
        "the earnings page no longer renders the payout state. The fields being "
        "typed is not the property — the P5 placement control was fully typed "
        "and submitted nothing."
    )
    assert "payouts={provider?.payouts}" in page, (
        "PayoutRequirements is present but is not being passed the provider's "
        "payout state, so it renders its unknown branch forever."
    )


def test_checked_live_is_still_the_field_that_gates_the_rendering():
    """The one property that makes this surface honest rather than reassuring."""
    component = (ROOT / "frontend/src/components/providers/payout-requirements.tsx").read_text(
        encoding="utf-8"
    )
    assert "!payouts.checked_live" in component, (
        "the component no longer gates on `checked_live`. When Stripe was not "
        "consulted every other field is a zero value, and an empty "
        "`currently_due` then reads as 'nothing outstanding, you are done' on "
        "the one surface whose job is explaining why someone is unpaid."
    )
