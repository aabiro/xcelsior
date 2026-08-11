"""The published privacy policy must agree with what the code actually does.

The policy is a promise to users. It sits in four locale files and nothing tied
it to the behaviour it describes, so it drifted three separate ways at once —
all three found by going to publish one line:

| The policy said | What was true |
|---|---|
| "Audit logs: retained for 1 year" | nothing dropped anything; retention was unbounded |
| "We do not transfer personal information outside Canada unless you consent via the cross-border toggle" | `cross_border_consent` is declared in `privacy.py` and **read by nothing**; placement never consults it |
| (fr) "les renseignements personnels des résidents de la C.-B. restent au Canada en tout temps" | the English text had already retracted exactly that promise |

The third is the worst shape: two languages of one legal document making
different promises, with the French carrying the one the English had removed.
A user reading the French page was told something no longer true, in a document
they are entitled to rely on.

## What is checked here, and why only this

Copy cannot be asserted in general — most of a privacy policy is a statement of
intent, and a test that pinned its wording would be changed to match whatever
the wording became. Two things are different:

1. **The retention period is a number the code enforces.** It appears in the
   policy and in `WORM_RETENTION_MONTHS`, and if those disagree, one of them is
   lying to a user. Derived from the constant, so changing the constant without
   the policy fails here rather than in front of a regulator.
2. **The two languages must offer the same keys.** Not the same words — the same
   set of statements. A key present in one and absent from the other is a
   promise made to one set of users and not the other.
"""

from __future__ import annotations

import pathlib
import re

import pytest

I18N = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "src" / "lib" / "i18n"

#: All four carry the same `privacy.*` keys: the marketing bundles and the
#: in-app ones. Editing one and not the others is how the French page came to
#: disagree with the English.
LOCALE_FILES = ("en-public", "en", "fr-public", "fr")

KEY = re.compile(r'"(privacy\.[a-z0-9_]+)":\s*"((?:[^"\\]|\\.)*)"')


def _strings(name: str) -> dict[str, str]:
    text = (I18N / f"{name}.ts").read_text(encoding="utf-8")
    return {k: v for k, v in KEY.findall(text)}


def test_the_locale_files_are_readable_and_populated():
    """A guard over an empty parse passes; this is what stops that reading green."""
    for name in LOCALE_FILES:
        strings = _strings(name)
        assert len(strings) > 20, (
            f"{name}.ts yielded {len(strings)} privacy strings; the parser has "
            "lost the file and every assertion below would be vacuous"
        )


@pytest.mark.parametrize("name", LOCALE_FILES)
def test_the_policy_states_the_retention_period_the_code_enforces(name: str):
    """The number users are shown is the number `drop_expired_partitions` uses.

    Derived from the constant rather than hardcoded here, so there is one place
    to change and this fails when only one side of it moves.
    """
    from control_plane.audit_partitions import WORM_RETENTION_MONTHS

    line = _strings(name).get("privacy.s7_p5")
    assert line, f"{name}.ts no longer has an audit-log retention line"

    numbers = [int(n) for n in re.findall(r"\b(\d+)\b", line)]
    assert WORM_RETENTION_MONTHS in numbers, (
        f"{name}.ts tells users audit logs are kept per {line!r}, but the code "
        f"enforces {WORM_RETENTION_MONTHS} months. One of them is lying to a "
        "user; change both together."
    )


@pytest.mark.parametrize("name", LOCALE_FILES)
def test_the_policy_says_erasure_does_not_reach_the_audit_trail(name: str):
    """The disclosure the retention basis is owed.

    Retaining under a documented basis is only legitimate if the subject is
    told. A policy that says "we erase your data" and omits the immutable trail
    has misled by omission whatever the basis says, so the sentence is not
    optional decoration — it is the half of the ruling users can see.
    """
    line = _strings(name).get("privacy.s7_p5", "")
    erasure_words = ("erasure", "effacement", "erase", "supprim")
    assert any(word in line.lower() for word in erasure_words), (
        f"{name}.ts no longer tells users that an erasure request does not "
        f"reach the audit trail: {line!r}"
    )


def test_every_promise_is_made_in_both_languages():
    """Same set of statements, not the same words.

    A key in one language and not the other is a promise made to one set of
    users and withheld from the other — or, as happened here, a retraction
    applied in one language and not the other.
    """
    english = set(_strings("en-public"))
    french = set(_strings("fr-public"))
    assert english == french, (
        "the privacy policy differs between languages: "
        f"English-only {sorted(english - french)}, French-only {sorted(french - english)}"
    )


def test_the_marketing_and_in_app_bundles_agree_on_which_keys_exist():
    """Four files, one document. Editing two of them is how this drifted."""
    for pair in (("en-public", "en"), ("fr-public", "fr")):
        first, second = (set(_strings(name)) for name in pair)
        assert first == second, (
            f"{pair[0]}.ts and {pair[1]}.ts disagree on which privacy keys "
            f"exist: {sorted(first ^ second)}"
        )


def test_no_language_still_promises_a_workload_stays_put():
    """The retracted promise, asserted gone from both languages.

    Capacity is selected on price, availability, GPU model and reputation,
    never on geography. A policy line promising otherwise is one the product
    cannot keep, and it survived in French for as long as nobody compared the
    two files.
    """
    broken = []
    for name in LOCALE_FILES:
        for key, value in _strings(name).items():
            lowered = value.lower()
            if "restent au canada" in lowered or "stays in canada" in lowered:
                broken.append(f"{name}.ts:{key}")
            if "remain in canada at all times" in lowered:
                broken.append(f"{name}.ts:{key}")
    assert not broken, (
        "a locale still promises data stays in one country: "
        + ", ".join(broken)
        + ". Placement does not honour it and there is no mechanism that could."
    )
