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

    # The number *attached to a unit*, not merely present somewhere in the
    # sentence. A looser check passes on any line that happens to contain the
    # digits — including one that says "1 year" and mentions 24 for an
    # unrelated reason — which is the failure mode a guard like this exists to
    # rule out rather than reproduce.
    stated = re.search(r"(\d+)\s*(months?|mois|years?|ans?|an)\b", line)
    assert stated, f"{name}.ts no longer states an audit retention period with a unit: {line!r}"
    value, unit = int(stated.group(1)), stated.group(2)
    months = value * 12 if unit.startswith(("year", "an")) else value
    assert months == WORM_RETENTION_MONTHS, (
        f"{name}.ts tells users audit logs are kept for {value} {unit} "
        f"({months} months), but the code enforces {WORM_RETENTION_MONTHS}. "
        "One of them is lying to a user; change both together."
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


# ── The mirror, as a mechanism rather than a state ────────────────────


def test_french_is_reconciled_to_the_english_it_was_written_against():
    """An English edit without a French edit fails here.

    Key parity above catches a *missing* statement. It cannot catch a
    **changed** one — which is what actually happened: English `s6_p3` was
    replaced with "compute may run on independent hosts in any country" while
    French went on promising a province's residents' data "restent au Canada en
    tout temps". Same keys, both files valid, every check passing.

    The missing fact was the reconciliation, and it is not recoverable from the
    files: nothing in either says whether the French was written against the
    English currently sitting beside it. So it is recorded, in
    `privacy-translation.lock.json`, and its staleness is the report.

    This is a lock file, not a hand-kept duplicate of a fact. It is generated by
    `scripts/sync_privacy_translations.py --write`, and the only correct time to
    regenerate it is *after* updating French.
    """
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/sync_privacy_translations.py"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        "the French privacy policy is not reconciled to the English it was "
        "written against:\n"
        + result.stdout.strip()
        + "\n\nUpdate fr-public.ts and fr.ts to match, then run "
        "`python3 scripts/sync_privacy_translations.py --write`. Regenerating "
        "the lock without touching French republishes the divergence."
    )


def test_the_lock_covers_every_english_string_it_claims_to():
    """A lock over an empty set passes; this is what stops that reading green."""
    import json

    lock = json.loads((I18N / "privacy-translation.lock.json").read_text(encoding="utf-8"))[
        "english_digests"
    ]
    english = _strings("en-public")
    assert len(lock) == len(english) > 20, (
        f"the lock records {len(lock)} strings against {len(english)} in "
        "en-public.ts; it is not covering the document it guards"
    )


def test_no_french_string_is_the_untranslated_english_one():
    """A copy-paste "translation" passes both parity and the lock.

    Both would be satisfied by pasting the English sentence into the French
    file — same key, reconciled digest, and a French-speaking user reading
    English in a document they are entitled to have in their language. Short
    strings can legitimately coincide (a product name, a date format), so this
    only looks at real prose.
    """
    english, french = _strings("en-public"), _strings("fr-public")
    identical = [
        key for key, value in english.items() if len(value.split()) > 8 and french.get(key) == value
    ]
    assert not identical, f"French carries the untranslated English text for: {identical}"


# ── The consent control that gated nothing ────────────────────────────


def test_the_cross_border_toggle_is_not_offered_in_settings():
    """A control that gates nothing must not be presented as one.

    `cross_border_consent` was read by no code path and placement has no
    geography input to gate, so a user who switched it off believed they had
    restricted transfers and had not. The policy sentence backing that belief
    is retracted; the control goes with it.
    """
    settings = (
        pathlib.Path(__file__).resolve().parent.parent
        / "frontend"
        / "src"
        / "app"
        / "(dashboard)"
        / "dashboard"
        / "settings"
        / "page.tsx"
    )
    source = settings.read_text(encoding="utf-8")
    offered = re.search(r"const consentTypes\s*=\s*\[([^\]]*)\]", source)
    assert offered, "the consent list is no longer readable from settings/page.tsx"
    assert "cross_border" not in offered.group(1), (
        "the cross-border toggle is being offered again. It gates nothing; if "
        "geography-aware placement now exists, this needs a fresh consent "
        "captured against a promise the product can keep."
    )


def test_nothing_reads_the_deprecated_cross_border_field():
    """Retained as a record, not resurrected as a gate.

    The column is kept on purpose — the stored values record what people chose
    while being told it mattered. What must not happen is it quietly becoming
    load-bearing again without a decision, which is how it came to be presented
    as a control in the first place.
    """
    from tests._source_tree import iter_source_files

    readers = []
    # The shared iterator, not an eighth hand-rolled `rglob`. Eight of those
    # once failed simultaneously on macOS AppleDouble sidecars with an error
    # naming neither the sidecar nor the gate's subject —
    # `test_source_tree_is_shared.py` asserts the convergence, and caught this
    # file the first time it ran.
    for path, rel in iter_source_files(exclude_prefixes=("migrations/",)):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for num, line in enumerate(text.splitlines(), 1):
            if "cross_border_consent" not in line:
                continue
            stripped = line.strip()
            # The declaration itself, and comments about it, are the point.
            if stripped.startswith("#") or "cross_border_consent: bool" in stripped:
                continue
            readers.append(f"{rel}:{num}")
    assert not readers, (
        "something now reads the deprecated cross_border_consent field: "
        + ", ".join(readers)
        + ". It was collected under a promise that has been retracted."
    )
