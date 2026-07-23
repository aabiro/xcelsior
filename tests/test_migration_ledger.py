"""Migration ledger gate (Track B B1.1).

Static enforcement of the rules in `migrations/README.md`. Pure parse — no
database, no Alembic import — so it runs everywhere and fast, and it fails
on a bad migration *before* anyone tries to apply it.

Enforces:
  1. exactly one head;
  2. `down_revision` links files in filename-numeric order (no skips,
     no reordering, no parallel branch);
  3. revision id equals the zero-padded filename prefix, for every
     migration from FIRST_NUMERIC_ENFORCED onward (one documented
     historical exception — see `migrations/README.md` §3);
  4. contract cleanup, if present, is the head;
  5. no duplicate revision ids and no duplicate filename numbers;
  6. the head matches what the other migration gates hardcode, so a new
     migration cannot land while `test_from_empty_bootstrap` still asserts
     the old head.

Every rule is driven both ways: once against the real chain, and once
against a synthetic chain that violates it. A check that has never been
observed to fail is not a gate.

Blueprint §13.8, companion §4.4/§14, Track B §B1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSIONS_DIR = PROJECT_ROOT / "migrations" / "versions"
LEDGER_README = PROJECT_ROOT / "migrations" / "README.md"
BOOTSTRAP_TEST = PROJECT_ROOT / "tests" / "test_from_empty_bootstrap.py"

# Current Alembic head. Update in the same commit as a new migration.
EXPECTED_HEAD = "070"

# Rule 2 of migrations/README.md is enforced from this filename prefix on.
# Everything below it predates the rule and is grandfathered.
FIRST_NUMERIC_ENFORCED = 66

# The one documented exception (README §3): authored without --rev-id.
# Rewriting an applied revision id risks an unmigratable deploy, so it is
# accommodated by id rather than corrected. Do not add a second entry.
GRANDFATHERED_REVISION_IDS = {"060": "a0985327493e"}

# A contract-cleanup migration drops transitional scaffolding, so nothing
# may land after it (blueprint §13.7, Track B §B16.2).
CONTRACT_CLEANUP_MARKERS = ("contract_cleanup", "contract_migration")

_FILENAME_RE = re.compile(r"^(?P<num>\d+)_(?P<slug>.+)\.py$")
# Matches both `revision = "054"` and `revision: str = 'a0985327493e'`.
_REVISION_RE = re.compile(
    r"^revision(?:\s*:\s*[^=]+?)?\s*=\s*(['\"])(?P<value>[^'\"]+)\1",
    re.MULTILINE,
)
_DOWN_REVISION_RE = re.compile(
    r"^down_revision(?:\s*:\s*[^=]+?)?\s*=\s*(?:(['\"])(?P<value>[^'\"]+)\1|(?P<none>None))",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Migration:
    """One parsed migration file."""

    name: str
    number: int
    slug: str
    revision: str
    down_revision: str | None

    @property
    def expected_revision(self) -> str:
        """The revision id this file should declare under rule 2."""
        return f"{self.number:03d}"


# ─────────────────────────── parsing ────────────────────────────


def _parse_migration(name: str, source: str) -> Migration:
    match = _FILENAME_RE.match(name)
    assert match, (
        f"{name} does not follow the NNN_slug.py convention required by "
        f"migrations/README.md rule 2."
    )
    rev_match = _REVISION_RE.search(source)
    assert rev_match, f"{name} declares no `revision`."
    down_match = _DOWN_REVISION_RE.search(source)
    assert down_match, f"{name} declares no `down_revision`."
    return Migration(
        name=name,
        number=int(match.group("num")),
        slug=match.group("slug"),
        revision=rev_match.group("value"),
        down_revision=down_match.group("value"),  # None when literal None
    )


def _parse_migrations() -> list[Migration]:
    """Parse every versioned migration, ordered by filename number."""
    parsed = [
        _parse_migration(path.name, path.read_text(encoding="utf-8"))
        for path in sorted(VERSIONS_DIR.glob("*.py"))
        if not path.name.startswith("_")
    ]
    assert parsed, f"no migrations found in {VERSIONS_DIR}"
    return sorted(parsed, key=lambda m: m.number)


# ────────────────────────── validators ──────────────────────────
# Pure functions over a migration list so each can be driven with
# synthetic violations as well as the real chain.


def validate_no_duplicate_revision_ids(migrations: list[Migration]) -> None:
    seen: dict[str, str] = {}
    for mig in migrations:
        prior = seen.get(mig.revision)
        assert prior is None, (
            f"duplicate revision id {mig.revision!r}: {prior} and {mig.name}. "
            f"Alembic cannot resolve a duplicated id."
        )
        seen[mig.revision] = mig.name


def validate_no_duplicate_filename_numbers(migrations: list[Migration]) -> None:
    seen: dict[int, str] = {}
    for mig in migrations:
        prior = seen.get(mig.number)
        assert prior is None, (
            f"two migrations share prefix {mig.number:03d}: {prior} and "
            f"{mig.name}. Read `alembic heads` before authoring "
            f"(migrations/README.md rule 1)."
        )
        seen[mig.number] = mig.name


def validate_single_head(migrations: list[Migration], expected_head: str) -> None:
    """No parallel heads — blueprint §13, migrations/README.md rule 1."""
    referenced = {m.down_revision for m in migrations if m.down_revision}
    heads = [m for m in migrations if m.revision not in referenced]
    assert len(heads) == 1, (
        "expected exactly one Alembic head, found "
        f"{[h.name for h in heads]}. A parallel head means two migrations "
        f"share a parent; merge them or fix the down_revision chain."
    )
    assert heads[0].revision == expected_head, (
        f"head is {heads[0].revision!r} ({heads[0].name}) but this test "
        f"declares EXPECTED_HEAD={expected_head!r}. Update EXPECTED_HEAD in "
        f"the same commit as the new migration (migrations/README.md §4)."
    )


def validate_chain_follows_filename_order(migrations: list[Migration]) -> None:
    """`down_revision` links files in numeric filename order — no skips.

    This is the check that catches a migration authored against a stale
    design document's revision number instead of the real head.
    """
    base = migrations[0]
    assert base.down_revision is None, (
        f"{base.name} is the lowest-numbered migration and must have "
        f"down_revision = None, not {base.down_revision!r}."
    )
    for previous, current in zip(migrations, migrations[1:]):
        assert current.down_revision == previous.revision, (
            f"chain break: {current.name} has "
            f"down_revision={current.down_revision!r}, but its filename "
            f"predecessor {previous.name} declares "
            f"revision={previous.revision!r}. Migrations must link in "
            f"filename-numeric order (migrations/README.md rule 3)."
        )


def validate_revision_ids_match_filename(
    migrations: list[Migration],
    *,
    first_enforced: int = FIRST_NUMERIC_ENFORCED,
    grandfathered: dict[str, str] | None = None,
) -> None:
    """Rule 2, enforced from `first_enforced` onward.

    Grandfathered ids are accommodated explicitly so the exception is
    visible rather than implied by a loose regex.
    """
    grandfathered = (
        GRANDFATHERED_REVISION_IDS if grandfathered is None else grandfathered
    )
    for mig in migrations:
        allowed = grandfathered.get(mig.expected_revision)
        if allowed is not None:
            assert mig.revision == allowed, (
                f"{mig.name} was grandfathered with revision {allowed!r} "
                f"(migrations/README.md §3) but now declares "
                f"{mig.revision!r}. If you are correcting it, remove the "
                f"GRANDFATHERED_REVISION_IDS entry and confirm no database "
                f"is stamped at the old id."
            )
            continue
        if mig.number < first_enforced:
            # Predates the rule; only the chain-order check applies.
            continue
        assert mig.revision == mig.expected_revision, (
            f"{mig.name} declares revision={mig.revision!r}; rule 2 requires "
            f"{mig.expected_revision!r} (the zero-padded filename prefix). "
            f"Create migrations with "
            f"`alembic revision --rev-id {mig.expected_revision}` — without "
            f"--rev-id Alembic generates a random hash."
        )


def find_contract_cleanups(migrations: list[Migration]) -> list[Migration]:
    return [
        m
        for m in migrations
        if any(marker in m.slug for marker in CONTRACT_CLEANUP_MARKERS)
    ]


def validate_contract_cleanup_is_last(migrations: list[Migration]) -> None:
    """Contract cleanup drops scaffolding, so nothing may land after it.

    Blueprint §13.7; companion §14 ("its contract-cleanup revision must
    remain last, so renumber the combined sequence"); Track B §B16.2.
    """
    cleanups = find_contract_cleanups(migrations)
    if not cleanups:
        return
    assert len(cleanups) == 1, (
        f"expected at most one contract-cleanup migration, found "
        f"{[m.name for m in cleanups]}."
    )
    cleanup = cleanups[0]
    assert cleanup == migrations[-1], (
        f"{cleanup.name} is a contract-cleanup migration but is not the "
        f"head — {migrations[-1].name} follows it. Contract cleanup removes "
        f"transitional columns, triggers, and legacy paths; anything landing "
        f"after it builds on removed scaffolding (blueprint §13.7)."
    )


# ─────────────────────── real-chain assertions ───────────────────────


@pytest.fixture(scope="module")
def migrations() -> list[Migration]:
    return _parse_migrations()


def test_ledger_readme_exists() -> None:
    """The rules this test enforces must be written down (Track B B1.1)."""
    assert LEDGER_README.is_file(), (
        "migrations/README.md is the migration ledger required by Track B "
        "B1.1. It records the numbering history and the rules this test "
        "enforces; without it the test's failures are unexplainable."
    )
    text = LEDGER_README.read_text(encoding="utf-8")
    for marker in (
        "One head, always",
        "--rev-id",
        "Contract cleanup is always the last",
    ):
        assert marker in text, f"migrations/README.md lost its rule text: {marker!r}"


def test_real_chain_has_no_duplicate_revision_ids(migrations: list[Migration]) -> None:
    validate_no_duplicate_revision_ids(migrations)


def test_real_chain_has_no_duplicate_filename_numbers(
    migrations: list[Migration],
) -> None:
    validate_no_duplicate_filename_numbers(migrations)


def test_real_chain_has_exactly_one_head(migrations: list[Migration]) -> None:
    validate_single_head(migrations, EXPECTED_HEAD)


def test_real_chain_follows_filename_order(migrations: list[Migration]) -> None:
    validate_chain_follows_filename_order(migrations)


def test_real_chain_revision_ids_match_filenames(migrations: list[Migration]) -> None:
    validate_revision_ids_match_filename(migrations)


def test_real_chain_contract_cleanup_placement(migrations: list[Migration]) -> None:
    validate_contract_cleanup_is_last(migrations)
    if not find_contract_cleanups(migrations):
        pytest.skip("no contract-cleanup migration yet (Track B B16.2)")


def test_no_new_grandfathered_exceptions() -> None:
    """One documented exception only (migrations/README.md §3)."""
    assert GRANDFATHERED_REVISION_IDS == {"060": "a0985327493e"}, (
        "GRANDFATHERED_REVISION_IDS changed. Rewriting an applied revision "
        "id risks an unmigratable deploy, and adding a *new* exception means "
        "a migration was authored without --rev-id. Fix the migration "
        "instead (it is unapplied if you are still in review)."
    )


def test_grandfathered_migration_still_matches_repository(
    migrations: list[Migration],
) -> None:
    """The exception must describe a file that actually exists as described.

    If 060 is ever corrected, this fails and forces the exception to be
    removed rather than left as stale dead configuration.
    """
    by_number = {m.expected_revision: m for m in migrations}
    for expected_rev, actual_rev in GRANDFATHERED_REVISION_IDS.items():
        mig = by_number.get(expected_rev)
        assert mig is not None, (
            f"GRANDFATHERED_REVISION_IDS names prefix {expected_rev} but no "
            f"such migration exists. Remove the stale entry."
        )
        assert mig.revision == actual_rev, (
            f"{mig.name} now declares revision={mig.revision!r}, not the "
            f"grandfathered {actual_rev!r}. Remove its "
            f"GRANDFATHERED_REVISION_IDS entry."
        )


def test_expected_head_agrees_with_bootstrap_gate() -> None:
    """The migration gates must not disagree about the head.

    `test_from_empty_bootstrap.py` hardcodes the head it expects a
    from-empty upgrade to reach. If a new migration updates one gate and
    not the other, the suite passes while one of them silently checks a
    stale revision.
    """
    source = BOOTSTRAP_TEST.read_text(encoding="utf-8")
    match = re.search(
        r"^EXPECTED_HEAD\s*=\s*(['\"])(?P<value>[^'\"]+)\1", source, re.MULTILINE
    )
    assert match, (
        f"{BOOTSTRAP_TEST.name} no longer declares EXPECTED_HEAD; this "
        f"cross-check needs updating."
    )
    other = match.group("value")
    assert other == EXPECTED_HEAD, (
        f"{BOOTSTRAP_TEST.name} expects head {other!r} but this ledger "
        f"expects {EXPECTED_HEAD!r}. Update both in the same commit as the "
        f"new migration (migrations/README.md §4)."
    )


def test_ledger_readme_records_the_real_head(migrations: list[Migration]) -> None:
    """The written ledger must not drift from the chain it documents."""
    text = LEDGER_README.read_text(encoding="utf-8")
    head = next(m for m in migrations if m.revision == EXPECTED_HEAD)
    assert head.name in text, (
        f"migrations/README.md does not mention the current head "
        f"{head.name}. Update the 'Repository head' line so the ledger stays "
        f"true (migrations/README.md §2)."
    )


# ──────────────── negative drives: prove each check fires ────────────────
# B0.1 rule 3 — a check that has never been observed to fail is not a gate.


def _mig(number: int, revision: str, down: str | None, slug: str = "thing") -> Migration:
    return Migration(
        name=f"{number:03d}_{slug}.py",
        number=number,
        slug=slug,
        revision=revision,
        down_revision=down,
    )


def test_catches_duplicate_revision_ids() -> None:
    chain = [_mig(1, "001", None), _mig(2, "001", "001")]
    with pytest.raises(AssertionError, match="duplicate revision id"):
        validate_no_duplicate_revision_ids(chain)


def test_catches_duplicate_filename_numbers() -> None:
    chain = [_mig(1, "001", None), _mig(1, "002", "001", slug="other")]
    with pytest.raises(AssertionError, match="share prefix"):
        validate_no_duplicate_filename_numbers(chain)


def test_catches_parallel_heads() -> None:
    """Two migrations sharing a parent — the classic bad merge."""
    chain = [_mig(1, "001", None), _mig(2, "002", "001"), _mig(3, "003", "001")]
    with pytest.raises(AssertionError, match="exactly one Alembic head"):
        validate_single_head(chain, "003")


def test_catches_head_disagreeing_with_expected() -> None:
    chain = [_mig(1, "001", None), _mig(2, "002", "001")]
    with pytest.raises(AssertionError, match="EXPECTED_HEAD"):
        validate_single_head(chain, "001")


def test_catches_chain_skip() -> None:
    """A migration authored against a stale design-doc number.

    This is the exact failure Track B B1 exists to prevent: 003 links back
    to 001 because someone read "§13.5 assigns 058" instead of running
    `alembic heads`.
    """
    chain = [_mig(1, "001", None), _mig(2, "002", "001"), _mig(3, "003", "001")]
    with pytest.raises(AssertionError, match="chain break"):
        validate_chain_follows_filename_order(chain)


def test_catches_non_none_base_down_revision() -> None:
    chain = [_mig(1, "001", "000"), _mig(2, "002", "001")]
    with pytest.raises(AssertionError, match="must have\ndown_revision = None|down_revision = None"):
        validate_chain_follows_filename_order(chain)


def test_catches_hash_revision_id_on_new_migration() -> None:
    """The 060 mistake, repeated after the rule took effect."""
    chain = [_mig(66, "beefcafe1234", None)]
    with pytest.raises(AssertionError, match="rule 2 requires"):
        validate_revision_ids_match_filename(
            chain, first_enforced=66, grandfathered={}
        )


def test_allows_grandfathered_hash_revision_id() -> None:
    """The documented exception is permitted — and only that exact id."""
    chain = [_mig(60, "a0985327493e", None)]
    validate_revision_ids_match_filename(
        chain, first_enforced=1, grandfathered={"060": "a0985327493e"}
    )
    with pytest.raises(AssertionError, match="was grandfathered"):
        validate_revision_ids_match_filename(
            [_mig(60, "somethingelse", None)],
            first_enforced=1,
            grandfathered={"060": "a0985327493e"},
        )


def test_pre_rule_migrations_are_not_retroactively_failed() -> None:
    """Grandfathering is by number, so history stays green."""
    chain = [_mig(5, "005", None), _mig(6, "006", "005")]
    validate_revision_ids_match_filename(chain, first_enforced=66, grandfathered={})


def test_catches_migration_after_contract_cleanup() -> None:
    chain = [
        _mig(1, "001", None),
        _mig(2, "002", "001", slug="contract_cleanup"),
        _mig(3, "003", "002", slug="one_more_thing"),
    ]
    with pytest.raises(AssertionError, match="not the\nhead|not the head"):
        validate_contract_cleanup_is_last(chain)


def test_allows_contract_cleanup_as_head() -> None:
    chain = [_mig(1, "001", None), _mig(2, "002", "001", slug="contract_cleanup")]
    validate_contract_cleanup_is_last(chain)


def test_catches_two_contract_cleanups() -> None:
    chain = [
        _mig(1, "001", None, slug="contract_cleanup"),
        _mig(2, "002", "001", slug="contract_migration_final"),
    ]
    with pytest.raises(AssertionError, match="at most one contract-cleanup"):
        validate_contract_cleanup_is_last(chain)


# ──────────────── parser drives: the regexes must handle both styles ────────────────


def test_parser_handles_bare_assignment_style() -> None:
    mig = _parse_migration(
        "054_control_plane_core_columns.py",
        'revision = "054"\ndown_revision = "053"\n',
    )
    assert (mig.revision, mig.down_revision) == ("054", "053")


def test_parser_handles_annotated_assignment_style() -> None:
    """Migration 060 onward uses `revision: str = '...'`."""
    mig = _parse_migration(
        "060_shared_state_to_pg.py",
        "revision: str = 'a0985327493e'\n"
        "down_revision: Union[str, None] = '059'\n",
    )
    assert (mig.revision, mig.down_revision) == ("a0985327493e", "059")


def test_parser_handles_none_down_revision() -> None:
    mig = _parse_migration(
        "001_initial.py", 'revision = "001"\ndown_revision = None\n'
    )
    assert mig.down_revision is None


def test_parser_rejects_missing_revision() -> None:
    with pytest.raises(AssertionError, match="declares no `revision`"):
        _parse_migration("001_initial.py", "down_revision = None\n")


def test_parser_rejects_bad_filename() -> None:
    with pytest.raises(AssertionError, match="NNN_slug.py convention"):
        _parse_migration("add_widgets.py", 'revision = "001"\ndown_revision = None\n')
