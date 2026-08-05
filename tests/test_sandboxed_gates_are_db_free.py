"""The sandboxed gates must actually run without a database.

`gates-sandboxed.yml` says its runner has "no database and no docker socket, by
design" and that everything it runs "needs no database, no network and no
credentials". That was a claim, not a check, and it was false.

The workflow filtered with

    -k "not scratch_db and not postgres"

which reads as "skip the tests that need a database" and does not do it. `-k`
matches **test names**; `scratch_db` is a **fixture**. The three tests that take
it — `test_one_transaction_across_both_tables_deadlocks`,
`test_per_table_transactions_survive_the_same_traffic`,
`test_a_shared_transaction_is_refused_rather_than_hanging` — have no
"scratch_db" in their names, so nothing was excluded. On the first run that
reached this step the job hung for ten minutes trying to open a database the
runner does not have.

It went unnoticed because the job had never got this far: the same workflow
installed three packages and needed a hundred, so every earlier run died at
collection. Two defects in the same file, the second hidden behind the first.

Now the workflow selects `-m "not needs_db"`, which matches markers. That only
holds while the marker and the fixture agree, so this asserts they do — the
filter is one edit away from silently excluding nothing again, and the symptom
is a ten-minute hang rather than a failure.
"""

from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "gates-sandboxed.yml"

#: Fixtures that mean "this test needs a live PostgreSQL server".
DB_FIXTURES = {"scratch_db"}

#: Direct routes to the database that take no fixture at all.
#:
#: The first version of this guard knew only about `scratch_db`, so it passed
#: while `tests/test_companion_schema_discipline.py` — which calls
#: `_get_pg_pool()` in nine tests and takes no fixture — stalled the runner for
#: fourteen minutes. Checking one of two mechanisms is how a guard reports clean
#: on a broken file.
DB_CALLS = ("_get_pg_pool", "scratch_db", "create_engine")


def _gate_files() -> list[pathlib.Path]:
    """The files the workflow actually runs, read from the workflow."""
    text = WORKFLOW.read_text(encoding="utf-8")
    names = re.findall(r"(tests/test_[a-z0-9_]+\.py)", text)
    return [ROOT / n for n in dict.fromkeys(names) if (ROOT / n).exists()]


def _tests_using_db_fixtures(path: pathlib.Path) -> list[tuple[str, bool]]:
    """Every test in *path* taking a DB fixture, and whether it is marked."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        params = {a.arg for a in node.args.args}
        if not (params & DB_FIXTURES):
            continue
        marked = any(
            "needs_db" in ast.unparse(dec) for dec in node.decorator_list
        )
        found.append((node.name, marked))
    return found


def _workflow_commands() -> str:
    """The workflow's shell, with comment lines removed.

    The first version of this guard scanned the raw file and failed on its own
    documentation: the workflow explains the bug it fixed, quoting the old
    `-k "not scratch_db ..."` filter, and the assertion that the filter is gone
    matched that sentence.

    Seventh time a text-scanning guard in this suite has flagged the prose
    *about* the thing it forbids — the banned-vocabulary guard did it twice,
    then the conditional-scope guard, the authz-assertion guard, the
    ratchet-literal guard, and the manual-top-up guard. Strip the comments and
    check the commands.
    """
    return "\n".join(
        line
        for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def test_the_workflow_selects_by_marker_not_by_name():
    """The defect itself, in the file that had it.

    `-k` against a fixture name is the bug. If it comes back, this fails before
    anyone waits ten minutes to find out.
    """
    commands = _workflow_commands()
    assert '-m "not needs_db"' in commands, (
        "the sandboxed gates no longer select by marker; if this reverted to "
        "`-k`, tests that take a database fixture are no longer excluded"
    )
    assert '-k "not scratch_db' not in commands, (
        'the `-k "not scratch_db"` filter is back in the commands. It matches '
        "test names, not fixture names, so it excludes nothing and the job hangs"
    )


def test_this_guard_reads_the_workflow_it_guards():
    """Prove the reach.

    Every assertion here is derived from the file list parsed out of the
    workflow. If that parse returned nothing — a rename, a reformat — the
    remaining tests would pass while checking no files at all.
    """
    files = _gate_files()
    assert len(files) >= 5, (
        f"parsed only {len(files)} gate files out of the workflow; the step's "
        "format changed and the assertions below check nothing"
    )


def test_every_database_backed_gate_is_marked():
    """The marker and the fixture must agree.

    A test that takes `scratch_db` and is not marked will be selected by
    `-m \"not needs_db\"` and will hang in a runner with no database. The
    failure is a timeout, not an error, which is the slowest possible way to
    learn about it.
    """
    unmarked = [
        f"{path.name}::{name}"
        for path in _gate_files()
        for name, marked in _tests_using_db_fixtures(path)
        if not marked
    ]
    assert not unmarked, (
        "these gate tests take a database fixture but carry no `needs_db` "
        f"marker, so the sandboxed runner will hang on them: {unmarked}"
    )


def test_the_marker_is_registered():
    """An unregistered marker is a warning, and warnings are not read.

    Without registration pytest treats `-m "not needs_db"` as matching nothing
    marked, which happens to work, while `--strict-markers` would reject it.
    Registering it makes the intent greppable and the filter durable.
    """
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "needs_db:" in pyproject, (
        "the `needs_db` marker is not registered in pyproject.toml"
    )


def _has_import_time_skip(path: pathlib.Path) -> bool:
    """Does the module probe the database at import and skip itself?

    `tests/test_no_runtime_ddl.py` established the pattern: open a connection
    once at import, and on failure set a module-level `pytestmark` skip. A file
    that does this runs where a database exists and skips cleanly where one does
    not, which is what makes it safe in the sandboxed runner without the
    workflow needing to know its name.
    """
    text = path.read_text(encoding="utf-8")
    return "pytestmark = pytest.mark.skip" in text


def test_every_gate_file_touching_a_database_can_survive_without_one():
    """The check that would have caught the fourteen-minute stall.

    A gate file may reach a database — several legitimately do — but it must
    then either skip itself at import or mark the tests that need one.
    Otherwise every such test retries the pool until pytest's 180-second
    timeout, and a file with nine of them exhausts the twenty-minute job budget.

    The run is then reported *cancelled*, not failed, which reads as
    infrastructure flakiness rather than a defect in the file.
    """
    offenders = []
    for path in _gate_files():
        text = path.read_text(encoding="utf-8")
        if not any(call in text for call in DB_CALLS):
            continue
        if _has_import_time_skip(path):
            continue
        unmarked = [
            name for name, marked in _tests_using_db_fixtures(path) if not marked
        ]
        # No fixture users and no import-time skip means the database is reached
        # directly inside the tests, which is the stalling shape.
        if unmarked or "_get_pg_pool" in text:
            offenders.append(path.name)
    assert not offenders, (
        "these gate files reach a database but neither skip at import nor mark "
        f"the tests that need one, so the sandboxed runner stalls on them: "
        f"{sorted(set(offenders))}"
    )


def test_at_least_one_test_is_actually_marked():
    """The calibration control.

    Every assertion above is satisfied if nothing in the repository uses a
    database fixture at all — the shape of a guard that reports clean because
    its subject vanished.
    """
    marked = [
        name
        for path in _gate_files()
        for name, is_marked in _tests_using_db_fixtures(path)
        if is_marked
    ]
    assert marked, (
        "no gate test carries `needs_db`, so this file is asserting nothing; "
        "either the fixture was renamed or the marker was stripped"
    )
