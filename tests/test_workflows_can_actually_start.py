"""A workflow GitHub cannot parse does not fail loudly — it fails invisibly.

Three of this repository's eleven workflows could not start at all. Each
produced a run that lasted 0s, created zero jobs, and appeared in the checks
list named by its *filename* rather than its `name:` — because GitHub never got
far enough to read the name. Nothing else said anything was wrong.

Two causes, both of which `yaml.safe_load` accepts without complaint:

* **Duplicate `if:` keys** (`mcp.yml`, `live-gates.yml`). A bulk edit added a
  `vars.HOSTED_ACTIONS` guard to jobs that already had a condition. YAML does
  not merge duplicate keys — a parser silently keeps one — but GitHub rejects
  the workflow outright. The comment introduced alongside that guard says "a
  job that fails on every push is worse than no job"; the edit produced neither
  a passing job nor a skipped one, but a workflow that never ran.
* **A step with neither `uses` nor `run`** (`publish.yml`). A stray `-` split
  `name:` from its action, leaving an empty step — and taking the whole
  workflow down with it, including the jobs either side.

So these checks are structural and cheap, and they run in the Python suite
because that is what actually runs here: the workflows themselves could not
report the problem, being the thing that was broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _workflow_files() -> list[Path]:
    return sorted(p for p in WORKFLOWS.glob("*.y*ml") if not p.name.startswith("._"))


def _ids(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


class _DuplicateKeyLoader(yaml.SafeLoader):
    """SafeLoader that records duplicate mapping keys instead of dropping them."""


def _no_duplicates(loader, node, deep=False):
    seen: dict = {}
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            loader.duplicates.append((key, key_node.start_mark.line + 1))
        seen[key] = True
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
)


def _load(path: Path) -> tuple[dict, list[tuple[str, int]]]:
    loader = _DuplicateKeyLoader(path.read_text(encoding="utf-8"))
    loader.duplicates = []
    try:
        data = loader.get_single_data()
    finally:
        loader.dispose()
    return data or {}, loader.duplicates


def test_there_are_workflows_to_check() -> None:
    """A rule about an empty set passes for the wrong reason."""
    assert _workflow_files(), "no workflow files found — this guard is looking in the wrong place"


@pytest.mark.parametrize("path", _workflow_files(), ids=_ids(_workflow_files()))
def test_no_duplicate_keys(path: Path) -> None:
    """The failure mode that took down two workflows.

    `yaml.safe_load` keeps the last of a duplicated key and says nothing, so a
    parse check alone would have passed on both broken files.
    """
    _, duplicates = _load(path)
    assert not duplicates, (
        f"{path.name} has duplicate keys, which GitHub rejects at startup — the "
        "whole workflow stops running, not just the job that carries them: "
        + ", ".join(f"{key!r} at line {line}" for key, line in duplicates)
    )


@pytest.mark.parametrize("path", _workflow_files(), ids=_ids(_workflow_files()))
def test_every_job_is_runnable(path: Path) -> None:
    """`runs-on`, real `needs`, and steps that actually do something."""
    data, _ = _load(path)
    jobs = data.get("jobs") or {}
    assert jobs, f"{path.name} declares no jobs"

    problems: list[str] = []
    for name, job in jobs.items():
        if not isinstance(job, dict):
            problems.append(f"{name}: not a mapping")
            continue
        if "uses" in job:
            continue  # a reusable-workflow call has no runs-on or steps

        if "runs-on" not in job:
            problems.append(f"{name}: no runs-on")

        needs = job.get("needs") or []
        for dep in [needs] if isinstance(needs, str) else needs:
            if dep not in jobs:
                problems.append(f"{name}: needs {dep!r}, which is not a job in this file")

        for index, step in enumerate(job.get("steps") or []):
            if not isinstance(step, dict):
                problems.append(f"{name}.steps[{index}]: not a mapping")
                continue
            if "uses" not in step and "run" not in step:
                label = step.get("name", "<unnamed>")
                problems.append(
                    f"{name}.steps[{index}] ({label!r}): neither `uses` nor `run` — "
                    "usually a stray `-` that split a step in two"
                )

    assert not problems, f"{path.name} cannot start:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("path", _workflow_files(), ids=_ids(_workflow_files()))
def test_every_workflow_has_a_trigger_and_a_name(path: Path) -> None:
    """`on:` parses as the boolean True in YAML 1.1, which is worth stating once."""
    data, _ = _load(path)
    trigger = data.get(True, data.get("on"))
    assert trigger, f"{path.name} declares no `on:` trigger, so nothing can ever run it"
    assert data.get("name"), (
        f"{path.name} has no `name:`; a run that cannot be parsed already shows as "
        "its filename, so a missing name hides the difference between the two"
    )
