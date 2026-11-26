"""Workflows run only code we pinned, with a read-only token unless a job says otherwise.

Three properties, each established once with `api/scripts/pin_actions.py` and
the zizmor audit, and held here because hosted CI does not run on this
repository and a workflow edit is reviewed by nobody else:

1. Every action is pinned to a full commit SHA. A tag can be repointed: on
   2026-03-19 an attacker force-pushed 76 of 77 `aquasecurity/trivy-action`
   tags to credential-stealing code for about 12 hours. Our `ci.yml` named
   `trivy-action@0.28.0`, one of the tags deleted in the cleanup. It had never
   run here (it was added in July, after hosted Actions stopped), but the next
   run would have failed to start, and a moved tag would have run on the
   self-hosted runner as readily as on GitHub's.
2. Every workflow declares a top-level `permissions:` block, so a job that does
   not ask for more gets a read-only token rather than the repository default.
3. Every `actions/checkout` sets `persist-credentials: false`, so the token is
   not left in `.git/config` for later steps (or uploaded artifacts) to read.

To fix a failure: `python3 api/scripts/pin_actions.py` pins, and the other two
are a line each in the workflow.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

_spec = importlib.util.spec_from_file_location(
    "_xcelsior_pin_actions", ROOT / "api" / "scripts" / "pin_actions.py"
)
assert _spec and _spec.loader
pin_actions = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pin_actions
_spec.loader.exec_module(pin_actions)


def test_there_are_workflows_to_check():
    assert len(WORKFLOWS) >= 10


def test_every_action_is_pinned_to_a_commit_sha_with_its_tag_beside_it():
    assert pin_actions.unpinned(WORKFLOWS) == []


def test_every_workflow_defaults_its_token_to_read_only():
    loose = []
    for path in WORKFLOWS:
        permissions = yaml.safe_load(path.read_text(encoding="utf-8")).get("permissions")
        if not isinstance(permissions, dict) or any(
            value != "read" for key, value in permissions.items() if key != "contents"
        ):
            loose.append(f"{path.name}: {permissions!r}")
        elif permissions.get("contents") != "read":
            loose.append(f"{path.name}: contents is {permissions.get('contents')!r}")
    assert loose == []


def test_every_checkout_drops_its_credentials_after_cloning():
    keeping = []
    for path in WORKFLOWS:
        for job_name, job in (
            yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs") or {}
        ).items():
            for step in job.get("steps") or []:
                if re.match(r"^actions/checkout@", str(step.get("uses", ""))):
                    if (step.get("with") or {}).get("persist-credentials") is not False:
                        keeping.append(f"{path.name}:{job_name}")
    assert keeping == []


def test_a_ref_that_is_an_input_is_spelled_out_once_pinned():
    # dtolnay/rust-toolchain installs whatever toolchain its ref is named; at
    # a SHA it needs the toolchain as an input or the job fails to start.
    missing = []
    for path in WORKFLOWS:
        for job_name, job in (
            yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs") or {}
        ).items():
            for step in job.get("steps") or []:
                action = str(step.get("uses", "")).split("@")[0]
                needed = pin_actions.REF_IS_INPUT.get(action)
                if needed and needed not in (step.get("with") or {}):
                    missing.append(f"{path.name}:{job_name}:{action}")
    assert missing == []
