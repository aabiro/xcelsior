"""Every place that tells people to run the wizard names the same release.

The pin used to be typed into five files across bash, TypeScript and Python.
install.sh was bumped to 0.1.8 while the dashboard's hosts page, the SDK docs and
Hexara's own prompt still said 0.1.7, so the wizard's assistant told people to
run an older wizard than the installer did.

Now install.sh is the one source (it needs the literal: it runs on a host with
no repository), `core/release_facts.py` reads it for the API, and
`api/scripts/generate_release_facts.py` copies it into the frontend's
`release-facts.json`. So a literal pin anywhere else is a copy that can drift.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"@xcelsior-gpu/wizard@(\d+\.\d+\.\d+)")


def _installer_pin() -> str:
    text = (ROOT / "api" / "scripts" / "install.sh").read_text()
    match = re.search(
        r'^readonly WIZARD_PACKAGE="@xcelsior-gpu/wizard@(\d+\.\d+\.\d+)"$', text, re.M
    )
    assert match is not None, "install.sh no longer pins the wizard"
    return match.group(1)


def _tracked_files_with_pins() -> list[Path]:
    out = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "-E",
            r"@xcelsior-gpu/wizard@[0-9]",
            "--",
            ":!**/package-lock.json",
            ":!archive/**",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [ROOT / line for line in out.stdout.splitlines() if line.strip()]


def test_only_the_installer_and_its_generated_copy_hold_a_pin():
    expected = _installer_pin()
    files = {str(path.relative_to(ROOT)) for path in _tracked_files_with_pins()}
    assert files == {"api/scripts/install.sh", "api/frontend/src/lib/release-facts.json"}, files
    drift = {
        path: sorted(set(PIN.findall((ROOT / path).read_text())) - {expected}) for path in files
    }
    drift = {path: versions for path, versions in drift.items() if versions}
    assert not drift, f"wizard pins differ from install.sh ({expected}): {drift}"
