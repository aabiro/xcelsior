"""Every place that tells people to run the wizard names the same release.

The pin lives in five files across bash, TypeScript and Python. install.sh was
bumped to 0.1.8 while the dashboard's hosts page, the SDK docs and Hexara's
own prompt still said 0.1.7, so the wizard's assistant told people to run an
older wizard than the installer did. install.sh is the reference: its test
already checks that it names a concrete release no newer than the source tree.
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


def test_every_wizard_pin_matches_the_installer():
    expected = _installer_pin()
    files = _tracked_files_with_pins()
    # The hosts page, the SDK docs and Hexara's prompt all pin it today.
    assert len(files) >= 4, files
    drift = {
        str(path.relative_to(ROOT)): sorted(set(PIN.findall(path.read_text())) - {expected})
        for path in files
    }
    drift = {path: versions for path, versions in drift.items() if versions}
    assert not drift, f"wizard pins differ from install.sh ({expected}): {drift}"
