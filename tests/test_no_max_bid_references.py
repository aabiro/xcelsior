"""CI guard: max_bid bidding model must not reappear in application code."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Paths allowed to mention max_bid (historical migration + this guard).
ALLOWED_FRAGMENTS = (
    "migrations/versions/040_retire_spot_bidding.py",
    "tests/test_no_max_bid_references.py",
    "tests/test_spot_e2e_staging.py",
    "docs/SPOT_INSTANCE_INTEGRATION_PLAN.md",
    "frontend/src/__tests__/launch-instance-modal-spot.test.tsx",
    "scripts/spot_staging_smoke.py",
    "CHANGELOG.md",
    "SPOT_ROLLOUT.md",
    "fern/pages/",
    "README.md",
)

SCAN_DIRS = (
    "routes",
    "scheduler.py",
    "marketplace.py",
    "billing.py",
    "db.py",
    "api.py",
    "templates",
    "frontend/src",
    "frontend/sdk",
    "public/openapi.json",
    "fern",
    "llms.txt",
    "scripts",
)


def test_no_max_bid_in_application_code():
    hits: list[str] = []
    for target in SCAN_DIRS:
        path = ROOT / target
        if not path.exists():
            continue
        cmd = ["rg", "-l", "max_bid", str(path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode not in (0, 1):
            raise RuntimeError(proc.stderr or proc.stdout)
        for line in proc.stdout.splitlines():
            rel = str(Path(line).resolve().relative_to(ROOT))
            if not any(frag in rel for frag in ALLOWED_FRAGMENTS):
                hits.append(rel)
    assert not hits, "max_bid found outside allowed paths:\n" + "\n".join(sorted(hits))