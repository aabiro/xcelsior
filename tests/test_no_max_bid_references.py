"""CI guard: max_bid bidding model must not reappear in application code."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Paths allowed to mention max_bid (historical migration + this guard).
ALLOWED_FRAGMENTS = (
    "api/migrations/versions/040_retire_spot_bidding.py",
    "tests/test_no_max_bid_references.py",
    "tests/test_spot_e2e_staging.py",
    "api/docs/SPOT_INSTANCE_INTEGRATION_PLAN.md",
    "api/frontend/src/__tests__/launch-instance-modal-spot.test.tsx",
    "api/scripts/spot_staging_smoke.py",
    "CHANGELOG.md",
    "SPOT_ROLLOUT.md",
    "api/fern/pages/",
    "README.md",
)

SCAN_DIRS = (
    "api/routes",
    "workers/scheduler.py",
    "billing/marketplace.py",
    "billing/billing.py",
    "core/db.py",
    "api/api.py",
    "api/templates",
    "api/frontend/src",
    "api/frontend/sdk",
    "api/public/openapi.json",
    "api/fern",
    "llms.txt",
    "api/scripts",
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
