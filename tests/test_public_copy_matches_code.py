"""Public copy may not say what the code does not do (Phase 0.7, 2026-11-22).

Each phrase below was on the site, in the docs or in an agent prompt while the
code did something else. The test reads the copy where people and agents see it
and fails if one comes back.

* Spot locks when the job is **submitted** (`control_plane/spot_admission.py`,
  called from `scheduler.submit_job`), and nothing re-locks it on requeue.
* Billing is **per second** (`rate_micros × seconds ÷ 3600`), not by the hour.
* Renters see live telemetry only for their own GPUs, from the per-GPU store
  (``hosts/gpu_telemetry.py``). Until 2026-11-22 the renter route read a store
  only filled when PostgreSQL was off, so copy promising it was false; the
  guard now checks that the route and the instance-page card behind the claim
  are the real ones.
* The local wizard checks and the Xcel AI explanation follow the server's
  thresholds and its seven checks.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._source_tree import iter_source_files

ROOT = Path(__file__).resolve().parent.parent
I18N = ROOT / "api/frontend/src/lib/i18n"
MARKETING = "api/frontend/src/app/(marketing)/"

COPY_FILES = [
    *sorted(I18N.glob("*.ts")),
    ROOT / "api/frontend/src/app/layout.tsx",
    ROOT / "api/frontend/src/app/manifest.ts",
    *sorted(
        path
        for path, rel in iter_source_files("*.tsx", include_prefixes=(MARKETING,))
        if rel.startswith(MARKETING)
    ),
    *sorted((ROOT / "api/frontend/content/blog").glob("*.mdx")),
    ROOT / "llms.txt",
    *sorted((ROOT / "api/support_docs").glob("*.mdx")),
    *sorted((ROOT / "api/fern/pages").glob("*.mdx")),
    ROOT / "api/mcp/src/server.ts",
    ROOT / "core/ai_assistant.py",
]

FALSE_PHRASES = [
    (r"lock\w* when (your|the) job is assigned", "spot locks at submission"),
    (r"locked at assignment", "spot locks at submission"),
    (r"(next|new) assignment may lock", "nothing re-locks a spot rate"),
    (r"set when the scheduler assigns a host", "spot locks at submission"),
    (r"verrouillé dès l'attribution", "spot locks at submission"),
    (r"locked at allocation|verrouillé à l'allocation", "spot locks at submission"),
    (r"locked at the moment the scheduler assigns", "spot locks at submission"),
    (r"\bby the hour\b", "billing is per second"),
    (r"billed hourly|hourly billing", "billing is per second"),
    (r"facturés? à l'heure", "billing is per second"),
    (r"see health before you rent", "listings show no host health yet"),
    (r"voir l'état avant de louer", "listings show no host health yet"),
    (r"see it before you rent|visible avant la location", "listings show no throttling data yet"),
    (r"every host, verified|chaque hôte, vérifié", "not every host is verified"),
]

# Renter-facing keys: the host dashboard's own live gauges are real and stay.
RENTER_KEY_PREFIXES = ("home.", "features.", "gpus.", "mcp.landing.", "pricing.")
RENTER_TELEMETRY = re.compile(
    r"(real[- ]time|live)\s+((nvml|gpu)\s+)?telemetry|telemetry[^.\"]*stream"
    r"|télémétrie (gpu )?(en temps réel|en direct|et journaux en direct)",
    re.I,
)


def _copy_files() -> list[Path]:
    files = [p for p in COPY_FILES if p.exists()]
    assert len(files) > 20, "the copy locations moved; update COPY_FILES"
    return files


@pytest.mark.parametrize("pattern, why", FALSE_PHRASES)
def test_no_copy_repeats_a_false_claim(pattern, why):
    rx = re.compile(pattern, re.I)
    hits = []
    for path in _copy_files():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()[:120]}")
    assert not hits, f"{why}:\n" + "\n".join(hits)


def _i18n_pairs(path: Path):
    for m in re.finditer(r'^\s*"([^"]+)":\s*"((?:[^"\\]|\\.)*)"', path.read_text("utf-8"), re.M):
        yield m.group(1), m.group(2)


def test_renter_telemetry_copy_has_the_real_route_behind_it():
    """Copy may promise a renter live GPU telemetry only while the code delivers it."""
    claims = [
        f"{path.name}: {key}"
        for path in sorted(I18N.glob("*.ts"))
        for key, value in _i18n_pairs(path)
        if key.startswith(RENTER_KEY_PREFIXES) and RENTER_TELEMETRY.search(value)
    ]
    for path in (
        ROOT / "api/frontend/src/app/layout.tsx",
        ROOT / "api/frontend/src/app/(marketing)/features/page.tsx",
        ROOT / "api/frontend/src/app/(marketing)/about/page.tsx",
    ):
        if RENTER_TELEMETRY.search(path.read_text("utf-8")):
            claims.append(str(path.relative_to(ROOT)))
    if not claims:
        return
    routes = (ROOT / "api/routes/instances.py").read_text("utf-8")
    start = routes.index("def api_instance_telemetry(")
    route = routes[start : routes.index("\n@router", start)]
    assert "latest_for_run" in route and "_host_telemetry" not in route, (
        "copy promises renters live telemetry, but the renter route no longer reads "
        "the per-GPU store: " + ", ".join(claims)
    )
    page = (ROOT / "api/frontend/src/app/(dashboard)/dashboard/instances/[id]/page.tsx").read_text(
        "utf-8"
    )
    assert "<GpuTelemetryCard" in page, (
        "copy promises renters live telemetry, but the instance page no longer shows it: "
        + ", ".join(claims)
    )


def test_the_spot_copy_says_submit():
    values = dict(_i18n_pairs(I18N / "en.ts"))
    assert "locks when you submit" in values["home.hero_desc"]
    assert "locks when you submit" in values["home.val_pricing_desc"]
    assert "billed by the second" in values["home.feat_marketplace_desc"]


def test_the_wizard_thresholds_are_the_servers():
    from verification import VERIFICATION_THRESHOLDS, verification_check_count

    source = (ROOT / "api/wizard/src/provider-checks.ts").read_text("utf-8")
    block = re.search(r"export const VERIFICATION_THRESHOLDS = \{(.*?)\};", source, re.S)
    assert block, "VERIFICATION_THRESHOLDS moved in provider-checks.ts"
    wizard = {k: float(v) for k, v in re.findall(r"(\w+):\s*([\d.]+)", block.group(1))}
    assert wizard, "no thresholds parsed"
    for key, value in wizard.items():
        assert key in VERIFICATION_THRESHOLDS, f"wizard threshold {key} has no server value"
        assert value == float(VERIFICATION_THRESHOLDS[key]), key

    checks = re.search(r"export function runVerificationChecks\((.*?)\n\}", source, re.S)
    assert checks
    assert checks.group(1).count("checks.push(") == verification_check_count()


def test_the_xcel_ai_explanation_names_the_servers_checks():
    import ai_assistant
    from verification import VERIFICATION_CHECKS, VERIFICATION_THRESHOLDS

    prompt = ai_assistant._prompt_verification({"gpu": "RTX 3060"})
    for fn in VERIFICATION_CHECKS:
        name = fn.__name__.removeprefix("check_").replace("_", " ")
        assert name.lower() in prompt.lower(), name
    assert f"{VERIFICATION_THRESHOLDS['max_gpu_temp_celsius']}°C" in prompt
    assert "Memory integrity" not in prompt
    assert "x16" not in prompt.split("WHEN CHECKS FAIL")[0]
    assert "fully verified and certified" not in prompt
