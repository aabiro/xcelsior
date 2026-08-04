"""Configured-but-ineffective environment variables, ratcheted to zero.

`docker-compose.yml` declares no `env_file:`, so a variable reaches a container only
by being named in the `x-api-environment` anchor. `.env` is otherwise consumed for
`${...}` interpolation alone. A variable set with a real value, read by server-side
code, and absent from the anchor is therefore **present on the host and absent in
the process that reads it** — and nothing reports it.

Two secrets in that state refused a production boot on 2026-08-04. They were not
instances of a small problem: they were the two that happened to be checked at
startup. Counting the rest found, among others, Stripe webhook signing secrets
(every delivery answered 503, behind a 401 nobody had reached), a storage cap
enforcing 2000 GB instead of the configured 100, and a Reply-To that had never been
live.

`test_startup_env_is_wired.py` gates the ~11 variables startup validation reads,
which is exactly why the rest survived it. This is the general case.

## Why a ratchet and not an assertion of zero

82 remain. Most are harmless — a value identical to its default changes nothing —
but "harmless" is a judgement per variable, and 82 judgements is not one commit.
A ratchet is the honest shape: the number may fall and may never rise, so the
problem is bounded now and closed incrementally, with each reduction a deliberate
act rather than a hope.

`docs/review/workaround-elimination-plan.md` §B is the route to zero: generate the
mapping from a manifest instead of maintaining it by hand, which removes the
failure mode rather than this instance of it.

## Why `env_file: .env` is not the fix

It would wire all 300-odd in one line and take this count to zero immediately —
while handing the complete production secret set to every container, including
services with no business holding it. `test_startup_env_is_wired.py` already
forbids it, and that test should stay.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
COMPOSE = REPO / "docker-compose.yml"
ENV_FILE = REPO / ".env"

#: Services whose environment comes from the shared anchor or their own block. A
#: variable reaching any of them counts as wired — several are worker-side
#: settings that only the scheduler or billing worker needs.
SERVICES = ("api", "api-blue", "scheduler-worker", "bg-worker")

#: Modules that run as a systemd unit on GPU hosts rather than in a container. A
#: variable read *only* by these is correctly absent from compose, so it is not a
#: finding — it would be one if it were mapped.
WORKER_ONLY_MODULES = {"worker_agent.py", "cli.py", "agent.py"}

#: The count on 2026-08-04, after wiring the ten with observable consequences.
#: **This may only decrease.** Raising it is not a fix; it is a record that the
#: problem grew.
MAX_CONFIGURED_BUT_INEFFECTIVE = 82

_READ = re.compile(r'environ(?:\.get)?\(?\[?["\'](XCELSIOR_[A-Z0-9_]+)["\']')
_SET = re.compile(r"^(XCELSIOR_[A-Z0-9_]+)=(.+)$")


def _source_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for pattern in ("*.py", "routes/*.py", "control_plane/*.py", "serverless/*.py"):
        out += [p for p in REPO.glob(pattern) if not p.name.startswith("._")]
    return out


def _vars_read_by_module() -> dict[str, set[str]]:
    read: dict[str, set[str]] = {}
    for path in _source_files():
        for var in set(_READ.findall(path.read_text(encoding="utf-8"))):
            read.setdefault(var, set()).add(path.name)
    return read


def _vars_set_with_a_value() -> set[str]:
    if not ENV_FILE.exists():
        pytest.skip(".env not present — nothing an operator has configured to check")
    found = set()
    for line in ENV_FILE.read_text().splitlines():
        m = _SET.match(line.strip())
        if m and m.group(2).strip().strip('"').strip("'"):
            found.add(m.group(1))
    return found


def _mapped_into_any_service() -> set[str]:
    compose = yaml.safe_load(COMPOSE.read_text())
    mapped: set[str] = set()
    for service in SERVICES:
        mapped |= set(compose["services"][service].get("environment") or {})
    return mapped


def configured_but_ineffective() -> list[str]:
    """Set by the operator, read by server-side code, reaching no container.

    Names only. **Never values** — a first run of this comparison printed two
    Stripe webhook secrets and a Headscale auth key into a transcript. The variable
    name is the finding; its content never is.
    """
    read = _vars_read_by_module()
    worker_only = {v for v, files in read.items() if files <= WORKER_ONLY_MODULES}
    return sorted((_vars_set_with_a_value() & set(read)) - _mapped_into_any_service() - worker_only)


def test_the_count_has_not_risen():
    unwired = configured_but_ineffective()
    assert len(unwired) <= MAX_CONFIGURED_BUT_INEFFECTIVE, (
        f"{len(unwired)} variables are set in .env, read by server-side code, and "
        f"reach no container — up from {MAX_CONFIGURED_BUT_INEFFECTIVE}. Newly "
        f"unwired: {unwired[:10]}{'…' if len(unwired) > 10 else ''}. Map them in "
        "docker-compose.yml, or the value an operator set does nothing."
    )


def test_the_ratchet_is_tightened_when_it_falls():
    """A ceiling well above the real count stops being a ratchet.

    If the number drops and this constant does not follow, the gate silently
    tolerates a regression back up to the old ceiling — which is how a ratchet
    becomes an allowlist.
    """
    unwired = configured_but_ineffective()
    assert len(unwired) >= MAX_CONFIGURED_BUT_INEFFECTIVE - 5, (
        f"only {len(unwired)} remain, against a ceiling of "
        f"{MAX_CONFIGURED_BUT_INEFFECTIVE}. Lower MAX_CONFIGURED_BUT_INEFFECTIVE to "
        f"{len(unwired)} in the same commit that fixed them."
    )


def test_the_measurement_finds_something_to_measure():
    """Guards against the detector silently matching nothing.

    A gate that reports zero because its regex stopped matching is worse than no
    gate: it reads as the problem being solved.
    """
    read = _vars_read_by_module()
    assert len(read) > 100, f"only {len(read)} XCELSIOR_* reads found — the detector broke"
    assert _vars_set_with_a_value(), ".env parsed to nothing — the parser broke"
    assert len(_mapped_into_any_service()) > 50, "compose parsed to almost nothing"


def test_the_two_secrets_that_refused_the_boot_stay_wired():
    """The specific regression, named.

    A ratchet on a count would not notice these two leaving, and their absence is
    what stopped production booting.
    """
    mapped = _mapped_into_any_service()
    for var in (
        "XCELSIOR_COMPAT_SESSION_SECRET",
        "XCELSIOR_AUDIT_SIGNING_KEYS",
        "XCELSIOR_STRIPE_THIN_WEBHOOK_SECRET",
        "XCELSIOR_STRIPE_CONNECT_WEBHOOK_SECRET",
    ):
        assert var in mapped, (
            f"{var} no longer reaches any container. Production refused to boot "
            "without the first two, and answered 503 to every Stripe delivery "
            "without the second two."
        )


def test_worker_only_variables_are_not_counted_as_findings():
    """The exclusion, asserted rather than assumed.

    `worker_agent.py` runs as a systemd unit on GPU hosts. A variable only it reads
    is correctly absent from compose — counting those would inflate the ratchet with
    non-problems and train someone to raise the ceiling.
    """
    read = _vars_read_by_module()
    worker_only = {v for v, files in read.items() if files <= WORKER_ONLY_MODULES}
    assert worker_only, (
        "no worker-only variables found — either the module list is wrong or the "
        "worker stopped reading configuration, and both matter"
    )
    assert not (set(configured_but_ineffective()) & worker_only)
