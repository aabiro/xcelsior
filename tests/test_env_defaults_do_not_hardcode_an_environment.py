"""An env var with a production literal fallback must be mapped in compose.

Issue #24: `XCELSIOR_PUBLIC_URL` was read in `routes/auth.py` with a fallback of
`https://xcelsior.ca` and **mapped in no compose file**. The container never
received it, so the fallback always won. On production that is invisible —
the fallback *is* production — which is exactly why the first place it would
have shown is **staging**, handing a staging user a Quick Connect config
pointing at prod.

Three more of the same shape existed when that was found (`XCELSIOR_CORS_ORIGINS`,
`XCELSIOR_PUBLIC_HOST`, `XCELSIOR_SSH_HOST`), which is why this is a guard and
not a one-line fix.

## What it does and does not flag

Only defaults that **name an environment** — a URL or an `xcelsior.ca` host.
A tuning knob defaulting to `30` is fine unmapped: the number means the same
thing everywhere. There are ~278 unmapped `XCELSIOR_*` vars and flagging all of
them would be noise that trains people to ignore this.

Both sides are derived: the reads from the Python source, the mapping from
`docker-compose.yml`. The single literal is `ACKNOWLEDGED_UNMAPPED` — entries
that are genuinely environment-independent or belong to scripts that never run
in a container.
"""

from __future__ import annotations

import re

from tests._source_tree import REPO, iter_source_files, read_source

COMPOSE = REPO / "docker-compose.yml"

#: Matches `os.environ.get("XCELSIOR_X", "literal")` and `os.getenv(...)`.
_READ = re.compile(
    r'os\.(?:environ\.get|getenv)\(\s*["\'](XCELSIOR_[A-Z0-9_]+)["\']\s*,\s*["\']([^"\']*)["\']'
)
_MAPPED = re.compile(r"^\s*(XCELSIOR_[A-Z0-9_]+):", re.M)

#: Reads that name an environment but do not need mapping, with the reason.
#:
#: Not a list of things to ignore — a list of judgements someone made. A new
#: entry should be a deliberate act.
ACKNOWLEDGED_UNMAPPED = {
    # Loopback: the same address inside every container, in every environment.
    "XCELSIOR_MCP_HEALTH_URL": "loopback readyz probe, identical everywhere",
    "XCELSIOR_LIVE_VLLM_URL": "operator script, never runs in a container",
    "XCELSIOR_TEST_BASE": "test script, never runs in a container",
    "XCELSIOR_DEMO_EMAIL": "seed script, never runs in a container",
    "XCELSIOR_REVIEWER_EMAIL": "seed script, never runs in a container",
}

#: Migrations run once against a database and never read a URL to talk to
#: themselves, so a literal there is not an environment leak.
_EXTRA_EXCLUDED = ("migrations/",)


def _environment_specific_reads() -> dict[str, tuple[str, str]]:
    """{var: (default, where)} for defaults that name a host or a URL.

    Walks via `tests._source_tree.iter_source_files`, not `rglob`: it excludes
    the macOS AppleDouble sidecars that broke four gates at once with a
    `UnicodeDecodeError` naming neither the sidecar nor the gate's subject. A
    gate that has to remember to skip junk is a gate that will forget.
    """
    found: dict[str, tuple[str, str]] = {}
    for path, rel in iter_source_files(exclude_prefixes=_EXTRA_EXCLUDED):
        source = read_source(path)
        for match in _READ.finditer(source):
            name, default = match.group(1), match.group(2)
            if default.startswith("http") or "xcelsior.ca" in default:
                line = source[: match.start()].count("\n") + 1
                found.setdefault(name, (default, f"{rel}:{line}"))
    return found


def _mapped_in_compose() -> set[str]:
    return set(_MAPPED.findall(COMPOSE.read_text(encoding="utf-8")))


def test_the_scan_finds_something_on_both_sides():
    """Calibration. Two empty sets make the assertion below vacuous."""
    reads = _environment_specific_reads()
    mapped = _mapped_in_compose()
    assert len(reads) >= 5, (
        f"only {len(reads)} environment-specific defaults found; the regex has "
        "stopped matching and this file now asserts nothing"
    )
    assert len(mapped) >= 100, (
        f"only {len(mapped)} vars parsed from docker-compose.yml; the mapping "
        "regex broke and every var would look unmapped"
    )


def test_no_production_literal_default_is_unreachable_from_compose():
    """The ratchet.

    Red means a var reads an environment-specific default and the container
    cannot be told otherwise — so a non-production deployment silently gets
    production's value. Map it in the compose anchor with its current default
    (production behaviour is then unchanged), or add it to
    `ACKNOWLEDGED_UNMAPPED` with the reason it does not need mapping.
    """
    mapped = _mapped_in_compose()
    offenders = {
        name: where
        for name, (default, where) in _environment_specific_reads().items()
        if name not in mapped and name not in ACKNOWLEDGED_UNMAPPED
    }
    assert not offenders, (
        "these read an environment-specific default and are mapped in no "
        "compose file, so a non-production deployment silently gets "
        f"production's value:\n"
        + "\n".join(f"  {name} at {where}" for name, where in sorted(offenders.items()))
    )


def test_the_acknowledged_list_does_not_rot():
    """An entry that no longer exists is a stale excuse, not a judgement."""
    reads = _environment_specific_reads()
    stale = sorted(set(ACKNOWLEDGED_UNMAPPED) - set(reads))
    assert not stale, (
        f"{stale} are in ACKNOWLEDGED_UNMAPPED but no longer read an "
        "environment-specific default. Remove them."
    )

    now_mapped = sorted(set(ACKNOWLEDGED_UNMAPPED) & _mapped_in_compose())
    assert not now_mapped, (
        f"{now_mapped} are both acknowledged-as-unmapped and mapped. Remove "
        "them from ACKNOWLEDGED_UNMAPPED — the mapping is the better fix and "
        "the list should say so."
    )


def test_the_var_from_issue_24_is_no_longer_the_only_name_that_works():
    """`XCELSIOR_PUBLIC_URL` is not mapped, and does not need to be.

    The fix was to read the *plumbed* name — `XCELSIOR_BASE_URL`, which staging
    already sets — rather than add a second name for one concept to the anchor.
    `XCELSIOR_PUBLIC_URL` still wins when explicitly set, so it never appears
    here: it is read with `or`, not with a literal default.
    """
    auth = (REPO / "routes" / "auth.py").read_text(encoding="utf-8")
    assert 'os.environ.get("XCELSIOR_PUBLIC_URL", "https://xcelsior.ca")' not in auth, (
        "the Quick Connect connector URL is back to reading only the unmapped "
        "name, so staging would hand out a production URL again"
    )
    assert 'os.environ.get("XCELSIOR_BASE_URL")' in auth
