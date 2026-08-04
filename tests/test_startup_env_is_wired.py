"""Every variable production startup validation reads must reach the container.

`docker-compose.yml` declares **no `env_file:`** on any service. A variable
reaches a container only by being named in the `x-api-environment` anchor, so a
value in `.env` that nothing maps is set on the host and unset in the process
that reads it — with no error anywhere, because `.env` is still consumed for
`${...}` interpolation and the file looks correctly configured.

That is not hypothetical. On 2026-08-04 the production deploy applied its
migrations and then failed to boot:

    StartupValidationError: production startup validation failed —
      compat_session_secret_missing: XCELSIOR_COMPAT_SESSION_SECRET is unset …
      audit_signing_key_default: Neither XCELSIOR_AUDIT_SIGNING_KEYS nor
        XCELSIOR_AUDIT_SIGNING_KEY is set …

Both were set in `.env`, correctly, with high-entropy values. Neither was named
in the anchor. The API had been down for hours and the secrets were sitting in
the file the whole time.

The rule this gate enforces is total on purpose: if
`control_plane/startup_validation.py` reads a variable, the container must
receive it. No exemption for variables whose default happens to be safe —
`XCELSIOR_VOLUME_PRIVILEGE` defaults to `host_ssh` and passes, but an operator
setting it to anything else would have been ignored the same way. An exemption
list here would be a list of variables allowed to lie about being configured.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSE = PROJECT_ROOT / "docker-compose.yml"
VALIDATION = PROJECT_ROOT / "control_plane" / "startup_validation.py"

#: Services whose environment comes from the shared `*api-env` anchor. Each one
#: runs code that calls `validate_startup` or the modules it validates, so a
#: variable missing from any of them is the same defect in a different process.
ANCHOR_SERVICES = ("api", "api-blue", "bg-worker")

_ENV_READ = re.compile(r'os\.environ\.get\(\s*"(XCELSIOR_[A-Z0-9_]+)"')


def _vars_read_by_validation() -> set[str]:
    return set(_ENV_READ.findall(VALIDATION.read_text()))


def _service_env(service: str) -> set[str]:
    compose = yaml.safe_load(COMPOSE.read_text())
    return set(compose["services"][service]["environment"])


def _unwired(read: set[str], mapped: set[str]) -> list[str]:
    return sorted(read - mapped)


def test_startup_validation_reads_nothing_the_container_cannot_see():
    read = _vars_read_by_validation()
    assert read, "found no XCELSIOR_* reads — the regex stopped matching"

    for service in ANCHOR_SERVICES:
        unwired = _unwired(read, _service_env(service))
        assert not unwired, (
            f"{service} does not receive {unwired}, which "
            "control_plane/startup_validation.py reads. There is no env_file: on "
            "this service, so setting these in .env has no effect on the "
            "container — the value is present on the host and absent in the "
            f"process that checks it. Add them to the x-api-environment anchor."
        )


def test_the_check_fails_on_the_configuration_that_caused_the_outage():
    """The failing arm: the anchor as it stood before 2026-08-04.

    Without this, the test above passes for as long as nobody breaks it and has
    never been observed to fail — which is the property that let the original
    gap ship.
    """
    read = {"XCELSIOR_COMPAT_SESSION_SECRET", "XCELSIOR_AUDIT_SIGNING_KEYS", "XCELSIOR_ENV"}
    mapped_before = {"XCELSIOR_ENV"}
    unwired = _unwired(read, mapped_before)
    assert unwired == [
        "XCELSIOR_AUDIT_SIGNING_KEYS",
        "XCELSIOR_COMPAT_SESSION_SECRET",
    ], unwired


def test_no_service_reintroduces_env_file_as_a_substitute():
    """`env_file:` would make this gate vacuous rather than satisfied.

    Adding `env_file: .env` would wire everything at once and pass the check
    above — while also handing every container the full production secret set,
    including services that have no business holding it. If that trade is ever
    made deliberately, it should be a deliberate edit here too.
    """
    compose = yaml.safe_load(COMPOSE.read_text())
    with_env_file = sorted(
        name for name, svc in compose["services"].items() if isinstance(svc, dict) and "env_file" in svc
    )
    assert not with_env_file, (
        f"services now use env_file: {with_env_file} — that grants the whole "
        "secret set to every listed service. Map variables explicitly, or "
        "update this test with the reasoning if the trade is intended."
    )


# ── Duplicate keys, which `yaml.safe_load` hides ───────────────────────


def _duplicate_keys(path: Path) -> list[tuple[str, int, int]]:
    """Every duplicated mapping key, as (key, first line, second line).

    `yaml.safe_load` silently keeps the last of a duplicated pair, so a compose
    file with two definitions of the same variable parses cleanly in these tests
    and is **rejected outright** by docker:

        failed to parse docker-compose.yml: yaml: construct errors:
          line 268: mapping key "XCELSIOR_STRIPE_SANDBOX_WEBHOOK_SECRET"
          already defined at line 84

    That happened on 2026-08-04: a variable was added to the anchor without
    checking whether it was already there, every test here passed, and the deploy
    then failed at the build step because `docker compose` could not read the file
    at all. The gate that was meant to keep the container's environment honest was
    blind to the one error that stops the environment existing.

    Merge keys (`<<`) are skipped: repeating one in a mapping is legal YAML and is
    how this file composes its anchors.
    """
    found: list[tuple[str, int, int]] = []

    def walk(node) -> None:
        if isinstance(node, yaml.MappingNode):
            seen: dict[str, int] = {}
            for key, value in node.value:
                if isinstance(key, yaml.ScalarNode) and key.value != "<<":
                    line = key.start_mark.line + 1
                    if key.value in seen:
                        found.append((key.value, seen[key.value], line))
                    else:
                        seen[key.value] = line
                walk(value)
        elif isinstance(node, yaml.SequenceNode):
            for item in node.value:
                walk(item)

    walk(yaml.compose(path.read_text()))
    # An anchor merged into four services is visited four times, so the same
    # duplicate is found repeatedly. Report each once — a gate that prints one
    # finding four times reads as four problems.
    return sorted(set(found))


def test_compose_has_no_duplicate_keys():
    dupes = _duplicate_keys(COMPOSE)
    assert not dupes, (
        "docker-compose.yml defines the same key twice, which docker refuses to "
        "parse even though yaml.safe_load accepts it: "
        + "; ".join(f"{k!r} at lines {a} and {b}" for k, a, b in dupes)
    )


def test_the_duplicate_check_catches_one():
    """The failing arm, on the exact shape that broke the deploy."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as fh:
        fh.write("services:\n  api:\n    environment:\n      A: 1\n      A: 2\n")
        probe = Path(fh.name)
    try:
        assert [k for k, _, _ in _duplicate_keys(probe)] == ["A"]
    finally:
        probe.unlink()


def test_merge_keys_are_not_reported_as_duplicates():
    """`<<` legitimately repeats; flagging it would make the gate unusable here."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as fh:
        fh.write("a: &a {x: 1}\nb: &b {y: 2}\nc:\n  <<: *a\n  <<: *b\n")
        probe = Path(fh.name)
    try:
        assert _duplicate_keys(probe) == []
    finally:
        probe.unlink()
