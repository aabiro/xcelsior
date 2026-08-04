"""The startup gate must refuse the boot, not log about it.

`control_plane/startup_validation.py` was made fail-closed: `enforcement_enabled()`
is `not env_config.is_relaxed_env()`, so any `error` finding raises
`StartupValidationError` everywhere except an explicitly named development
environment. That is correct and tested inside the module.

Its caller was not read against it. `api.lifespan` held:

    try:
        from control_plane.startup_validation import validate_startup

        for finding in validate_startup():
            log.warning(...)
    except Exception as exc:
        if type(exc).__name__ == "StartupValidationError":
            log.critical(...)
            raise
        log.warning("startup validation could not run: %s", exc)

Three defects compounding, and the fix removes all three rather than patching
each:

1. The re-raise compared a **class name string** — the same shape as
   `os.environ.get("XCELSIOR_ENV") == "production"`, one layer up. A rename, a
   subclass, or a wrapped exception downgraded a hard boot refusal to a log
   line.
2. The import sat **inside** the `try`, so an `ImportError` — a syntax error, a
   circular import introduced later, a missing dependency in a slim image — was
   caught by the bare `except` and the process booted with **zero** configuration
   validation. The comment directly above said "must fail the deploy, not
   discover it in prod".
3. The only test of the wiring was

       source = inspect.getsource(api.lifespan)
       assert "validate_startup" in source
       assert "StartupValidationError" in source

   A substring check. Both strings appear in the swallowing version above and
   would survive deleting the `raise`, so it passed whether the boot refused or
   not. Fourth "cannot fail" assertion found on this branch, and it was guarding
   the gate that guards the other three.

The import is now at module scope and the call is bare. Nothing is caught, so
nothing can be swallowed, and the class-name comparison is deleted rather than
corrected.

`test_a_validator_failure_that_is_not_the_named_exception_still_refuses` is the
one that would have caught the defect: under the old code it passed straight
through the `if type(exc).__name__ ==` branch into `log.warning` and the boot
continued.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import api
from control_plane import startup_validation
from control_plane.startup_validation import Finding, StartupValidationError

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _always_raises(exc: BaseException):
    def _fn(*args, **kwargs):
        raise exc

    return _fn


def _pin_validator(monkeypatch, exc: BaseException) -> None:
    """Make `validate_startup` raise, wherever the lifespan resolves it from.

    Both targets are patched deliberately. With the import at module scope
    `api.validate_startup` is the binding the lifespan uses; with the import
    inside the function it re-resolves from `control_plane.startup_validation`
    on every call. Patching one only would make this test pass for a reason
    unrelated to what it claims to check on whichever version it did not cover.
    """
    boom = _always_raises(exc)
    monkeypatch.setattr(startup_validation, "validate_startup", boom)
    monkeypatch.setattr(api, "validate_startup", boom, raising=False)


async def _run_startup() -> None:
    """Enter the real lifespan. Returning means the process would serve traffic."""
    async with api.lifespan(api.app):
        pass


async def test_an_error_finding_refuses_the_boot(monkeypatch):
    """The gate's headline claim, asserted by booting rather than by reading."""
    planted = Finding(
        code="probe",
        severity="error",
        message="planted by test_startup_gate_refuses_boot",
        remediation="none — this finding exists to prove the boot refuses",
    )
    _pin_validator(monkeypatch, StartupValidationError([planted]))

    with pytest.raises(StartupValidationError):
        await _run_startup()


async def test_a_validator_failure_that_is_not_the_named_exception_still_refuses(
    monkeypatch,
):
    """The one that would have caught it.

    `type(exc).__name__ == "StartupValidationError"` admitted exactly one
    spelling. Anything else — and an `ImportError` from the import that sat
    inside the same `try` was the realistic one — reached `log.warning` and the
    boot continued unvalidated.

    A validator that cannot run is not a validator that passed.
    """
    _pin_validator(monkeypatch, ImportError("control_plane.startup_validation is broken"))

    with pytest.raises(ImportError):
        await _run_startup()


def _lifespan_node() -> ast.AsyncFunctionDef:
    tree = ast.parse((ROOT / "api.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan":
            return node
    raise AssertionError("api.lifespan not found")


def test_the_validator_is_imported_at_module_scope():
    """An import inside the gate is an import the gate can swallow.

    Structural rather than textual: the two boot tests above cannot observe an
    `ImportError` from the import statement itself, because by the time a test
    runs, `api` has already been imported successfully. This is what pins the
    second half of the fix.
    """
    inner_imports = [
        alias.name
        for node in ast.walk(_lifespan_node())
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert "validate_startup" not in inner_imports, (
        "validate_startup is imported inside api.lifespan. An ImportError there "
        "is catchable, and a caught ImportError means the process boots with no "
        "configuration validation at all — import it at module scope so the "
        "failure kills the process before it can serve traffic"
    )

    module_level = [
        alias.name
        for node in ast.parse((ROOT / "api.py").read_text(encoding="utf-8")).body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    ]
    assert "validate_startup" in module_level, (
        "validate_startup must be imported at module scope in api.py"
    )


def test_the_validator_call_is_not_wrapped_in_a_try():
    """Nothing may catch the refusal.

    A `try` around the call is how the swallow returns, whether or not it
    re-raises correctly today — the correctness would live in an `except` body
    that a later edit can weaken without any test noticing.
    """
    guarded = [
        node
        for node in ast.walk(_lifespan_node())
        if isinstance(node, ast.Try)
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "validate_startup"
    ]
    assert not guarded, (
        "the validate_startup call is inside a try block in api.lifespan; the "
        "refusal must propagate uncaught"
    )
