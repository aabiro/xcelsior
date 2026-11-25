"""Account emails must read naturally when the account has no name.

On 2026-11-24 a sign-up with no name got "Verify your email," and
"Welcome to Xcelsior, !" with "Hi ," in the body, under a "[Xcelsior] "
subject prefix. `user.get("name", fallback)` returns "" when the key exists
with an empty value, so the fallback never ran.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

from routes.teams import greeting_name, hi_line  # noqa: E402
from provider_expectations import provider_welcome_email_text  # noqa: E402


def test_empty_or_missing_name_greets_without_one():
    for user in ({"name": ""}, {"name": "   "}, {}, None, {"name": None}):
        assert greeting_name(user) == ""
        assert hi_line(greeting_name(user)) == "Hi there,"


def test_a_real_name_is_used_and_trimmed():
    assert hi_line(greeting_name({"name": "  Ada "})) == "Hi Ada,"


def test_provider_welcome_never_reads_hi_comma():
    assert provider_welcome_email_text("").startswith("Hi there,\n\n")
    assert provider_welcome_email_text("Ada").startswith("Hi Ada,\n\n")


def test_no_subject_prefix_and_no_cad_in_the_welcome():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    teams = (root / "api" / "routes" / "teams.py").read_text()
    scheduler = (root / "workers" / "scheduler.py").read_text()
    auth = (root / "api" / "routes" / "auth.py").read_text()
    assert 'f"[Xcelsior] {subject}"' not in teams + scheduler
    assert "billed in CAD" not in auth
    assert "Hi {display_name}," not in auth
    # The logo sits beside the "Xcelsior" wordmark; alt text repeated it
    # ("Xcelsior Xcelsior") whenever images were blocked.
    assert 'alt="Xcelsior"' not in teams
