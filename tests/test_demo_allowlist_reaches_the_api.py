"""The demo login's per-deployment settings reach the API container.

`docker-compose.yml` has no `env_file:`; a variable reaches a container only by
being named in the `x-api-environment` anchor. None of the demo settings were,
so `XCELSIOR_DEMO_IP_WHITELIST_EXTRA` in production's `.env` (the documented
way to admit the owner's network) was set on the host and absent in the API:
only the built-in tailnet and LAN defaults ever applied.
"""

from __future__ import annotations

from pathlib import Path

import yaml

COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"


def test_demo_settings_are_named_in_the_api_environment() -> None:
    api_env = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["x-api-environment"]
    for name in (
        "XCELSIOR_DEMO_ENABLED",
        "XCELSIOR_DEMO_IP_WHITELIST",
        "XCELSIOR_DEMO_IP_WHITELIST_EXTRA",
    ):
        assert name in api_env, f"{name} never reaches the API container"
        assert f"${{{name}" in str(api_env[name]), f"{name} is not taken from .env"
