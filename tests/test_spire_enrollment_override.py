"""The production override has to enroll whichever API slot is live.

On 2026-10-12 the deploy patched only `api`. Nginx had already moved traffic
to `api-blue`, so the live container had neither the SPIRE socket nor the CLI.
"""

from __future__ import annotations

import importlib.util
import stat
from pathlib import Path

import yaml

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "api" / "scripts" / "spire_enrollment_override.py"
)
spec = importlib.util.spec_from_file_location("spire_enrollment_override", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ENV_KEYS = tuple(line.split(":", 1)[0] for line in mod.ENV_LINES)
VOLUMES = mod.VOL_LINES

PROD_SHAPE = """\
# Production - uses host PostgreSQL
services:
  api:
    environment:
      XCELSIOR_POSTGRES_DSN: postgres://example
      # Host enrollment. Values stay in /opt/xcelsior/.env; this names them so
      # they reach the container. Compose does not inject .env on its own.
      XCELSIOR_SPIRE_SERVER_CMD: ${XCELSIOR_SPIRE_SERVER_CMD:-}
      XCELSIOR_SPIRE_SERVER_SOCKET: ${XCELSIOR_SPIRE_SERVER_SOCKET:-}
      XCELSIOR_SPIRE_SERVER_ADDRESS: ${XCELSIOR_SPIRE_SERVER_ADDRESS:-}
      XCELSIOR_SPIRE_SERVER_PORT: ${XCELSIOR_SPIRE_SERVER_PORT:-8095}
      XCELSIOR_HEADSCALE_URL: ${XCELSIOR_HEADSCALE_URL:-}
      XCELSIOR_HEADSCALE_API_KEY: ${XCELSIOR_HEADSCALE_API_KEY:-}
      XCELSIOR_HEADSCALE_USER_ID: ${XCELSIOR_HEADSCALE_USER_ID:-}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - /var/lib/xcelsior-spire/sock:/run/xcelsior-spire
      - /usr/local/lib/xcelsior/spire-server:/usr/local/bin/spire-server:ro

  scheduler-worker:
    environment:
      XCELSIOR_POSTGRES_DSN: postgres://example

  api-blue:
    environment:
      XCELSIOR_SPIRE_SERVER_CMD: ${XCELSIOR_SPIRE_SERVER_CMD:-}
      XCELSIOR_SPIRE_SERVER_SOCKET: ${XCELSIOR_SPIRE_SERVER_SOCKET:-}
      XCELSIOR_SPIRE_SERVER_ADDRESS: ${XCELSIOR_SPIRE_SERVER_ADDRESS:-}
      XCELSIOR_SPIRE_SERVER_PORT: ${XCELSIOR_SPIRE_SERVER_PORT:-8095}
      XCELSIOR_HEADSCALE_URL: ${XCELSIOR_HEADSCALE_URL:-}
      XCELSIOR_HEADSCALE_API_KEY: ${XCELSIOR_HEADSCALE_API_KEY:-}
      XCELSIOR_HEADSCALE_USER_ID: ${XCELSIOR_HEADSCALE_USER_ID:-}
    volumes:
      - /var/lib/xcelsior-spire/sock:/run/xcelsior-spire
      - /usr/local/lib/xcelsior/spire-server:/usr/local/bin/spire-server:ro
"""

API_ONLY = """\
# Production - uses host PostgreSQL
services:
  api:
    environment:
      XCELSIOR_POSTGRES_DSN: postgres://example
      XCELSIOR_SPIRE_SERVER_CMD: ${XCELSIOR_SPIRE_SERVER_CMD:-}
      XCELSIOR_SPIRE_SERVER_SOCKET: ${XCELSIOR_SPIRE_SERVER_SOCKET:-}
      XCELSIOR_SPIRE_SERVER_ADDRESS: ${XCELSIOR_SPIRE_SERVER_ADDRESS:-}
      XCELSIOR_SPIRE_SERVER_PORT: ${XCELSIOR_SPIRE_SERVER_PORT:-8095}
      XCELSIOR_HEADSCALE_URL: ${XCELSIOR_HEADSCALE_URL:-}
      XCELSIOR_HEADSCALE_API_KEY: ${XCELSIOR_HEADSCALE_API_KEY:-}
      XCELSIOR_HEADSCALE_USER_ID: ${XCELSIOR_HEADSCALE_USER_ID:-}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - /var/lib/xcelsior-spire/sock:/run/xcelsior-spire
      - /usr/local/lib/xcelsior/spire-server:/usr/local/bin/spire-server:ro

  scheduler-worker:
    environment:
      XCELSIOR_POSTGRES_DSN: postgres://example
"""

BOTH_BARE = """\
services:
  api:
    environment:
      XCELSIOR_POSTGRES_DSN: postgres://example
    extra_hosts:
      - "host.docker.internal:host-gateway"
  scheduler-worker:
    environment:
      XCELSIOR_POSTGRES_DSN: postgres://scheduler
  api-blue:
    environment:
      XCELSIOR_POSTGRES_DSN: postgres://blue
"""


def _services(text: str) -> dict:
    loaded = yaml.safe_load(text)
    return loaded["services"]


def _wired(service: dict) -> None:
    env = service["environment"]
    for key in ENV_KEYS:
        assert key in env
    for mount in VOLUMES:
        assert mount in service["volumes"]


def test_a_fully_wired_override_is_left_byte_for_byte():
    updated, touched = mod.patch_override(PROD_SHAPE)
    assert touched == []
    assert updated == PROD_SHAPE


JSON_WIRED = """\
{
  "services": {
    "api": {
      "environment": {
        "XCELSIOR_POSTGRES_DSN": "postgres://example",
        "XCELSIOR_SPIRE_SERVER_CMD": "${XCELSIOR_SPIRE_SERVER_CMD:-}",
        "XCELSIOR_SPIRE_SERVER_SOCKET": "${XCELSIOR_SPIRE_SERVER_SOCKET:-}",
        "XCELSIOR_SPIRE_SERVER_ADDRESS": "${XCELSIOR_SPIRE_SERVER_ADDRESS:-}",
        "XCELSIOR_SPIRE_SERVER_PORT": "${XCELSIOR_SPIRE_SERVER_PORT:-8095}",
        "XCELSIOR_HEADSCALE_URL": "${XCELSIOR_HEADSCALE_URL:-}",
        "XCELSIOR_HEADSCALE_API_KEY": "${XCELSIOR_HEADSCALE_API_KEY:-}",
        "XCELSIOR_HEADSCALE_USER_ID": "${XCELSIOR_HEADSCALE_USER_ID:-}"
      },
      "volumes": [
        "/var/lib/xcelsior-spire/sock:/run/xcelsior-spire",
        "/usr/local/lib/xcelsior/spire-server:/usr/local/bin/spire-server:ro"
      ]
    },
    "api-blue": {
      "environment": {
        "XCELSIOR_POSTGRES_DSN": "postgres://blue",
        "XCELSIOR_SPIRE_SERVER_CMD": "${XCELSIOR_SPIRE_SERVER_CMD:-}",
        "XCELSIOR_SPIRE_SERVER_SOCKET": "${XCELSIOR_SPIRE_SERVER_SOCKET:-}",
        "XCELSIOR_SPIRE_SERVER_ADDRESS": "${XCELSIOR_SPIRE_SERVER_ADDRESS:-}",
        "XCELSIOR_SPIRE_SERVER_PORT": "${XCELSIOR_SPIRE_SERVER_PORT:-8095}",
        "XCELSIOR_HEADSCALE_URL": "${XCELSIOR_HEADSCALE_URL:-}",
        "XCELSIOR_HEADSCALE_API_KEY": "${XCELSIOR_HEADSCALE_API_KEY:-}",
        "XCELSIOR_HEADSCALE_USER_ID": "${XCELSIOR_HEADSCALE_USER_ID:-}"
      },
      "volumes": [
        "/var/lib/xcelsior-spire/sock:/run/xcelsior-spire",
        "/usr/local/lib/xcelsior/spire-server:/usr/local/bin/spire-server:ro"
      ]
    }
  }
}
"""


def test_json_override_already_wired_is_left_unchanged():
    updated, touched = mod.patch_override(JSON_WIRED)
    assert touched == []
    assert updated == JSON_WIRED


def test_json_override_gains_the_missing_slot_and_keeps_its_dsn():
    text = """\
{
  "services": {
    "api": {
      "environment": {"XCELSIOR_POSTGRES_DSN": "postgres://example"},
      "volumes": []
    },
    "scheduler-worker": {
      "environment": {"XCELSIOR_POSTGRES_DSN": "postgres://scheduler"}
    }
  }
}
"""
    updated, touched = mod.patch_override(text)
    assert touched == ["api", "api-blue"]
    services = _services(updated)
    _wired(services["api"])
    _wired(services["api-blue"])
    assert services["api"]["environment"]["XCELSIOR_POSTGRES_DSN"] == "postgres://example"
    assert services["scheduler-worker"]["environment"] == {
        "XCELSIOR_POSTGRES_DSN": "postgres://scheduler"
    }


def test_volume_lines_on_api_do_not_count_for_api_blue():
    updated, touched = mod.patch_override(API_ONLY)
    assert touched == ["api-blue"]
    services = _services(updated)
    _wired(services["api-blue"])
    assert list(services["scheduler-worker"]["environment"]) == ["XCELSIOR_POSTGRES_DSN"]
    assert "volumes" not in services["scheduler-worker"]
    assert services["api"]["extra_hosts"] == ["host.docker.internal:host-gateway"]
    again, again_touched = mod.patch_override(updated)
    assert again_touched == []
    assert again == updated


def test_both_slots_gain_wiring_and_the_scheduler_does_not():
    updated, touched = mod.patch_override(BOTH_BARE)
    assert touched == ["api", "api-blue"]
    services = _services(updated)
    _wired(services["api"])
    _wired(services["api-blue"])
    assert services["api-blue"]["environment"]["XCELSIOR_POSTGRES_DSN"] == "postgres://blue"
    assert list(services["scheduler-worker"]["environment"]) == ["XCELSIOR_POSTGRES_DSN"]
    assert services["scheduler-worker"]["environment"]["XCELSIOR_POSTGRES_DSN"] == (
        "postgres://scheduler"
    )


def test_each_slot_receives_only_the_piece_it_lacks():
    text = """\
services:
  api:
    environment:
      XCELSIOR_SPIRE_SERVER_CMD: ${XCELSIOR_SPIRE_SERVER_CMD:-}
      XCELSIOR_SPIRE_SERVER_SOCKET: ${XCELSIOR_SPIRE_SERVER_SOCKET:-}
      XCELSIOR_SPIRE_SERVER_ADDRESS: ${XCELSIOR_SPIRE_SERVER_ADDRESS:-}
      XCELSIOR_SPIRE_SERVER_PORT: ${XCELSIOR_SPIRE_SERVER_PORT:-8095}
      XCELSIOR_HEADSCALE_URL: ${XCELSIOR_HEADSCALE_URL:-}
      XCELSIOR_HEADSCALE_API_KEY: ${XCELSIOR_HEADSCALE_API_KEY:-}
      XCELSIOR_HEADSCALE_USER_ID: ${XCELSIOR_HEADSCALE_USER_ID:-}
  api-blue:
    volumes:
      - /var/lib/xcelsior-spire/sock:/run/xcelsior-spire
      - /usr/local/lib/xcelsior/spire-server:/usr/local/bin/spire-server:ro
"""
    updated, touched = mod.patch_override(text)
    assert touched == ["api", "api-blue"]
    services = _services(updated)
    assert services["api"]["volumes"] == list(VOLUMES)
    for key in ENV_KEYS:
        assert key in services["api-blue"]["environment"]
    assert services["api-blue"]["volumes"] == list(VOLUMES)


def test_merge_env_adds_missing_keys_and_keeps_existing_values():
    enrollment = (
        "# generated\n"
        "XCELSIOR_SPIRE_SERVER_ADDRESS=100.64.0.1\n"
        "XCELSIOR_HEADSCALE_API_KEY=from-enrollment\n"
        "\n"
    )
    existing = "XCELSIOR_HEADSCALE_API_KEY=already-set\n"
    merged, added = mod.merge_env(enrollment, existing)
    assert added == ["XCELSIOR_SPIRE_SERVER_ADDRESS"]
    assert "XCELSIOR_HEADSCALE_API_KEY=already-set" in merged
    assert "from-enrollment" not in merged
    assert "XCELSIOR_SPIRE_SERVER_ADDRESS=100.64.0.1" in merged
    second, second_added = mod.merge_env(enrollment, merged)
    assert second_added == []
    assert second == merged


def test_main_skips_a_host_that_has_not_enrolled(tmp_path: Path, capsys):
    override = tmp_path / "docker-compose.override.yml"
    override.write_text(API_ONLY)
    env = tmp_path / ".env"
    env.write_text("UNRELATED=1\n")
    code = mod.main([str(tmp_path / "missing.env"), str(override), str(env)])
    assert code == 0
    assert "leaving compose unchanged" in capsys.readouterr().out
    assert override.read_text() == API_ONLY


def test_main_writes_the_live_slot_and_leaves_existing_env_values(tmp_path: Path, capsys):
    enrollment = tmp_path / "api-enrollment.env"
    enrollment.write_text(
        "XCELSIOR_SPIRE_SERVER_ADDRESS=100.64.0.1\n" "XCELSIOR_HEADSCALE_API_KEY=from-enrollment\n"
    )
    override = tmp_path / "docker-compose.override.yml"
    override.write_text(API_ONLY)
    env = tmp_path / ".env"
    env.write_text("XCELSIOR_HEADSCALE_API_KEY=keep-me\n")
    code = mod.main([str(enrollment), str(override), str(env)])
    assert code == 0
    captured = capsys.readouterr().out
    assert "spire_override updated" in captured
    assert "spire_services api-blue" in captured
    assert "spire_env_keys_added XCELSIOR_SPIRE_SERVER_ADDRESS" in captured
    assert "from-enrollment" not in env.read_text()
    assert "keep-me" in env.read_text()
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    services = _services(override.read_text())
    _wired(services["api"])
    _wired(services["api-blue"])
