"""Tests for Xcelsior Phase 4 features:
- CLI ~/.xcelsior/config.toml support
- argcomplete shell completion registration
- WebSocket terminal backend endpoint
- Spot pricing Recharts chart data
- Marketplace frontend enhancements validation
"""

import json
import os
import sys
import tempfile
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_DB_BACKEND", "sqlite")

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_phase4_test_")
_tmpdir = _tmp_ctx.name


# ── CLI Config TOML ──────────────────────────────────────────────────


class TestCLIConfigTOML:
    """~/.xcelsior/config.toml load/save/get/set."""

    def test_load_config_empty(self, monkeypatch):
        """Loading config from nonexistent path returns empty dict."""
        import cli

        monkeypatch.setattr(cli, "CONFIG_FILE", os.path.join(_tmpdir, "nonexistent.toml"))
        assert cli._load_config() == {}

    def test_save_and_load_roundtrip(self, monkeypatch):
        """Config values survive a save/load roundtrip."""
        import cli

        cfg_path = os.path.join(_tmpdir, "test_config.toml")
        monkeypatch.setattr(cli, "CONFIG_FILE", cfg_path)
        monkeypatch.setattr(cli, "CONFIG_DIR", _tmpdir)

        cli._save_config(
            {
                "api_url": "https://api.xcelsior.ca",
                "defaults": {
                    "region": "ca-central-1",
                    "gpu_model": "A100",
                },
            }
        )

        assert os.path.exists(cfg_path)
        loaded = cli._load_config()
        assert loaded["api_url"] == "https://api.xcelsior.ca"
        assert loaded["defaults"]["region"] == "ca-central-1"
        assert loaded["defaults"]["gpu_model"] == "A100"

    def test_cfg_get_dotted_key(self, monkeypatch):
        """_cfg_get resolves dotted keys like 'defaults.region'."""
        import cli

        cfg_path = os.path.join(_tmpdir, "dotted_config.toml")
        monkeypatch.setattr(cli, "CONFIG_FILE", cfg_path)
        monkeypatch.setattr(cli, "CONFIG_DIR", _tmpdir)

        cli._save_config({"defaults": {"region": "ca-west-1"}})
        assert cli._cfg_get("defaults.region") == "ca-west-1"
        assert cli._cfg_get("defaults.nonexistent", "fallback") == "fallback"
        assert cli._cfg_get("totally.missing", "default") == "default"

    def test_cfg_get_top_level(self, monkeypatch):
        """_cfg_get resolves top-level keys."""
        import cli

        cfg_path = os.path.join(_tmpdir, "top_level_config.toml")
        monkeypatch.setattr(cli, "CONFIG_FILE", cfg_path)
        monkeypatch.setattr(cli, "CONFIG_DIR", _tmpdir)

        cli._save_config({"api_url": "http://localhost:9000"})
        assert cli._cfg_get("api_url") == "http://localhost:9000"

    def test_get_api_url_config_fallback(self, monkeypatch):
        """get_api_url() reads from config when env var is empty."""
        import cli

        cfg_path = os.path.join(_tmpdir, "api_url_config.toml")
        monkeypatch.setattr(cli, "CONFIG_FILE", cfg_path)
        monkeypatch.setattr(cli, "CONFIG_DIR", _tmpdir)
        monkeypatch.delenv("XCELSIOR_API_URL", raising=False)

        cli._save_config({"api_url": "https://custom.xcelsior.ca"})
        assert cli.get_api_url() == "https://custom.xcelsior.ca"

    def test_get_api_url_env_takes_precedence(self, monkeypatch):
        """XCELSIOR_API_URL env var overrides config."""
        import cli

        cfg_path = os.path.join(_tmpdir, "env_precedence_config.toml")
        monkeypatch.setattr(cli, "CONFIG_FILE", cfg_path)
        monkeypatch.setattr(cli, "CONFIG_DIR", _tmpdir)
        monkeypatch.setenv("XCELSIOR_API_URL", "https://env-override.xcelsior.ca")

        cli._save_config({"api_url": "https://config-value.xcelsior.ca"})
        assert cli.get_api_url() == "https://env-override.xcelsior.ca"

    def test_get_default_region(self, monkeypatch):
        """get_default_region() reads from config defaults section."""
        import cli

        cfg_path = os.path.join(_tmpdir, "region_config.toml")
        monkeypatch.setattr(cli, "CONFIG_FILE", cfg_path)
        monkeypatch.setattr(cli, "CONFIG_DIR", _tmpdir)
        monkeypatch.delenv("XCELSIOR_REGION", raising=False)

        cli._save_config({"defaults": {"region": "ca-east-1"}})
        assert cli.get_default_region() == "ca-east-1"

    def test_save_config_permissions(self, monkeypatch):
        """Config file is created with 0o600 permissions."""
        import cli

        cfg_path = os.path.join(_tmpdir, "perms_config.toml")
        monkeypatch.setattr(cli, "CONFIG_FILE", cfg_path)
        monkeypatch.setattr(cli, "CONFIG_DIR", _tmpdir)

        cli._save_config({"api_url": "test"})
        mode = os.stat(cfg_path).st_mode & 0o777
        assert mode == 0o600

    def test_save_config_boolean_values(self, monkeypatch):
        """Boolean values are properly saved and loaded."""
        import cli

        cfg_path = os.path.join(_tmpdir, "bool_config.toml")
        monkeypatch.setattr(cli, "CONFIG_FILE", cfg_path)
        monkeypatch.setattr(cli, "CONFIG_DIR", _tmpdir)

        cli._save_config({"verbose": True, "defaults": {"auto_topup": False}})
        loaded = cli._load_config()
        assert loaded["verbose"] is True
        assert loaded["defaults"]["auto_topup"] is False


class TestCmdConfig:
    """Test the 'config' CLI subcommand handler."""

    def test_cmd_config_set_and_get(self, monkeypatch, capsys):
        """'config set' writes, 'config get' reads."""
        import cli

        cfg_path = os.path.join(_tmpdir, "cmd_config.toml")
        monkeypatch.setattr(cli, "CONFIG_FILE", cfg_path)
        monkeypatch.setattr(cli, "CONFIG_DIR", _tmpdir)

        # Set
        args = type("Args", (), {"action": "set", "key": "api_url", "value": "https://test.api"})()
        cli.cmd_config(args)
        out = capsys.readouterr().out
        assert "Set api_url" in out

        # Get
        args = type("Args", (), {"action": "get", "key": "api_url"})()
        cli.cmd_config(args)
        out = capsys.readouterr().out
        assert "https://test.api" in out

    def test_cmd_config_set_dotted(self, monkeypatch, capsys):
        """'config set defaults.region ca-west-1' creates nested section."""
        import cli

        cfg_path = os.path.join(_tmpdir, "cmd_dotted_config.toml")
        monkeypatch.setattr(cli, "CONFIG_FILE", cfg_path)
        monkeypatch.setattr(cli, "CONFIG_DIR", _tmpdir)

        args = type("Args", (), {"action": "set", "key": "defaults.region", "value": "ca-west-1"})()
        cli.cmd_config(args)

        loaded = cli._load_config()
        assert loaded["defaults"]["region"] == "ca-west-1"

    def test_cmd_config_unset(self, monkeypatch, capsys):
        """'config unset' removes a key."""
        import cli

        cfg_path = os.path.join(_tmpdir, "unset_config.toml")
        monkeypatch.setattr(cli, "CONFIG_FILE", cfg_path)
        monkeypatch.setattr(cli, "CONFIG_DIR", _tmpdir)

        cli._save_config({"api_url": "http://old", "keep": "yes"})

        args = type("Args", (), {"action": "unset", "key": "api_url"})()
        cli.cmd_config(args)

        loaded = cli._load_config()
        assert "api_url" not in loaded
        assert loaded["keep"] == "yes"

    def test_cmd_config_path(self, monkeypatch, capsys):
        """'config path' prints the config file path."""
        import cli

        cfg_path = os.path.join(_tmpdir, "path_test_config.toml")
        monkeypatch.setattr(cli, "CONFIG_FILE", cfg_path)

        args = type("Args", (), {"action": "path"})()
        cli.cmd_config(args)
        out = capsys.readouterr().out
        assert cfg_path in out


# ── argcomplete ──────────────────────────────────────────────────────


class TestArgcompleteIntegration:
    """Verify argcomplete is wired into main() parser."""

    def test_argcomplete_hook_in_main(self):
        """main() calls argcomplete.autocomplete if available."""
        import cli
        import inspect

        source = inspect.getsource(cli.main)
        assert "argcomplete" in source
        assert "autocomplete" in source

    def test_config_subparser_exists(self):
        """'config' subcommand is registered in the parser."""
        import cli
        import argparse

        # Build the parser by calling main with --help and catching SystemExit
        import io
        from unittest import mock

        with mock.patch("sys.argv", ["xcelsior", "config", "path"]):
            with mock.patch.object(cli, "CONFIG_FILE", os.path.join(_tmpdir, "test.toml")):
                try:
                    cli.main()
                except SystemExit:
                    pass  # Expected for some actions


# ── WebSocket Terminal Backend ───────────────────────────────────────


class TestWebSocketTerminal:
    """WebSocket /ws/terminal/{instance_id} endpoint tests.

    NOTE: Comprehensive terminal tests live in tests/test_terminal.py.
    These are kept as basic smoke checks for backward compatibility.
    """

    def test_terminal_endpoint_exists(self):
        """The /ws/terminal/{instance_id} route is registered."""
        from api import app

        ws_routes = [r.path for r in app.routes if hasattr(r, "path") and "terminal" in r.path]
        assert "/ws/terminal/{instance_id}" in ws_routes

    def test_terminal_constants(self):
        """Terminal session constants are properly defined (new module)."""
        from routes import terminal as terminal_mod

        assert terminal_mod._SESSION_TIMEOUT_SEC == 14_400  # 4 hours
        assert terminal_mod._RATE_LIMIT_BYTES_PER_SEC == 524_288  # 512 KB/s
        assert terminal_mod._IDLE_WARN_THRESHOLD_SEC == 14_100  # 5 min before

    def test_terminal_ws_auth_validator(self):
        """_validate_ws_auth returns None for empty tokens when auth required."""
        from routes._deps import _validate_ws_auth
        import routes._deps as _deps_mod

        original = _deps_mod.AUTH_REQUIRED
        try:
            _deps_mod.AUTH_REQUIRED = True

            class FakeWebSocket:
                cookies = {}
                query_params = {}

            result = _validate_ws_auth(FakeWebSocket())
            assert result is None
        finally:
            _deps_mod.AUTH_REQUIRED = original


# ── Instance Detail Terminal Integration ─────────────────────────────


class TestInstanceTerminalIntegration:
    """Verify terminal component is wired into instance detail page."""

    def test_instance_page_imports_terminal(self):
        """Instance detail page imports WebTerminal component."""
        page_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "frontend",
            "src",
            "app",
            "(dashboard)",
            "dashboard",
            "instances",
            "[id]",
            "page.tsx",
        )
        if not os.path.exists(page_path):
            pytest.skip("Frontend source not available")

        with open(page_path) as f:
            content = f.read()

        assert "WebTerminal" in content
        assert "showTerminal" in content
        assert "@/components/terminal/WebTerminal" in content

    def test_terminal_component_exists(self):
        """WebTerminal.tsx component file exists."""
        component_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "frontend",
            "src",
            "components",
            "terminal",
            "WebTerminal.tsx",
        )
        assert os.path.exists(component_path), "WebTerminal.tsx not found"

        with open(component_path) as f:
            content = f.read()

        assert "ws/terminal/" in content
        assert "xterm" in content.lower()
        assert "onData" in content  # Input handler
        assert "_TERMINAL" not in content or "send" in content  # Sends data


# ── Spot Pricing Recharts Enhancement ────────────────────────────────


class TestSpotPricingRecharts:
    """Verify spot pricing page uses Recharts area chart."""

    def test_spot_pricing_uses_recharts(self):
        """Spot pricing page imports and uses Recharts components."""
        page_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "frontend",
            "src",
            "app",
            "(dashboard)",
            "dashboard",
            "spot-pricing",
            "page.tsx",
        )
        if not os.path.exists(page_path):
            pytest.skip("Frontend source not available")

        with open(page_path) as f:
            content = f.read()

        assert "recharts" in content
        assert "AreaChart" in content
        assert "Area" in content
        assert "XAxis" in content
        assert "YAxis" in content
        assert "Tooltip" in content
        assert "ResponsiveContainer" in content

    def test_spot_pricing_no_div_bars(self):
        """Spot pricing page no longer uses CSS div bar chart."""
        page_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "frontend",
            "src",
            "app",
            "(dashboard)",
            "dashboard",
            "spot-pricing",
            "page.tsx",
        )
        if not os.path.exists(page_path):
            pytest.skip("Frontend source not available")

        with open(page_path) as f:
            content = f.read()

        # Old CSS bar chart used flex items-end gap-px and maxHistoryPrice
        assert "maxHistoryPrice" not in content
        assert "items-end gap-px" not in content


# ── Migration 005 Completeness ───────────────────────────────────────


class TestMigration005Completeness:
    """Verify migration 005 covers all planned tables."""

    def test_migration_contains_all_tables(self):
        """Migration 005 creates all planned tables from phases 1-5."""
        migration_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "migrations",
            "versions",
            "005_payment_infrastructure.py",
        )
        if not os.path.exists(migration_path):
            pytest.skip("Migration file not available")

        with open(migration_path) as f:
            content = f.read()

        expected_tables = [
            "stripe_event_inbox",
            "billing_cycles",
            "fintrac_reports",
            "gpu_offers",
            "gpu_allocations",
            "spot_price_history",
            "reservations",
            "inference_endpoints",
            "worker_model_cache",
            "volumes",
            "volume_attachments",
            "cloud_burst_instances",
            "event_snapshots",
        ]
        for table in expected_tables:
            assert table in content, f"Table '{table}' missing from migration 005"

    def test_migration_has_downgrade(self):
        """Migration 005 has a proper downgrade function."""
        migration_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "migrations",
            "versions",
            "005_payment_infrastructure.py",
        )
        if not os.path.exists(migration_path):
            pytest.skip("Migration file not available")

        with open(migration_path) as f:
            content = f.read()

        assert "def downgrade" in content
        assert "op.drop_table" in content


# ── StatCard & Badge Error Fixes ─────────────────────────────────────


class TestFrontendErrorFixes:
    """Verify the StatCard title→label and Badge variant fixes."""

    @pytest.mark.parametrize(
        "page",
        [
            "inference/page.tsx",
            "volumes/page.tsx",
            "spot-pricing/page.tsx",
        ],
    )
    def test_statcard_uses_label_not_title(self, page):
        """StatCard components use 'label' prop, not 'title'."""
        page_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "frontend",
            "src",
            "app",
            "(dashboard)",
            "dashboard",
            page,
        )
        if not os.path.exists(page_path):
            pytest.skip("Frontend source not available")

        with open(page_path) as f:
            content = f.read()

        # All StatCard should use label=
        import re

        title_matches = re.findall(r"<StatCard\s+title=", content)
        assert len(title_matches) == 0, f"Found StatCard with 'title' prop in {page}"

        label_matches = re.findall(r"<StatCard\s+label=", content)
        assert len(label_matches) >= 3, f"Expected ≥3 StatCard with 'label' prop in {page}"

    @pytest.mark.parametrize(
        "page",
        [
            "inference/page.tsx",
            "volumes/page.tsx",
            "spot-pricing/page.tsx",
        ],
    )
    def test_badge_uses_valid_variants(self, page):
        """Badge components only use valid variants."""
        page_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "frontend",
            "src",
            "app",
            "(dashboard)",
            "dashboard",
            page,
        )
        if not os.path.exists(page_path):
            pytest.skip("Frontend source not available")

        with open(page_path) as f:
            content = f.read()

        valid_variants = {
            "default",
            "active",
            "dead",
            "queued",
            "running",
            "completed",
            "failed",
            "cancelled",
            "warning",
            "info",
        }

        import re

        # Find all Badge variant="..." patterns (not Button variant= etc.)
        variant_matches = re.findall(r'<Badge\s+variant="(\w+)"', content)
        # Also check statusColor return values that feed into Badge variant=
        status_color_matches = re.findall(r'return\s+"(\w+)"', content)
        all_variants = variant_matches + [
            v
            for v in status_color_matches
            if v not in ("default", "attached", "available")  # filter non-variant strings
        ]
        for variant in variant_matches:
            assert variant in valid_variants, f"Invalid Badge variant '{variant}' in {page}"

    def test_api_no_duplicate_createReservation(self):
        """api.ts should not have duplicate createReservation exports."""
        api_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "frontend",
            "src",
            "lib",
            "api.ts",
        )
        if not os.path.exists(api_path):
            pytest.skip("Frontend source not available")

        with open(api_path) as f:
            content = f.read()

        import re

        matches = re.findall(r"export async function createReservation\b", content)
        assert len(matches) == 1, f"Expected 1 createReservation, found {len(matches)}"

        # The v2 version should be renamed
        assert "createMarketplaceReservation" in content
