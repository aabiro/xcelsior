"""Phase 15 — managed engines, GitHub deploy, vanity, cache replication, Redis RPM."""

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from serverless.cache import cache_replicate_regions, cache_volume_name
from serverless.github_deploy import (
    GitHubSourceError,
    apply_github_source,
    parse_github_repo,
    resolve_github_image,
)
from serverless.rate_limit_store import redis_rate_limits_enabled
from serverless.vanity import (
    clean_endpoint_display_name,
    endpoint_vanity_slug,
    vanity_invoke_path,
)


class TestVanitySlugs:
    def test_slug_from_name(self):
        assert endpoint_vanity_slug("My Llama Endpoint", "ep-123") == "my-llama-endpoint"

    def test_slug_fallback_to_id(self):
        assert endpoint_vanity_slug("", "abcdef123456") == "abcdef123456"

    def test_invoke_path_uses_endpoint_id_and_slug(self):
        assert vanity_invoke_path("ep-99", "my-slug") == "/v1/serverless/ep-99/my-slug"

    def test_display_name_removes_provider_prefix(self):
        assert clean_endpoint_display_name("meta-llama/Llama-3.1-8B") == "Llama-3.1-8B"


class TestGitHubDeploy:
    def test_parse_https_url(self):
        assert parse_github_repo("https://github.com/acme/infer") == ("acme", "infer")

    def test_parse_ssh_url(self):
        assert parse_github_repo("git@github.com:acme/infer.git") == ("acme", "infer")

    def test_resolve_ghcr_image(self):
        img = resolve_github_image("https://github.com/Acme/Infer", ref="main")
        assert img == "ghcr.io/acme/infer:main"

    def test_apply_github_source_sets_image(self):
        image, env = apply_github_source(
            mode="custom",
            image_ref="",
            source_type="github",
            source_ref="https://github.com/acme/infer",
            source_ref_branch="dev",
        )
        assert image == "ghcr.io/acme/infer:dev"
        assert env["XCELSIOR_SOURCE_TYPE"] == "github"

    def test_rejects_invalid_url(self):
        with pytest.raises(GitHubSourceError):
            parse_github_repo("https://gitlab.com/acme/infer")


class TestCacheReplication:
    def test_cache_volume_name_includes_region(self):
        name = cache_volume_name("meta-llama/Llama-3.1-8B", "main", "ca-west")
        assert "ca-west" in name

    def test_replicate_regions_from_env(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_SERVERLESS_CACHE_REPLICATE_REGIONS", "ca-west,ca-east")
        assert cache_replicate_regions() == ["ca-west", "ca-east"]


class TestRedisRateLimits:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("XCELSIOR_SERVERLESS_REDIS_RATE_LIMITS", raising=False)
        assert redis_rate_limits_enabled() is False
