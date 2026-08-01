"""Durable agent API keys — issuance, validation, rotation, revocation.

These credentials are pasted into editor configs and never expire, so the
properties that matter are: the plaintext is never stored, a revoked key stops
working immediately, rotating invalidates what it replaced, and use is
recorded so the dashboard can tell a live key from an unused one.
"""

import os
import time

import pytest
from unittest import mock

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

import oauth_service as oa
from db import AgentKeyStore

CLIENT = "test-mcp-client"


def _user(suffix: str = "") -> dict:
    return {
        "user_id": f"agentkey-user{suffix}",
        "email": f"agentkey{suffix}@xcelsior.ca",
        "role": "submitter",
    }


@pytest.fixture
def user():
    u = _user()
    AgentKeyStore.revoke_all_for_client(u["user_id"], CLIENT, time.time())
    yield u
    AgentKeyStore.revoke_all_for_client(u["user_id"], CLIENT, time.time())


def _issue(u, **kw):
    return oa.issue_agent_api_key(
        user=u, client_id=CLIENT, scopes=kw.pop("scopes", ["instances:read"]), **kw
    )


class TestKeyShape:
    def test_prefix_is_scannable(self, user):
        """Secret scanners match known prefixes; a leaked key must be findable."""
        assert _issue(user)["access_token"].startswith("xcel_ai_")

    def test_key_is_short_enough_to_show_whole(self, user):
        # The UI shows the key inline rather than truncating it.
        assert len(_issue(user)["access_token"]) < 80

    def test_keys_are_unique(self, user):
        a = _issue(user)["access_token"]
        b = _issue(user)["access_token"]
        assert a != b

    def test_no_expiry_is_reported(self, user):
        """Advertising an expiry would make clients schedule a pointless refresh."""
        assert _issue(user)["expires_in"] is None


class TestStorage:
    def test_plaintext_is_never_stored(self, user):
        token = _issue(user)["access_token"]
        row = AgentKeyStore.get_live_by_hash(oa._hash_agent_key(token))
        assert row is not None
        assert token not in str(row)

    def test_stored_digest_is_sha256(self, user):
        token = _issue(user)["access_token"]
        row = AgentKeyStore.get_live_by_hash(oa._hash_agent_key(token))
        assert len(row["key_hash"]) == 64

    def test_display_prefix_leaks_nothing_usable(self, user):
        bundle = _issue(user)
        assert bundle["access_token"] not in bundle["key_prefix"]
        assert len(bundle["key_prefix"]) < len(bundle["access_token"])


class TestValidation:
    def test_valid_key_resolves_to_its_owner(self, user):
        token = _issue(user, scopes=["instances:read", "billing:read"])["access_token"]
        principal = oa.validate_agent_api_key(token)
        assert principal is not None
        assert principal["user_id"] == user["user_id"]
        assert principal["scopes"] == ["instances:read", "billing:read"]
        assert principal["auth_type"] == "agent_api_key"

    def test_unknown_key_is_rejected(self, user):
        assert oa.validate_agent_api_key("xcel_ai_not_a_real_key") is None

    def test_non_key_bearer_is_ignored(self, user):
        """A JWT must not be routed down the key path."""
        assert oa.validate_agent_api_key("eyJhbGciOiJSUzI1NiI.x.y") is None
        assert oa.validate_agent_api_key("") is None

    def test_shape_check_avoids_a_lookup(self):
        assert oa.looks_like_agent_key("xcel_ai_abc")
        assert not oa.looks_like_agent_key("xoa_abc")
        assert not oa.looks_like_agent_key(None)

    def test_repeat_use_does_not_rewrite_on_every_request(self, user):
        """last_used_at must not put a write on the hot auth path.

        Every authenticated agent call resolves through this function. Writing
        the timestamp each time would add a row-level write — and lock
        contention on a single hot row — to every request, which is exactly
        what the JWT it replaced avoided by verifying locally.
        """
        token = _issue(user)["access_token"]
        oa.validate_agent_api_key(token)
        first = AgentKeyStore.get_live_by_hash(oa._hash_agent_key(token))["last_used_at"]
        assert first is not None
        for _ in range(5):
            assert oa.validate_agent_api_key(token) is not None
        again = AgentKeyStore.get_live_by_hash(oa._hash_agent_key(token))["last_used_at"]
        assert again == first, "last_used_at was rewritten inside the throttle window"

    def test_bookkeeping_failure_never_fails_authentication(self, user):
        """A write problem must not lock every agent out of the platform."""
        token = _issue(user)["access_token"]
        with mock.patch.object(
            AgentKeyStore, "touch_last_used", side_effect=RuntimeError("db down")
        ):
            assert oa.validate_agent_api_key(token) is not None

    def test_use_is_recorded(self, user):
        """last_used_at is how the dashboard knows a key is live in a config."""
        token = _issue(user)["access_token"]
        before = AgentKeyStore.get_live_by_hash(oa._hash_agent_key(token))
        assert before["last_used_at"] is None, "a fresh key has never been used"
        oa.validate_agent_api_key(token)
        after = AgentKeyStore.get_live_by_hash(oa._hash_agent_key(token))
        assert after["last_used_at"] is not None


class TestRevocation:
    def test_revoked_key_stops_working(self, user):
        bundle = _issue(user)
        assert oa.validate_agent_api_key(bundle["access_token"]) is not None
        assert AgentKeyStore.revoke(bundle["key_id"], user["user_id"], time.time())
        assert oa.validate_agent_api_key(bundle["access_token"]) is None

    def test_revocation_is_scoped_to_the_owner(self, user):
        """A key id alone must not let someone revoke another user's key."""
        bundle = _issue(user)
        assert not AgentKeyStore.revoke(bundle["key_id"], "someone-else", time.time())
        assert oa.validate_agent_api_key(bundle["access_token"]) is not None

    def test_revoking_twice_is_not_an_error_the_second_time(self, user):
        bundle = _issue(user)
        assert AgentKeyStore.revoke(bundle["key_id"], user["user_id"], time.time())
        assert not AgentKeyStore.revoke(bundle["key_id"], user["user_id"], time.time())


class TestRotation:
    def test_rotation_revokes_what_it_replaces(self, user):
        """Otherwise rotating just accumulates working credentials."""
        old = _issue(user)["access_token"]
        new = _issue(user, replace_existing=True)
        assert new["replaced_keys"] == 1
        assert oa.validate_agent_api_key(old) is None
        assert oa.validate_agent_api_key(new["access_token"]) is not None

    def test_without_replace_the_old_key_survives(self, user):
        old = _issue(user)["access_token"]
        new = _issue(user)["access_token"]
        assert oa.validate_agent_api_key(old) is not None
        assert oa.validate_agent_api_key(new) is not None

    def test_listing_excludes_revoked_keys(self, user):
        bundle = _issue(user)
        assert any(k["key_id"] == bundle["key_id"] for k in AgentKeyStore.list_for_user(user["user_id"]))
        AgentKeyStore.revoke(bundle["key_id"], user["user_id"], time.time())
        assert not any(
            k["key_id"] == bundle["key_id"] for k in AgentKeyStore.list_for_user(user["user_id"])
        )


class TestRename:
    def test_rename_changes_the_label_only(self, user):
        bundle = _issue(user)
        assert AgentKeyStore.rename(bundle["key_id"], user["user_id"], "Cursor laptop")
        row = AgentKeyStore.get_live_by_hash(oa._hash_agent_key(bundle["access_token"]))
        assert row["name"] == "Cursor laptop"
        assert oa.validate_agent_api_key(bundle["access_token"]) is not None

    def test_rename_is_scoped_to_the_owner(self, user):
        bundle = _issue(user)
        assert not AgentKeyStore.rename(bundle["key_id"], "someone-else", "stolen")
