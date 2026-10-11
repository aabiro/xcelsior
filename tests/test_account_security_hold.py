"""A persistent account suspension must defeat new IPs and all token types."""
import time
import uuid
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

import abuse_policy
import oauth_service as oauth
from db import get_engine


@pytest.fixture
def suspended():
    uid = 'security-test-' + uuid.uuid4().hex
    with get_engine().transaction() as (conn, backend):
        abuse_policy.set_account_suspension(conn, user_id=uid, active=True, actor='admin-test', reason='test', backend=backend)
    oauth.reset_auth_cache_for_tests()
    yield uid
    with get_engine().transaction() as (conn, backend):
        conn.execute('DELETE FROM state WHERE namespace = %s', (f'security.suspended.{uid}',))
    oauth.reset_auth_cache_for_tests()


def test_cached_bearer_stops_without_waiting_for_expiry(suspended):
    for uid in (suspended, 'unrelated-account'):
        token = 'xoa_' + uid
        oauth._cache_set_json('access_token', token, {'user_id':uid, 'expires_at':time.time()+3600}, 3600)
    assert oauth.resolve_opaque_access_token('xoa_' + suspended) is None
    assert oauth.resolve_opaque_access_token('xoa_unrelated-account') is not None


def test_new_login_and_agent_key_cannot_bypass_hold(suspended, monkeypatch):
    store = Mock()
    monkeypatch.setattr(oauth.UserStore, 'create_session', store)
    for mint in (
        lambda: oauth.issue_user_tokens({'user_id':suspended}, Mock()),
        lambda: oauth.issue_agent_api_key(user={'user_id':suspended}, client_id='test', scopes=[]),
        lambda: oauth._issue_opaque_access_token({'user_id':suspended}, client_id='test', scopes=[], session_token='new', session_type='browser'),
    ):
        with pytest.raises(oauth.OAuthGrantError) as exc:
            mint()
        assert exc.value.status_code == 403
    store.assert_not_called()


def test_legacy_session_uses_fresh_account_policy(suspended):
    from routes._deps import _merge_auth_user
    with pytest.raises(HTTPException) as exc:
        _merge_auth_user({'user_id':suspended, 'role':'submitter'})
    assert exc.value.status_code == 403


def test_machine_client_creator_and_workspace_are_checked(suspended):
    assert abuse_policy.account_suspended({'created_by_email':suspended})
    assert abuse_policy.account_suspended({'workspace_customer_id':suspended})
    assert abuse_policy.account_suspended({'client_id':suspended})
    assert abuse_policy.account_suspended({'sub':suspended})
    assert not abuse_policy.account_suspended({'created_by_email':'someone-else'})


def test_store_failure_does_not_become_permission_to_mint(monkeypatch):
    monkeypatch.setattr(abuse_policy, 'get_engine', Mock(side_effect=RuntimeError('unavailable')))
    with pytest.raises(RuntimeError):
        oauth.issue_user_tokens({'user_id':'must-check'}, Mock())
