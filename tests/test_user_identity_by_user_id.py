"""A user is one row, found by user_id; everything they own follows it (164).

Before 164, thirteen tables named their user by email with no foreign key.
Changing an address re-keyed six of them by hand and missed `mfa_methods`, so
the next login skipped the second factor and turned it off. Deleting a user
left sessions behind that still authenticated. A wallet notice stored the
billing account id where an email belonged, so nobody ever saw one.

These pin the replacement: rows refer to `users.user_id`, the address lives
only on `users`, deleting a user removes what it owned, and the email copies
that the previous release still reads stay correct while it serves.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from db import (  # noqa: E402
    ConsentStore,
    MfaStore,
    NotificationStore,
    UserStore,
    auth_connection,
)


def _new_user() -> dict:
    tag = uuid.uuid4().hex[:10]
    user = {
        "email": f"ident-{tag}@xcelsior.ca",
        "user_id": f"user-ident-{tag}",
        "customer_id": f"cust-ident-{tag}",
        "name": "Identity Test",
    }
    UserStore.create_user(user)
    return user


@pytest.fixture
def user():
    created = _new_user()
    team_ids: list[str] = []
    created["team_ids"] = team_ids
    yield created
    with auth_connection() as conn:
        for team_id in team_ids:
            conn.execute("DELETE FROM team_members WHERE team_id = %s", (team_id,))
            conn.execute("DELETE FROM teams WHERE team_id = %s", (team_id,))
        conn.execute("DELETE FROM users WHERE user_id = %s", (created["user_id"],))


def _give_everything(user: dict) -> dict:
    """One of each thing a user owns, written through the stores."""
    now = time.time()
    tag = user["user_id"].rsplit("-", 1)[-1]
    method_id = MfaStore.create_method(
        {"email": user["email"], "method_type": "totp", "secret": "S" * 32}
    )
    UserStore.add_ssh_key(
        {
            "id": f"sshk-{tag}",
            "user_id": user["user_id"],
            "name": "laptop",
            "public_key": f"ssh-ed25519 AAAA{tag} test",
            "fingerprint": f"SHA256:{tag}",
        }
    )
    ConsentStore.record(
        grant_id=f"grant-{tag}",
        tenant_id=user["customer_id"],
        user_id=user["user_id"],
        email=user["email"],
        client_id=f"client-{tag}",
        scopes=["instances:read"],
    )
    team_id = f"team-ident-{tag}"
    UserStore.create_team({"team_id": team_id, "name": "Ident", "owner_email": user["email"]})
    user["team_ids"].append(team_id)
    session_token = f"sess-ident-{tag}"
    UserStore.create_session(
        {"token": session_token, "user_id": user["user_id"], "expires_at": now + 3600}
    )
    api_key = f"xcel-ident-{tag}"
    UserStore.create_api_key({"key": api_key, "user_id": user["user_id"]})
    notification_id = NotificationStore.create(user["email"], "info", "Hello")
    return {
        "method_id": method_id,
        "ssh_key_id": f"sshk-{tag}",
        "client_id": f"client-{tag}",
        "team_id": team_id,
        "session": session_token,
        "api_key": api_key,
        "notification_id": notification_id,
    }


def test_changing_an_address_keeps_two_factor_and_everything_else(user):
    owned = _give_everything(user)
    new_email = user["email"].replace("ident-", "moved-")

    UserStore.change_email(user["email"], new_email)

    methods = MfaStore.list_methods(new_email)
    assert [m["id"] for m in methods] == [owned["method_id"]], (
        "the second factor must survive an address change; losing it here is "
        "how a login skipped 2FA and then switched it off"
    )
    assert methods[0]["email"] == new_email
    assert [k["id"] for k in UserStore.list_ssh_keys(new_email)] == [owned["ssh_key_id"]]
    grant = ConsentStore.get(user["user_id"], owned["client_id"])
    assert grant is not None and grant["email"] == new_email
    teams = UserStore.get_user_teams(new_email)
    assert [t["team_id"] for t in teams] == [owned["team_id"]]
    assert teams[0]["owner_email"] == new_email
    session = UserStore.get_session(owned["session"])
    assert session is not None and session["email"] == new_email
    assert [k["key"] for k in UserStore.list_api_keys(new_email)] == [owned["api_key"]]
    assert [n["id"] for n in NotificationStore.list_for_user(new_email)] == [
        owned["notification_id"]
    ]
    assert MfaStore.list_methods(user["email"]) == []


def test_the_previous_release_reads_the_new_address_from_its_copies(user):
    """The release still serving during a deploy reads the email columns."""
    owned = _give_everything(user)
    new_email = user["email"].replace("ident-", "copy-")

    UserStore.change_email(user["email"], new_email)

    with auth_connection() as conn:
        for table, column in (
            ("mfa_methods", "email"),
            ("user_ssh_keys", "email"),
            ("oauth_consent_grants", "email"),
            ("team_members", "email"),
            ("sessions", "email"),
            ("api_keys", "email"),
            ("notifications", "user_email"),
        ):
            stale = conn.execute(
                f"SELECT count(*) AS n FROM {table} WHERE user_id = %s AND {column} <> %s",
                (user["user_id"], new_email),
            ).fetchone()["n"]
            assert stale == 0, f"{table}.{column} still holds the old address"
        owner = conn.execute(
            "SELECT owner_email FROM teams WHERE team_id = %s", (owned["team_id"],)
        ).fetchone()
        assert owner["owner_email"] == new_email


def test_a_row_written_by_user_id_alone_gets_its_email_copy(user):
    MfaStore.create_method({"email": user["email"], "method_type": "totp", "secret": "x"})
    with auth_connection() as conn:
        row = conn.execute(
            "SELECT email FROM mfa_methods WHERE user_id = %s", (user["user_id"],)
        ).fetchone()
    assert row["email"] == user["email"]


def test_a_row_written_by_the_previous_release_gets_its_user_id(user):
    """The previous release inserts by email only; the row must still belong."""
    with auth_connection() as conn:
        conn.execute(
            "INSERT INTO mfa_methods (email, method_type, secret, enabled, created_at) "
            "VALUES (%s, 'totp', 'x', 1, %s)",
            (user["email"].upper(), time.time()),
        )
        rows = conn.execute(
            "SELECT user_id, email FROM mfa_methods WHERE user_id = %s", (user["user_id"],)
        ).fetchall()
    assert [(r["user_id"], r["email"]) for r in rows] == [(user["user_id"], user["email"])]
    assert len(MfaStore.list_methods(user["email"])) == 1


def test_a_row_naming_two_users_belongs_to_its_user_id(user):
    """Both releases must agree whose row it is, so the email copy follows user_id."""
    other = _new_user()
    try:
        with auth_connection() as conn:
            conn.execute(
                "INSERT INTO mfa_methods (email, user_id, method_type, enabled, created_at) "
                "VALUES (%s, %s, 'totp', 1, %s)",
                (user["email"], other["user_id"], time.time()),
            )
            owners = conn.execute(
                "SELECT user_id, email FROM mfa_methods WHERE user_id IN (%s, %s)",
                (user["user_id"], other["user_id"]),
            ).fetchall()
        assert [(r["user_id"], r["email"]) for r in owners] == [(other["user_id"], other["email"])]
    finally:
        with auth_connection() as conn:
            conn.execute("DELETE FROM users WHERE user_id = %s", (other["user_id"],))


def test_the_previous_release_moving_a_row_by_email_moves_its_owner(user):
    other = _new_user()
    try:
        MfaStore.create_method({"email": user["email"], "method_type": "totp", "secret": "x"})
        with auth_connection() as conn:
            conn.execute(
                "UPDATE mfa_methods SET email = %s WHERE user_id = %s",
                (other["email"], user["user_id"]),
            )
        assert MfaStore.list_methods(user["email"]) == []
        assert len(MfaStore.list_methods(other["email"])) == 1
    finally:
        with auth_connection() as conn:
            conn.execute("DELETE FROM users WHERE user_id = %s", (other["user_id"],))


def test_deleting_a_user_removes_what_it_owned(user):
    owned = _give_everything(user)

    UserStore.delete_user(user["email"])

    assert UserStore.get_session(owned["session"]) is None, "a session must not outlive its user"
    assert UserStore.get_api_key(owned["api_key"]) is None
    with auth_connection() as conn:
        for table in (
            "mfa_methods",
            "user_ssh_keys",
            "oauth_consent_grants",
            "team_members",
            "notifications",
            "sessions",
            "api_keys",
        ):
            left = conn.execute(
                f"SELECT count(*) AS n FROM {table} WHERE user_id = %s", (user["user_id"],)
            ).fetchone()["n"]
            assert left == 0, f"{table} kept rows of a deleted user"
        team = conn.execute(
            "SELECT owner_user_id FROM teams WHERE team_id = %s", (owned["team_id"],)
        ).fetchone()
    assert team is not None and team["owner_user_id"] is None


def test_a_notice_for_someone_without_an_account_is_not_stored():
    assert NotificationStore.create("nobody-ident@xcelsior.ca", "info", "Hello") == ""


def test_a_wallet_notice_reaches_the_wallets_user(user, monkeypatch):
    """A wallet is a billing account; the notice goes to the person behind it."""
    import control_plane.billing_effects as effects
    from control_plane.outbox import OutboxEvent

    delivered: list[str] = []
    monkeypatch.setattr(effects, "current_service", lambda: "api")
    monkeypatch.setattr(NotificationStore, "deliver", staticmethod(delivered.append))
    event = OutboxEvent(
        event_id=str(uuid.uuid4()),
        aggregate_type="wallet",
        aggregate_id=user["customer_id"],
        event_type="billing.v1.wallet_notice_requested",
        payload={
            "customer_id": user["customer_id"],
            "kind": "billing_low_balance",
            "title": "Balance low",
            "body": "Top up",
            "data": {},
        },
        headers={},
        destination_class="billing_notice",
        idempotency_key="wallet_notice:" + uuid.uuid4().hex,
        attempt_count=0,
    )

    effects.handle_wallet_notice(event)
    effects.handle_wallet_notice(event)  # an outbox retry adds nothing

    notices = NotificationStore.list_for_user(user["email"])
    assert [n["type"] for n in notices] == ["billing_low_balance"]
    assert delivered == ["notif-billing-" + event.event_id]
