"""The monthly product-update email (core/product_updates.py).

It is a commercial message under CASL, so most of this file is about what
must not happen: nobody without consent is written to, nobody gets an issue
twice, a draft is not sent, and an unsubscribe link works for exactly the
person and purpose it was signed for.
"""

import json
import os
import smtplib
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest

import product_updates as pu

ISSUE = {
    "slug": "2026-10",
    "status": "approved",
    "month_label": "October 2026",
    "subject": "What's new in Xcelsior: October 2026",
    "preview": "Three things shipped.",
    "intro": "Here is what shipped.",
    "hero_image": "https://xcelsior.ca/email/product-updates/2026-10/hero.png",
    "hero_alt": "Xcelsior, what's new",
    "features": [
        {
            "title": "Xcelsior MCP",
            "body": "Connect an agent & run compute.",
            "image": "https://xcelsior.ca/email/product-updates/2026-10/mcp.png",
            "image_alt": "MCP",
            "cta_label": "Connect your agent",
            "cta_url": "https://xcelsior.ca/dashboard/mcp",
        }
    ],
    "improvements": [{"title": "Device sign-in", "body": "approve a terminal from a browser."}],
    "outro": "More in the docs.",
}
CONFIG = pu.MailConfig(
    "smtp.test", 587, "user", "pass", "updates@xcelsior.ca", "support@xcelsior.ca"
)
ALICE = pu.Recipient("u-alice", "alice@example.com", "Alice Ng")
BOB = pu.Recipient("u-bob", "bob@example.com", "")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("XCELSIOR_UNSUBSCRIBE_SECRET", "test-unsubscribe-secret")
    monkeypatch.setenv("XCELSIOR_MAILING_ADDRESS", "1 Example St\\nToronto, ON A1A 1A1, Canada")
    monkeypatch.setenv("XCELSIOR_BASE_URL", "https://xcelsior.ca")


class FakeSMTP:
    def __init__(self, fail_for=()):
        self.sent, self.fail_for, self.quits = [], set(fail_for), 0

    def send_message(self, message):
        if message["To"] in self.fail_for:
            raise smtplib.SMTPRecipientsRefused({message["To"]: (550, b"no such user")})
        self.sent.append(message)

    def quit(self):
        self.quits += 1


def _send(audience, log, server, **kwargs):
    kwargs.setdefault("consents", lambda _user_id: True)
    return pu.send_issue(
        pu.parse_issue(ISSUE),
        audience,
        log,
        dry_run=False,
        config=CONFIG,
        connect=lambda _cfg: server,
        pause_seconds=0,
        **kwargs,
    )


# ── Content ──────────────────────────────────────────────────────────


def test_the_issue_in_the_repo_is_valid_and_its_images_exist():
    from pathlib import Path

    public = Path(__file__).resolve().parent.parent / "api" / "frontend" / "public"
    for path in sorted(pu.ISSUES_DIR.glob("*.json")):
        issue = pu.load_issue(path.stem)
        for url in [issue.hero_image, *(f.image for f in issue.features)]:
            assert url.startswith("https://xcelsior.ca/"), url
            assert (public / url.removeprefix("https://xcelsior.ca/")).is_file(), url


@pytest.mark.parametrize(
    "change, message",
    [
        ({"slug": "october"}, "must look like 2026-10"),
        ({"status": "sent"}, "status must be one of"),
        ({"features": []}, "at least one feature"),
        ({"hero_image": "http://xcelsior.ca/hero.png"}, "must be an https:// URL"),
        ({"intro": "  "}, "'intro' is missing or empty"),
    ],
)
def test_a_malformed_issue_is_refused(change, message):
    with pytest.raises(pu.ProductUpdateError, match=message):
        pu.parse_issue({**ISSUE, **change})


def test_a_button_needs_both_its_label_and_its_link():
    feature = {**ISSUE["features"][0], "cta_url": None}
    with pytest.raises(pu.ProductUpdateError, match="both cta_label and cta_url"):
        pu.parse_issue({**ISSUE, "features": [feature]})


# ── Rendering ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name, greeting",
    [
        ("Aaryn Biro", "Aaryn"),
        ("  mel  ", "mel"),
        ("", "there"),
        (None, "there"),
        ("a@b.co", "there"),
    ],
)
def test_greets_by_first_name(name, greeting):
    assert pu.first_name(name) == greeting


def test_render_personalises_identifies_the_sender_and_links_unsubscribe():
    rendered = pu.render(
        pu.parse_issue(ISSUE),
        "Alice Ng",
        "https://xcelsior.ca/unsubscribe?token=t",
        "1 Example St\nToronto",
    )
    assert rendered.subject == "What's new in Xcelsior: October 2026"
    for part in (rendered.html, rendered.text):
        assert "Hey Alice," in part
        assert "Xcelsior Compute Inc." in part
        assert "1 Example St" in part and "Toronto" in part
        assert "https://xcelsior.ca/unsubscribe?token=t" in part
        assert "https://xcelsior.ca/dashboard/mcp" in part
        assert "Device sign-in" in part
    assert 'alt="MCP"' in rendered.html
    assert "Connect an agent &amp; run compute." in rendered.html


def test_render_escapes_a_hostile_name():
    rendered = pu.render(
        pu.parse_issue(ISSUE), "<img src=x onerror=alert(1)>", "https://x/u", "addr"
    )
    assert "<img src=x" not in rendered.html
    assert "&lt;img" in rendered.html


# ── Unsubscribe tokens ───────────────────────────────────────────────


def test_a_token_names_one_user_and_one_purpose():
    token = pu.make_unsubscribe_token("u-alice")
    assert pu.read_unsubscribe_token(token) == ("u-alice", "product_updates")
    assert pu.unsubscribe_url("u-alice").startswith("https://xcelsior.ca/unsubscribe?token=")


def test_a_tampered_or_foreign_token_is_refused(monkeypatch):
    token = pu.make_unsubscribe_token("u-alice")
    payload, signature = token.split(".")
    forged = pu._b64(json.dumps(["u-bob", "product_updates"]).encode())
    assert pu.read_unsubscribe_token(f"{forged}.{signature}") is None
    assert pu.read_unsubscribe_token(payload) is None
    assert pu.read_unsubscribe_token("") is None
    assert pu.read_unsubscribe_token("a.b.c") is None
    monkeypatch.setenv("XCELSIOR_UNSUBSCRIBE_SECRET", "another-secret")
    assert pu.read_unsubscribe_token(token) is None


def test_no_secret_means_no_token_rather_than_an_unsigned_one(monkeypatch):
    monkeypatch.delenv("XCELSIOR_UNSUBSCRIBE_SECRET")
    monkeypatch.delenv("XCELSIOR_PRIVACY_REFERENCE_SECRET", raising=False)
    with pytest.raises(pu.ProductUpdateError, match="XCELSIOR_UNSUBSCRIBE_SECRET"):
        pu.make_unsubscribe_token("u-alice")


def test_the_fallback_key_is_derived_not_reused(monkeypatch):
    monkeypatch.delenv("XCELSIOR_UNSUBSCRIBE_SECRET")
    monkeypatch.setenv("XCELSIOR_PRIVACY_REFERENCE_SECRET", "shared")
    assert pu._unsubscribe_key() != b"shared"
    assert pu.read_unsubscribe_token(pu.make_unsubscribe_token("u-alice")) == (
        "u-alice",
        "product_updates",
    )


# ── Sending ──────────────────────────────────────────────────────────


def test_a_dry_run_counts_and_sends_nothing(tmp_path):
    log = pu.SendLog(tmp_path / "log.jsonl")
    report = pu.send_issue(pu.parse_issue(ISSUE), [ALICE, BOB], log, consents=lambda _u: True)
    assert report.dry_run and report.sent == 2 and report.audience == 2
    assert not (tmp_path / "log.jsonl").exists()
    assert "would send 2" in report.summary()


def test_each_message_carries_the_one_click_unsubscribe_headers(tmp_path):
    server = FakeSMTP()
    report = _send([ALICE], pu.SendLog(tmp_path / "log.jsonl"), server)
    assert report.sent == 1 and server.quits == 1
    message = server.sent[0]
    assert message["To"] == "alice@example.com"
    assert message["From"] == "Xcelsior <updates@xcelsior.ca>"
    assert message["Reply-To"] == "support@xcelsior.ca"
    assert message["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"
    link = message["List-Unsubscribe"].strip("<>")
    assert link.startswith("https://xcelsior.ca/api/v2/privacy/unsubscribe?token=")
    token = link.split("token=")[1]
    assert pu.read_unsubscribe_token(token) == ("u-alice", "product_updates")
    assert [part.get_content_type() for part in message.get_payload()] == [
        "text/plain",
        "text/html",
    ]


def test_nobody_gets_an_issue_twice(tmp_path):
    path = tmp_path / "log.jsonl"
    first = FakeSMTP()
    assert _send([ALICE], pu.SendLog(path), first).sent == 1
    # A later run, as after a crash: the log is read back from disk.
    second = FakeSMTP()
    report = _send([ALICE, BOB], pu.SendLog(path), second)
    assert report.already_sent == 1 and report.sent == 1
    assert [m["To"] for m in second.sent] == ["bob@example.com"]


def test_consent_withdrawn_since_the_audience_was_read_is_respected(tmp_path):
    server = FakeSMTP()
    report = _send(
        [ALICE, BOB], pu.SendLog(tmp_path / "log.jsonl"), server, consents=lambda u: u != "u-alice"
    )
    assert report.withdrawn == 1 and report.sent == 1
    assert [m["To"] for m in server.sent] == ["bob@example.com"]


def test_one_refused_address_does_not_stop_the_run_and_is_retried_later(tmp_path):
    path = tmp_path / "log.jsonl"
    report = _send([ALICE, BOB], pu.SendLog(path), FakeSMTP(fail_for={"alice@example.com"}))
    assert report.failed == 1 and report.sent == 1
    assert "alice@example.com" in report.failures[0]
    log = pu.SendLog(path)
    assert log.has("bob@example.com") and not log.has("alice@example.com")
    assert _send([ALICE, BOB], log, FakeSMTP()).sent == 1


def test_a_draft_is_never_sent(tmp_path):
    draft = pu.parse_issue({**ISSUE, "status": "draft"})
    server = FakeSMTP()
    with pytest.raises(pu.ProductUpdateError, match="'draft'"):
        pu.send_issue(
            draft,
            [ALICE],
            pu.SendLog(tmp_path / "l"),
            dry_run=False,
            config=CONFIG,
            connect=lambda _c: server,
        )
    assert server.sent == []


def test_no_mailing_address_means_no_send(tmp_path, monkeypatch):
    monkeypatch.delenv("XCELSIOR_MAILING_ADDRESS")
    with pytest.raises(pu.ProductUpdateError, match="XCELSIOR_MAILING_ADDRESS"):
        _send([ALICE], pu.SendLog(tmp_path / "l"), FakeSMTP())


def test_limit_caps_one_run(tmp_path):
    server = FakeSMTP()
    assert _send([ALICE, BOB], pu.SendLog(tmp_path / "l"), server, limit=1).sent == 1
    assert len(server.sent) == 1


# ── The unsubscribe endpoint and the audience, against the database ──


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api import app

    return TestClient(app)


def _user(name="Una Updates", verified=True, consent=True):
    """A user row, as production stores it. Returns (email, user_id)."""
    from db import UserStore, auth_connection
    from privacy import get_consent_manager

    tag = uuid.uuid4().hex[:10]
    email, user_id = f"updates-{tag}@xcelsior.ca", f"user-{tag}"
    UserStore.create_user({"email": email, "user_id": user_id, "name": name})
    if verified:
        with auth_connection() as conn:
            conn.execute("UPDATE users SET email_verified = 1 WHERE email = %s", (email,))
    if consent:
        get_consent_manager().record_consent(user_id, "express", "product_updates", source="test")
    return email, user_id


@pytest.fixture()
def subscriber():
    return _user()


def test_the_audience_is_verified_users_who_opted_in(subscriber):
    email, user_id = subscriber
    never_asked, _ = _user(consent=False)
    unverified, _ = _user(verified=False)
    audience = {r.email: r for r in pu.load_audience()}
    assert audience[email] == pu.Recipient(user_id, email, "Una Updates")
    assert never_asked not in audience
    assert unverified not in audience


def test_the_link_unsubscribes_without_signing_in_and_twice_is_fine(client, subscriber):
    email, user_id = subscriber
    token = pu.make_unsubscribe_token(user_id)
    # As the web page sends it.
    r = client.post("/api/v2/privacy/unsubscribe", json={"token": token})
    assert r.status_code == 200 and r.json() == {"ok": True, "purpose": "product_updates"}
    assert not pu.still_consents(user_id)
    assert email not in {r.email for r in pu.load_audience()}
    # As a mail client's one-click unsubscribe sends it (RFC 8058).
    again = client.post(
        f"/api/v2/privacy/unsubscribe?token={token}",
        data={"List-Unsubscribe": "One-Click"},
    )
    assert again.status_code == 200


def test_the_unsubscribe_path_is_open_to_a_caller_with_no_session():
    """The half the in-process suite cannot see.

    These tests run with auth enforcement relaxed, where an anonymous request
    reaches the handler anyway. With enforcement on (staging, production) the
    middleware answered 401 before the handler ran, so every link and every
    one-click unsubscribe failed. Found on staging on 2026-11-10.
    """
    from routes._deps import PUBLIC_PATH_PREFIXES, PUBLIC_PATHS

    path = "/api/v2/privacy/unsubscribe"
    assert path in PUBLIC_PATHS or path.startswith(PUBLIC_PATH_PREFIXES)
    # The handler limits itself because public paths skip the general limiter.
    import inspect

    from routes.privacy import api_privacy_unsubscribe

    assert "_check_auth_rate_limit" in inspect.getsource(api_privacy_unsubscribe)


def test_a_bad_link_unsubscribes_nobody(client, subscriber):
    _email, user_id = subscriber
    for body in ({"token": "not-a-token"}, {"token": ""}, {}):
        assert client.post("/api/v2/privacy/unsubscribe", json=body).status_code == 400
    assert pu.still_consents(user_id)
