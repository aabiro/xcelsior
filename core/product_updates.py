"""Monthly product-update email: content, rendering, audience and sending.

One issue is one JSON file in ``core/product_update_issues/``. Rendering is a
pure function of the issue, the recipient's first name and their unsubscribe
link, so it can be previewed and tested without a database or a mail server.

This is a commercial electronic message under CASL, so three things are not
optional and the sender refuses to run without them:

- **Consent.** Only users with an active ``product_updates`` record in
  ``casl_consent`` are in the audience (``ConsentManager``). Nobody is
  enrolled here.
- **Unsubscribe.** Every message carries a signed link that works without
  signing in, and the ``List-Unsubscribe`` / ``List-Unsubscribe-Post`` headers
  for one-click unsubscribe (RFC 8058).
- **Identification.** The footer carries the sender's legal name and mailing
  address, from ``XCELSIOR_MAILING_ADDRESS``.

An issue is sent at most once per recipient. The record of who has it is an
append-only JSONL log beside the issue (see ``SendLog``): a re-run after a
failure picks up where it stopped. The log is a file, not a table, so the
campaign must be sent from one machine.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import logging
import os
import re
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote

log = logging.getLogger("xcelsior")

ISSUES_DIR = Path(__file__).resolve().parent / "product_update_issues"
CONSENT_PURPOSE = "product_updates"
LEGAL_NAME = "Xcelsior Compute Inc."
SLUG = re.compile(r"^\d{4}-\d{2}$")
#: An issue stays ``draft`` until a person has read the copy and approved it.
STATUSES = ("draft", "approved")

# Brand tokens, as in api/frontend/src/app/globals.css.
NAVY = "#0a0e1a"
SURFACE = "#0d1320"
BORDER = "#1a2235"
TEXT = "#e8edf5"
TEXT_SECONDARY = "#a3b1c6"
TEXT_MUTED = "#7e8fa0"
CYAN = "#00d4ff"
VIOLET = "#7c3aed"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


class ProductUpdateError(Exception):
    """The campaign cannot go ahead as asked. The message says why."""


# ── Issue content ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Feature:
    title: str
    body: str
    image: str
    image_alt: str
    cta_label: str | None = None
    cta_url: str | None = None


@dataclass(frozen=True)
class Improvement:
    title: str
    body: str


@dataclass(frozen=True)
class Issue:
    slug: str
    status: str
    month_label: str
    subject: str
    preview: str
    intro: str
    hero_image: str
    hero_alt: str
    features: tuple[Feature, ...]
    improvements: tuple[Improvement, ...] = ()
    outro: str = ""


def _require_text(data: dict, key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProductUpdateError(f"{where}: {key!r} is missing or empty")
    return value.strip()


def _require_https(url: str, where: str) -> str:
    if not url.startswith("https://"):
        raise ProductUpdateError(f"{where}: {url!r} must be an https:// URL")
    return url


def parse_issue(data: dict) -> Issue:
    """Validate an issue's JSON. Every image and link must be absolute https."""
    slug = _require_text(data, "slug", "issue")
    if not SLUG.match(slug):
        raise ProductUpdateError(f"issue: slug {slug!r} must look like 2026-10")
    where = f"issue {slug}"
    status = data.get("status", "draft")
    if status not in STATUSES:
        raise ProductUpdateError(f"{where}: status must be one of {', '.join(STATUSES)}")
    raw_features = data.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise ProductUpdateError(f"{where}: at least one feature is required")
    features = []
    for n, raw in enumerate(raw_features, 1):
        at = f"{where}, feature {n}"
        cta_label, cta_url = raw.get("cta_label"), raw.get("cta_url")
        if bool(cta_label) != bool(cta_url):
            raise ProductUpdateError(f"{at}: give both cta_label and cta_url, or neither")
        features.append(
            Feature(
                title=_require_text(raw, "title", at),
                body=_require_text(raw, "body", at),
                image=_require_https(_require_text(raw, "image", at), at),
                image_alt=_require_text(raw, "image_alt", at),
                cta_label=cta_label or None,
                cta_url=_require_https(cta_url, at) if cta_url else None,
            )
        )
    improvements = tuple(
        Improvement(
            title=_require_text(raw, "title", f"{where}, improvement {n}"),
            body=_require_text(raw, "body", f"{where}, improvement {n}"),
        )
        for n, raw in enumerate(data.get("improvements") or [], 1)
    )
    return Issue(
        slug=slug,
        status=status,
        month_label=_require_text(data, "month_label", where),
        subject=_require_text(data, "subject", where),
        preview=_require_text(data, "preview", where),
        intro=_require_text(data, "intro", where),
        hero_image=_require_https(_require_text(data, "hero_image", where), where),
        hero_alt=_require_text(data, "hero_alt", where),
        features=tuple(features),
        improvements=improvements,
        outro=(data.get("outro") or "").strip(),
    )


def load_issue(slug: str, issues_dir: Path | None = None) -> Issue:
    if not SLUG.match(slug):
        raise ProductUpdateError(f"issue slug {slug!r} must look like 2026-10")
    path = (issues_dir or ISSUES_DIR) / f"{slug}.json"
    if not path.is_file():
        raise ProductUpdateError(f"no issue file at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ProductUpdateError(f"{path} is not valid JSON: {e}") from e
    issue = parse_issue(data)
    if issue.slug != slug:
        raise ProductUpdateError(f"{path} declares slug {issue.slug!r}")
    return issue


# ── Unsubscribe tokens ───────────────────────────────────────────────


def _unsubscribe_key() -> bytes:
    """The signing key, or an error: an unsigned unsubscribe link is never sent."""
    own = os.environ.get("XCELSIOR_UNSUBSCRIBE_SECRET", "")
    if own:
        return own.encode()
    shared = os.environ.get("XCELSIOR_PRIVACY_REFERENCE_SECRET", "")
    if shared:
        # Derived, so this key never equals the one it comes from.
        return hmac.new(shared.encode(), b"xcelsior.unsubscribe.v1", hashlib.sha256).digest()
    raise ProductUpdateError(
        "set XCELSIOR_UNSUBSCRIBE_SECRET (or XCELSIOR_PRIVACY_REFERENCE_SECRET) "
        "to sign unsubscribe links"
    )


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_unsubscribe_token(user_id: str, purpose: str = CONSENT_PURPOSE) -> str:
    """A token that withdraws one purpose for one user, and does nothing else.

    It does not expire: a message can be read, and unsubscribed from, years
    after it was sent, and CASL requires the mechanism to keep working.
    """
    payload = _b64(json.dumps([user_id, purpose], separators=(",", ":")).encode())
    signature = hmac.new(_unsubscribe_key(), payload.encode(), hashlib.sha256).digest()
    return f"{payload}.{_b64(signature)}"


def read_unsubscribe_token(token: str) -> tuple[str, str] | None:
    """(user_id, purpose) for a genuine token, else None."""
    try:
        payload, signature = token.split(".")
        expected = hmac.new(_unsubscribe_key(), payload.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_unb64(signature), expected):
            return None
        user_id, purpose = json.loads(_unb64(payload))
    except (ValueError, TypeError):
        return None
    if not isinstance(user_id, str) or not isinstance(purpose, str) or not user_id or not purpose:
        return None
    return user_id, purpose


def base_url() -> str:
    return os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca").rstrip("/")


def unsubscribe_url(user_id: str) -> str:
    return f"{base_url()}/unsubscribe?token={quote(make_unsubscribe_token(user_id))}"


def one_click_url(user_id: str) -> str:
    """Where a mail client POSTs for one-click unsubscribe (RFC 8058)."""
    return f"{base_url()}/api/v2/privacy/unsubscribe?token={quote(make_unsubscribe_token(user_id))}"


# ── Rendering ────────────────────────────────────────────────────────


def first_name(name: str | None) -> str:
    """The name to greet someone by: "Aaryn Biro" → "Aaryn"; nothing → "there"."""
    words = (name or "").strip().split()
    if not words or "@" in words[0]:
        return "there"
    return words[0][:40]


@dataclass(frozen=True)
class Rendered:
    subject: str
    html: str
    text: str


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _button(label: str, url: str) -> str:
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" align="center" '
        'style="margin:20px auto 4px"><tr>'
        f'<td align="center" bgcolor="{VIOLET}" style="border-radius:8px;'
        f'background-color:{VIOLET};background-image:linear-gradient(135deg,#2f7df6,{VIOLET})">'
        f'<a href="{_esc(url)}" style="display:inline-block;padding:13px 30px;font-family:{FONT};'
        "font-size:16px;font-weight:600;line-height:1;color:#ffffff;text-decoration:none;"
        f'border-radius:8px">{_esc(label)}</a></td></tr></table>'
    )


def _rule() -> str:
    return (
        '<tr><td style="padding:28px 32px 0"><div style="height:1px;line-height:1px;font-size:1px;'
        f'background-color:{BORDER};background-image:linear-gradient(90deg,{CYAN},{VIOLET})">&nbsp;</div>'
        "</td></tr>"
    )


def _feature(feature: Feature) -> str:
    button = (
        _button(feature.cta_label, feature.cta_url) if feature.cta_url and feature.cta_label else ""
    )
    return (
        '<tr><td style="padding:28px 32px 0">'
        f'<img src="{_esc(feature.image)}" width="536" alt="{_esc(feature.image_alt)}" '
        'style="display:block;width:100%;max-width:536px;height:auto;border:0;border-radius:14px"/>'
        f'<p style="margin:20px 0 0;font-family:{FONT};font-size:16px;line-height:1.65;color:{TEXT_SECONDARY}">'
        f'<strong style="color:{TEXT};font-weight:700">{_esc(feature.title)}:</strong> {_esc(feature.body)}</p>'
        f"{button}</td></tr>"
    )


def render(issue: Issue, name: str | None, unsubscribe_link: str, mailing_address: str) -> Rendered:
    """One recipient's message. Everything that came from a person is escaped."""
    hello = first_name(name)
    root = base_url()
    features = _rule().join(_feature(f) for f in issue.features)

    improvements = ""
    if issue.improvements:
        items = "".join(
            f'<li style="margin:0 0 10px;font-family:{FONT};font-size:16px;line-height:1.6;color:{TEXT_SECONDARY}">'
            f'<strong style="color:{TEXT};font-weight:700">{_esc(i.title)}:</strong> {_esc(i.body)}</li>'
            for i in issue.improvements
        )
        improvements = (
            _rule() + '<tr><td style="padding:28px 32px 0">'
            f'<p style="margin:0 0 14px;font-family:{FONT};font-size:17px;font-weight:700;color:{TEXT}">'
            "Other improvements</p>"
            f'<ul style="margin:0;padding:0 0 0 22px">{items}</ul></td></tr>'
        )

    outro = (
        f'<p style="margin:0 0 16px;font-family:{FONT};font-size:16px;line-height:1.65;color:{TEXT_SECONDARY}">'
        f"{_esc(issue.outro)}</p>"
        if issue.outro
        else ""
    )
    address = _esc(mailing_address).replace("\n", "<br/>")

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta name="color-scheme" content="dark"/>
<meta name="supported-color-schemes" content="dark"/>
<title>{_esc(issue.subject)}</title>
</head>
<body style="margin:0;padding:0;background-color:{NAVY}">
<div style="display:none;max-height:0;overflow:hidden;opacity:0;color:{NAVY}">{_esc(issue.preview)}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="{NAVY}" style="background-color:{NAVY}">
<tr><td align="center" style="padding:24px 12px">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" style="width:100%;max-width:600px;background-color:{SURFACE};border:1px solid {BORDER};border-radius:16px;overflow:hidden">
<tr><td style="padding:0">
<a href="{root}" style="text-decoration:none"><img src="{_esc(issue.hero_image)}" width="600" alt="{_esc(issue.hero_alt)}" style="display:block;width:100%;max-width:600px;height:auto;border:0"/></a>
</td></tr>
<tr><td style="padding:32px 32px 0">
<p style="margin:0 0 16px;font-family:{FONT};font-size:17px;line-height:1.5;color:{TEXT}">Hey {_esc(hello)},</p>
<p style="margin:0;font-family:{FONT};font-size:16px;line-height:1.65;color:{TEXT_SECONDARY}">{_esc(issue.intro)}</p>
</td></tr>
{features}
{improvements}
{_rule()}
<tr><td style="padding:28px 32px 36px">
{outro}
<p style="margin:0 0 16px;font-family:{FONT};font-size:16px;line-height:1.65;color:{TEXT_SECONDARY}">Thanks for building on Xcelsior. We&#8217;ll be back next month with more.</p>
<p style="margin:0;font-family:{FONT};font-size:16px;line-height:1.65;color:{TEXT}">&#8211; The Xcelsior Team</p>
</td></tr>
<tr><td bgcolor="{NAVY}" style="padding:28px 32px;background-color:{NAVY};border-top:1px solid {BORDER}">
<a href="{root}" style="text-decoration:none"><img src="{root}/site-assets/assets/png/lockup-light.png" width="150" alt="Xcelsior" style="display:block;border:0;width:150px;height:auto"/></a>
<p style="margin:18px 0 0;font-family:{FONT};font-size:14px;line-height:1.6">
<a href="{root}/dashboard" style="color:{CYAN};font-weight:600;text-decoration:underline">Open your dashboard &rarr;</a>
&nbsp;&nbsp;&nbsp;
<a href="https://docs.xcelsior.ca" style="color:{CYAN};font-weight:600;text-decoration:underline">Docs &rarr;</a>
</p>
<p style="margin:16px 0 0;font-family:{FONT};font-size:13px;line-height:1.6;color:{TEXT_MUTED}">Questions? Reply to this email or write to <a href="mailto:support@xcelsior.ca" style="color:{TEXT_SECONDARY};text-decoration:underline">support@xcelsior.ca</a>.</p>
<p style="margin:16px 0 0;font-family:{FONT};font-size:13px;line-height:1.6;color:{TEXT_MUTED}">&copy; {time.gmtime().tm_year} {LEGAL_NAME}<br/>{address}</p>
<p style="margin:16px 0 0;font-family:{FONT};font-size:13px;line-height:1.6;color:{TEXT_MUTED}">You&#8217;re getting this because you asked for product updates from Xcelsior. <a href="{_esc(unsubscribe_link)}" style="color:{TEXT_SECONDARY};text-decoration:underline">Unsubscribe</a> &middot; <a href="{root}/privacy" style="color:{TEXT_SECONDARY};text-decoration:underline">Privacy</a></p>
</td></tr>
</table>
</td></tr>
</table>
</body>
</html>"""

    lines = [
        f"What's new in Xcelsior: {issue.month_label}",
        "",
        f"Hey {hello},",
        "",
        issue.intro,
        "",
    ]
    for f in issue.features:
        lines += [f"{f.title}: {f.body}"]
        if f.cta_url:
            lines += [f"{f.cta_label}: {f.cta_url}"]
        lines += [""]
    if issue.improvements:
        lines += ["Other improvements:", ""]
        lines += [f"- {i.title}: {i.body}" for i in issue.improvements]
        lines += [""]
    if issue.outro:
        lines += [issue.outro, ""]
    lines += [
        "Thanks for building on Xcelsior. We'll be back next month with more.",
        "",
        "- The Xcelsior Team",
        "",
        "--",
        LEGAL_NAME,
        mailing_address,
        "",
        "You're getting this because you asked for product updates from Xcelsior.",
        f"Unsubscribe: {unsubscribe_link}",
    ]
    return Rendered(subject=issue.subject, html=body, text="\n".join(lines))


# ── Audience ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Recipient:
    user_id: str
    email: str
    name: str = ""


def load_audience() -> list[Recipient]:
    """Verified users with active, unexpired consent to product updates.

    Consent is keyed by ``user_id``, or by email for accounts that predate
    user ids, so both are matched.
    """
    from db import auth_connection

    with auth_connection() as conn:
        rows = conn.execute(
            """
            SELECT u.user_id, u.email, u.name
              FROM users u
              JOIN casl_consent c
                ON c.user_id = u.user_id OR c.user_id = u.email
             WHERE c.purpose = %s
               AND c.active = TRUE
               AND c.withdrawn_at IS NULL
               AND (c.expires_at IS NULL OR c.expires_at > now())
               AND u.email_verified = 1
             ORDER BY u.email
            """,
            (CONSENT_PURPOSE,),
        ).fetchall()
    seen: set[str] = set()
    audience = []
    for row in rows:
        email = (row["email"] or "").strip().lower()
        if not email or email in seen:
            continue
        seen.add(email)
        audience.append(
            Recipient(user_id=row["user_id"] or email, email=email, name=row["name"] or "")
        )
    return audience


def still_consents(user_id: str) -> bool:
    """Asked again right before each send: a campaign can run for minutes."""
    from privacy import get_consent_manager

    has, _kind = get_consent_manager().has_consent(user_id, CONSENT_PURPOSE)
    return bool(has)


# ── Send log ─────────────────────────────────────────────────────────


class SendLog:
    """Who has been sent an issue. Append-only JSONL, flushed to disk per line.

    The line is written *after* SMTP accepts the message, so a crash between
    the two can send one person the issue twice on the re-run, and can never
    record someone as sent who was not.
    """

    def __init__(self, path: Path):
        self.path = path
        self._sent: set[str] = set()
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("status") == "sent":
                    self._sent.add(entry["email"])

    def has(self, email: str) -> bool:
        return email in self._sent

    def __len__(self) -> int:
        return len(self._sent)

    def record(self, email: str, status: str, detail: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"email": email, "status": status, "at": int(time.time())}
        if detail:
            entry["detail"] = detail
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if status == "sent":
            self._sent.add(email)


def default_log_path(slug: str) -> Path:
    root = os.environ.get("XCELSIOR_PRODUCT_UPDATE_LOG_DIR") or str(
        Path(__file__).resolve().parent.parent / "data" / "product_updates"
    )
    return Path(root) / f"{slug}.sent.jsonl"


# ── Sending ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MailConfig:
    host: str
    port: int
    user: str
    password: str
    sender: str
    reply_to: str = ""

    @classmethod
    def from_env(cls) -> "MailConfig":
        config = cls(
            host=os.environ.get("XCELSIOR_SMTP_HOST", ""),
            port=int(os.environ.get("XCELSIOR_SMTP_PORT", "587")),
            user=os.environ.get("XCELSIOR_SMTP_USER", ""),
            password=os.environ.get("XCELSIOR_SMTP_PASS", ""),
            sender=os.environ.get("XCELSIOR_EMAIL_FROM", ""),
            reply_to=os.environ.get("XCELSIOR_EMAIL_REPLY_TO", ""),
        )
        if not (config.host and config.user and config.sender):
            raise ProductUpdateError(
                "mail is not configured: set XCELSIOR_SMTP_HOST, XCELSIOR_SMTP_USER and "
                "XCELSIOR_EMAIL_FROM"
            )
        return config


def mailing_address() -> str:
    """The sender's postal address, which CASL requires in every message."""
    address = os.environ.get("XCELSIOR_MAILING_ADDRESS", "").replace("\\n", "\n").strip()
    if not address:
        raise ProductUpdateError(
            "set XCELSIOR_MAILING_ADDRESS to the company's mailing address; "
            "a commercial message cannot be sent without one"
        )
    return address


def build_message(rendered: Rendered, recipient: Recipient, config: MailConfig) -> MIMEMultipart:
    message = MIMEMultipart("alternative")
    message["Subject"] = rendered.subject
    address = parseaddr(config.sender)[1] or config.sender
    message["From"] = formataddr(("Xcelsior", address))
    message["To"] = recipient.email
    if config.reply_to:
        message["Reply-To"] = config.reply_to
    message["List-Unsubscribe"] = f"<{one_click_url(recipient.user_id)}>"
    message["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    message.attach(MIMEText(rendered.text, "plain", "utf-8"))
    message.attach(MIMEText(rendered.html, "html", "utf-8"))
    return message


@dataclass
class SendReport:
    issue: str
    dry_run: bool
    audience: int = 0
    sent: int = 0
    already_sent: int = 0
    withdrawn: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verb = "would send" if self.dry_run else "sent"
        return (
            f"{self.issue}: audience {self.audience}, {verb} {self.sent}, "
            f"already sent {self.already_sent}, consent withdrawn {self.withdrawn}, "
            f"failed {self.failed}"
        )


def send_issue(
    issue: Issue,
    audience: Iterable[Recipient],
    send_log: SendLog,
    *,
    dry_run: bool = True,
    limit: int | None = None,
    config: MailConfig | None = None,
    consents: Callable[[str], bool] = still_consents,
    connect: Callable[[MailConfig], smtplib.SMTP] | None = None,
    pause_seconds: float = 0.5,
) -> SendReport:
    """Send an approved issue to everyone in the audience who lacks it.

    ``dry_run`` (the default) renders every message and sends none. One
    failed recipient does not stop the run; they stay unsent and a re-run
    tries them again.
    """
    recipients = list(audience)
    report = SendReport(issue=issue.slug, dry_run=dry_run, audience=len(recipients))
    if not dry_run and issue.status != "approved":
        raise ProductUpdateError(
            f"issue {issue.slug} is {issue.status!r}; set its status to 'approved' once the "
            "copy has been reviewed"
        )
    address = mailing_address()
    if not dry_run:
        config = config or MailConfig.from_env()

    def _connect(cfg: MailConfig) -> smtplib.SMTP:
        server = smtplib.SMTP(cfg.host, cfg.port, timeout=30)
        server.starttls()
        server.login(cfg.user, cfg.password)
        return server

    server: smtplib.SMTP | None = None
    try:
        for recipient in recipients:
            if limit is not None and report.sent >= limit:
                break
            if send_log.has(recipient.email):
                report.already_sent += 1
                continue
            if not consents(recipient.user_id):
                report.withdrawn += 1
                continue
            rendered = render(issue, recipient.name, unsubscribe_url(recipient.user_id), address)
            if dry_run:
                report.sent += 1
                continue
            assert config is not None
            try:
                if server is None:
                    server = (connect or _connect)(config)
                server.send_message(build_message(rendered, recipient, config))
            except (smtplib.SMTPException, OSError) as e:
                report.failed += 1
                report.failures.append(f"{recipient.email}: {e}")
                send_log.record(recipient.email, "failed", str(e)[:200])
                log.warning("PRODUCT UPDATE FAILED: %s -> %s | %s", issue.slug, recipient.email, e)
                # The connection may be the thing that broke: open a new one.
                if server is not None:
                    try:
                        server.quit()
                    except (smtplib.SMTPException, OSError):
                        pass
                    server = None
                continue
            send_log.record(recipient.email, "sent")
            report.sent += 1
            log.info("PRODUCT UPDATE SENT: %s -> %s", issue.slug, recipient.email)
            if pause_seconds:
                time.sleep(pause_seconds)
    finally:
        if server is not None:
            try:
                server.quit()
            except (smtplib.SMTPException, OSError):
                pass
    return report


def send_test(
    issue: Issue, to_email: str, name: str = "", config: MailConfig | None = None
) -> None:
    """Send one copy to one address, for proofreading. Not logged, any status.

    The unsubscribe link is real but signed for an id no account has, so
    clicking it in a proof withdraws nobody.
    """
    config = config or MailConfig.from_env()
    recipient = Recipient(user_id=f"proof:{to_email}", email=to_email, name=name)
    rendered = render(issue, name, unsubscribe_url(recipient.user_id), mailing_address())
    rendered = Rendered(
        subject=f"[Proof] {rendered.subject}", html=rendered.html, text=rendered.text
    )
    with smtplib.SMTP(config.host, config.port, timeout=30) as server:
        server.starttls()
        server.login(config.user, config.password)
        server.send_message(build_message(rendered, recipient, config))
