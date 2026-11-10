#!/usr/bin/env python3
"""Preview, proof and send a monthly product-update email.

    # Write the HTML to a file and open it in a browser. No database, no mail.
    python api/scripts/send_product_update.py 2026-10 preview --out /tmp/2026-10.html

    # Send one proof to yourself.
    python api/scripts/send_product_update.py 2026-10 proof --to you@xcelsior.ca --name Aaryn

    # Count who would get it. Sends nothing.
    python api/scripts/send_product_update.py 2026-10 send

    # Send it. Needs the issue's status set to "approved".
    python api/scripts/send_product_update.py 2026-10 send --confirm 2026-10

`send` is a dry run unless `--confirm` repeats the issue slug. It goes only to
verified users with active consent to product updates, skips anyone the send
log already has, and can be re-run after a failure. See core/product_updates.py.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
for _dir in ("core", "privacy", "workers", "auth", "billing", "hosts", "api", ""):
    _path = str(ROOT / _dir) if _dir else str(ROOT)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import product_updates as pu  # noqa: E402


def _preview(args: argparse.Namespace, issue: pu.Issue) -> int:
    try:
        address = pu.mailing_address()
    except pu.ProductUpdateError:
        address = "[XCELSIOR_MAILING_ADDRESS is not set]"
    link = f"{pu.base_url()}/unsubscribe?token=preview"
    rendered = pu.render(issue, args.name, link, address)
    out = pathlib.Path(args.out)
    out.write_text(rendered.html if not args.text else rendered.text, encoding="utf-8")
    print(f"{issue.slug} ({issue.status}): {rendered.subject}")
    print(f"wrote {out}")
    return 0


def _proof(args: argparse.Namespace, issue: pu.Issue) -> int:
    pu.send_test(issue, args.to, name=args.name)
    print(f"proof of {issue.slug} sent to {args.to}")
    return 0


def _send(args: argparse.Namespace, issue: pu.Issue) -> int:
    if args.confirm is not None and args.confirm != issue.slug:
        print(f"--confirm must repeat the issue slug ({issue.slug}); nothing sent", file=sys.stderr)
        return 2
    dry_run = args.confirm is None
    log_path = pathlib.Path(args.log) if args.log else pu.default_log_path(issue.slug)
    send_log = pu.SendLog(log_path)
    audience = pu.load_audience()
    report = pu.send_issue(issue, audience, send_log, dry_run=dry_run, limit=args.limit)
    print(report.summary())
    for failure in report.failures:
        print(f"  failed: {failure}", file=sys.stderr)
    if dry_run:
        print(f"dry run: nothing sent. To send: --confirm {issue.slug}")
    else:
        print(f"send log: {log_path}")
    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("issue", help="issue slug, e.g. 2026-10")
    commands = parser.add_subparsers(dest="command", required=True)

    preview = commands.add_parser("preview", help="write the rendered email to a file")
    preview.add_argument("--out", required=True)
    preview.add_argument("--name", default="Aaryn Biro", help="recipient name to greet")
    preview.add_argument("--text", action="store_true", help="write the plain-text part instead")
    preview.set_defaults(run=_preview)

    proof = commands.add_parser("proof", help="send one copy to one address")
    proof.add_argument("--to", required=True)
    proof.add_argument("--name", default="")
    proof.set_defaults(run=_proof)

    send = commands.add_parser("send", help="send to everyone who consented (dry run by default)")
    send.add_argument("--confirm", metavar="SLUG", help="repeat the issue slug to really send")
    send.add_argument("--limit", type=int, help="send to at most this many people this run")
    send.add_argument(
        "--log", help="send-log path (default: data/product_updates/<slug>.sent.jsonl)"
    )
    send.set_defaults(run=_send)

    args = parser.parse_args(argv)
    try:
        issue = pu.load_issue(args.issue)
        return args.run(args, issue)
    except pu.ProductUpdateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
