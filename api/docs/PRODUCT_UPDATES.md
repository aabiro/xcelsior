# Monthly product-update email

A "what's new in Xcelsior" email, once a month, to people who asked for it.
Code: `core/product_updates.py`. Tests: `tests/test_product_updates.py`.

## What goes out

A hero image, "Hey <first name>,", a short intro, up to a few features (each a
picture, one paragraph and one button), a list of smaller improvements, and a
footer with the unsubscribe link and the company's mailing address. HTML and
plain text are both sent.

## Who gets it

Only verified users with active consent to `product_updates` in
`casl_consent`. People opt in with the **Product updates** switch under
Settings → Notifications; it is off by default. Nobody is enrolled by the
sender. Consent is checked again right before each message is sent.

Unsubscribing never needs a sign-in:

- the link in the footer opens `/unsubscribe`, which asks once and then
  withdraws consent;
- mail clients that show their own Unsubscribe button POST to
  `/api/v2/privacy/unsubscribe` (`List-Unsubscribe-Post`, RFC 8058);
- the Settings switch does the same thing.

## Each month

1. **Write the issue.** Copy `core/product_update_issues/2026-10.json` to the
   new month and rewrite it. Keep `"status": "draft"`. Only describe what is
   live for customers.
2. **Draw the images.** Add the month's scenes to
   `api/frontend/scripts/build-product-update-assets.mjs` and run, from
   `api/frontend`:

   ```bash
   node scripts/build-product-update-assets.mjs 2026-11
   ```

   They are written to `public/email/product-updates/<slug>/` and must be
   deployed before the email is sent, because the email links to them on
   `https://xcelsior.ca`. For an animation, put the finished state first: a
   client that does not animate shows only the first frame.
3. **Preview.**

   ```bash
   python api/scripts/send_product_update.py 2026-11 preview --out /tmp/2026-11.html
   ```

4. **Proof.** Send yourself one copy and read it on a phone and a desktop.

   ```bash
   python api/scripts/send_product_update.py 2026-11 proof --to you@xcelsior.ca --name Aaryn
   ```

5. **Approve.** Set `"status": "approved"` in the issue file. A draft cannot
   be sent.
6. **Dry run, then send.**

   ```bash
   python api/scripts/send_product_update.py 2026-11 send
   python api/scripts/send_product_update.py 2026-11 send --confirm 2026-11
   ```

   The first prints how many people would get it. The second sends.

## Configuration

| Variable | Why |
| --- | --- |
| `XCELSIOR_MAILING_ADDRESS` | Postal address shown in the footer. Required by CASL; nothing sends without it. `\n` between lines. |
| `XCELSIOR_UNSUBSCRIBE_SECRET` | Signs unsubscribe links. Falls back to a key derived from `XCELSIOR_PRIVACY_REFERENCE_SECRET`. Must be the same on the machine that sends and on the API. Do not rotate it casually: links in mail already sent stop working. |
| `XCELSIOR_SMTP_*`, `XCELSIOR_EMAIL_FROM`, `XCELSIOR_EMAIL_REPLY_TO` | The same mail settings the rest of the platform uses. |
| `XCELSIOR_BASE_URL` | Where links point. Defaults to `https://xcelsior.ca`. |

## Sending twice, and failures

Each successful send is appended to
`data/product_updates/<slug>.sent.jsonl`. A re-run skips everyone in that
file, so it is safe to run `send` again after a failure; anyone whose send
failed is tried again.

The log is a file on the machine that sends. Send an issue from one machine,
and keep the file. If two machines send the same issue, people get it twice.
Moving this to a database table needs a migration and has not been done.
