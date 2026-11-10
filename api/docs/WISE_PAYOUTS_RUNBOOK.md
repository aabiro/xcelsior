# Wise payouts runbook

Wise is the third provider payout rail, beside Stripe Connect and PayPal. Code: `billing/wise_payouts.py`, `api/routes/wise.py`, migration `159`. Tests: `tests/test_wise_payouts.py`.

It runs on a Wise **business** account with a personal API token. No partner agreement is involved.

## How a payout moves

1. A provider's finished job is recorded on the Wise rail: `POST /api/providers/{id}/payout?job_id=…&payment_rail=wise`. The amount comes from the ledger, as on the other rails. The row is `queued`.
2. Once a day (or when an admin calls `POST /api/admin/wise/payouts/batch`) each provider's queued rows are added up into one `wise_payouts` row, `pending_approval`. Nothing has been sent to Wise.
3. An admin approves it: `POST /api/admin/wise/payouts/{id}/approve`.
4. Within two minutes the worker creates a quote and a transfer at Wise. The payout is `awaiting_funding`.
5. A person funds the transfer in the Wise app (default), or the worker funds it from the Wise balance (`XCELSIOR_WISE_AUTO_FUND=1`).
6. Wise reports `outgoing_payment_sent`. The payout is `sent` and its ledger rows are `paid`.

The provider bears Wise's fee: the transfer's source amount is the provider's share, and Wise takes its fee from that. The fee quoted is stored on the payout (`fee_micros`).

## Funding and top-ups

Wise gives a personal token no way to pull money into the balance. The balance is topped up by a person: in the Wise app, or by bank transfer to the account details Wise shows for the CAD balance.

What is automatic:

- Every two minutes the worker compares the CAD balance with the payouts that are approved or already created at Wise. When the balance is short it emails the admin address the exact amount to add. It says so again only if the shortfall grows, or after six hours.
- `GET /api/admin/wise/status` shows `balance_cad`, `committed_cad`, `top_up_needed_cad` and `top_up_needed_if_all_approved_cad`, so the amount is known before approving.
- With `XCELSIOR_WISE_AUTO_FUND=1`, a transfer whose amount the balance covers is funded without a person. A transfer the balance does not cover waits; it is funded on a later pass once the balance is topped up. Default is off: a person funds each transfer in the Wise app.

## Controls

| Control | Where | Default |
|---|---|---|
| Kill switch | `XCELSIOR_WISE_PAYOUTS_ENABLED` | off |
| Funding by a person in the Wise app | `XCELSIOR_WISE_AUTO_FUND` | manual |
| Cool-off after a destination is set or changed | `XCELSIOR_WISE_RECIPIENT_COOLOFF_HOURS` | 72 h; production never below 24 h |
| Cap per payout | `XCELSIOR_WISE_MAX_PAYOUT_CAD` | 2000 |
| Cap on approvals in 24 h | `XCELSIOR_WISE_DAILY_CAP_CAD` | 5000 |
| Minimum payout | `XCELSIOR_WISE_MIN_PAYOUT_CAD` | 20 |
| Who may approve, link, reject or release | a platform admin signed in to the dashboard | API keys, the master token, agent keys and connector tokens are refused |
| Destination check before every transfer | Wise's recipient is read back and compared with the fingerprint stored at batching | always |
| Webhook signature | RSA-SHA256 against Wise's published key, pinned in source | always |

Bank details are never stored. A destination is Wise's recipient id, a masked hint and a fingerprint.

## First-time setup

1. In Wise, create a personal API token on the business account. Restrict it to the API server's IP address. Put it in the server's `.env` as `XCELSIOR_WISE_API_TOKEN`. Do not paste it anywhere else.
2. Find the business profile id and set `XCELSIOR_WISE_PROFILE_ID`:

   ```bash
   curl -s -H "Authorization: Bearer $XCELSIOR_WISE_API_TOKEN" https://api.wise.com/v2/profiles
   ```

3. Set `XCELSIOR_WISE_ENV=production`. Leave `XCELSIOR_WISE_PAYOUTS_ENABLED=0` for now.
4. Apply migration 159 and deploy.
5. Subscribe the webhook (once per environment):

   ```bash
   curl -s -X POST -H "Authorization: Bearer $XCELSIOR_WISE_API_TOKEN" -H "Content-Type: application/json" "https://api.wise.com/v3/profiles/$XCELSIOR_WISE_PROFILE_ID/subscriptions" -d '{"name":"xcelsior-transfers","trigger_on":"transfers#state-change","delivery":{"version":"4.0.0","url":"https://xcelsior.ca/api/billing/wise/webhook"}}'
   ```

   Wise sends a test delivery; it is recorded in `wise_webhook_inbox` with outcome `test`. The poller covers a missed delivery, so the webhook is an accelerator, not a dependency.
6. Check `GET /api/admin/wise/status`: it shows the switches, the caps and the CAD balance.
7. Set `XCELSIOR_WISE_PAYOUTS_ENABLED=1` and restart the API and the background worker.

If payment approvals are configured on the Wise account, transfers created by API are refused ("missing approval"). The payout stops at `needs_review` with that message.

## Adding a provider

1. In Wise, add the provider as a recipient (Wise collects and checks the bank details, or use their Wise email or Wisetag).
2. Copy the recipient id and link it: `POST /api/admin/wise/providers/{provider_id}/recipient` with `{"recipient_id": …}`.
3. The provider is emailed at their address on file. The cool-off starts.

Provider self-service by Wise email (`POST /api/providers/{id}/wise/recipient`) is off unless `XCELSIOR_WISE_EMAIL_RECIPIENTS=1`. Wise asks to be contacted before email recipients are used, they are not available on every route, and they cannot be tested in sandbox.

## When something goes wrong

| Payout status | Meaning | What to do |
|---|---|---|
| `unknown` | Wise did not answer, or answered 5xx, when the transfer was created | Nothing. The worker retries with the same idempotency id, so there is at most one transfer. |
| `awaiting_funding` with an error | Auto-fund was refused or the balance is short | Top up the CAD balance or fund the transfer in the Wise app. Wise cancels an unfunded transfer after 14 days. |
| `needs_review`, transfer `cancelled` or `funds_refunded` | The money is back in the balance | `POST …/{id}/release`. The rows return to the queue and are batched again. |
| `needs_review`, transfer `bounced_back` | Wise is still deciding: it redelivers or refunds | Wait. If it is redelivered, the payout returns to `sent` on its own. Release is refused until Wise says cancelled or refunded. |
| `needs_review`, no transfer id | Wise refused the transfer outright | Search Wise for reference `XCL-<payout id>`. If there is none, release with `{"confirm_no_transfer": true}`. |
| `needs_review`, "destination no longer matches" | The recipient was edited or deactivated at Wise after approval | Confirm the change with the provider out of band, link the recipient again, release. |

A provider with a payout in `needs_review` gets no new payout until it is released.

To stop everything: set `XCELSIOR_WISE_PAYOUTS_ENABLED=0` and restart. Nothing is approved or created after that. Transfers already created at Wise are cancelled in the Wise app.

## Sandbox

`XCELSIOR_WISE_ENV=sandbox` uses `api.wise-sandbox.com` and the sandbox signing key. Wise notes the sandbox is most stable for GBP, USD and EUR; CAD may misbehave. Transfer states are advanced with `GET /v1/simulation/transfers/{id}/{processing|funds_converted|outgoing_payment_sent|bounced_back|funds_refunded}`.

## Not built

- A dashboard screen. The routes exist; the provider and admin pages do not call them yet.
- Bank-detail entry by the provider (Wise's dynamic account-requirements form).
- Payouts in a source currency other than CAD: settlement itself requires CAD.
- MCP tools. Approval is deliberately not available to agents.
