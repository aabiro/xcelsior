# Billing money path — runbook

How credits enter customer wallets and how provider payouts leave the platform.

## Architecture decisions (2026-07-28)

| Decision | Choice |
|----------|--------|
| Customer SoT | Prepaid CAD wallet (not Stripe Subscriptions) |
| Stripe Billing Meters | Dual-write observability only (`stripe_meter_event_outbox`) |
| Usage cadence | ~5 min hosted ticks; serverless ≥60s slices — not literal per-ms |
| Zero balance | **Hard stop** (no free-ride grace charges); low-balance warning before $0 |
| Provider settlement | **Daily** queue + **manual instant** payout API when float allows |
| Connect | Express + embedded AccountSession; global cross-border countries |
| Crypto top-up | BTC + Lightning only (no Stripe Crypto Onramp) |
| Stripe SDK | `stripe>=15.3` / API `2026-06-24.dahlia` |

## Customer credits (wallet deposits)

| Path | Endpoint | Proof required | Idempotency key |
|------|----------|----------------|-----------------|
| Stripe card | `POST /api/billing/payment-intent` → webhook | `payment_intent.succeeded` | `stripe:{pi_id}` |
| PayPal | `POST /api/billing/paypal/create-order` → capture | PayPal capture / webhook | `paypal-{order_id}` |
| Crypto | `POST /api/billing/crypto/deposit` | On-chain confirmation | per deposit_id |
| Lightning | `POST /api/billing/lightning/deposit` | Invoice paid | per deposit_id |
| Direct API | `POST /api/billing/wallet/{id}/deposit` | **Blocked in prod** (403) | — |
| Promo | `POST /api/billing/wallet/{id}/free-credits` | One-time per account | `free-credits-{customer_id}` |

**Important:** PayPal on the Billing page only **tops up the customer wallet**. It does not route money to GPU providers. Jobs always charge wallet balance regardless of how credits were added.

## Provider payouts

| Rail | When used | Requires |
|------|-----------|----------|
| Stripe (default) | `POST /api/providers/{id}/payout` | Stripe Connect `active` |
| PayPal marketplace | `payment_rail=paypal` on payout API | PayPal seller `active` |

Each job creates **one** `payout_splits` row with **one** `payment_rail`. Connecting both Stripe and PayPal does not double-pay.

## Spot instance metering

Spot jobs use `pricing_mode=spot` with a **locked** `spot_rate_cad` at allocation time.

| Stage | Renter wallet | Provider payout |
|-------|---------------|-----------------|
| Launch preflight | ~1 hr at live spot quote | — |
| Running | `spot_rate_cad` × seconds | Split from allocation `price_cents_per_hour` |
| Preemption | Meter closes at `preempted_at` | Paid for actual runtime only |
| Requeue | Next assignment may lock a new rate | — |

Bidding (`max_bid`) is retired — spot price is the published rate, not a customer bid.

## Security gates (production)

- All customer-scoped billing routes: `_require_customer_access` (auth + ownership)
- Direct wallet deposit: `_allow_direct_wallet_deposit` → false in prod (admins/test only)
- Payment mutations: `_check_billing_payment_rate_limit` (per-customer, per-action)
- PayPal webhooks: signature verification via `PAYPAL_WEBHOOK_ID`

## Observability

Search logs for these patterns:

| Log prefix / message | Meaning |
|---------------------|---------|
| `billing.direct_deposit_blocked` | User hit blocked direct-deposit path (stale client or abuse probe) |
| `PayPal webhook credit:` | Wallet credited from webhook backup |
| `PayPal CAPTURE.COMPLETED marketplace` | Marketplace order skipped for wallet credit |
| `PayPal provider onboarding webhook` | Provider PayPal status updated |

## Ops scripts

```bash
# 1. Wallet audit — deposits without payment proof
python scripts/audit_wallet_deposits.py
# On VPS: docker compose exec -T api-blue python scripts/audit_wallet_deposits.py

# 2. Production smoke (auth, 403 deposit, payment-intent IDOR, PayPal enabled)
python scripts/billing_prod_smoke.py
```

Exit code **1** from wallet audit means suspicious deposits found — investigate `suspicious` rows in JSON output.

## Billing hardening checklist (items 1–6)

| # | Item | Status | How to verify |
|---|------|--------|---------------|
| 1 | Wallet audit | `scripts/audit_wallet_deposits.py` | Run on prod DB; expect 0 suspicious |
| 2 | E2E / regression tests | `tests/test_billing_*.py`, `tests/test_paypal_*.py` | `pytest tests/test_billing_endpoints_coverage.py tests/test_billing_security_sweep.py tests/test_paypal_webhook.py` |
| 3 | PayPal + Lightning smoke | `scripts/billing_prod_smoke.py` | Checks enabled flags + auth gates |
| 4 | Rate limits | `routes/_deps.py` `_check_billing_payment_rate_limit` | `test_billing_payment_rate_limit` |
| 5 | Observability + runbook | This doc + structured logs in `routes/billing.py` | Grep prod logs for patterns above |
| 6 | Security sweep | `tests/test_billing_security_sweep.py` | Auth + IDOR on all payment mutations |

## PayPal vs Stripe clarity

- **Customers:** Choose funding method when adding credits (Billing → Add Credits). Jobs debit wallet only.
- **Providers:** Stripe required to register. PayPal optional for marketplace disbursement rail.
- **No linkage:** Customer PayPal deposit ≠ provider PayPal payout for that job.

See also: [`docs/paypal-marketplace-e2e.md`](paypal-marketplace-e2e.md) for webhooks and sandbox E2E.

## PayPal provider payout failures (on-call)

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `400 Provider has not completed PayPal onboarding` | `paypal_status` ≠ `active` | Provider completes Connect card on Earnings; check `GET /api/providers/{id}/paypal` |
| `400 PayPal is not configured` | Missing `PAYPAL_CLIENT_ID` / secret on API | Set env on api-blue/green; verify `GET /api/billing/paypal/enabled` |
| `502 PayPal provider onboarding failed` | Partner referral API error | Check PayPal partner dashboard; sandbox vs live mode |
| Duplicate `payout_splits` concern | Re-capture same job | Safe — `capture_marketplace_order` is idempotent per `job_id` + `payment_rail=paypal` |
| Wallet credited on marketplace order | Webhook misclassified | `custom_id` must be `provider_id:job_id`; wallet credit skipped for marketplace pattern |

**Support macro:** Customer PayPal wallet top-up does **not** pay the provider for a job. Provider marketplace payouts use `payment_rail=paypal` on `POST /api/providers/{id}/payout` after seller onboarding is `active`.