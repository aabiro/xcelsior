# PayPal marketplace — webhooks, E2E testing, sandbox credentials

Sandbox credentials live in [`.env.paypal-sandbox.example`](../.env.paypal-sandbox.example) (committed, fake money only).

---

## Webhook checklist

**Endpoint URL:** `https://xcelsior.ca/api/billing/paypal/webhook`

**Env:** `PAYPAL_WEBHOOK_ID` must match the webhook ID shown in the Developer Dashboard after creation.

### Dashboard setup (step-by-step)

1. Open [developer.paypal.com → Apps & Credentials](https://developer.paypal.com/dashboard/applications) → your **REST app** (sandbox or live).
2. Scroll to **Webhooks** → **Add Webhook**.
3. **Webhook URL:** `https://xcelsior.ca/api/billing/paypal/webhook`
4. Subscribe to the events below (search by name in the event picker).
5. Save and copy the **Webhook ID** into `PAYPAL_WEBHOOK_ID`.
6. Use **Simulate webhook** in the dashboard to verify `200` responses from production.

### Events to subscribe

#### Wallet deposits (customer credits)

| Event | Required | Handler behavior |
|-------|----------|------------------|
| `PAYMENT.CAPTURE.COMPLETED` | Yes | Credits wallet from `custom_id` (customer email/id); skips marketplace `provider:job` orders |
| `CHECKOUT.ORDER.COMPLETED` | Yes | Backup credit path when capture event is delayed |

Optional (logging only today): `PAYMENT.CAPTURE.DENIED`, `PAYMENT.CAPTURE.REFUNDED`, `CHECKOUT.ORDER.CANCELLED`

#### Provider onboarding (marketplace sellers)

PayPal’s sandbox event picker typically offers **only these two** onboarding-related events — that is sufficient. Do **not** wait for `PRODUCT-SUBSCRIPTION.CREATED`; it is not listed in sandbox and is not required.

| Event | Required | Handler behavior |
|-------|----------|------------------|
| `MERCHANT.ONBOARDING.COMPLETED` | Yes | Sets `provider_accounts.paypal_status = active`, stores merchant/payer IDs |
| `CUSTOMER.MERCHANT-INTEGRATION.PRODUCT-SUBSCRIPTION.UPDATED` | Yes | Same handler — refreshes seller status when capabilities or PARTNER_FEE consent change |

### Post-setup verification

- [ ] `PAYPAL_WEBHOOK_ID` set in production `.env`
- [ ] `GET /api/billing/paypal/enabled` → `{ "enabled": true, "platform_mode": true }`
- [ ] Simulate `MERCHANT.ONBOARDING.COMPLETED` → server logs `PayPal provider onboarding webhook`
- [ ] Wallet deposit credits **once** per order (idempotency key `paypal-{order_id}`)

---

## E2E walkthrough

### A. Wallet deposit (customer credits)

**Goal:** Customer adds CAD credits via PayPal popup; balance increases exactly once.

1. Log in at https://xcelsior.ca
2. **Dashboard → Billing → Add Credits**
3. Enter amount ≥ **$5.00 CAD**
4. Click **Pay with PayPal** — popup opens (`sandbox.paypal.com` in sandbox mode)
5. Approve payment:
   - **Sandbox buyer login:** see `PAYPAL_SANDBOX_BUYER_EMAIL` in `.env.paypal-sandbox.example`
   - **Guest card:** `4032036301886261` / `03/2028` / CVC `335`
6. Popup closes; toast shows credited amount
7. **Verify:**
   - Billing page wallet balance increased
   - `GET /api/billing/wallet/{customer_id}` shows new balance
   - No duplicate credit on page refresh (idempotent capture + webhook)

### B. Provider PayPal Connect (earnings UI)

**Goal:** Provider completes PayPal seller onboarding; Earnings page shows connected state.

**Prerequisite:** User has a provider account (complete Stripe setup on Earnings first, or existing provider).

1. Log in as provider → **Dashboard → Earnings**
2. Scroll to **Payout Methods** — two cards: **Stripe Connect** and **PayPal**
3. On the PayPal card, confirm:
   - PayPal logo + “Instant marketplace payouts” subtitle
   - Three benefit bullets (instant disbursement, platform fee, dual-rail)
   - Yellow **Connect PayPal** button
4. Click **Connect PayPal** → redirect to PayPal hosted seller signup (sandbox)
5. Complete business/seller verification on PayPal
6. Return to Xcelsior (`/dashboard/earnings?paypal=return`)
   - Toast: “Verifying your PayPal seller status…”
   - Card polls automatically (up to ~30 s)
7. **Connected state** (production UI):
   - Blue gradient card border + “Connected” badge
   - “PayPal seller account connected” banner
   - “Connected since {date}” when onboarded
8. If webhook is slow, click **Check status** (manual `POST /api/providers/{id}/paypal/refresh`)
9. **Verify API:** `GET /api/providers/{id}/paypal` → `"status": "active"`

**Return URL params (handled automatically):**

| Param | Meaning |
|-------|---------|
| `?paypal=return` | Onboarding finished — poll until active |
| `?paypal=refresh` | User left early — show “Resume PayPal Setup” |

### C. Marketplace payment with platform fee

**Prerequisite:** Provider PayPal **active** (step B).

1. Authenticated customer creates order:
   ```http
   POST /api/billing/paypal/marketplace/create-order
   Authorization: Bearer <token>
   Content-Type: application/json

   {
     "customer_id": "you@example.com",
     "provider_id": "your-provider-id",
     "job_id": "job-e2e-001",
     "amount_cad": 50.0
   }
   ```
2. Open checkout: `https://www.sandbox.paypal.com/checkoutnow?token={order_id}`
3. Approve as sandbox buyer
4. Capture:
   ```http
   POST /api/billing/paypal/marketplace/capture-order
   {
     "customer_id": "you@example.com",
     "provider_id": "your-provider-id",
     "order_id": "{order_id}"
   }
   ```
5. **Verify:**
   - `payout_splits` row: `payment_rail=paypal`, `platform_share_cad` ≈ 15% of total
   - Earnings → **Payout History** shows **PayPal** rail badge
   - Wallet **not** credited (marketplace `custom_id` is `provider_id:job_id`)

### D. Shortcut payout API (admin/provider)

```http
POST /api/providers/{id}/payout?job_id=job-1&total_cad=50&payment_rail=paypal
```

Returns `order_id` — complete popup + marketplace capture as in step C.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| 403 on `create-order` | App missing payment scopes | Enable `payment/authcapture` on REST app |
| `PAYEE_ACCOUNT_INVALID` in sandbox | Live merchant_id as payee | Set `PAYPAL_PLATFORM_PAYEE_EMAIL` for sandbox |
| PayPal card shows “not configured” | Missing client id/secret | Set `PAYPAL_CLIENT_ID` + `PAYPAL_CLIENT_SECRET` |
| Stuck “In progress” after onboarding | Webhook not subscribed or slow | Subscribe `MERCHANT.ONBOARDING.COMPLETED` + `PRODUCT-SUBSCRIPTION.UPDATED`; click **Check status** |
| Double wallet credit | — | Should not happen; idempotency on `paypal-{order_id}` |

---

## Environment reference

| Variable | Sandbox value | Notes |
|----------|---------------|-------|
| `PAYPAL_PLATFORM_PAYEE_EMAIL` | `aaryn.biro@xcelsior.ca` | Wallet deposit payee |
| `PAYPAL_PLATFORM_MERCHANT_ID` | `BEC6DEHNQBV32` | Live marketplace payee |
| `PAYPAL_PARTNER_ATTRIBUTION_ID` | `4819757953265067229` | Partner BN code (not merchant_id) |
| `PAYPAL_SANDBOX_BUYER_EMAIL` | `aaryn.biro@xcelsior.ca` | Checkout popup login |
| `PAYPAL_WEBHOOK_ID` | `5Y031871B5304413V` | Sandbox webhook |

Platform fee: `XCELSIOR_PLATFORM_CUT` (default `0.15` = 15%).