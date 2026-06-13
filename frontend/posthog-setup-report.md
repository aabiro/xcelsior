<wizard-report>
# PostHog post-wizard report

The wizard has completed a deep integration of PostHog analytics into the Xcelsior GPU compute platform (Next.js App Router). The integration covers client-side initialization via `instrumentation-client.ts` (Next.js 15.3+ pattern), a server-side PostHog client for API routes, reverse-proxy rewrites in `next.config.ts` to route analytics traffic through `/ingest`, user identification on login and page-load session probes, and 11 carefully chosen business events spanning the full user lifecycle — from pricing discovery through GPU launch and billing.

| Event | Description | File |
|---|---|---|
| `user_registered` | User completes email/password registration | `src/app/(marketing)/register/page.tsx` |
| `user_logged_in` | User authenticates and session is established (method: password, mfa_totp, mfa_sms, mfa_passkey) | `src/app/(marketing)/login/page.tsx` |
| `oauth_initiated` | User clicks GitHub / Google / Hugging Face OAuth button | `src/app/(marketing)/login/page.tsx`, `register/page.tsx` |
| `mfa_challenged` | MFA challenge triggered after password login | `src/app/(marketing)/login/page.tsx` |
| `pricing_page_viewed` | Public pricing page visited (top of conversion funnel) | `src/app/(marketing)/pricing/content.tsx` |
| `gpu_instance_launched` | GPU compute instance launched successfully | `src/components/instances/launch-instance-modal.tsx` |
| `serverless_endpoint_deployed` | Serverless inference endpoint deployed via Deploy Studio | `src/features/serverless/deploy-studio.tsx` |
| `free_credits_claimed` | $10 CAD free signup credits claimed from billing page | `src/app/(dashboard)/dashboard/billing/page.tsx` |
| `credits_deposit_initiated` | User opens the Stripe top-up deposit modal | `src/app/(dashboard)/dashboard/billing/page.tsx` |
| `auto_topup_configured` | Auto-reload wallet settings saved | `src/app/(dashboard)/dashboard/billing/page.tsx` |
| `payment_method_added` | Credit/debit card saved via Stripe SetupIntent | `src/components/billing/payment-method-modal.tsx` |

**User identification** is performed in `src/lib/auth.tsx` on both fresh logins (`login()` callback) and returning visitor session probes (page-load `getMe()` call), using `user_id` as the distinct ID with `email`, `name`, `role`, `country`, and `province` as person properties. `posthog.reset()` is called on logout.

**Error tracking** (`posthog.captureException`) is wired to all critical catch blocks: registration, login, OAuth initiation, GPU instance launch errors, serverless deploy errors, and card-save errors.

**Files created:**
- `frontend/instrumentation-client.ts` — PostHog client-side init (Next.js 15.3+ pattern, uses reverse proxy)
- `frontend/src/lib/posthog-server.ts` — Singleton PostHog Node.js client for server-side use
- `frontend/.env.local` — `NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN` and `NEXT_PUBLIC_POSTHOG_HOST`

**Files modified:**
- `frontend/next.config.ts` — Added PostHog reverse proxy rewrites (`/ingest/*`) and CSP `connect-src` entries
- `frontend/src/lib/auth.tsx` — Added `posthog.identify()` on login/session-probe, `posthog.reset()` on logout
- `frontend/src/app/(marketing)/login/page.tsx` — `user_logged_in`, `oauth_initiated`, `mfa_challenged`
- `frontend/src/app/(marketing)/register/page.tsx` — `user_registered`, `oauth_initiated`
- `frontend/src/app/(marketing)/pricing/content.tsx` — `pricing_page_viewed`
- `frontend/src/components/instances/launch-instance-modal.tsx` — `gpu_instance_launched`
- `frontend/src/features/serverless/deploy-studio.tsx` — `serverless_endpoint_deployed`
- `frontend/src/app/(dashboard)/dashboard/billing/page.tsx` — `free_credits_claimed`, `credits_deposit_initiated`, `auto_topup_configured`
- `frontend/src/components/billing/payment-method-modal.tsx` — `payment_method_added`

> **Install required:** Run `npm install posthog-js posthog-node` in the `frontend/` directory to complete setup.

## Next steps

We've built some insights and a dashboard for you to keep an eye on user behavior, based on the events we just instrumented:

- [Analytics basics (wizard) — Dashboard](https://us.posthog.com/project/397400/dashboard/1707940)
- [Signups & Logins (wizard)](https://us.posthog.com/project/397400/insights/GhDg0LRc)
- [GPU & Inference Launches (wizard)](https://us.posthog.com/project/397400/insights/0g4Y75iC)
- [Pricing to Signup Conversion Funnel (wizard)](https://us.posthog.com/project/397400/insights/CQDlxJbq)
- [Signup to First GPU Launch Funnel (wizard)](https://us.posthog.com/project/397400/insights/vondttnE)
- [Billing & Credits Activity (wizard)](https://us.posthog.com/project/397400/insights/FE61Acua)

### Agent skill

We've left an agent skill folder in your project at `.claude/skills/integration-nextjs-app-router/`. You can use this context for further agent development when using Claude Code. This will help ensure the model provides the most up-to-date approaches for integrating PostHog.

</wizard-report>
