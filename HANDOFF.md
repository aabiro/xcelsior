# Xcelsior — Work Handoff

Living handoff for the next agent. Pick up the **Open items** below.

## How to work in this repo
- Frontend: Next.js in `frontend/`. Typecheck with `cd frontend && npx tsc --noEmit` (must be exit 0, no `error TS`).
- i18n: flat dicts in `frontend/src/lib/i18n/en.ts` + `fr.ts`. **After editing, run `node scripts/split-i18n.mjs`** (regenerates the `*-dashboard.ts` / `*-public.ts` bundles). Keep en + fr in sync.
- Backend: FastAPI at repo root (`routes/*.py`, etc.). Smoke-check with `venv/bin/python -c "import routes.<mod>"`.
- **Three runtimes** exist — don't conflate: dev DB (`_get_pg_pool` → `xcelsior`), docker-test (`127.0.0.1:9501`, `xcelsior_test`, A100 seed data), and real prod (`https://xcelsior.ca`). Query prod via API with `Authorization: Bearer $XCELSIOR_API_TOKEN` (from `.env`); its Postgres isn't reachable with the dev password.
- **Leave alone** (modified outside these sessions): `bg_worker.py`, `worker_agent.py`, and the PostHog additions in `pricing/content.tsx` + `deploy-studio.tsx`.

## Completed in prior sessions (verified, tsc green)
Deploy-studio draft persistence + Save&Exit; serverless/mobile i18n keys; artifacts 404; volumes; earnings Stripe/PayPal; PWA banner; templates flash; instance rename dialog; silent-logout refresh retry; SMS 2FA; pricing page redesign; reputation journey; right-rail AI spark; light-theme cards; launch-modal global bus; session copy; serverless worker fleet; web terminal refactor; Beta badge/logo crop; download-page copy cleanup.

## Completed this session (verified, tsc green, wizard auth tests pass)
- **ConfirmDialog sweep** — settings SecurityTab (TOTP/SMS/all-MFA/passkey disable) + PrivacyTab export; templates single-delete; inference endpoint delete. No native `confirm()` left in dashboard.
- **API settings** — OAuth `SecretBanner` click-to-copy on ID/secret; auto-copy secret on create/rotate; callout clarifying `xoa_` session tokens vs `oauth_` client IDs; wizard now provisions OAuth client for **all modes** (rent scopes: instances+billing; provide/both: hosts) — visible in Settings → API as "CLI Wizard …".
- **GPU marketing page** — full redesign in `gpu-availability/content.tsx`: aurora hero, stat strip, sovereign story + canada map, live fleet grid with spot badges, CTA. i18n keys `gpus.*` in en+fr.
- **Wizard auth** — `requestDeviceCode` + `pollDeviceToken` now send `client_id: "xcelsior-cli"` (fixes HTTP 401); manual token validation rejects `oauth_` IDs, requires `xoa_` prefix.
- **Hexara context-aware movement** — `index.tsx` branch map extended: idle state reacts to step type (auto-check pass/fail/running, mode select, confirm, device-auth waiting, etc.).

---

## OPEN ITEMS

_(none from the prior handoff batch — pick up new work from user)_
