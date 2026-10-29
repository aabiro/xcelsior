# Demo Account & IP Whitelist

A standing demo **admin** account that makes one-click login possible from the
owner's own networks, and gives Playwright / E2E automation a reliable way past
the auth gate. Set up once, then left alone.

## Credentials (fixed — do not rotate casually)

| Field | Value |
|-------|-------|
| Email | `demo@xcelsior.ca` |
| Password | `DemoUser123abc!` |
| Role | admin (`is_admin=1`, `role=admin`), email-verified, no MFA |

Source of truth: **`demo_account.py`** (repo root). Everything else reads from it.

## How it works

1. **Login page** (`/login`) calls `GET /api/auth/demo-credentials` on mount.
2. That endpoint returns the credentials **only if the request IP is whitelisted**
   (otherwise `403`). On success the page shows a **"Demo account"** button right
   below *Sign in*; one click fills the email + password fields.
3. You click *Sign in* as normal — it's a real login, not an auth bypass.

The IP whitelist only controls whether the button renders and whether the
password is handed out. See the security note below.

## The whitelist

Determined live on the ASUS host (`aaryn-tuf-rtx2060`) on 2026-07-15:

| Network | CIDR | What it is |
|---------|------|------------|
| Public IP | `153.68.20.238/32` | Home / ASUS egress |
| Tailscale — ASUS | `100.64.0.6/32` | `aaryn-tuf-rtx2060` |
| Tailscale — Mac | `100.64.0.3/32` | Volume/NFS server |
| Tailscale IPv6 | `fd7a:115c:a1e0::/48` | Whole tailnet (v6) |
| Home LAN | `192.168.1.0/24` | Covers both machines' LAN IPs |
| Localhost | `127.0.0.1/32`, `::1/128` | Local dev + smoke server |

Overrides (env):

- `XCELSIOR_DEMO_IP_WHITELIST` — comma-separated CIDRs/IPs; **replaces** the defaults.
- `XCELSIOR_DEMO_IP_WHITELIST_EXTRA` — comma-separated; **appended** to whatever's active.
- `XCELSIOR_DEMO_ENABLED=0` — kill switch: endpoint 404s, button disappears (account stays).
- `XCELSIOR_DEMO_EMAIL` / `XCELSIOR_DEMO_PASSWORD` — override the creds.

## Seeding (run once per database)

Idempotent, clearly-labelled seed script — safe to re-run, converges the account
to the known state and touches nothing else:

```bash
# Test DB
XCELSIOR_ENV=test python scripts/seed_demo_account.py
# Production (inside the api container / on the VPS with real DB env)
python scripts/seed_demo_account.py
# Report state without writing
python scripts/seed_demo_account.py --check
```

## Playwright / E2E

Helper: **`frontend/e2e/helpers/demo-auth.ts`** — two modes:

- `loginAsDemo(page)` — **real** login against a live backend (fetches the IP-gated
  creds, POSTs `/api/auth/login`, cookie lands in the context). Use when `baseURL`
  points at a running API.
- `installDemoAuthMock(context)` — **frontend-only** bypass (no backend): mocks
  `/api/auth/me` + `/api/auth/refresh` as the demo admin. Use for local smoke runs
  where `/api/**` rewrites to prod.
- `ensureDemoAuth(page)` — auto-picks: real when `DEMO_REAL_AUTH=1` or
  `XCELSIOR_PWA_SMOKE_BASE_URL` is set, else the mock.

Wired in: `e2e/demo-auth.spec.ts` (proves it), `e2e/pwa-push-staging.spec.ts`
(real login via the demo button), `scripts/audit-dashboard.mjs` (defaults to the
demo creds). `serverless-smoke` / `mcp-visual` already carry their own auth mock.

Run the proof against the live dev server (no rebuild):

```bash
cd frontend
XCELSIOR_PWA_SMOKE_BASE_URL=http://127.0.0.1:3000 npx playwright test e2e/demo-auth.spec.ts
```

## ⚠️ Security note — read before widening the whitelist

- This is a **real admin account with a fixed, known password.** The whitelist
  decides whether the password is *revealed* and whether the button *renders* —
  the account is still reachable by anyone who has the password. Treat the
  password as semi-public; the whitelist is its only guard.
- The client IP comes from proxy headers (`cf-connecting-ip` / `x-real-ip` /
  `x-forwarded-for`). That is trustworthy **only** if Cloudflare/nginx set them and
  strip client-supplied ones. Confirm that before trusting the gate in prod, or an
  attacker could spoof `X-Forwarded-For` to be handed the password.
- To fully disable without deleting anything: `XCELSIOR_DEMO_ENABLED=0`.
