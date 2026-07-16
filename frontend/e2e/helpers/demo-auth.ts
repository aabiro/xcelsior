// Demo-account auth for Playwright — the standing "get past the auth gate" helper.
//
// Two modes, because our e2e runs in two very different setups:
//
//   1. REAL backend (staging/prod/full local stack) → loginAsDemo(): fetches the
//      IP-gated demo credentials from the server (falls back to the known
//      defaults) and POSTs /api/auth/login so the httpOnly session cookie lands
//      in the browser context. After that, page.goto() is authenticated.
//
//   2. FRONTEND-ONLY smoke (the default: standalone Next server with no API, where
//      /api/** rewrites to prod) → installDemoAuthMock(): intercepts the auth
//      probes so the app believes the demo admin is signed in. No backend needed.
//
// ensureDemoAuth() picks the right mode automatically. Credentials mirror the
// backend single source of truth (demo_account.py); env vars override.
import type { BrowserContext, Page, Route } from "@playwright/test";

export const DEMO_EMAIL = process.env.DEMO_EMAIL?.trim() || "demo@xcelsior.ca";
export const DEMO_PASSWORD = process.env.DEMO_PASSWORD?.trim() || "DemoUser123abc!";

/** Shape of /api/auth/me's user for the demo admin (used by the mock). */
export const DEMO_USER = {
  user_id: "user-demo",
  email: DEMO_EMAIL,
  name: "Demo Account",
  role: "admin",
  is_admin: true,
  customer_id: "cust-demo",
  team_can_write_instances: true,
  team_can_manage_billing: true,
} as const;

function json(body: unknown, status = 200) {
  return { status, contentType: "application/json", body: JSON.stringify(body) };
}

/**
 * REAL login against a live backend. Returns true when the session cookie is set.
 * Use when baseURL points at a running API. No-op-safe: returns false instead of
 * throwing so callers can fall back to the mock.
 */
export async function loginAsDemo(page: Page): Promise<boolean> {
  let email = DEMO_EMAIL;
  let password = DEMO_PASSWORD;
  // Prefer the server's IP-gated creds when reachable (keeps this in lockstep
  // with whatever demo_account.py hands out), else use the defaults above.
  try {
    const credsRes = await page.request.get("/api/auth/demo-credentials");
    if (credsRes.ok()) {
      const c = await credsRes.json();
      if (c?.email && c?.password) {
        email = c.email;
        password = c.password;
      }
    }
  } catch {
    /* not reachable — use defaults */
  }
  try {
    const res = await page.request.post("/api/auth/login", { data: { email, password } });
    if (!res.ok()) return false;
    const body = await res.json().catch(() => ({}));
    if (body?.mfa_required) return false; // demo has no MFA, but never hang on a challenge
    return true;
  } catch {
    return false;
  }
}

/**
 * FRONTEND-ONLY bypass — no backend required. Sets the session cookie and
 * intercepts /api/auth/me + /api/auth/refresh so the SPA renders as the demo
 * admin. Register this BEFORE any test-specific /api mocks (Playwright runs the
 * most recently registered matching route first, so auth stays authoritative).
 */
export async function installDemoAuthMock(
  context: BrowserContext,
  opts: { user?: Record<string, unknown>; baseURL?: string } = {},
): Promise<void> {
  const user = opts.user ?? DEMO_USER;
  const base = opts.baseURL ?? "http://127.0.0.1:3100";
  try {
    await context.addCookies([
      { name: "xcelsior_session", value: "demo-session", url: base },
    ]);
  } catch {
    /* cookie best-effort */
  }
  await context.route("**/api/auth/me", (route: Route) =>
    route.fulfill(json({ ok: true, user })),
  );
  await context.route("**/api/auth/refresh", (route: Route) =>
    route.fulfill(json({ ok: true, access_token: "demo", expires_in: 3600 })),
  );
}

/**
 * Auto-pick the mode. Real login when DEMO_REAL_AUTH=1 or a remote base URL is
 * configured (XCELSIOR_PWA_SMOKE_BASE_URL); otherwise the frontend mock. Returns
 * which mode was used.
 */
export async function ensureDemoAuth(page: Page): Promise<"real" | "mock"> {
  const preferReal =
    process.env.DEMO_REAL_AUTH === "1" || !!process.env.XCELSIOR_PWA_SMOKE_BASE_URL?.trim();
  if (preferReal && (await loginAsDemo(page))) return "real";
  await installDemoAuthMock(page.context());
  return "mock";
}
