#!/usr/bin/env node
/**
 * Authenticated dashboard crawl for MCP / post-deploy audits.
 * Requires .env.audit (see scripts/provision_audit_user.sh) or AUDIT_* env vars.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = process.env.AUDIT_OUT || "/tmp/xcelsior-audit";

const DASHBOARD_ROUTES = [
  "/dashboard",
  "/dashboard/instances",
  "/dashboard/hosts",
  "/dashboard/billing",
  "/dashboard/settings",
  "/dashboard/notifications",
  "/dashboard/marketplace",
  "/dashboard/volumes",
  "/dashboard/analytics",
  "/dashboard/inference",
  "/dashboard/compliance",
];

// Falls back to the standing demo admin (demo@xcelsior.ca) so this never dies at
// the auth gate — the demo login works from the owner's whitelisted networks.
// Mirrors demo_account.py; override with AUDIT_EMAIL/AUDIT_PASSWORD or .env.audit.
const DEMO_EMAIL = process.env.DEMO_EMAIL || "demo@xcelsior.ca";
const DEMO_PASSWORD = process.env.DEMO_PASSWORD || "DemoUser123abc!";

function loadEnvAudit() {
  const env = {
    base: (process.env.AUDIT_BASE || "https://xcelsior.ca").replace(/\/$/, ""),
    email: process.env.AUDIT_EMAIL || "",
    password: process.env.AUDIT_PASSWORD || "",
  };
  const p = path.join(REPO, ".env.audit");
  if (fs.existsSync(p)) {
    for (const line of fs.readFileSync(p, "utf8").split("\n")) {
      const s = line.trim();
      if (!s || s.startsWith("#") || !s.includes("=")) continue;
      const [k, v] = s.split("=", 2);
      const val = v.trim().replace(/^["']|["']$/g, "");
      if (k === "AUDIT_BASE" && !process.env.AUDIT_BASE) env.base = val.replace(/\/$/, "");
      if (k === "AUDIT_EMAIL" && !process.env.AUDIT_EMAIL) env.email = val;
      if (k === "AUDIT_PASSWORD" && !process.env.AUDIT_PASSWORD) env.password = val;
    }
  }
  if (!env.email) env.email = DEMO_EMAIL;
  if (!env.password) env.password = DEMO_PASSWORD;
  return env;
}

const NAV_TIMEOUT_MS = Number(process.env.AUDIT_NAV_TIMEOUT_MS || 90000);
const SETTLE_MS = Number(process.env.AUDIT_ROUTE_SETTLE_MS || 2000);

async function sessionOk(page, base) {
  const authMe = await page
    .request.get(`${base}/api/auth/me`)
    .then((r) => r.json())
    .catch(() => null);
  const finalUrl = page.url();
  const authed = Boolean(authMe?.user?.email);
  const onDashboard = authed && !finalUrl.includes("/login");
  return {
    authEmail: authMe?.user?.email || null,
    onDashboard,
    finalUrl,
  };
}

async function waitForDashboard(page, base, routePath, { timeoutMs = 15000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  let last = await sessionOk(page, base);
  while (Date.now() < deadline) {
    if (last.onDashboard && last.finalUrl.includes(routePath)) return last;
    if (last.authEmail && last.finalUrl.includes("/login")) {
      await page.waitForTimeout(500);
      last = await sessionOk(page, base);
      continue;
    }
    if (last.onDashboard) return last;
    await page.waitForTimeout(500);
    last = await sessionOk(page, base);
  }
  return last;
}

async function gotoResilient(page, url, { attempts = 2 } = {}) {
  const waitStrategies = ["networkidle", "load"];
  let lastError = null;
  let status = null;

  for (let attempt = 0; attempt < attempts; attempt++) {
    for (const waitUntil of waitStrategies) {
      try {
        const res = await page.goto(url, { waitUntil, timeout: NAV_TIMEOUT_MS });
        status = res?.status() ?? null;
        await page.waitForTimeout(SETTLE_MS);
        return { status, error: null };
      } catch (e) {
        lastError = e;
        if (!/ERR_ABORTED|Timeout/i.test(e.message)) break;
      }
    }
  }

  return {
    status,
    error: lastError ? String(lastError.message).split("\n")[0] : "navigation failed",
  };
}

async function login(page, base, email, password) {
  const loginRes = await page.request.post(`${base}/api/auth/login`, {
    data: { email, password },
    headers: { "Content-Type": "application/json" },
    timeout: Number(process.env.AUDIT_FETCH_TIMEOUT_MS || 30000),
  });
  const body = await loginRes.json().catch(() => ({}));
  if (!loginRes.ok()) {
    throw new Error(`login ${loginRes.status()}: ${JSON.stringify(body).slice(0, 200)}`);
  }
  if (body.mfa_required) {
    throw new Error("MFA required — audit user must have MFA disabled");
  }
  const nav = await gotoResilient(page, `${base}/dashboard`);
  const session = await waitForDashboard(page, base, "/dashboard", { timeoutMs: 20000 });
  if (nav.error && !session.onDashboard) {
    throw new Error(`dashboard login navigation failed: ${nav.error}`);
  }
  if (!session.onDashboard) {
    throw new Error(`still on login after API auth: ${session.finalUrl}`);
  }
  await page.waitForTimeout(1000);
}

function benignConsoleError(text) {
  return /favicon|404|410|google-analytics|googletagmanager|Content Security Policy|g\/collect|API keys are permanently disabled|Failed to fetch unread count|PostHog was initialized without a token/i.test(
    text,
  );
}

async function probeRoute(page, base, routePath) {
  const errors = [];
  const onConsole = (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  };
  page.on("console", onConsole);
  const url = base + routePath;
  let nav = await gotoResilient(page, url);
  let session = await waitForDashboard(page, base, routePath);
  if (!session.onDashboard && session.authEmail) {
    nav = await gotoResilient(page, url, { attempts: 1 });
    session = await waitForDashboard(page, base, routePath);
  }
  page.off("console", onConsole);
  const title = await page.title().catch(() => "");
  const consoleErrors = errors.filter((e) => !benignConsoleError(e)).slice(0, 10);
  const hardFail = nav.error && !session.onDashboard;
  return {
    routePath,
    status: nav.status,
    finalUrl: session.finalUrl,
    title,
    authEmail: session.authEmail,
    onDashboard: session.onDashboard,
    consoleErrors,
    ...(hardFail ? { error: nav.error } : nav.error ? { navWarning: nav.error } : {}),
  };
}

async function main() {
  const { base, email, password } = loadEnvAudit();
  if (!email || !password) {
    console.error("Missing AUDIT_EMAIL/AUDIT_PASSWORD — run: bash scripts/provision_audit_user.sh");
    process.exit(1);
  }

  fs.mkdirSync(path.join(OUT, "raw"), { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    await login(page, base, email, password);
    const results = {
      capturedAt: new Date().toISOString(),
      base,
      email,
      routes: [],
    };
    for (const routePath of DASHBOARD_ROUTES) {
      console.log("[dashboard]", routePath);
      results.routes.push(await probeRoute(page, base, routePath));
    }
    const outPath = path.join(OUT, "raw", "dashboard-all.json");
    fs.writeFileSync(outPath, JSON.stringify(results, null, 2));
    console.log("Wrote", outPath);
    const bad = results.routes.filter((r) => r.error || !r.onDashboard || (r.consoleErrors?.length > 0));
    if (bad.length) {
      console.error("Issues:", bad.map((r) => r.routePath).join(", "));
      process.exit(1);
    }
  } finally {
    await browser.close();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});