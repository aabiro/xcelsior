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
  "/dashboard/compliance",
];

function loadEnvAudit() {
  const env = {
    base: (process.env.AUDIT_BASE || "https://xcelsior.ca").replace(/\/$/, ""),
    email: process.env.AUDIT_EMAIL || "",
    password: process.env.AUDIT_PASSWORD || "",
  };
  const p = path.join(REPO, ".env.audit");
  if (!fs.existsSync(p)) return env;
  for (const line of fs.readFileSync(p, "utf8").split("\n")) {
    const s = line.trim();
    if (!s || s.startsWith("#") || !s.includes("=")) continue;
    const [k, v] = s.split("=", 2);
    const val = v.trim().replace(/^["']|["']$/g, "");
    if (k === "AUDIT_BASE" && !process.env.AUDIT_BASE) env.base = val.replace(/\/$/, "");
    if (k === "AUDIT_EMAIL" && !process.env.AUDIT_EMAIL) env.email = val;
    if (k === "AUDIT_PASSWORD" && !process.env.AUDIT_PASSWORD) env.password = val;
  }
  return env;
}

async function login(page, base, email, password) {
  const loginRes = await page.request.post(`${base}/api/auth/login`, {
    data: { email, password },
    headers: { "Content-Type": "application/json" },
  });
  const body = await loginRes.json().catch(() => ({}));
  if (!loginRes.ok()) {
    throw new Error(`login ${loginRes.status()}: ${JSON.stringify(body).slice(0, 200)}`);
  }
  if (body.mfa_required) {
    throw new Error("MFA required — audit user must have MFA disabled");
  }
  await page.goto(`${base}/dashboard`, { waitUntil: "networkidle", timeout: 90000 });
  const url = page.url();
  if (url.includes("/login")) {
    throw new Error(`still on login after API auth: ${url}`);
  }
}

async function probeRoute(page, base, routePath) {
  const errors = [];
  const onConsole = (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  };
  page.on("console", onConsole);
  const url = base + routePath;
  let status = null;
  let finalUrl = url;
  try {
    const res = await page.goto(url, { waitUntil: "networkidle", timeout: 90000 });
    status = res?.status() ?? null;
    finalUrl = page.url();
  } catch (e) {
    return { routePath, error: e.message, status, finalUrl, consoleErrors: errors.slice(0, 10) };
  }
  page.off("console", onConsole);
  const authMe = await page
    .request.get(`${base}/api/auth/me`)
    .then((r) => r.json())
    .catch(() => null);
  const title = await page.title();
  return {
    routePath,
    status,
    finalUrl,
    title,
    authEmail: authMe?.user?.email || null,
    onDashboard: !finalUrl.includes("/login"),
    consoleErrors: errors.filter((e) => !/favicon|404/.test(e)).slice(0, 10),
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