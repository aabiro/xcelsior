#!/usr/bin/env node
/**
 * Authenticated API probe for dashboard audit (no browser — works when CF is flaky).
 * Uses AUDIT_* from .env.audit; obtains bearer via /api/auth/login.
 *
 * For origin-only (SSH): AUDIT_BASE=http://127.0.0.1:9501 with ssh -L 9501:127.0.0.1:9501
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = process.env.AUDIT_OUT || "/tmp/xcelsior-audit";

function loadEnv() {
  const cfg = {
    base: (process.env.AUDIT_BASE || "https://xcelsior.ca").replace(/\/$/, ""),
    email: process.env.AUDIT_EMAIL || "",
    password: process.env.AUDIT_PASSWORD || "",
  };
  const p = path.join(REPO, ".env.audit");
  if (!fs.existsSync(p)) return cfg;
  for (const line of fs.readFileSync(p, "utf8").split("\n")) {
    const s = line.trim();
    if (!s || s.startsWith("#") || !s.includes("=")) continue;
    const [k, v] = s.split("=", 2);
    const val = v.trim();
    if (k === "AUDIT_BASE" && !process.env.AUDIT_BASE) cfg.base = val.replace(/\/$/, "");
    if (k === "AUDIT_EMAIL" && !process.env.AUDIT_EMAIL) cfg.email = val;
    if (k === "AUDIT_PASSWORD" && !process.env.AUDIT_PASSWORD) cfg.password = val;
  }
  return cfg;
}

const API_ROUTES = [
  ["GET", "/api/auth/me"],
  ["GET", "/api/v2/gpu/available"],
  ["GET", "/api/notifications"],
  ["GET", "/api/billing/wallet"],
  ["GET", "/api/jobs"],
  ["GET", "/api/volumes"],
];

async function main() {
  const cfg = loadEnv();
  if (!cfg.email || !cfg.password) {
    console.error("Missing .env.audit — run: bash scripts/provision_audit_user.sh");
    process.exit(1);
  }

  const loginRes = await fetch(`${cfg.base}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: cfg.email, password: cfg.password }),
  });
  const loginBody = await loginRes.json().catch(() => ({}));
  if (!loginRes.ok) {
    console.error("Login failed", loginRes.status, loginBody);
    process.exit(1);
  }
  if (loginBody.mfa_required) {
    console.error("MFA required on audit user");
    process.exit(1);
  }
  const token = loginBody.access_token || loginBody.token;
  if (!token) {
    console.error("No access_token in login response");
    process.exit(1);
  }

  const checks = [];
  for (const [method, route] of API_ROUTES) {
    const res = await fetch(`${cfg.base}${route}`, {
      method,
      headers: { Authorization: `Bearer ${token}` },
    });
    let snippet = "";
    try {
      const text = await res.text();
      snippet = text.slice(0, 120);
    } catch {
      /* */
    }
    checks.push({
      route,
      method,
      status: res.status,
      ok: res.status >= 200 && res.status < 400,
      snippet,
    });
  }

  const out = {
    capturedAt: new Date().toISOString(),
    base: cfg.base,
    email: cfg.email,
    loginStatus: loginRes.status,
    checks,
    summary: {
      total: checks.length,
      passed: checks.filter((c) => c.ok).length,
    },
  };

  fs.mkdirSync(path.join(OUT, "raw"), { recursive: true });
  const outPath = path.join(OUT, "raw", "dashboard-api.json");
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
  console.log(JSON.stringify(out.summary));
  console.log("Wrote", outPath);
  if (checks.some((c) => !c.ok)) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});