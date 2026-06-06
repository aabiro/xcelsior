#!/usr/bin/env node
/**
 * Capture hydration / React #418 on marketing routes.
 *
 * - Dev (non-minified): BASE_URL=http://127.0.0.1:3000  (next dev; uses bypassCSP)
 * - Local prod:         BASE_URL=http://127.0.0.1:3457  (next build && next start)
 * - Production:         BASE_URL=https://xcelsior.ca    (#418 on /privacy,/terms is often
 *                       Cloudflare Email Obfuscation — see hydration-diff.mjs)
 * - Tailscale origin:   AUDIT_ORIGIN_IP=100.64.0.1 BASE_URL=https://xcelsior.ca
 */
import { chromium } from "playwright";

const BASE = process.env.BASE_URL || "https://xcelsior.ca";
const ORIGIN_IP = process.env.AUDIT_ORIGIN_IP || "";
const ROUTES = ["/about", "/support", "/privacy", "/terms", "/blog"];

const NOISE =
  /webpack-hmr|WebSocket connection|eval\(\) is not supported|Download the React DevTools/i;

async function main() {
  const launchArgs = [];
  if (ORIGIN_IP) {
    const host = new URL(BASE).hostname;
    launchArgs.push(`--host-resolver-rules=MAP ${host} ${ORIGIN_IP}`);
  }
  const browser = await chromium.launch({ headless: true, args: launchArgs });
  const context = await browser.newContext({ bypassCSP: true });

  for (const routePath of ROUTES) {
    const routeErrors = [];
    const page = await context.newPage();
    page.on("console", (msg) => {
      const text = msg.text();
      if (NOISE.test(text)) return;
      if (msg.type() === "error" || /hydration|418|did not match/i.test(text)) {
        routeErrors.push({ type: msg.type(), text });
      }
    });
    page.on("pageerror", (err) => {
      routeErrors.push({ type: "pageerror", text: String(err.message || err) });
    });

    await page.goto(BASE + routePath, { waitUntil: "networkidle", timeout: 90000 });
    await page.waitForTimeout(2500);
    await page.close();

    console.log("\n===", routePath, "===");
    if (!routeErrors.length) {
      console.log("OK — no hydration/console errors");
    } else {
      for (const e of routeErrors) {
        console.log(`[${e.type}]`, e.text);
      }
    }
  }

  await context.close();
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});