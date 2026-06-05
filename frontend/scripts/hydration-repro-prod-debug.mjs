#!/usr/bin/env node
/** Isolate production hydration #418 — third-party scripts, service worker, etc. */
import { chromium } from "playwright";

const BASE = process.env.BASE_URL || "https://xcelsior.ca";
const ROUTES = ["/privacy", "/terms"];

async function probe(label, options) {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    bypassCSP: true,
    serviceWorkers: options.serviceWorkers ?? "allow",
  });
  if (options.blockThirdParty) {
    await context.route("**/*", (route) => {
      const url = route.request().url();
      if (
        url.includes("googletagmanager.com") ||
        url.includes("google-analytics.com") ||
        url.includes("cloudflareinsights.com")
      ) {
        return route.abort();
      }
      return route.continue();
    });
  }

  for (const routePath of ROUTES) {
    const errors = [];
    const page = await context.newPage();
    page.on("pageerror", (err) => {
      const text = String(err.message || err);
      if (/418|hydration|did not match/i.test(text)) errors.push(text);
    });

    if (options.doubleVisit) {
      await page.goto(BASE + routePath, { waitUntil: "domcontentloaded", timeout: 90000 });
      await page.waitForTimeout(3000);
    }
    await page.goto(BASE + routePath, { waitUntil: "networkidle", timeout: 90000 });
    await page.waitForTimeout(2500);
    console.log(`  ${routePath}: ${errors.length ? errors[0].slice(0, 200) : "OK"}`);
    await page.close();
  }

  await context.close();
  await browser.close();
}

async function main() {
  console.log("BASE", BASE);
  console.log("\n[baseline]");
  await probe("baseline", {});
  console.log("\n[block third-party]");
  await probe("block", { blockThirdParty: true });
  console.log("\n[no service workers]");
  await probe("no-sw", { serviceWorkers: "block" });
  console.log("\n[block 3p + no sw + double visit]");
  await probe("combo", {
    blockThirdParty: true,
    serviceWorkers: "block",
    doubleVisit: true,
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});