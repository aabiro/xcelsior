#!/usr/bin/env node
/** Compare SSR main text vs post-hydration client text on production legal routes. */
import { chromium } from "playwright";

const BASE = process.env.BASE_URL || "https://xcelsior.ca";
const ROUTE = process.env.ROUTE || "/privacy";

function normalize(text) {
  return text.replace(/\s+/g, " ").trim();
}

function firstDiff(a, b) {
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    if (a[i] !== b[i]) {
      const start = Math.max(0, i - 40);
      return {
        index: i,
        server: a.slice(start, i + 60),
        client: b.slice(start, i + 60),
      };
    }
  }
  if (a.length !== b.length) {
    return { index: len, server: a.slice(len - 40, len + 80), client: b.slice(len - 40, len + 80), lenDiff: a.length - b.length };
  }
  return null;
}

async function main() {
  const ssrRes = await fetch(BASE + ROUTE, { headers: { "User-Agent": "hydration-diff/1.0" } });
  const ssrHtml = await ssrRes.text();
  const mainMatch = ssrHtml.match(/<main[^>]*>([\s\S]*?)<\/main>/i);
  const ssrMainHtml = mainMatch?.[1] ?? "";
  const ssrText = normalize(ssrMainHtml.replace(/<[^>]+>/g, " "));

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ bypassCSP: true, serviceWorkers: "block" });
  const page = await context.newPage();
  let hydrationError = null;
  page.on("pageerror", (err) => {
    if (/418|hydration|did not match/i.test(String(err.message))) {
      hydrationError = String(err.message);
    }
  });

  await page.goto(BASE + ROUTE, { waitUntil: "networkidle", timeout: 90000 });
  await page.waitForTimeout(2000);
  const clientText = normalize(await page.locator("main").innerText());

  console.log("route", ROUTE);
  console.log("hydrationError", hydrationError ? "yes" : "no");
  console.log("ssrTextLen", ssrText.length, "clientTextLen", clientText.length);
  if (ssrText === clientText) {
    console.log("main text: IDENTICAL");
  } else {
    const diff = firstDiff(ssrText, clientText);
    console.log("main text: DIFFER");
    console.log(JSON.stringify(diff, null, 2));
  }

  await context.close();
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});