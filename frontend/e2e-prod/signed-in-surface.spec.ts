/**
 * Every dashboard surface the 2026-10-05 release changed, as production serves
 * it to a signed-in admin. Read-only: pages are loaded and inspected, and no
 * button that creates, changes or pays for anything is pressed.
 *
 * Beyond errors, it looks for the tells of a page that rendered but rendered
 * wrong — `undefined`, `NaN`, `[object Object]`, `Invalid Date`, i18n keys shown
 * as text — and for layouts that overflow sideways on a phone. Screenshots are
 * kept for every page so the result can be looked at, not just counted.
 */
import { expect, test, type Page } from "@playwright/test";
import { clippedContent } from "./layout-checks";

const PAGES: Array<{ path: string; heading?: RegExp }> = [
  { path: "/dashboard" },
  { path: "/dashboard/admin/control-plane" },
  { path: "/dashboard/ai" },
  { path: "/dashboard/analytics" },
  { path: "/dashboard/artifacts" },
  { path: "/dashboard/billing" },
  { path: "/dashboard/compliance" },
  { path: "/dashboard/earnings" },
  { path: "/dashboard/events" },
  { path: "/dashboard/hosts" },
  { path: "/dashboard/hpc" },
  { path: "/dashboard/inference" },
  { path: "/dashboard/instances" },
  { path: "/dashboard/marketplace" },
  { path: "/dashboard/mcp" },
  { path: "/dashboard/notifications" },
  { path: "/dashboard/reputation" },
  { path: "/dashboard/settings" },
  { path: "/dashboard/spot-pricing" },
  { path: "/dashboard/telemetry" },
  { path: "/dashboard/templates" },
  { path: "/dashboard/trust" },
  { path: "/dashboard/volumes" },
  // Not `/dashboard/launch-plans`: there is no index. A plan is reached by the
  // approval link an agent hands out, `/dashboard/launch-plans/{id}`.
];

const EXPECTED_RESPONSES: Array<{ pattern: RegExp; why: string }> = [
  { pattern: /^FAILED \/[^ ]*\?_rsc=[^ ]+ \(net::ERR_ABORTED\)$/, why: "cancelled RSC prefetch" },
  { pattern: /^FAILED \/api\/stream\b/, why: "SSE stream closed by navigation" },
];
const isExpected = (line: string) => EXPECTED_RESPONSES.some((e) => e.pattern.test(line));
const HYDRATION = /hydrat|did not match|server rendered|react\.dev\/errors\/(418|423|425)/i;
const RESOURCE_NOISE = /^Failed to load resource: the server responded with a status of \d+/;

type Findings = Record<"pageErrors" | "consoleErrors" | "cspViolations" | "failedRequests" | "hydration", string[]>;

async function watch(page: Page, origin: string): Promise<Findings> {
  const f: Findings = { pageErrors: [], consoleErrors: [], cspViolations: [], failedRequests: [], hydration: [] };
  await page.addInitScript(() => {
    (window as unknown as { __csp: string[] }).__csp = [];
    document.addEventListener("securitypolicyviolation", (e) => {
      (window as unknown as { __csp: string[] }).__csp.push(`${e.violatedDirective} blocked ${e.blockedURI || "(inline)"}`);
    });
  });
  page.on("pageerror", (err) => (HYDRATION.test(err.message) ? f.hydration : f.pageErrors).push(err.message.slice(0, 300)));
  page.on("console", (msg) => {
    const t = msg.text();
    if (HYDRATION.test(t)) f.hydration.push(t.slice(0, 300));
    else if (msg.type() === "error" && !RESOURCE_NOISE.test(t)) f.consoleErrors.push(t.slice(0, 300));
  });
  page.on("response", (res) => {
    if (!res.url().startsWith(origin) || res.status() < 400) return;
    const line = `${res.status()} ${res.url().replace(origin, "")}`;
    if (!isExpected(line)) f.failedRequests.push(line);
  });
  page.on("requestfailed", (req) => {
    if (!req.url().startsWith(origin)) return;
    const line = `FAILED ${req.url().replace(origin, "")} (${req.failure()?.errorText})`;
    if (!isExpected(line)) f.failedRequests.push(line);
  });
  return f;
}

/** Visible text that means something rendered wrong. */
async function brokenText(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const bad: string[] = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const KEY = /^[a-z][a-z0-9_]*(\.[a-z0-9_]+){2,}$/; // dashboard.billing.title
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const el = n.parentElement;
      if (!el || el.closest("script,style,code,pre,[aria-hidden=true]")) continue;
      const style = getComputedStyle(el);
      if (style.display === "none" || style.visibility === "hidden") continue;
      const text = (n.textContent ?? "").trim();
      if (!text) continue;
      if (/\bundefined\b|\bNaN\b|\[object Object\]|Invalid Date/.test(text) || KEY.test(text)) {
        bad.push(text.slice(0, 120));
      }
    }
    return Array.from(new Set(bad)).slice(0, 20);
  });
}

for (const { path } of PAGES) {
  test(`${path} — signed in`, async ({ page, baseURL }, info) => {
    const origin = new URL(baseURL!).origin;
    const findings = await watch(page, origin);
    // Not "networkidle": the dashboard holds an event stream open for live
    // updates, so the network is never idle and every page timed out waiting
    // for it. Load, then a visible heading, then a settle window for the
    // client-side fetches that fill the page.
    const response = await page.goto(path, { waitUntil: "load" });
    expect(response?.status() ?? 0, `${path} document`).toBeLessThan(400);
    expect(new URL(page.url()).pathname, "redirected away — the session did not hold").toBe(path);
    await page.locator("h1, h2").first().waitFor({ state: "visible", timeout: 20_000 }).catch(() => {});
    await page.waitForTimeout(3000);
    findings.cspViolations.push(...(await page.evaluate(() => (window as unknown as { __csp: string[] }).__csp ?? [])));
    const broken = await brokenText(page);
    const overflow = await page.evaluate(() => {
      const doc = document.documentElement;
      return doc.scrollWidth - doc.clientWidth;
    });
    const clipped = await clippedContent(page);
    const brokenImages = await page.evaluate(() =>
      Array.from(document.images).filter((i) => i.complete && i.naturalWidth === 0).map((i) => i.src).slice(0, 10),
    );

    await page.screenshot({ path: info.outputPath(`${path.replace(/\W+/g, "_")}.png`), fullPage: true });
    await info.attach("findings", {
      body: JSON.stringify({ ...findings, broken, overflow, clipped, brokenImages }, null, 2),
      contentType: "application/json",
    });

    await expect(page.locator("h1, h2").first(), "no heading rendered").toBeVisible();
    expect.soft(findings.pageErrors, "uncaught page errors").toEqual([]);
    expect.soft(findings.hydration, "hydration mismatches").toEqual([]);
    expect.soft(findings.cspViolations, "CSP violations").toEqual([]);
    expect.soft(findings.failedRequests, "same-origin requests that failed").toEqual([]);
    expect.soft(findings.consoleErrors, "console errors").toEqual([]);
    expect.soft(broken, "text that means something rendered wrong").toEqual([]);
    expect.soft(overflow, "page scrolls sideways (px)").toBeLessThanOrEqual(1);
    expect.soft(clipped, "content past the edge of the screen").toEqual([]);
    expect.soft(brokenImages, "images that failed to load").toEqual([]);
  });
}

test("an instance detail page renders its new cards — signed in", async ({ page }) => {
  await page.goto("/dashboard/instances", { waitUntil: "load" });
  await page.waitForTimeout(3000);
  const first = page.locator('a[href^="/dashboard/instances/"]').first();
  test.skip((await first.count()) === 0, "no instances on this account to open");
  await first.click();
  await page.waitForLoadState("load");
  await page.waitForTimeout(3000);
  await expect(page.locator("h1, h2").first()).toBeVisible();
  const broken = await brokenText(page);
  expect.soft(broken, "text that means something rendered wrong").toEqual([]);
});
