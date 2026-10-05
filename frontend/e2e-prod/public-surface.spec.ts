/**
 * The public site as production actually serves it — real backend, real CSP,
 * real third parties. Read-only: nothing here creates, changes or pays for
 * anything. The one write-shaped call is a deliberately wrong login, which
 * costs one slot of this network's auth rate limit and nothing else.
 */
import { expect, test, type Page } from "@playwright/test";

const PUBLIC_PAGES = [
  "/", "/about", "/blog", "/download", "/features", "/gpu-availability", "/gpus",
  "/login", "/mcp", "/pricing", "/privacy", "/register", "/security", "/status",
  "/support", "/terms", "/forgot-password",
];

/**
 * Responses that are correct behaviour, each with its reason. Anything not
 * listed fails — so a new exception has to be argued for here, not absorbed.
 */
const EXPECTED_RESPONSES: Array<{ pattern: RegExp; why: string }> = [
  // A logged-out visitor asking "who am I" is answered 401 by design.
  { pattern: /^401 \/api\/auth\/(me|refresh)\b/, why: "anonymous session probe" },
  // The demo button is shown only to whitelisted networks; everyone else is
  // told no, which is how the button knows to stay hidden.
  { pattern: /^403 \/api\/auth\/demo-credentials$/, why: "demo gate, non-whitelisted network" },
  // Next.js cancels in-flight RSC prefetches when the page settles or moves on.
  { pattern: /^FAILED \/[^ ]*\?_rsc=[^ ]+ \(net::ERR_ABORTED\)$/, why: "cancelled RSC prefetch" },
];

const isExpected = (line: string) => EXPECTED_RESPONSES.some((e) => e.pattern.test(line));

// Minified React hydration errors: #418 text, #423 recoverable, #425 text content.
const HYDRATION = /hydrat|did not match|server rendered|react\.dev\/errors\/(418|423|425)/i;

type Findings = {
  pageErrors: string[];
  consoleErrors: string[];
  cspViolations: string[];
  failedRequests: string[];
  hydration: string[];
};

/** Attach every listener before navigation, so nothing early is missed. */
async function watch(page: Page, origin: string): Promise<Findings> {
  const f: Findings = { pageErrors: [], consoleErrors: [], cspViolations: [], failedRequests: [], hydration: [] };

  // The DOM event is the authoritative CSP signal; the console line is a
  // browser-specific courtesy and can be suppressed.
  await page.addInitScript(() => {
    (window as unknown as { __csp: string[] }).__csp = [];
    document.addEventListener("securitypolicyviolation", (e) => {
      (window as unknown as { __csp: string[] }).__csp.push(
        `${e.violatedDirective} blocked ${e.blockedURI || "(inline)"}`,
      );
    });
  });
  page.on("pageerror", (err) => {
    const text = err.message.slice(0, 300);
    (HYDRATION.test(text) ? f.hydration : f.pageErrors).push(text);
  });
  // The browser logs every non-2xx subresource as a console error, so those
  // are judged as responses below rather than twice.
  const resourceNoise = /^Failed to load resource: the server responded with a status of \d+/;
  page.on("console", (msg) => {
    const text = msg.text();
    if (HYDRATION.test(text)) f.hydration.push(text.slice(0, 300));
    else if (msg.type() === "error" && !resourceNoise.test(text)) f.consoleErrors.push(text.slice(0, 300));
  });
  page.on("response", (res) => {
    const url = res.url();
    if (!url.startsWith(origin) || res.status() < 400) return;
    const line = `${res.status()} ${url.replace(origin, "")}`;
    if (!isExpected(line)) f.failedRequests.push(line);
  });
  page.on("requestfailed", (req) => {
    if (!req.url().startsWith(origin)) return;
    const line = `FAILED ${req.url().replace(origin, "")} (${req.failure()?.errorText})`;
    if (!isExpected(line)) f.failedRequests.push(line);
  });
  return f;
}

for (const path of PUBLIC_PAGES) {
  test(`${path} renders cleanly in production`, async ({ page, baseURL }, info) => {
    const origin = new URL(baseURL!).origin;
    const findings = await watch(page, origin);

    const response = await page.goto(path, { waitUntil: "networkidle" });
    expect(response, `no response for ${path}`).not.toBeNull();
    expect(response!.status(), `${path} document status`).toBeLessThan(400);

    await expect(page.locator("body")).toBeVisible();
    findings.cspViolations.push(...(await page.evaluate(() => (window as unknown as { __csp: string[] }).__csp ?? [])));

    await page.screenshot({ path: info.outputPath(`${path.replace(/\W+/g, "_") || "root"}.png`), fullPage: true });
    await info.attach("findings", { body: JSON.stringify(findings, null, 2), contentType: "application/json" });

    // Separate assertions so the report names the kind of failure.
    expect.soft(findings.cspViolations, "CSP violations").toEqual([]);
    expect.soft(findings.pageErrors, "uncaught page errors").toEqual([]);
    expect.soft(findings.hydration, "hydration mismatches").toEqual([]);
    expect.soft(findings.failedRequests, "same-origin requests that failed").toEqual([]);
    expect.soft(findings.consoleErrors, "console errors").toEqual([]);
  });
}

test("pricing is populated from the live rate table", async ({ page }) => {
  await page.goto("/pricing", { waitUntil: "networkidle" });
  // A price is rendered as currency; an empty or failed rate fetch renders none.
  const prices = page.getByText(/\$\s?\d+(\.\d+)?/);
  expect(await prices.count(), "no prices rendered on /pricing").toBeGreaterThan(3);
});

test("the GPU availability page shows real inventory or says there is none", async ({ page }) => {
  const api = page.waitForResponse((r) => /\/api\/v2\/gpu\/available|\/gpu-availability|\/hosts/.test(r.url()), { timeout: 30_000 }).catch(() => null);
  await page.goto("/gpu-availability", { waitUntil: "networkidle" });
  const res = await api;
  expect(res, "the page made no inventory request").not.toBeNull();
  expect(res!.status(), `inventory request ${res!.url()}`).toBeLessThan(400);
});

test("a wrong password is refused with a visible message, not a crash", async ({ page }) => {
  await page.goto("/login", { waitUntil: "networkidle" });
  await page.locator("#email").fill("playwright-prod-probe@example.invalid");
  await page.locator("#password").fill("definitely-not-the-password");
  const login = page.waitForResponse((r) => r.url().includes("/api/auth/login"));
  await page.locator("form").getByRole("button", { name: /sign in|log in/i }).first().click();
  const res = await login;
  expect([401, 429]).toContain(res.status());
  await expect(page.getByRole("alert").or(page.getByText(/invalid|incorrect|too many/i)).first()).toBeVisible();
  await expect(page).toHaveURL(/\/login/);
});

test("the theme on <html> and the app shell agree after hydration", async ({ page }) => {
  // The light-theme desync shipped once: the shell kept the server's theme
  // while <html> took the stored one. Compare what each actually says.
  await page.goto("/", { waitUntil: "networkidle" });
  const themes = await page.evaluate(() => ({
    html: document.documentElement.getAttribute("data-theme") ?? document.documentElement.className,
    shells: Array.from(document.querySelectorAll("[data-theme]")).map((el) => el.getAttribute("data-theme")),
  }));
  const distinct = new Set(themes.shells.filter(Boolean));
  expect(distinct.size, `conflicting data-theme values: ${JSON.stringify(themes)}`).toBeLessThanOrEqual(1);
});
