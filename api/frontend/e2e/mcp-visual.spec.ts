import { expect, test, type BrowserContext, type Route } from "@playwright/test";

const AUTH_COOKIE = "xcelsior_session";

const mockUser = {
  user_id: "mcp-visual-user",
  email: "mcp-visual@xcelsior.ca",
  name: "MCP Visual",
  role: "user",
  customer_id: "cust-mcp-visual",
  team_can_write_instances: true,
  team_can_manage_billing: true,
};

function jsonResponse(body: unknown, status = 200) {
  return {
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  };
}

async function installAuthenticatedContext(ctx: BrowserContext, baseURL: string) {
  const { hostname } = new URL(baseURL);
  await ctx.addCookies([
    { name: AUTH_COOKIE, value: "mcp-visual-session", domain: hostname, path: "/" },
  ]);
}

async function installDashboardMocks(ctx: BrowserContext, baseURL: string) {
  const origin = baseURL.replace(/\/$/, "");
  await ctx.route(`${origin}/healthz`, (route) => route.fulfill({ status: 200, body: "ok" }));
  await ctx.route(`${origin}/readyz`, (route) => route.fulfill({ status: 200, body: "ok" }));

  const apiGlob = `${origin}/api/**`;
  await ctx.route(apiGlob, async (route: Route) => {
    const url = route.request().url();
    const method = route.request().method();

    if (url.includes("/api/auth/me")) {
      return route.fulfill(jsonResponse({ ok: true, user: mockUser }));
    }
    if (url.includes("/api/auth/refresh")) {
      return route.fulfill(jsonResponse({ ok: true, access_token: "visual", expires_in: 3600 }));
    }
    if (url.includes("/api/oauth/clients") && method === "GET") {
      return route.fulfill(jsonResponse({ ok: true, clients: [] }));
    }
    if (url.includes("/api/v2/gpu/available")) {
      return route.fulfill(
        jsonResponse({
          ok: true,
          gpus: [{ gpu_model: "RTX 4090", vram_gb: 24, count_available: 2, price_cad: 1.2 }],
        }),
      );
    }
    if (url.includes("/api/stream")) {
      return route.fulfill({ status: 204, body: "" });
    }
    return route.fulfill(jsonResponse({ ok: true }));
  });
}

async function installEventSourceMock(ctx: BrowserContext) {
  await ctx.addInitScript(() => {
    class MockEventSource extends EventTarget {
      readonly url: string;
      readonly readyState = 1;
      onopen: ((ev: Event) => void) | null = null;
      onerror: ((ev: Event) => void) | null = null;
      onmessage: ((ev: MessageEvent) => void) | null = null;
      constructor(url: string | URL) {
        super();
        this.url = String(url);
        queueMicrotask(() => {
          const ev = new Event("open");
          this.onopen?.(ev);
          this.dispatchEvent(ev);
        });
      }
      close() {}
      addEventListener() {}
      removeEventListener() {}
    }
    window.EventSource = MockEventSource as unknown as typeof EventSource;
  });
}

test.describe("MCP marketing page", () => {
  test("renders hero and key sections", async ({ page }) => {
    await page.goto("/mcp", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { level: 1 })).toContainText(/AI agents control real GPUs/i);
    await expect(page.getByText(/missing backend for agentic AI/i)).toBeVisible();
    await expect(page.getByText(/Cost guardrails built in/i)).toBeVisible();
  });

  test("marketing page visual snapshot", async ({ page }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/mcp", { waitUntil: "domcontentloaded" });
    const hero = page.locator("section").first();
    await expect(hero).toBeVisible({ timeout: 15_000 });
    await expect(hero).toHaveScreenshot("mcp-landing-hero.png", {
      maxDiffPixelRatio: 0.03,
      animations: "disabled",
      timeout: 30_000,
    });
  });
});

test.describe("MCP settings tab", () => {
  test("settings MCP wizard visual snapshot", async ({ browser, baseURL }) => {
    test.setTimeout(90_000);
    const origin = baseURL ?? "http://127.0.0.1:3100";
    const context = await browser.newContext({ serviceWorkers: "block" });
    await installAuthenticatedContext(context, origin);
    await installDashboardMocks(context, origin);
    await installEventSourceMock(context);
    const page = await context.newPage();
    try {
      await page.setViewportSize({ width: 1280, height: 900 });
      const authResponse = page.waitForResponse(
        (res) => res.url().includes("/api/auth/me") && res.ok(),
      );
      await page.goto("/dashboard/settings#mcp", { waitUntil: "domcontentloaded" });
      await expect(page).toHaveURL(/\/dashboard\/settings/, { timeout: 15_000 });
      await authResponse;
      await expect(page.getByRole("button", { name: /Create MCP Client/i })).toBeVisible({
        timeout: 20_000,
      });
      const panel = page.getByRole("button", { name: /Create MCP Client/i }).locator("xpath=ancestor::section[1]");
      await expect(panel).toHaveScreenshot("settings-mcp-tab.png", {
        maxDiffPixelRatio: 0.03,
        animations: "disabled",
        timeout: 30_000,
      });
    } finally {
      await context.close();
    }
  });
});