import { test, expect, type BrowserContext, type Page } from "@playwright/test";

// The instance page's app bar, web terminal and expand view, against mocked
// APIs and a scripted WebSocket. No backend or real shell involved.

test.use({ serviceWorkers: "block", screenshot: "only-on-failure", actionTimeout: 15_000 });
test.setTimeout(120_000);

const INSTANCE_ID = "ae1d03c3";
const INSTANCE_NAME = "fierce-apex";
const user = {
  user_id: "terminal-review-user", customer_id: "terminal-review-wallet",
  email: "review@example.test", name: "Terminal Review", role: "admin", is_admin: true,
  email_verified: true, team_can_write_instances: true, team_can_manage_billing: true,
};

type SocketMode = "normal" | "exit" | "refused";

async function installFixtures(context: BrowserContext, baseURL: string, mode: SocketMode) {
  const origin = new URL(baseURL).origin;
  await context.addCookies([{ name: "xcelsior_session", value: "isolated-review-session", domain: new URL(origin).hostname, path: "/" }]);
  await context.addInitScript((socketMode: SocketMode) => {
    localStorage.setItem("xcelsior-locale", "en");
    const w = window as unknown as Record<string, unknown> & { __terminalSockets: unknown[] };
    w.__terminalSockets = [];
    w.__hidden = false;
    Object.defineProperty(document, "hidden", { configurable: true, get: () => Boolean(w.__hidden) });
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      get: () => (w.__hidden ? "hidden" : "visible"),
    });

    class TestEventSource extends EventTarget {
      static OPEN = 1;
      readonly readyState = 1;
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;
      constructor(readonly url: string) {
        super();
        queueMicrotask(() => this.onopen?.(new Event("open")));
      }
      close() {}
    }
    window.EventSource = TestEventSource as unknown as typeof EventSource;

    class TestWebSocket extends EventTarget {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSING = 2;
      static CLOSED = 3;
      readyState = 0;
      binaryType = "blob";
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onclose: ((event: CloseEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;
      constructor(readonly url: string) {
        super();
        const terminal = url.includes("/ws/terminal/");
        if (terminal) w.__terminalSockets.push(this);
        setTimeout(() => this.run(terminal), 20);
      }
      private open() {
        this.readyState = 1;
        this.onopen?.(new Event("open"));
      }
      private message(data: string | ArrayBuffer) {
        this.onmessage?.(new MessageEvent("message", { data }));
      }
      drop(code: number) {
        if (this.readyState === 3) return;
        this.readyState = 3;
        this.onclose?.(new CloseEvent("close", { code }));
      }
      private run(terminal: boolean) {
        this.open();
        if (!terminal) return;
        if (socketMode === "refused") {
          this.message(JSON.stringify({
            type: "error",
            code: 4429,
            message: "Too many open terminals: 3 at a time per account. Close one in another tab or device, then reconnect.",
          }));
          this.drop(4429);
          return;
        }
        this.message(new TextEncoder().encode("root@xcelsior-ae1d03c3:/workspace# ").buffer);
        if (socketMode === "exit") this.message(JSON.stringify({ type: "exit", code: 0 }));
      }
      send() {}
      close(code = 1000) {
        this.drop(code);
      }
    }
    // Only the app's own sockets are scripted; the dev server's hot-reload
    // socket goes to the real WebSocket.
    const RealWebSocket = window.WebSocket;
    function PatchedWebSocket(url: string | URL, protocols?: string | string[]) {
      const href = String(url);
      if (!href.includes("/ws/")) return new RealWebSocket(href, protocols);
      return new TestWebSocket(href);
    }
    Object.assign(PatchedWebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
    window.WebSocket = PatchedWebSocket as unknown as typeof WebSocket;
  }, mode);

  const instance = {
    job_id: INSTANCE_ID, name: INSTANCE_NAME, status: "running", gpu_model: "RTX 3060",
    docker_image: "pytorch/pytorch:latest", submitted_at: 1_790_000_000, started_at: "2026-11-11T09:00:00Z",
    host_id: "host-review", interactive: true, ssh_port: 0, num_gpus: 1,
  };

  await context.route("**/*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== origin) {
      if (["fetch", "xhr", "eventsource"].includes(request.resourceType())) return route.abort();
      return route.continue();
    }
    const path = url.pathname;
    const json = (body: unknown, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (path === "/api/auth/me") return json({ ok: true, user: { ...user, team_id: null, team_name: null, team_role: "owner" } });
    if (path === "/api/auth/refresh") return json({ ok: true, access_token: "review", expires_in: 3600 });
    if (path === "/api/teams/me") return json({ ok: true, teams: [], active_team_id: null });
    if (path === `/instance/${INSTANCE_ID}`) return json({ ok: true, instance });
    if (path === `/instances/${INSTANCE_ID}/logs`) return json({ ok: true, instance_id: INSTANCE_ID, logs: [], total: 0 });
    if (path === "/api/terminal/ticket" || path.endsWith("/stream-ticket")) return json({ ok: true, ticket: "review-ticket", expires_in: 60 });
    if (path.endsWith("/timeline")) return json({ ok: true, job_id: INSTANCE_ID, attempts: [] });
    if (path.endsWith("/active-lease")) return json({ ok: true, job_id: INSTANCE_ID, lease: null });
    if (path === "/api/mcp/quick-connect") return json({ ok: true, access_token: "xcel_ai_mcp_review_0123456789", client_id: "mcp-client", mcp_url: "https://mcp.example.test/mcp", api_url: origin, scopes: [], in_use: false });
    if (path.startsWith("/api/billing/wallet/")) return json({ ok: true, wallet: { customer_id: user.customer_id, balance_cad: 15.53, total_deposited_cad: 20, status: "active" }, transactions: [] });
    if (path === "/api/users/me/preferences") return json({ ok: true, preferences: { onboarding: { profile: true, api_key: true, mcp_connect: true, browse: true, instance: true } } });
    if (["/instances", "/hosts", "/marketplace"].includes(path)) return json({ ok: true, instances: [instance], hosts: [], listings: [] });
    if (["/healthz", "/readyz"].includes(path)) return json({ ok: true });
    if (path.startsWith("/api/")) return json({ ok: true, items: [], keys: [], methods: [], consents: [], sessions: [], notifications: [], unread_count: 0, available: false });
    return route.continue();
  });
}

const sockets = (page: Page) => page.evaluate(() => (window as unknown as { __terminalSockets: unknown[] }).__terminalSockets.length);

/** Label of the button actually hit at the centre of `name`'s box, or null. */
async function buttonHitAt(page: Page, name: string) {
  const box = await page.getByRole("button", { name, exact: true }).boundingBox();
  if (!box) return null;
  return page.evaluate(({ x, y }) => (
    document.elementFromPoint(x, y)?.closest("button")?.getAttribute("aria-label") ?? null
  ), { x: box.x + box.width / 2, y: box.y + box.height / 2 });
}

async function openInstance(page: Page) {
  await page.goto(`/dashboard/instances/${INSTANCE_ID}`);
  await expect(page.getByText("Connected", { exact: true })).toBeVisible({ timeout: 60_000 });
}

test("app bar names the instance, and Connect reads Connect MCP", async ({ context, page, baseURL }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await installFixtures(context, baseURL!, "normal");
  await openInstance(page);
  const crumbs = page.getByRole("navigation", { name: "Breadcrumb" });
  await expect(crumbs).toContainText(INSTANCE_NAME);
  await expect(crumbs).not.toContainText(/ae1d03c3/i);
  await expect(page.locator('aside a[href="/dashboard/mcp"]')).toContainText("Connect MCP");

  await page.goto("/dashboard/mcp");
  await expect(page.getByRole("navigation", { name: "Breadcrumb" })).toContainText("Connect MCP");
});

test("expanded terminal is above the sidebar and app bar, and collapses back", async ({ context, page, baseURL }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await installFixtures(context, baseURL!, "normal");
  await openInstance(page);

  await page.getByRole("button", { name: "Full screen", exact: true }).click();
  const overlay = page.getByRole("dialog", { name: "Terminal" });
  await expect(overlay).toBeVisible();
  await expect(overlay.locator(".xterm")).toHaveCount(1);
  // Its own controls are on top, not under the app bar.
  expect(await buttonHitAt(page, "Exit full screen")).toBe("Exit full screen");
  // And it covers the sidebar instead of sliding under it.
  const sidebarCovered = await page.evaluate(() => {
    const hit = document.elementFromPoint(120, 300);
    return Boolean(hit?.closest('[role="dialog"][aria-label="Terminal"]'));
  });
  expect(sidebarCovered).toBe(true);
  await page.screenshot({ path: "test-results/web-terminal-expanded.png" });

  await page.getByRole("button", { name: "Exit full screen", exact: true }).click();
  await expect(overlay).toHaveCount(0);
  await expect(page.locator(".xterm")).toHaveCount(1);
  const sidebarBack = await page.evaluate(() => Boolean(document.elementFromPoint(120, 300)?.closest("aside")));
  expect(sidebarBack).toBe(true);
  // Same terminal and socket throughout: expanding never remounted it.
  expect(await sockets(page)).toBe(1);
  await page.waitForTimeout(500);
  await expect(overlay).toHaveCount(0);

  await page.getByRole("button", { name: "Full screen", exact: true }).click();
  await expect(overlay).toBeVisible();
  await page.getByRole("button", { name: "Hide terminal", exact: true }).click();
  await expect(overlay).toHaveCount(0);
  expect(await sockets(page)).toBe(1);
});

test("phone: status badge and Reconnect don't overlap", async ({ context, page, baseURL }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installFixtures(context, baseURL!, "exit");
  await page.goto(`/dashboard/instances/${INSTANCE_ID}`);
  const reconnect = page.getByRole("button", { name: "Reconnect" });
  await expect(reconnect).toBeVisible({ timeout: 60_000 });
  // The terminal toolbar: Reconnect's controls group, then its row.
  const badge = reconnect.locator("xpath=../..").getByText("Disconnected", { exact: true });
  await expect(badge).toBeVisible();
  await reconnect.scrollIntoViewIfNeeded();
  const [b, r] = [await badge.boundingBox(), await reconnect.boundingBox()];
  expect(b && r).toBeTruthy();
  expect(b!.x + b!.width).toBeLessThanOrEqual(r!.x);
  await page.screenshot({ path: "test-results/web-terminal-phone-toolbar.png" });

  await page.getByRole("button", { name: "Full screen", exact: true }).click();
  const overlay = page.getByRole("dialog", { name: "Terminal" });
  const box = await overlay.boundingBox();
  expect(box!.width).toBeGreaterThanOrEqual(389);
  expect(box!.height).toBeGreaterThanOrEqual(800);
  expect(await buttonHitAt(page, "Exit full screen")).toBe("Exit full screen");
  await page.screenshot({ path: "test-results/web-terminal-phone-expanded.png" });
});

test("a refused connection stops at once with the server's reason", async ({ context, page, baseURL }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await installFixtures(context, baseURL!, "refused");
  await page.goto(`/dashboard/instances/${INSTANCE_ID}`);
  await expect(page.getByText("Too many open terminals", { exact: false }).first()).toBeVisible({ timeout: 60_000 });
  await page.waitForTimeout(4_000);
  expect(await sockets(page)).toBe(1);
  await expect(page.getByRole("button", { name: "Reconnect" })).toBeVisible();
});

test("a drop while the tab is hidden waits, then reconnects on return", async ({ context, page, baseURL }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await installFixtures(context, baseURL!, "normal");
  await openInstance(page);

  await page.evaluate(() => {
    const w = window as unknown as { __hidden: boolean; __terminalSockets: { drop(code: number): void }[] };
    w.__hidden = true;
    document.dispatchEvent(new Event("visibilitychange"));
    w.__terminalSockets[0].drop(1006);
  });
  await expect(page.getByText("Paused while the tab is hidden")).toBeVisible();
  await page.waitForTimeout(4_000);
  expect(await sockets(page)).toBe(1);

  await page.evaluate(() => {
    (window as unknown as { __hidden: boolean }).__hidden = false;
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect.poll(() => sockets(page), { timeout: 2_000 }).toBe(2);
  await expect(page.getByText("Connected", { exact: true })).toBeVisible();
});
