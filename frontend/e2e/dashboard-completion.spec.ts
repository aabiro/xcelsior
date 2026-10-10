import { test, expect, type BrowserContext, type Page } from "@playwright/test";

test.use({ serviceWorkers: "block", screenshot: "only-on-failure", actionTimeout: 15_000 });
test.setTimeout(90_000);

const user = {
  user_id: "dashboard-review-user", customer_id: "dashboard-review-wallet",
  email: "review@example.test", name: "Dashboard Review", role: "admin", is_admin: true,
  email_verified: true, team_can_write_instances: true, team_can_manage_billing: true,
};
const secret = `xcs_review_${"abcdef0123456789".repeat(6)}`;

async function installFixtures(context: BrowserContext, baseURL: string) {
  const origin = new URL(baseURL).origin;
  let teamId: string | null = null;
  const eventRequests: string[] = [];
  const keyRequests: string[] = [];
  const events = Array.from({ length: 60 }, (_, index) => ({
    event_id: `event-${index}`, event_type: "instance_started", severity: index % 2 ? "warning" : "info",
    timestamp: 1_790_000_000 - index, message: `Review event ${index}`, data: {},
  }));
  await context.addCookies([{ name: "xcelsior_session", value: "isolated-review-session", domain: new URL(origin).hostname, path: "/" }]);
  await context.addInitScript(() => {
    localStorage.setItem("xcelsior-locale", "en");
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
  });
  await context.route("**/*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== origin) {
      if (["fetch", "xhr", "eventsource"].includes(request.resourceType())) return route.abort();
      return route.continue();
    }
    const path = url.pathname;
    const json = (body: unknown, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (path === "/api/auth/me") return json({ ok: true, user: { ...user, team_id: teamId, team_name: teamId ? "Research Team" : null, team_role: "owner" } });
    if (path === "/api/auth/refresh") return json({ ok: true, access_token: "review", expires_in: 3600 });
    if (path === "/api/teams/me") return json({ ok: true, teams: [{ team_id: "team-review", name: "Research Team", plan: "pro", owner_email: user.email }], active_team_id: teamId });
    if (path === "/api/teams/active") {
      teamId = request.postDataJSON().team_id;
      return json({ ok: true, active_team_id: teamId, team_name: teamId ? "Research Team" : null });
    }
    if (path === "/api/events") {
      eventRequests.push(url.search);
      const severity = url.searchParams.get("severity");
      const rows = severity ? events.filter((event) => event.severity === severity) : events;
      const start = Number(url.searchParams.get("before") || 0);
      return json({ ok: true, events: rows.slice(start, start + 25), total: rows.length, event_types: ["instance_started"], next_cursor: start + 25 < rows.length ? String(start + 25) : null });
    }
    if (path === "/api/mcp/quick-connect") {
      const surface = url.searchParams.get("surface") || "mcp";
      keyRequests.push(surface);
      return json({ ok: true, access_token: `xcel_ai_${surface}_review_0123456789`, client_id: `${surface}-client`, mcp_url: "https://mcp.example.test/mcp", api_url: origin, scopes: [], in_use: false });
    }
    if (path === "/api/oauth/clients") {
      if (request.method() === "POST") return json({ ok: true, client: { client_id: "oauth_review", client_name: "Review Agent", client_secret: secret, scopes: ["instances:read", "instances:write", "billing:read"] } });
      return json({ ok: true, clients: [] });
    }
    if (path.startsWith("/api/billing/wallet/")) return json({ ok: true, wallet: { customer_id: user.customer_id, balance_cad: 125, total_deposited_cad: 100, status: "active" }, transactions: [] });
    if (path === "/api/users/me/preferences") return json({ ok: true, preferences: { onboarding: { profile: true, api_key: true, mcp_connect: true, browse: true, instance: true } } });
    if (["/instances", "/hosts", "/marketplace"].includes(path)) return json({ ok: true, instances: [], hosts: [], listings: [] });
    if (["/healthz", "/readyz"].includes(path)) return json({ ok: true });
    if (path.startsWith("/api/")) return json({ ok: true, items: [], keys: [], methods: [], consents: [], sessions: [], notifications: [], unread_count: 0, available: false });
    return route.continue();
  });
  return { eventRequests, keyRequests };
}

async function openAvatar(page: Page) {
  await page.locator("header button").filter({ hasText: user.name }).click();
}

for (const theme of ["light", "dark"] as const) {
  test(`${theme}: menus remain usable and CLI shows its own key`, async ({ context, page, baseURL }, info) => {
    const state = await installFixtures(context, baseURL!);
    await context.addInitScript((value) => localStorage.setItem("xcelsior-theme", value), theme);
    await page.goto("/dashboard/mcp");
    await expect(page.locator(".mcp-connect-card")).toBeVisible();
    await page.getByRole("button", { name: "Personal workspace", exact: true }).click();
    const menu = page.getByRole("listbox", { name: "Workspace" });
    await expect(menu).toBeVisible();
    await menu.getByRole("option", { name: "Research Team" }).hover();
    await expect(page.locator(".mcp-connect-card")).toBeVisible();
    await page.screenshot({ path: info.outputPath(`team-menu-${theme}.png`) });
    await page.keyboard.press("Escape");
    await expect(menu).toBeHidden();
    await page.getByRole("tab", { name: "CLI", exact: true }).click();
    await expect(page.getByText("xcel_ai_cli_review_0123456789", { exact: true })).toBeVisible();
    const command = page.getByRole("button", { name: /npx skills add/ });
    await expect(command).toBeVisible();
    expect(await command.evaluate((element) => getComputedStyle(element).fontFamily)).toMatch(/mono/i);
    expect(state.keyRequests).toContain("mcp");
    expect(state.keyRequests).toContain("cli");
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await command.click();
    await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe("npx skills add xcelsior-gpu/skill");
    const toast = page.locator("[data-sonner-toast]").first();
    await expect(toast).toBeVisible();
    await expect.poll(() => toast.evaluate((element) => element.getBoundingClientRect().bottom)).toBeLessThanOrEqual(700);
    await expect(page.locator("[data-sonner-toaster]")).toHaveAttribute("data-sonner-theme", theme);
    await page.screenshot({ path: info.outputPath(`cli-toast-${theme}.png`) });
    const ring = page.locator("header .user-avatar-ring").first();
    expect(await ring.evaluate((element) => getComputedStyle(element).backgroundImage)).not.toBe("none");
    await openAvatar(page);
    await page.locator('.dashboard-site-header-dropdown a[href$="#team"]').click();
    await expect(page).toHaveURL(/settings#team$/);
    await openAvatar(page);
    await page.locator('.dashboard-site-header-dropdown a[href$="#api-keys"]').click();
    await expect(page).toHaveURL(/settings#api-keys$/);
    await expect(page.getByRole("heading", { name: /SSH Keys/i }).first()).toBeVisible();
    await openAvatar(page);
    await page.locator('.dashboard-site-header-dropdown a[href$="#profile"]').click();
    await expect(page).toHaveURL(/settings#profile$/);
    await expect(page.getByRole("heading", { name: "Profile", exact: true })).toBeVisible();
  });

  test(`${theme}: newly created client secret stays single-line`, async ({ context, page, baseURL }, info) => {
    await installFixtures(context, baseURL!);
    await context.addInitScript((value) => localStorage.setItem("xcelsior-theme", value), theme);
    await page.goto("/dashboard/settings#mcp");
    await page.getByRole("button", { name: "Create MCP Client", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Save your client secret" });
    await expect(dialog).toBeFocused();
    await page.keyboard.press("Tab");
    await expect(dialog.getByRole("button", { name: "Close", exact: true })).toBeFocused();
    const value = page.locator(".oauth-secret-value");
    await expect(value).toHaveText(secret);
    expect(await value.evaluate((element) => getComputedStyle(element).whiteSpace)).toBe("nowrap");
    await page.screenshot({ path: info.outputPath(`secret-${theme}.png`) });
    await page.keyboard.press("Escape");
    await expect(value).toBeVisible();
    await page.setViewportSize({ width: 390, height: 844 });
    await expect(value).toBeVisible();
    expect(await value.evaluate((element) => element.scrollWidth > element.clientWidth)).toBe(true);
    await page.screenshot({ path: info.outputPath(`secret-mobile-${theme}.png`) });
    await page.getByRole("checkbox", { name: /copied and stored/i }).check();
    await page.getByRole("button", { name: "Done", exact: true }).click();
    await expect(value).toBeHidden();
  });
}

test("events traverse pages and filtering resets to the first page", async ({ context, page, baseURL }) => {
  const state = await installFixtures(context, baseURL!);
  await page.goto("/dashboard/events");
  await expect(page.getByText("Review event 0", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Newer events" })).toBeDisabled();
  await page.getByRole("button", { name: "Older events" }).click();
  await expect(page.getByText("Review event 25", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Older events" }).click();
  await expect(page.getByText("Review event 50", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Older events" })).toBeDisabled();
  await page.getByRole("button", { name: "Newer events" }).click();
  await expect(page.getByText("Review event 25", { exact: true })).toBeVisible();
  await page.getByLabel("Severity").selectOption("warning");
  await expect(page.getByText("Review event 1", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Newer events" })).toBeDisabled();
  expect(state.eventRequests.at(-1)).toContain("severity=warning");
  expect(state.eventRequests.at(-1)).not.toContain("before=");
});

test("a platform admin credits a wallet without paying, and the wallet shows it", async ({ context, page, baseURL }, info) => {
  await installFixtures(context, baseURL!);
  let balance = 125;
  const grants: { amount_cad: number; reason: string; idempotency_key: string }[] = [];
  // Registered after the fixtures, so it answers wallet requests first.
  await context.route(/\/api\/billing\/wallet\//, async (route) => {
    const request = route.request();
    const json = (body: unknown) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
    if (request.method() === "POST" && request.url().endsWith("/admin-credit")) {
      const body = request.postDataJSON();
      grants.push(body);
      balance += body.amount_cad;
      return json({ ok: true, tx_id: "tx-review", balance_cad: balance, amount_cad: body.amount_cad });
    }
    return json({ ok: true, wallet: { customer_id: user.customer_id, balance_cad: balance, total_deposited_cad: 100, status: "active" }, transactions: [] });
  });
  await page.goto("/dashboard/billing?topup=true");
  await page.getByText("Admin credit", { exact: true }).click();
  await page.getByRole("button", { name: "$100", exact: true }).click();
  await page.getByLabel(/Reason/).fill("End-to-end launch test");
  await page.screenshot({ path: info.outputPath("admin-credit.png") });
  await page.getByRole("button", { name: "Credit $100.00" }).click();
  await expect(page.locator("[data-sonner-toast]").filter({ hasText: "$100.00 CAD credited" })).toBeVisible();
  expect(grants).toHaveLength(1);
  expect(grants[0]).toMatchObject({ amount_cad: 100, reason: "End-to-end launch test" });
  expect(grants[0].idempotency_key).toMatch(/\S{8,}/);
  await expect(page.locator("header").getByText("$225.00")).toBeVisible();
});

test("the Xcel AI panel is closed again after leaving the dashboard and coming back", async ({ context, page, baseURL }) => {
  await installFixtures(context, baseURL!);
  await context.addInitScript(() => localStorage.setItem("xcelsior-ai-onboarding-v1", "1"));
  const panel = page.locator(".dashboard-site-ai-panel");
  await page.goto("/dashboard/mcp");
  await page.locator(".dashboard-site-ai-toggle").click();
  await expect(panel).toBeVisible();
  // Moving between dashboard pages keeps it open.
  await page.locator('a[href="/dashboard/settings"]:visible').first().click();
  await expect(page).toHaveURL(/\/dashboard\/settings/);
  await expect(panel).toBeVisible();

  // Out to the product site in the same tab, then back through its own link.
  await page.locator('a[href="/features"]:visible').first().click();
  await expect(page).toHaveURL(/\/features$/);
  await page.locator('a[href="/dashboard"]:visible').first().click();
  await expect(page).toHaveURL(/\/dashboard/);
  await expect(page.locator(".dashboard-site-ai-toggle")).toBeVisible();
  await expect(panel).toBeHidden();

  // And with the browser's Back button.
  await page.locator(".dashboard-site-ai-toggle").click();
  await expect(panel).toBeVisible();
  await page.locator('a[href="/features"]:visible').first().click();
  await expect(page).toHaveURL(/\/features$/);
  await page.goBack();
  await expect(page).toHaveURL(/\/dashboard/);
  await expect(page.locator(".dashboard-site-ai-toggle")).toBeVisible();
  await expect(panel).toBeHidden();
});
