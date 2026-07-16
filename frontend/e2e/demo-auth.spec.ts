// Proves the demo-account auth path end-to-end: the IP-gated button fills the
// login form, and the frontend mock gets past the dashboard auth gate. This is
// the "successful test" the rest of the suite's bypass is modelled on.
import { expect, test } from "@playwright/test";
import { DEMO_EMAIL, DEMO_PASSWORD, installDemoAuthMock } from "./helpers/demo-auth";

function json(body: unknown, status = 200) {
  return { status, contentType: "application/json", body: JSON.stringify(body) };
}

test.describe("demo account auth", () => {
  test("Demo button appears for whitelisted IPs and fills the form", async ({ page, context }) => {
    // Simulate a whitelisted network: the endpoint hands back the creds.
    await context.route("**/api/auth/demo-credentials", (route) =>
      route.fulfill(json({ email: DEMO_EMAIL, password: DEMO_PASSWORD })),
    );

    await page.goto("/login", { waitUntil: "domcontentloaded" });

    const demoButton = page.getByTestId("demo-account-button");
    await expect(demoButton).toBeVisible();

    await demoButton.click();
    await expect(page.locator("#email")).toHaveValue(DEMO_EMAIL);
    await expect(page.locator("#password")).toHaveValue(DEMO_PASSWORD);
  });

  test("Demo button is hidden when the network is not whitelisted", async ({ page, context }) => {
    await context.route("**/api/auth/demo-credentials", (route) =>
      route.fulfill(json({ detail: "not available" }, 403)),
    );

    await page.goto("/login", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("demo-account-button")).toHaveCount(0);
  });

  test("installDemoAuthMock gets past the dashboard auth gate", async ({ page, context, baseURL }) => {
    // Catch-all first so it has LOWER precedence than the auth routes that
    // installDemoAuthMock registers next (Playwright runs newest route first).
    await context.route("**/api/**", (route) => route.fulfill(json({ ok: true })));
    await installDemoAuthMock(context, { baseURL: baseURL ?? undefined });

    await page.goto("/dashboard", { waitUntil: "domcontentloaded" });
    // The whole point: we are NOT bounced to /login.
    await expect(page).not.toHaveURL(/\/login/);
  });
});
