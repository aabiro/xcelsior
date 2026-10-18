/**
 * One real login, saved for the signed-in projects.
 *
 * Through the demo-account button, which the server only renders when
 * `/api/auth/demo-credentials` agrees this network is whitelisted. No
 * credentials are typed from this file: if the gate says no, this fails, and
 * the signed-in examination does not run against a session it never had.
 */
import { expect, test as setup } from "@playwright/test";

const SESSION = "test-results-prod/.auth/demo.json";

setup("sign in through the demo gate", async ({ page }) => {
  const gate = page.waitForResponse((r) => r.url().includes("/api/auth/demo-credentials"));
  await page.goto("/login", { waitUntil: "domcontentloaded" });
  const res = await gate;
  expect(res.status(), "the demo gate refused this network — whitelist it in demo_account.py").toBe(200);

  await page.getByTestId("demo-account-button").click();
  const login = page.waitForResponse((r) => r.url().includes("/api/auth/login"));
  await page.locator("form").getByRole("button", { name: /sign in|log in/i }).first().click();
  expect((await login).status(), "the demo login itself failed").toBeLessThan(400);

  await page.waitForURL(/\/dashboard/, { timeout: 30_000 });
  await page.context().storageState({ path: SESSION });
});
