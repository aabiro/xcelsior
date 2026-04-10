import { expect, test, type Page } from "@playwright/test";

async function waitForActiveServiceWorker(page: Page) {
  await page.waitForFunction(async () => {
    const registration = await navigator.serviceWorker.getRegistration("/");
    return Boolean(registration?.active);
  });
}

async function createControlledPage(page: Page) {
  await waitForActiveServiceWorker(page);

  const controlledPage = await page.context().newPage();
  await controlledPage.goto("/");
  await controlledPage.waitForFunction(() => Boolean(navigator.serviceWorker.controller));
  return controlledPage;
}

test("desktop PWA exposes manifest metadata and registers the service worker", async ({ page, request }) => {
  const manifestResponse = await request.get("/manifest.webmanifest");
  expect(manifestResponse.ok()).toBeTruthy();

  const manifest = await manifestResponse.json();
  expect(manifest.display).toBe("standalone");
  expect(manifest.id).toBe("/");
  expect(manifest.scope).toBe("/");
  expect(manifest.shortcuts).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ url: "/dashboard" }),
      expect.objectContaining({ url: "/dashboard/marketplace" }),
      expect.objectContaining({ url: "/dashboard/billing" }),
    ]),
  );
  expect(
    manifest.icons.some(
      (icon: { src?: string; purpose?: string }) =>
        icon.src === "/xcelsior_icon_192x192.png" && icon.purpose === "maskable",
    ),
  ).toBe(true);

  await page.goto("/");
  const controlledPage = await createControlledPage(page);

  const registration = await controlledPage.evaluate(async () => {
    const serviceWorker = await navigator.serviceWorker.getRegistration("/");
    const manifestHref = document.querySelector('link[rel="manifest"]')?.getAttribute("href");

    return {
      controlled: Boolean(navigator.serviceWorker.controller),
      manifestHref,
      scope: serviceWorker?.scope ?? null,
      hasActiveWorker: Boolean(serviceWorker?.active),
    };
  });

  expect(registration.controlled).toBe(true);
  expect(registration.hasActiveWorker).toBe(true);
  expect(registration.manifestHref).toBe("/manifest.webmanifest");
  expect(registration.scope).toContain("127.0.0.1");

  await controlledPage.close();
});

test("desktop PWA falls back to the offline shell for uncached navigations", async ({ browser }) => {
  const context = await browser.newContext({
    serviceWorkers: "allow",
  });
  const page = await context.newPage();

  await page.goto("http://127.0.0.1:3100/");
  const controlledPage = await createControlledPage(page);

  await context.setOffline(true);
  await controlledPage.goto("http://127.0.0.1:3100/dashboard/offline-smoke-check", {
    waitUntil: "domcontentloaded",
  });

  await expect(
    controlledPage.getByRole("heading", { name: "Xcelsior is waiting for your connection to come back." }),
  ).toBeVisible();
  await expect(
    controlledPage.getByText("Cached pages and assets are still available, but live actions like launches,"),
  ).toBeVisible();

  await controlledPage.close();
  await page.close();
  await context.close();
});
