import { expect, test, type BrowserContext, type Page } from "@playwright/test";
import { WEB_PUSH_LIFECYCLE_MESSAGE_TYPE } from "../src/lib/pwa/web-push";

const SMOKE_EMAIL = process.env.XCELSIOR_PWA_SMOKE_EMAIL?.trim();
const SMOKE_PASSWORD = process.env.XCELSIOR_PWA_SMOKE_PASSWORD?.trim();
const REQUIRE_CLICK = process.env.XCELSIOR_PWA_SMOKE_REQUIRE_CLICK !== "0";
const CLICK_TIMEOUT_MS = Number(process.env.XCELSIOR_PWA_SMOKE_CLICK_TIMEOUT_MS ?? "45000");

async function waitForActiveServiceWorker(page: Page) {
  await page.waitForFunction(async () => {
    const registration = await navigator.serviceWorker.getRegistration("/");
    return Boolean(registration?.active);
  });
}

async function ensureControlledPage(page: Page) {
  await waitForActiveServiceWorker(page);
  const controlled = await page.evaluate(() => Boolean(navigator.serviceWorker.controller));
  if (!controlled) {
    await page.reload();
    await page.waitForFunction(() => Boolean(navigator.serviceWorker.controller));
  }
}

async function registerPushLifecycleListener(page: Page) {
  await page.evaluate((messageType) => {
    const win = window as Window & {
      __xcelsiorPushLifecycleListenerInstalled?: boolean;
      __xcelsiorPushLifecycleEvents?: Array<Record<string, unknown>>;
    };

    win.__xcelsiorPushLifecycleEvents = [];
    if (win.__xcelsiorPushLifecycleListenerInstalled) return;

    navigator.serviceWorker.addEventListener("message", (event) => {
      if (event.data?.type !== messageType) return;
      win.__xcelsiorPushLifecycleEvents?.push(event.data as Record<string, unknown>);
    });

    win.__xcelsiorPushLifecycleListenerInstalled = true;
  }, WEB_PUSH_LIFECYCLE_MESSAGE_TYPE);
}

async function login(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Email").fill(SMOKE_EMAIL!);
  await page.getByLabel("Password").fill(SMOKE_PASSWORD!);
  await Promise.all([
    page.waitForURL(/\/dashboard(?:\/)?$/),
    page.locator("form button[type='submit']").click(),
  ]);
}

async function enableDesktopNotifications(page: Page) {
  await page.goto("/dashboard/settings");
  await ensureControlledPage(page);

  const enableButton = page.getByRole("button", { name: /Enable notifications|Disable notifications/i });
  await expect(enableButton).toBeVisible();

  const currentLabel = await enableButton.innerText();
  if (/Enable notifications/i.test(currentLabel)) {
    await enableButton.click();
    await expect(page.getByRole("button", { name: /Disable notifications/i })).toBeVisible({
      timeout: 20_000,
    });
  }
}

async function disableDesktopNotifications(page: Page) {
  await page.goto("/dashboard/settings");
  const disableButton = page.getByRole("button", { name: /Disable notifications/i });
  if (await disableButton.isVisible().catch(() => false)) {
    await disableButton.click();
    await expect(page.getByRole("button", { name: /Enable notifications/i })).toBeVisible({
      timeout: 20_000,
    });
  }
}

async function grantNotificationPermission(context: BrowserContext, baseURL: string) {
  const origin = new URL(baseURL).origin;
  await context.grantPermissions(["notifications"], { origin });
}

test.describe("authenticated desktop push staging smoke", () => {
  test.skip(!SMOKE_EMAIL || !SMOKE_PASSWORD, "Set XCELSIOR_PWA_SMOKE_EMAIL and XCELSIOR_PWA_SMOKE_PASSWORD.");

  test("admin can subscribe, receive a real push, confirm click-through, and unsubscribe", async ({ page, baseURL }) => {
    test.setTimeout(180_000);

    await grantNotificationPermission(page.context(), baseURL!);

    await login(page);
    await enableDesktopNotifications(page);

    await page.goto("/dashboard/admin");
    await ensureControlledPage(page);
    await registerPushLifecycleListener(page);

    await page.getByRole("button", { name: "Send Test Notification" }).click();

    const receivedHandle = await page.waitForFunction((messageType) => {
      const events = (window as Window & {
        __xcelsiorPushLifecycleEvents?: Array<Record<string, unknown>>;
      }).__xcelsiorPushLifecycleEvents || [];

      return (
        events.find((entry) =>
          entry.type === messageType &&
          entry.event === "received" &&
          typeof entry.data === "object" &&
          entry.data !== null &&
          (entry.data as Record<string, unknown>).smoke_test === true,
        ) || null
      );
    }, WEB_PUSH_LIFECYCLE_MESSAGE_TYPE, {
      timeout: 30_000,
    });

    const receivedEvent = await receivedHandle.jsonValue() as {
      url: string;
      notificationId: string | null;
      data?: { smoke_id?: string };
    };

    expect(receivedEvent.url).toContain("/dashboard/admin?push_smoke=1");

    await expect.poll(async () => {
      return page.evaluate(async () => {
        const registration = await navigator.serviceWorker.getRegistration("/");
        const notifications = await registration?.getNotifications();
        return (notifications || []).map((notification) => ({
          title: notification.title,
          data: notification.data,
        }));
      });
    }, { timeout: 30_000 }).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          title: "Desktop push smoke test",
        }),
      ]),
    );

    if (REQUIRE_CLICK) {
      const clickedHandle = await page.waitForFunction((messageType) => {
        const events = (window as Window & {
          __xcelsiorPushLifecycleEvents?: Array<Record<string, unknown>>;
        }).__xcelsiorPushLifecycleEvents || [];

        return (
          events.find((entry) =>
            entry.type === messageType &&
            entry.event === "clicked" &&
            typeof entry.data === "object" &&
            entry.data !== null &&
            (entry.data as Record<string, unknown>).smoke_test === true,
          ) || null
        );
      }, WEB_PUSH_LIFECYCLE_MESSAGE_TYPE, {
        timeout: CLICK_TIMEOUT_MS,
      });

      const clickedEvent = await clickedHandle.jsonValue() as { url: string };
      await expect(page).toHaveURL(new RegExp(clickedEvent.url.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    }

    await page.evaluate(async () => {
      const registration = await navigator.serviceWorker.getRegistration("/");
      const notifications = await registration?.getNotifications();
      await Promise.all((notifications || []).map((notification) => notification.close()));
    });

    await disableDesktopNotifications(page);
  });
});
