import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

/**
 * The marketing navbar (.site-nav) is sticky at the top, at z-index 60, and
 * stays over the page. Content that scrolls passes under it and comes back
 * out, but anything pinned to the viewport (position: fixed) never does: if
 * the navbar covers one of its controls, that control can't be tapped at all.
 *
 * That is how the chat panel's close button got trapped on phones: the panel
 * opened full-screen at z-50, so its header sat under the navbar, and a tap on
 * the X landed on the navbar's menu button instead.
 *
 * These checks hit-test the real pages rather than reading z-index values, so
 * a new fixed overlay, a z-index change or a new marketing page is covered
 * without editing this file.
 *
 * Run against a dev or prod server with XCELSIOR_E2E_BASE_URL, or let the
 * Playwright config build and start the standalone server.
 */

const PHONE = { width: 375, height: 812 };
/** Wider than Tailwind's sm breakpoint, so the chat opens as the floating
 * panel; a phone on its side is the shortest screen that gets it. */
const PHONE_LANDSCAPE = { width: 812, height: 375 };
const DESKTOP = { width: 1280, height: 800 };

/** Every static marketing route, read from the app directory so that new
 * pages are checked without editing this list. Dynamic segments are skipped. */
function marketingRoutes(): string[] {
  const root = path.resolve(__dirname, "../src/app/(marketing)");
  const routes: string[] = [];
  const walk = (dir: string, route: string) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.isFile() && entry.name === "page.tsx") routes.push(route || "/");
      if (!entry.isDirectory() || entry.name.startsWith("[")) continue;
      const segment = entry.name.startsWith("(") ? "" : `/${entry.name}`;
      walk(path.join(dir, entry.name), route + segment);
    }
  };
  walk(root, "");
  return routes.sort();
}

/**
 * Controls inside fixed layers whose centre the navbar covers, described for
 * the failure message. Elements scrolled under the navbar are not counted:
 * they are reachable by scrolling, so only pinned ones can be trapped.
 */
async function pinnedControlsUnderNavbar(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const nav = document.querySelector(".site-nav");
    if (!nav) return [];
    const inFixedLayer = (el: Element) => {
      for (let node: Element | null = el; node; node = node.parentElement) {
        if (nav.contains(node)) return false;
        if (getComputedStyle(node).position === "fixed") return true;
      }
      return false;
    };
    const describe = (el: Element) => {
      const name = el.getAttribute("aria-label") || el.textContent?.trim().slice(0, 40) || "";
      return `${el.tagName.toLowerCase()} "${name}"`;
    };
    const covered: string[] = [];
    const controls = Array.from(document.querySelectorAll(
      'a[href], button, input:not([type="hidden"]), textarea, select, [role="button"]',
    ));
    for (const el of controls) {
      if (!inFixedLayer(el)) continue;
      const box = el.getBoundingClientRect();
      if (box.width < 1 || box.height < 1) continue;
      const x = box.left + box.width / 2;
      const y = box.top + box.height / 2;
      if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) continue;
      const hit = document.elementFromPoint(x, y);
      if (hit && nav.contains(hit)) {
        covered.push(`${describe(el)} at y=${Math.round(box.top)}-${Math.round(box.bottom)}, under ${describe(hit)}`);
      }
    }
    return covered;
  });
}

async function navbarBottom(page: Page): Promise<number> {
  return page.locator(".site-nav").evaluate((nav) => nav.getBoundingClientRect().bottom);
}

for (const viewport of [PHONE, PHONE_LANDSCAPE, DESKTOP]) {
  test(`chat panel controls stay clear of the navbar at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.goto("/");
    // A first visit gets the cookie notice, which covers the chat button
    // until it's answered.
    await page.getByRole("region", { name: "Cookie notice" }).getByRole("button", { name: "Decline" }).click();
    const fab = page.getByRole("button", { name: "Open AI chat assistant" });
    // The widget mounts when the browser is idle, up to five seconds in.
    await fab.click({ timeout: 15_000 });

    const close = page.getByRole("button", { name: "Close chat" });
    await expect(close).toBeVisible();
    // On a phone the sheet covers the navbar; on wider screens the panel
    // stays below it. Either way, nothing in it may be under the navbar.
    expect(await pinnedControlsUnderNavbar(page)).toEqual([]);

    // A real tap, not a programmatic click: Playwright refuses it if another
    // element would receive it, so it has to land on the X.
    await close.click();
    await expect(close).toBeHidden();
    await expect(fab).toBeVisible();
  });
}

test("nothing pinned to the viewport sits under the navbar on any marketing page", async ({ page }) => {
  test.setTimeout(240_000);
  const failures: string[] = [];
  for (const viewport of [PHONE, DESKTOP]) {
    await page.setViewportSize(viewport);
    for (const route of marketingRoutes()) {
      await page.goto(route, { waitUntil: "load" });
      // Check mid-page too: fixed layers often appear only after a scroll.
      for (const scroll of [0, 0.5]) {
        await page.evaluate((f) => window.scrollTo(0, document.body.scrollHeight * f), scroll);
        await page.waitForTimeout(300);
        for (const item of await pinnedControlsUnderNavbar(page)) {
          failures.push(`${route} at ${viewport.width}px, scrolled ${scroll * 100}%: ${item}`);
        }
      }
    }
  }
  expect(failures).toEqual([]);
});

test("in-page links stop below the navbar instead of under it", async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize(PHONE);
  let checked = 0;
  const failures: string[] = [];
  for (const route of marketingRoutes()) {
    await page.goto(route, { waitUntil: "load" });
    const ids = await page.evaluate(() =>
      Array.from(document.querySelectorAll<HTMLAnchorElement>('a[href^="#"]'))
        .map((a) => a.getAttribute("href")!.slice(1))
        .filter((id) => id && document.getElementById(id)),
    );
    for (const id of new Set(ids)) {
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.locator(`a[href="#${id}"]`).first().click();
      await page.waitForFunction((target) => {
        const box = document.getElementById(target)!.getBoundingClientRect();
        return box.top < innerHeight && box.top > -innerHeight;
      }, id);
      await page.waitForTimeout(400); // let smooth scrolling settle
      const top = await page.evaluate((target) => document.getElementById(target)!.getBoundingClientRect().top, id);
      const bottom = await navbarBottom(page);
      checked += 1;
      if (top < bottom - 1) failures.push(`${route}#${id} stops at y=${Math.round(top)}, under the navbar (bottom ${Math.round(bottom)})`);
    }
  }
  expect(checked).toBeGreaterThan(0);
  expect(failures).toEqual([]);
});
