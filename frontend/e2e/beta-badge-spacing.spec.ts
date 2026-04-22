import { expect, test } from "@playwright/test";

/**
 * Visual-parity test: the Beta badge in the top-left header must sit the same
 * number of pixels away from the end of the Xcelsior wordmark as the sidebar's
 * "Xcel AI" item badge sits from the end of its label.
 *
 * We render the exact same Tailwind class lists used in dashboard-shell.tsx
 * into an isolated page so the test doesn't depend on auth/session. If those
 * class strings drift in dashboard-shell.tsx, update the fixtures below in
 * lockstep — they are the contract this test enforces.
 */
test("header Beta badge and sidebar badge share identical left-gap", async ({ page }) => {
  // Fixtures mirror dashboard-shell.tsx exactly.
  //   - Header wrapper:   "flex items-center gap-2 …"
  //   - Sidebar wrapper:  "flex items-center gap-2 …"
  //   - Badge span:       "shrink-0 rounded-full bg-accent-cyan/8 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-widest text-accent-cyan/70"
  const html = `<!doctype html>
<html>
<head>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { margin: 0; padding: 40px; background: #0A1222; color: #fff; font-family: system-ui, -apple-system, sans-serif; }
  /* Stand-in for the wordmark — we only need its box width. */
  .wordmark { display:inline-block; width: 172px; height: 52px; background:#1a2340; }
</style>
</head>
<body>
  <!-- Header variant -->
  <div id="header" class="flex items-center gap-2 transition-all duration-300 ease-in-out">
    <span class="wordmark" data-testid="wordmark"></span>
    <span data-testid="header-badge" class="shrink-0 rounded-full bg-accent-cyan/8 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-widest text-accent-cyan/70" style="background:rgba(0,212,255,0.08); color:rgba(0,212,255,0.7);">Beta</span>
  </div>

  <!-- Sidebar variant -->
  <div id="sidebar" class="flex items-center gap-2" style="margin-top:40px;">
    <span data-testid="sidebar-label" style="font-size:16px; line-height:24px;">Xcel AI</span>
    <span data-testid="sidebar-badge" class="shrink-0 rounded-full bg-accent-cyan/8 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-widest text-accent-cyan/70" style="background:rgba(0,212,255,0.08); color:rgba(0,212,255,0.7);">AI</span>
  </div>
</body>
</html>`;

  await page.setContent(html, { waitUntil: "load" });
  // Give Tailwind CDN time to apply styles.
  await page.waitForFunction(() => {
    const badge = document.querySelector('[data-testid="header-badge"]') as HTMLElement | null;
    if (!badge) return false;
    return getComputedStyle(badge).paddingLeft !== "0px";
  });

  const gaps = await page.evaluate(() => {
    const rect = (sel: string) =>
      (document.querySelector(sel) as HTMLElement).getBoundingClientRect();

    const wordmark = rect('[data-testid="wordmark"]');
    const headerBadge = rect('[data-testid="header-badge"]');
    const label = rect('[data-testid="sidebar-label"]');
    const sidebarBadge = rect('[data-testid="sidebar-badge"]');

    return {
      headerGap: Math.round(headerBadge.left - wordmark.right),
      sidebarGap: Math.round(sidebarBadge.left - label.right),
    };
  });

  // Both wrappers use `gap-2` (0.5rem = 8px by Tailwind default) so the gap
  // should be identical. Allow a 1px tolerance for sub-pixel rounding.
  expect(gaps.headerGap).toBeGreaterThan(0);
  expect(gaps.sidebarGap).toBeGreaterThan(0);
  expect(Math.abs(gaps.headerGap - gaps.sidebarGap)).toBeLessThanOrEqual(1);
});
