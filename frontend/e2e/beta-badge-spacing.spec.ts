import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

/**
 * Visual-parity test: the Beta badge in the top-left header must sit the same
 * number of pixels away from the *visible* end of the Xcelsior wordmark as the
 * sidebar's "Xcel AI" item badge sits from the end of its label.
 *
 * Important: this test uses the REAL wordmark SVG (not a placeholder div) so
 * that any transparent padding baked into the SVG viewBox is caught here. The
 * visual gap the user perceives is (badge.left - svg_text_right), NOT
 * (badge.left - img.right) — if the SVG has extra transparent space after the
 * text, the <img> right edge will be further out than the text and the visual
 * gap will look larger than the sidebar's.
 *
 * Fixtures mirror dashboard-shell.tsx exactly — if the class strings or the
 * wordmark SVG change, update both together.
 */
test("header Beta badge and sidebar badge share identical left-gap (SVG-aware)", async ({ page }) => {
    const svgPath = path.resolve(
        __dirname,
        "../public/xcelsior-logo-wordmark-iconbg.svg"
    );
    const wordmarkSvg = fs.readFileSync(svgPath, "utf8");

    const html = `<!doctype html>
<html>
<head>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { margin: 0; padding: 40px; background: #0A1222; color: #fff; font-family: system-ui, -apple-system, sans-serif; }
  /* Matches dashboard-shell.tsx wordmark sizing (h-[47px]). */
  #header svg { height: 47px; width: auto; display: block; }
</style>
</head>
<body>
  <!-- Header variant: REAL wordmark SVG inlined so getBBox is authoritative -->
  <div id="header" class="flex items-center gap-2 transition-all duration-300 ease-in-out">
    <span data-testid="wordmark-wrapper" class="shrink-0">${wordmarkSvg}</span>
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
    await page.waitForFunction(() => {
        const badge = document.querySelector('[data-testid="header-badge"]') as HTMLElement | null;
        if (!badge) return false;
        return getComputedStyle(badge).paddingLeft !== "0px";
    });

    const gaps = await page.evaluate(() => {
        const rect = (sel: string) =>
            (document.querySelector(sel) as HTMLElement).getBoundingClientRect();

        // Compute the DOM-coordinate right edge of the SVG's <text> element by
        // mapping its SVG-space bbox through the SVG's current transform.
        const svg = document.querySelector("#header svg") as SVGSVGElement;
        const text = svg.querySelector("text") as SVGTextElement;
        const tb = text.getBBox();
        const ctm = svg.getScreenCTM()!;
        const pt = svg.createSVGPoint();
        pt.x = tb.x + tb.width;
        pt.y = tb.y + tb.height / 2;
        const textRight = pt.matrixTransform(ctm).x;

        const headerBadge = rect('[data-testid="header-badge"]');
        const label = rect('[data-testid="sidebar-label"]');
        const sidebarBadge = rect('[data-testid="sidebar-badge"]');

        return {
            headerGap: Math.round(headerBadge.left - textRight),
            sidebarGap: Math.round(sidebarBadge.left - label.right),
            textRight: Math.round(textRight),
            imgRight: Math.round(svg.getBoundingClientRect().right),
        };
    });

    // The SVG's DOM right edge must be close to the <text> right edge —
    // i.e. the viewBox's transparent padding is trimmed. If this fails,
    // someone re-introduced extra viewBox padding in the wordmark SVG.
    expect(
        gaps.imgRight - gaps.textRight,
        `wordmark SVG has ${gaps.imgRight - gaps.textRight}px of transparent padding after the text; trim the viewBox`
    ).toBeLessThanOrEqual(6);

    // Both wrappers use `gap-2` (0.5rem = 8px). Visual gaps must match within
    // a small tolerance. `headerGap` uses the text edge, not the <img> edge.
    expect(gaps.headerGap).toBeGreaterThan(0);
    expect(gaps.sidebarGap).toBeGreaterThan(0);
    expect(
        Math.abs(gaps.headerGap - gaps.sidebarGap),
        `header gap=${gaps.headerGap}px, sidebar gap=${gaps.sidebarGap}px`
    ).toBeLessThanOrEqual(4);
});
