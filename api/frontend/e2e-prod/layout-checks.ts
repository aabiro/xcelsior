import type { Page } from "@playwright/test";

/**
 * Content a reader cannot reach: text, controls and images that end past the
 * right edge of the viewport and are not inside something that scrolls
 * sideways on purpose.
 *
 * `document.scrollWidth` alone missed every case found on 2026-10-06. The
 * dashboard scrolls inside `<main>`, not the document, and the shell clips
 * overflow, so a 640px Settings page on a 412px phone, a top bar that pushed
 * the wallet and account menu off-screen, and an Analytics toolbar running off
 * the edge all reported zero overflow. `<main>` is a page scroller, not a
 * sideways one, so content past the edge inside it still counts.
 */
export async function clippedContent(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const vw = window.innerWidth;
    const found: string[] = [];
    const scrollsSideways = (el: Element) => {
      for (let a = el.parentElement; a && a !== document.body && a.tagName !== "MAIN"; a = a.parentElement) {
        const o = getComputedStyle(a).overflowX;
        if (o === "auto" || o === "scroll") return true;
      }
      return false;
    };
    for (const el of Array.from(document.querySelectorAll("body *"))) {
      if (el.closest("[aria-hidden=true], script, style, noscript")) continue;
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height || r.right <= vw + 1) continue;
      const st = getComputedStyle(el);
      if (st.visibility === "hidden" || st.opacity === "0") continue;
      const ownText = Array.from(el.childNodes).some((n) => n.nodeType === 3 && (n.textContent ?? "").trim());
      const control = el.matches("a, button, input, select, textarea, img, [role=button], [role=tab]");
      if (!ownText && !control) continue;
      if (scrollsSideways(el)) continue;
      const label = (el.textContent || el.getAttribute("aria-label") || el.getAttribute("alt") || "")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 40);
      found.push(`<${el.tagName.toLowerCase()}> "${label}" ends at ${Math.round(r.right)}px of ${vw}`);
    }
    return Array.from(new Set(found)).slice(0, 15);
  });
}
