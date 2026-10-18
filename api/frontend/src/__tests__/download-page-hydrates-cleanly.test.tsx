/**
 * /download must hydrate without React rewriting its text.
 *
 * The detected platform came from a `useState` initializer guarded by
 * `typeof window === "undefined"`: the server rendered the generic label, the
 * client's first render rendered "Linux" or "macOS", and hydration threw
 * minified React error #418 on every visit. Found on the real production site by
 * the Playwright examination in `e2e-prod/`, not by any test here — a test that
 * renders only in jsdom never has a server pass to disagree with.
 *
 * This one does both passes: server markup with `window` hidden, then
 * `hydrateRoot` under a Linux user agent, failing on any recoverable error.
 */
import { act } from "react";
import { hydrateRoot } from "react-dom/client";
import { renderToString } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({
    t: (key: string, vars?: Record<string, string>) =>
      vars ? `${key}:${Object.values(vars).join(",")}` : key,
    locale: "en",
  }),
}));
vi.mock("@/components/marketing/auth-aware-link", () => ({
  AuthAwareLink: ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

import { DownloadContent } from "@/app/(marketing)/download/content";

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = "";
});

function serverMarkup(): string {
  // What Node sees: no `window`, no `navigator`.
  vi.stubGlobal("window", undefined);
  vi.stubGlobal("navigator", undefined);
  try {
    return renderToString(<DownloadContent />);
  } finally {
    vi.unstubAllGlobals();
  }
}

describe("/download hydration", () => {
  it("hydrates under a Linux user agent with no recoverable error", async () => {
    const html = serverMarkup();
    expect(html).toContain("download.primary_generic");

    vi.stubGlobal("navigator", { userAgent: "Mozilla/5.0 (X11; Linux x86_64)" });
    const container = document.createElement("div");
    container.innerHTML = html;
    document.body.appendChild(container);

    const recoverable: unknown[] = [];
    await act(async () => {
      hydrateRoot(container, <DownloadContent />, {
        onRecoverableError: (error) => recoverable.push(error),
      });
    });

    expect(recoverable, "hydration disagreed with the server's markup").toEqual([]);
    // And detection still happens, just after hydration rather than during it.
    expect(container.textContent).toContain("Linux");
  });
});
