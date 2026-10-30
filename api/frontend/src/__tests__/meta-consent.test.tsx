import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { CookieBanner } from "@/components/CookieBanner";
import {
  CONSENT_CHANGED_EVENT,
  hasMarketingConsent,
  openCookiePreferences,
  readConsent,
  setMarketingConsent,
} from "@/lib/cookie-consent";
import {
  _resetMetaEventsForTests,
  metaRequestHeaders,
  syncConsentToAccount,
  syncMetaCookies,
  trackBrowserEvent,
  trackPageView,
} from "@/lib/meta-events";

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({ t: (key: string) => key, locale: "en", displayLocale: "en", toggleLocale: () => {} }),
}));

function clearCookies() {
  for (const part of document.cookie.split(";")) {
    const name = part.split("=")[0]?.trim();
    if (name) document.cookie = `${name}=; Max-Age=0; Path=/`;
  }
}

beforeEach(() => {
  clearCookies();
  window.localStorage.clear();
  _resetMetaEventsForTests();
  window.fbq = undefined;
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("cookie consent", () => {
  it("is no consent until the visitor chooses", () => {
    expect(readConsent()).toBeNull();
    expect(hasMarketingConsent()).toBe(false);
  });

  it("stores the choice and announces it", () => {
    const heard: unknown[] = [];
    const onChange = (event: Event) => heard.push((event as CustomEvent).detail);
    window.addEventListener(CONSENT_CHANGED_EVENT, onChange);
    setMarketingConsent(true);
    window.removeEventListener(CONSENT_CHANGED_EVENT, onChange);
    expect(hasMarketingConsent()).toBe(true);
    expect(heard).toHaveLength(1);
    expect(document.cookie).toContain("xcelsior_consent=");
  });

  it("falls back to localStorage when the cookie is gone", () => {
    setMarketingConsent(true, "settings");
    clearCookies();
    expect(readConsent()).toMatchObject({ marketing: true, source: "settings" });
  });
});

describe("Meta helpers follow consent", () => {
  it("send denied, and no click ids, without consent", () => {
    document.cookie = "_fbp=fb.1.1710000000000.1234567890; Path=/";
    expect(metaRequestHeaders()).toEqual({ "X-Marketing-Consent": "denied" });
  });

  it("send granted with click ids after opting in", () => {
    setMarketingConsent(true);
    document.cookie = "_fbp=fb.1.1710000000000.1234567890; Path=/";
    expect(metaRequestHeaders()).toEqual({
      "X-Marketing-Consent": "granted",
      "X-FB-FBP": "fb.1.1710000000000.1234567890",
    });
  });

  it("set no _fbp without consent and clear an old one", () => {
    document.cookie = "_fbp=fb.1.1710000000000.1234567890; Path=/";
    syncMetaCookies();
    expect(document.cookie).not.toContain("_fbp=");
    setMarketingConsent(true);
    syncMetaCookies();
    expect(document.cookie).toContain("_fbp=fb.1.");
  });

  it("fire no Pixel event without consent", () => {
    const fbq = vi.fn();
    window.fbq = fbq as unknown as Window["fbq"];
    trackBrowserEvent("Purchase", { value: 10 });
    trackPageView("/pricing");
    expect(fbq).not.toHaveBeenCalled();
    setMarketingConsent(true);
    trackBrowserEvent("Purchase", { value: 10 }, "purchase_1");
    expect(fbq).toHaveBeenCalledWith("track", "Purchase", { value: 10 }, { eventID: "purchase_1" });
  });
});

describe("saving the choice to the account", () => {
  it("sends nothing before a choice or when signed out", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}"));
    await syncConsentToAccount("u_1");
    setMarketingConsent(true);
    await syncConsentToAccount(null);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("sends each change once per user", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}"));
    setMarketingConsent(true);
    await syncConsentToAccount("u_1");
    await syncConsentToAccount("u_1");
    setMarketingConsent(false, "settings");
    await syncConsentToAccount("u_1");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const [url, init] = fetchMock.mock.calls[1];
    expect(url).toBe("/api/meta/consent");
    expect(init).toMatchObject({ method: "PUT", credentials: "include" });
    expect(JSON.parse(String(init?.body))).toEqual({ marketing: false, source: "settings" });
  });

  it("retries after a failed save", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("{}", { status: 500 }))
      .mockResolvedValueOnce(new Response("{}"));
    setMarketingConsent(true);
    await syncConsentToAccount("u_1");
    await syncConsentToAccount("u_1");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("CookieBanner", () => {
  it("asks until the visitor chooses, and declining is one click", async () => {
    render(<CookieBanner />);
    fireEvent.click(await screen.findByRole("button", { name: "cookies.decline" }));
    expect(readConsent()).toMatchObject({ marketing: false, source: "banner" });
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("stays hidden after a choice and reopens from the footer", async () => {
    setMarketingConsent(false);
    render(<CookieBanner />);
    await act(() => new Promise((resolve) => requestAnimationFrame(() => resolve(undefined))));
    expect(screen.queryByRole("dialog")).toBeNull();
    act(() => openCookiePreferences());
    fireEvent.click(screen.getByRole("button", { name: "cookies.accept" }));
    expect(hasMarketingConsent()).toBe(true);
  });
});
