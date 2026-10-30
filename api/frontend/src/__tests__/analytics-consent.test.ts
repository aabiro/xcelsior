import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const ph = vi.hoisted(() => ({
  opt_in_capturing: vi.fn(),
  opt_out_capturing: vi.fn(),
  has_opted_in_capturing: vi.fn(() => false),
  has_opted_out_capturing: vi.fn(() => false),
}));

vi.mock("posthog-js", () => ({ default: ph }));

import { applyAnalyticsConsent } from "@/components/analytics/AnalyticsConsent";
import { setMarketingConsent } from "@/lib/cookie-consent";

function clearCookies() {
  for (const part of document.cookie.split(";")) {
    const name = part.split("=")[0]?.trim();
    if (name) document.cookie = `${name}=; Max-Age=0; Path=/`;
  }
}

describe("analytics follow the cookie choice", () => {
  const gtag = vi.fn();

  beforeEach(() => {
    clearCookies();
    window.localStorage.clear();
    (window as unknown as { gtag: typeof gtag }).gtag = gtag;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("leaves PostHog in its cookieless default and GA denied before a choice", () => {
    applyAnalyticsConsent();
    expect(ph.opt_in_capturing).not.toHaveBeenCalled();
    expect(ph.opt_out_capturing).not.toHaveBeenCalled();
    expect(gtag).toHaveBeenCalledWith("consent", "update", expect.objectContaining({ analytics_storage: "denied", ad_storage: "denied" }));
  });

  it("opts in once on accept", () => {
    setMarketingConsent(true);
    applyAnalyticsConsent();
    expect(ph.opt_in_capturing).toHaveBeenCalledTimes(1);
    expect(gtag).toHaveBeenCalledWith("consent", "update", expect.objectContaining({ analytics_storage: "granted" }));
    ph.has_opted_in_capturing.mockReturnValueOnce(true);
    applyAnalyticsConsent();
    expect(ph.opt_in_capturing).toHaveBeenCalledTimes(1);
  });

  it("opts out on decline", () => {
    setMarketingConsent(false);
    applyAnalyticsConsent();
    expect(ph.opt_out_capturing).toHaveBeenCalledTimes(1);
    expect(ph.opt_in_capturing).not.toHaveBeenCalled();
  });
});
