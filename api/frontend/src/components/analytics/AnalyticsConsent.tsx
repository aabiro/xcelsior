"use client";

import { useEffect } from "react";
import posthog from "posthog-js";
import { CONSENT_CHANGED_EVENT, readConsent } from "@/lib/cookie-consent";

type Gtag = (...args: unknown[]) => void;

/** Applies the cookie-banner choice to PostHog and Google Analytics, on load and on change. */
export function applyAnalyticsConsent() {
  const consent = readConsent();
  const granted = consent?.marketing === true;
  try {
    // Each opt-in records an event, so only call on a real change.
    if (granted && !posthog.has_opted_in_capturing()) posthog.opt_in_capturing();
    else if (consent && !granted && !posthog.has_opted_out_capturing()) posthog.opt_out_capturing();
  } catch {
    // PostHog is not initialised without a project token.
  }
  const gtag = (window as unknown as { gtag?: Gtag }).gtag;
  if (typeof gtag === "function") {
    const state = granted ? "granted" : "denied";
    gtag("consent", "update", {
      ad_storage: state,
      analytics_storage: state,
      ad_user_data: state,
      ad_personalization: state,
    });
  }
}

export function AnalyticsConsent() {
  useEffect(() => {
    applyAnalyticsConsent();
    window.addEventListener(CONSENT_CHANGED_EVENT, applyAnalyticsConsent);
    return () => window.removeEventListener(CONSENT_CHANGED_EVENT, applyAnalyticsConsent);
  }, []);
  return null;
}
