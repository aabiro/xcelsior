/** First-party Meta click ids and browser Pixel helpers. Nothing runs without marketing consent. */

import { hasMarketingConsent, readConsent } from "./cookie-consent";

const MAX_AGE = 60 * 60 * 24 * 90;
const CONSENT_URL = "/api/meta/consent";

type Fbq = ((...args: unknown[]) => void) & { queue?: unknown[]; loaded?: boolean; version?: string };

declare global {
  interface Window {
    fbq?: Fbq;
    _fbq?: Fbq;
  }
}

function readCookie(name: string): string {
  if (typeof document === "undefined") return "";
  const prefix = `${name}=`;
  for (const part of document.cookie.split(";")) {
    const trimmed = part.trim();
    if (trimmed.startsWith(prefix)) return decodeURIComponent(trimmed.slice(prefix.length));
  }
  return "";
}

function writeCookie(name: string, value: string) {
  const secure = window.location.protocol === "https:" ? "; Secure" : "";
  document.cookie = `${name}=${encodeURIComponent(value)}; Max-Age=${MAX_AGE}; Path=/; SameSite=Lax${secure}`;
}

/** Remove _fbp/_fbc, including copies fbevents set on the parent domain. */
export function clearMetaCookies() {
  if (typeof document === "undefined") return;
  const host = window.location.hostname;
  const parts = host.split(".");
  const parentDomain =
    parts.length >= 2 && host !== "localhost" && !/^\d+\.\d+\.\d+\.\d+$/.test(host)
      ? `.${parts.slice(-2).join(".")}`
      : "";
  for (const name of ["_fbp", "_fbc"]) {
    document.cookie = `${name}=; Max-Age=0; expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/; SameSite=Lax`;
    if (parentDomain) {
      document.cookie = `${name}=; Max-Age=0; expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/; domain=${parentDomain}; SameSite=Lax`;
    }
  }
}

export function syncMetaCookies() {
  if (typeof window === "undefined") return;
  if (!hasMarketingConsent()) {
    clearMetaCookies();
    return;
  }
  const fbclid = new URLSearchParams(window.location.search).get("fbclid");
  if (fbclid) {
    writeCookie("_fbc", `fb.1.${Date.now()}.${fbclid}`);
  }
  if (!readCookie("_fbp")) {
    const random = Math.floor(Math.random() * 1_000_000_0000);
    writeCookie("_fbp", `fb.1.${Date.now()}.${random}`);
  }
}

/** Click ids plus the visitor's marketing choice. The API sends no Conversions API event without "granted". */
export function metaRequestHeaders(): Record<string, string> {
  if (typeof document === "undefined") return {};
  if (!hasMarketingConsent()) return { "X-Marketing-Consent": "denied" };
  const headers: Record<string, string> = { "X-Marketing-Consent": "granted" };
  const fbp = readCookie("_fbp");
  const fbc = readCookie("_fbc");
  if (fbp) headers["X-FB-FBP"] = fbp;
  if (fbc) headers["X-FB-FBC"] = fbc;
  return headers;
}

export function newMetaEventId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `evt_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export function trackBrowserEvent(
  eventName: string,
  customData: Record<string, unknown> = {},
  eventId?: string,
) {
  if (typeof window === "undefined" || !hasMarketingConsent()) return;
  if (typeof window.fbq !== "function") return;
  const id = eventId || newMetaEventId();
  window.fbq("track", eventName, customData, { eventID: id });
}

let lastPageKey = "";

export function trackPageView(pathname: string) {
  if (!hasMarketingConsent()) return;
  if (typeof window === "undefined" || typeof window.fbq !== "function") return;
  if (lastPageKey === pathname) return;
  lastPageKey = pathname;
  const eventId = newMetaEventId();
  trackBrowserEvent("PageView", {}, eventId);
  if (pathname === "/pricing" || pathname.startsWith("/gpus") || pathname.startsWith("/gpu-availability")) {
    trackBrowserEvent("ViewContent", { content_category: "gpu_compute", currency: "CAD" }, newMetaEventId());
  }
}

let lastSyncedConsent: string | null = null;

/**
 * Save a signed-in visitor's banner choice to their account, so the Stripe
 * webhook and auto top-ups (which have no browser) can respect it. Only an
 * explicit choice is sent, and only when it differs from the last one sent
 * for this user on this page load.
 */
export async function syncConsentToAccount(userId: string | null | undefined) {
  if (typeof window === "undefined" || !userId) return;
  const consent = readConsent();
  if (!consent) return;
  const key = `${userId}:${consent.marketing}`;
  if (lastSyncedConsent === key) return;
  lastSyncedConsent = key;
  try {
    const res = await fetch(CONSENT_URL, {
      method: "PUT",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ marketing: consent.marketing, source: consent.source }),
    });
    if (!res.ok) lastSyncedConsent = null;
  } catch {
    lastSyncedConsent = null;
  }
}

/** Test hook: forget page-load state. */
export function _resetMetaEventsForTests() {
  lastPageKey = "";
  lastSyncedConsent = null;
}
