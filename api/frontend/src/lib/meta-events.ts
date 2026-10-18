/** First-party Meta click ids and browser Pixel helpers. */

const MAX_AGE = 60 * 60 * 24 * 90;

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

export function syncMetaCookies() {
  if (typeof window === "undefined") return;
  const fbclid = new URLSearchParams(window.location.search).get("fbclid");
  if (fbclid) {
    writeCookie("_fbc", `fb.1.${Date.now()}.${fbclid}`);
  }
  if (!readCookie("_fbp")) {
    const random = Math.floor(Math.random() * 1_000_000_0000);
    writeCookie("_fbp", `fb.1.${Date.now()}.${random}`);
  }
}

export function metaRequestHeaders(): Record<string, string> {
  if (typeof document === "undefined") return {};
  const headers: Record<string, string> = {};
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
  if (typeof window === "undefined" || typeof window.fbq !== "function") return;
  const id = eventId || newMetaEventId();
  window.fbq("track", eventName, customData, { eventID: id });
}

let lastPageKey = "";

export function trackPageView(pathname: string) {
  if (lastPageKey === pathname) return;
  lastPageKey = pathname;
  const eventId = newMetaEventId();
  trackBrowserEvent("PageView", {}, eventId);
  if (pathname === "/pricing" || pathname.startsWith("/gpus") || pathname.startsWith("/gpu-availability")) {
    trackBrowserEvent("ViewContent", { content_category: "gpu_compute", currency: "CAD" }, newMetaEventId());
  }
}
