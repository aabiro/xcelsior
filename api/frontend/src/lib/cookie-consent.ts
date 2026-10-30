/** The visitor's cookie-banner choice. Only "marketing" (Meta Pixel and Conversions API) needs opt-in. */

const COOKIE_NAME = "xcelsior_consent";
const STORAGE_KEY = "xcelsior.cookieConsent";
const MAX_AGE = 60 * 60 * 24 * 365;

export const CONSENT_CHANGED_EVENT = "xcelsior:cookie-consent-changed";
export const OPEN_COOKIE_PREFERENCES_EVENT = "xcelsior:open-cookie-preferences";

export type ConsentSource = "banner" | "settings";

export interface ConsentPreferences {
  marketing: boolean;
  updated_at: string;
  source: ConsentSource;
}

function parse(raw: string | null | undefined): ConsentPreferences | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<ConsentPreferences>;
    if (typeof parsed.marketing !== "boolean") return null;
    return {
      marketing: parsed.marketing,
      updated_at: typeof parsed.updated_at === "string" ? parsed.updated_at : new Date(0).toISOString(),
      source: parsed.source === "settings" ? "settings" : "banner",
    };
  } catch {
    return null;
  }
}

function readCookieValue(): string {
  if (typeof document === "undefined") return "";
  const prefix = `${COOKIE_NAME}=`;
  for (const part of document.cookie.split(";")) {
    const trimmed = part.trim();
    if (trimmed.startsWith(prefix)) {
      try {
        return decodeURIComponent(trimmed.slice(prefix.length));
      } catch {
        return "";
      }
    }
  }
  return "";
}

function writeCookie(prefs: ConsentPreferences) {
  const secure = window.location.protocol === "https:" ? "; Secure" : "";
  const value = encodeURIComponent(JSON.stringify(prefs));
  document.cookie = `${COOKIE_NAME}=${value}; Max-Age=${MAX_AGE}; Path=/; SameSite=Lax${secure}`;
}

/** The stored choice, or null when the visitor has not made one. */
export function readConsent(): ConsentPreferences | null {
  if (typeof window === "undefined") return null;
  const fromCookie = parse(readCookieValue());
  if (fromCookie) return fromCookie;
  try {
    const fromStorage = parse(window.localStorage.getItem(STORAGE_KEY));
    if (fromStorage) writeCookie(fromStorage);
    return fromStorage;
  } catch {
    return null;
  }
}

/** True only after an explicit opt-in. No choice is no consent. */
export function hasMarketingConsent(): boolean {
  return readConsent()?.marketing === true;
}

export function setMarketingConsent(marketing: boolean, source: ConsentSource = "banner") {
  if (typeof window === "undefined") return;
  const prefs: ConsentPreferences = { marketing, updated_at: new Date().toISOString(), source };
  try {
    writeCookie(prefs);
  } catch {
    // A blocked cookie still leaves the localStorage copy.
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    // Private mode or quota: the cookie copy is enough.
  }
  window.dispatchEvent(new CustomEvent(CONSENT_CHANGED_EVENT, { detail: prefs }));
}

/** Reopen the banner so the visitor can change their choice. */
export function openCookiePreferences() {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(OPEN_COOKIE_PREFERENCES_EVENT));
}
