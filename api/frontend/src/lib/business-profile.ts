/**
 * A business account's public face, as the dashboard shows it.
 *
 * The API is the authority on every rule here (api/core/business_profile.py);
 * the checks below only let the form say what is wrong before a round trip.
 */
import type { BusinessLogoTile, User } from "@/lib/auth";

export const DISPLAY_NAME_MAX = 80;
export const TAGLINE_MAX = 120;
export const WEBSITE_MAX = 200;
export const LOGO_MAX_BYTES = 2 * 1024 * 1024;
export const LOGO_ACCEPT = "image/png,image/jpeg,image/webp";

export type BusinessIdentity = Pick<
  User,
  | "account_type"
  | "business_name"
  | "business_display_name"
  | "business_website"
  | "business_tagline"
  | "business_logo_url"
  | "business_logo_tile"
>;

export function isBusinessAccount(user?: Pick<User, "account_type"> | null): boolean {
  return user?.account_type === "business";
}

/** The name a business is shown as: its display name, else its legal name. */
export function businessName(user?: Pick<User, "business_display_name" | "business_name"> | null): string {
  return (user?.business_display_name || "").trim() || (user?.business_name || "").trim();
}

export function normalizeTile(tile?: string | null): BusinessLogoTile {
  return tile === "light" || tile === "dark" ? tile : "none";
}

/** `https://www.example.com/about` → `example.com/about`, for display only. */
export function websiteLabel(url?: string | null): string {
  if (!url) return "";
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.replace(/^www\./, "");
    const path = parsed.pathname === "/" ? "" : parsed.pathname.replace(/\/$/, "");
    return `${host}${path}`;
  } catch {
    return url;
  }
}

const HOST_RE = /^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,61}[a-z0-9]$/;

/** Mirrors the API's check: blank, or an http(s) address on a public domain name. */
export function websiteIsValid(raw: string): boolean {
  const value = raw.trim();
  if (!value) return true;
  if (/\s/.test(value)) return false;
  const withScheme = value.includes("://") ? value : `https://${value}`;
  let parsed: URL;
  try {
    parsed = new URL(withScheme);
  } catch {
    return false;
  }
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return false;
  if (parsed.username || parsed.password) return false;
  // URL gives the host in ASCII (punycode) already, as the API stores it.
  return HOST_RE.test(parsed.hostname.toLowerCase()) && withScheme.length <= WEBSITE_MAX;
}

/** Collapse whitespace the way the API does, so "dirty" compares like for like. */
export function cleanLine(raw: string): string {
  return raw.replace(/\s+/g, " ").trim();
}

/**
 * Up to two initials: "North Labs" → "NL", "Acme" → "A". Letters, not code
 * units, so an accented or non-Latin first letter is kept whole.
 */
export function monogram(name: string, max: 1 | 2 = 2): string {
  const words = name
    .split(/[\s\-_.&/]+/)
    .map((word) => word.replace(/^[^\p{L}\p{N}]+/u, ""))
    .filter(Boolean);
  const letters = words.slice(0, max).map((word) => Array.from(word)[0] ?? "");
  return letters.join("").toLocaleUpperCase() || "?";
}

/**
 * Gradient pairs for a business without a logo. Each is dark enough at both
 * ends for white initials, and the pick is stable for a given name, so a
 * business keeps its colour from one visit to the next.
 */
const MONOGRAM_GRADIENTS: ReadonlyArray<readonly [string, string]> = [
  ["#0e7490", "#6d28d9"],
  ["#6d28d9", "#be185d"],
  ["#047857", "#0e7490"],
  ["#c2410c", "#be185d"],
  ["#1d4ed8", "#7c3aed"],
  ["#0f766e", "#4d7c0f"],
];

export function monogramGradient(name: string): string {
  let hash = 0;
  for (const ch of name.trim().toLowerCase()) {
    hash = (hash * 31 + (ch.codePointAt(0) ?? 0)) >>> 0;
  }
  const [from, to] = MONOGRAM_GRADIENTS[hash % MONOGRAM_GRADIENTS.length];
  return `linear-gradient(135deg, ${from}, ${to})`;
}
