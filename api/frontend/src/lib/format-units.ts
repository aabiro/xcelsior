/**
 * Durations and money, formatted one way across the dashboard.
 *
 * Money arrives as integer micros (1/1,000,000 of a dollar), the ledger unit.
 * No currency code is printed: copy says "/hr", not "CAD/hr".
 */

/** 42 → "42s", 90 → "1m 30s", 7380 → "2h 03m", 200000 → "2d 07h". */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  const pad = (n: number) => String(n).padStart(2, "0");
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${pad(s % 60)}s`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ${pad(m % 60)}m`;
  return `${Math.floor(h / 24)}d ${pad(h % 24)}h`;
}

/**
 * Micros as dollars. Below a dollar four places, else two: 389_083 → "$0.3891".
 * `exact` keeps every non-zero micro (up to six places) for receipts, where the
 * figure has to match the wallet to the micro: 7_500 → "$0.0075".
 */
export function formatMicros(
  micros: number | null | undefined,
  { exact = false }: { exact?: boolean } = {},
): string {
  if (micros === null || micros === undefined || !Number.isFinite(micros)) return "—";
  const dollars = micros / 1_000_000;
  const min = exact ? 2 : dollars !== 0 && Math.abs(dollars) < 1 ? 4 : 2;
  const max = exact ? 6 : min;
  return `$${dollars.toLocaleString("en-CA", { minimumFractionDigits: min, maximumFractionDigits: max })}`;
}

/**
 * A Unix-seconds timestamp as a short date: 1763800000 → "Nov 22, 2026".
 * The API sends times in seconds; `new Date(seconds)` reads them as
 * milliseconds and lands in January 1970, which the trust page showed.
 */
export function formatEpochDate(seconds: number | null | undefined, locale?: string): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds <= 0) {
    return "—";
  }
  return new Date(seconds * 1000).toLocaleDateString(locale, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}
