/** Stable date strings for SSR + hydration (UTC, locale-independent). */

const MONTHS_EN = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
] as const;

const MONTHS_FR = [
  "janvier",
  "février",
  "mars",
  "avril",
  "mai",
  "juin",
  "juillet",
  "août",
  "septembre",
  "octobre",
  "novembre",
  "décembre",
] as const;

export function formatBlogDate(iso: string, locale: "en" | "fr" = "en"): string {
  const part = (iso || "").split("T")[0];
  const [y, m, d] = part.split("-").map((n) => parseInt(n, 10));
  if (!y || !m || !d) return part || iso;
  const months = locale === "fr" ? MONTHS_FR : MONTHS_EN;
  const month = months[m - 1] ?? String(m);
  if (locale === "fr") {
    return `${d} ${month} ${y}`;
  }
  return `${month} ${d}, ${y}`;
}