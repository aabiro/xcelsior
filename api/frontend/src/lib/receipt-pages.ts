/**
 * Server-side reads of the public workload pages (`/api/receipt-pages`).
 *
 * Every figure on a page is what the ledger billed for real runs their owners
 * shared; nothing here estimates or fills in. A failed read is reported as
 * such (null), never as "no pages".
 */
import type { ReceiptPage } from "@/lib/api";

const BACKEND = process.env.NEXT_PUBLIC_API_URL || "https://xcelsior.ca";
// Next's data cache keeps only 200 answers, so an error or a 404 is asked
// again on the next view instead of sticking for this long.
export const RECEIPT_PAGES_REVALIDATE_SECONDS = 900;

export async function fetchReceiptPages(): Promise<{ pages: ReceiptPage[]; minRuns: number } | null> {
  try {
    const res = await fetch(`${BACKEND}/api/receipt-pages`, {
      next: { revalidate: RECEIPT_PAGES_REVALIDATE_SECONDS },
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null;
    const body = await res.json();
    return { pages: Array.isArray(body.pages) ? body.pages : [], minRuns: Number(body.min_runs) || 3 };
  } catch {
    return null;
  }
}

/** The page, `"missing"` for a slug that does not qualify, or null when the read failed. */
export async function fetchReceiptPage(slug: string): Promise<ReceiptPage | "missing" | null> {
  if (!/^[a-z0-9][a-z0-9-]{0,127}$/.test(slug)) return "missing";
  try {
    const res = await fetch(`${BACKEND}/api/receipt-pages/${encodeURIComponent(slug)}`, {
      next: { revalidate: RECEIPT_PAGES_REVALIDATE_SECONDS },
      signal: AbortSignal.timeout(5000),
    });
    if (res.status === 404) return "missing";
    if (!res.ok) return null;
    const body = await res.json();
    return body.page ?? "missing";
  } catch {
    return null;
  }
}

export function pageTitle(page: Pick<ReceiptPage, "template_label" | "gpu_model">): string {
  return `${page.template_label} on ${page.gpu_model}: what real runs cost`;
}

/** "March 2026" from "2026-03". */
export function monthLabel(month: string | null): string | null {
  if (!month || !/^\d{4}-\d{2}$/.test(month)) return null;
  const [y, m] = month.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, 15)).toLocaleDateString("en-CA", {
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function pricingModeLabel(mode: string): string {
  return mode === "on_demand" ? "on-demand" : mode === "spot" ? "spot" : mode === "reserved" ? "reserved" : mode;
}
