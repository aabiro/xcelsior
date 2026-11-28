import { curatePricingGpus, type PricingCardRow } from "@/lib/marketing-gpu";

/**
 * The pricing page's rows, its share image and its metadata all read the
 * published rate card (`/api/pricing/reference`, the researched per-GPU rates)
 * and the live spot market. Nothing here is invented: a rate the card does not
 * carry is `null` and shows as "—", and when the card cannot be read the page
 * says so instead of showing a stand-in list.
 *
 * It replaced a hard-coded six-GPU fallback list and spot/reserved rates
 * derived by fixed ratios (x0.4, x0.8, x0.55) — prices no one had set.
 */

const BACKEND = process.env.NEXT_PUBLIC_API_URL || process.env.INTERNAL_API_URL || "https://xcelsior.ca";

interface RateCardEntry {
  vram_gb?: number;
  base_rate_cad: number;
  spot_cad?: number | null;
  reserved_1mo_cad?: number | null;
  reserved_1yr_cad?: number | null;
}

interface FleetRow {
  gpu_model: string;
  vram_gb: number;
  count_total?: number;
  count_available: number;
}

export interface PricingData {
  rows: PricingCardRow[];
  /** Models with at least one listed GPU in the live fleet, and how many are free now. */
  fleet: Record<string, number>;
  maxSpotSaving: number | null;
  maxReservedSaving: number | null;
}

async function fetchJson<T>(path: string, revalidate: number): Promise<T | null> {
  try {
    const res = await fetch(`${BACKEND}${path}`, { next: { revalidate }, signal: AbortSignal.timeout(5000) });
    return res.ok ? ((await res.json()) as T) : null;
  } catch {
    return null;
  }
}

function savings(rows: PricingCardRow[], pick: (row: PricingCardRow) => number | null): number | null {
  const pct = rows
    .map((row) => {
      const rate = pick(row);
      return rate === null || !row.onDemand ? null : Math.round((1 - rate / row.onDemand) * 100);
    })
    .filter((value): value is number => value !== null);
  return pct.length ? Math.max(...pct) : null;
}

export async function loadPricing(max = 10): Promise<PricingData> {
  const [card, spot, available] = await Promise.all([
    fetchJson<{ pricing?: Record<string, RateCardEntry> }>("/api/pricing/reference", 3600),
    fetchJson<{ prices?: Record<string, number>; spot_prices?: Record<string, number> }>("/spot-prices", 300),
    fetchJson<{ inventory?: FleetRow[]; gpus?: FleetRow[] }>("/api/v2/gpu/available", 300),
  ]);
  const liveSpot = spot?.prices || spot?.spot_prices || {};

  const fleet: Record<string, number> = {};
  for (const gpu of available?.inventory?.length ? available.inventory : available?.gpus || []) {
    if ((gpu.count_total ?? gpu.count_available) <= 0) continue;
    fleet[gpu.gpu_model] = (fleet[gpu.gpu_model] ?? 0) + Math.max(0, gpu.count_available);
  }

  const all: PricingCardRow[] = Object.entries(card?.pricing || {})
    // The table and calculator may quote only hardware someone actually lists,
    // including busy inventory. A published price is not fleet capacity.
    .filter(([model]) => Object.hasOwn(fleet, model))
    // A row needs its published rate and its memory; neither is guessed.
    .filter(([, rate]) => rate.base_rate_cad > 0 && (rate.vram_gb ?? 0) > 0)
    .map(([model, rate]) => ({
      model,
      vram: rate.vram_gb as number,
      onDemand: rate.base_rate_cad,
      spot: liveSpot[model] ?? rate.spot_cad ?? null,
      reserved1m: rate.reserved_1mo_cad ?? null,
      reserved1y: rate.reserved_1yr_cad ?? null,
    }));

  // GPUs a renter can take today are always listed, whatever the marquee order.
  const rows = curatePricingGpus(all, max, Object.keys(fleet));
  return {
    rows,
    fleet,
    maxSpotSaving: savings(rows, (row) => row.spot),
    maxReservedSaving: savings(rows, (row) => row.reserved1y),
  };
}
