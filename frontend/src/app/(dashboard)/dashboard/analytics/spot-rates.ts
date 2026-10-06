import type { SpotPriceRow } from "@/lib/api";

/**
 * The highest live spot rates, one per GPU model, for the Platform Pulse chart.
 *
 * `spot_prices` is a list of rows. The page used to treat it as a
 * `{model: price}` map, so `Object.entries` on the list produced array indices
 * as GPU names and whole rows as prices: the "0", "1", "$NaN" chart that
 * production showed. Rows without a model or a finite rate are dropped,
 * because the feed does carry a row with an empty `gpu_model`.
 */
export function topSpotRates(
  res: { spot_prices?: SpotPriceRow[] } | null | undefined,
  limit = 10,
): { model: string; price: number }[] {
  return (res?.spot_prices ?? [])
    .filter((r) => r.gpu_model?.trim() && Number.isFinite(r.rate_cad))
    .map((r) => ({ model: r.gpu_model, price: r.rate_cad }))
    .sort((a, b) => b.price - a.price)
    .slice(0, limit);
}
