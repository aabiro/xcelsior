// marketplace-stats.ts — Derive live marketplace figures for the Learn pane.
//
// The Learn slideshow's charts are data-driven; this fetches a real snapshot of
// the marketplace (once a token is available) and reduces it to the few numbers
// the charts need. Best-effort: returns null on any error so the Learn pane
// falls back to its built-in concept cards.

import { searchMarketplace, type MarketplaceListing } from "./api-client.js";

export interface ModelStat {
    model: string;
    avgPrice: number;
    count: number;
    totalJobs: number;
}

export interface MarketplaceStats {
    count: number;
    activeCount: number;
    avgPrice: number;
    minPrice: number;
    maxPrice: number;
    /** Per-GPU-model rollup, sorted by listing count desc. */
    byModel: ModelStat[];
    /** All listing prices (for a distribution sparkline), ascending. */
    prices: number[];
}

/** Reduce raw listings into chart-ready stats. Pure; exported for tests. */
export function buildStatsFromListings(listings: MarketplaceListing[]): MarketplaceStats {
    const prices = listings
        .map((l) => Number(l.price_per_hour))
        .filter((p) => Number.isFinite(p) && p >= 0)
        .sort((a, b) => a - b);

    const groups = new Map<string, { sum: number; count: number; jobs: number }>();
    for (const l of listings) {
        const model = (l.gpu_model || "Unknown").trim();
        const g = groups.get(model) ?? { sum: 0, count: 0, jobs: 0 };
        g.sum += Number(l.price_per_hour) || 0;
        g.count += 1;
        g.jobs += Number(l.total_jobs) || 0;
        groups.set(model, g);
    }
    const byModel: ModelStat[] = [...groups.entries()]
        .map(([model, g]) => ({
            model,
            avgPrice: Math.round((g.sum / Math.max(1, g.count)) * 100) / 100,
            count: g.count,
            totalJobs: g.jobs,
        }))
        .sort((a, b) => b.count - a.count);

    const sum = prices.reduce((a, b) => a + b, 0);
    return {
        count: listings.length,
        activeCount: listings.filter((l) => l.active).length,
        avgPrice: prices.length ? Math.round((sum / prices.length) * 100) / 100 : 0,
        minPrice: prices.length ? prices[0] : 0,
        maxPrice: prices.length ? prices[prices.length - 1] : 0,
        byModel,
        prices,
    };
}

/** Fetch a live marketplace snapshot. Best-effort — resolves null on any error. */
export async function fetchMarketplaceStats(
    baseUrl: string,
    token: string,
): Promise<MarketplaceStats | null> {
    try {
        const result = await searchMarketplace(baseUrl, token, { limit: 50, sort_by: "score" });
        if (!result.listings || result.listings.length === 0) return null;
        return buildStatsFromListings(result.listings);
    } catch {
        return null;
    }
}
