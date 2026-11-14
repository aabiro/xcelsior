// marketplace-stats.ts — Derive live marketplace figures for the Learn pane.
//
// The Learn slideshow's charts are data-driven; this fetches a real snapshot
// (once a token is available) and reduces it to the few numbers the charts
// need. It reads the same feed the GPU pick does (`/api/v2/gpu/available`):
// free GPUs on admitted, ready hosts, at the rate a launch is charged. The
// legacy listing table it used to read kept test rows for hosts that never
// existed, so the "live" chart showed GPUs nobody could rent. Best-effort:
// returns null on any error so the pane falls back to its concept cards.

import { listAvailableGpus, type AvailableGpu } from "./api-client.js";

export interface ModelStat {
    model: string;
    avgPrice: number;
    /** Free GPUs of this model. */
    count: number;
    totalJobs: number;
}

export interface MarketplaceStats {
    /** Free GPUs right now. */
    count: number;
    activeCount: number;
    avgPrice: number;
    minPrice: number;
    maxPrice: number;
    /** Per-GPU-model rollup, sorted by free count desc. */
    byModel: ModelStat[];
    /** One price per free GPU (for a distribution sparkline), ascending. */
    prices: number[];
}

/** Reduce the live feed into chart-ready stats. Pure; exported for tests. */
export function buildStatsFromAvailable(gpus: AvailableGpu[]): MarketplaceStats {
    const prices: number[] = [];
    const groups = new Map<string, { sum: number; count: number }>();
    for (const gpu of gpus) {
        const price = Number(gpu.price_per_hour_cad);
        const count = Math.max(0, Math.floor(Number(gpu.count_available) || 0));
        if (!Number.isFinite(price) || price <= 0 || count === 0) continue;
        const model = (gpu.gpu_model || "Unknown").trim();
        const g = groups.get(model) ?? { sum: 0, count: 0 };
        g.sum += price * count;
        g.count += count;
        groups.set(model, g);
        for (let i = 0; i < count; i += 1) prices.push(price);
    }
    prices.sort((a, b) => a - b);
    const byModel: ModelStat[] = [...groups.entries()]
        .map(([model, g]) => ({
            model,
            avgPrice: Math.round((g.sum / Math.max(1, g.count)) * 100) / 100,
            count: g.count,
            // The availability feed carries no job history.
            totalJobs: 0,
        }))
        .sort((a, b) => b.count - a.count);

    const sum = prices.reduce((a, b) => a + b, 0);
    return {
        count: prices.length,
        activeCount: prices.length,
        avgPrice: prices.length ? Math.round((sum / prices.length) * 100) / 100 : 0,
        minPrice: prices.length ? prices[0] : 0,
        maxPrice: prices.length ? prices[prices.length - 1] : 0,
        byModel,
        prices,
    };
}

/** Fetch a live snapshot. Best-effort — resolves null on any error or an empty fleet. */
export async function fetchMarketplaceStats(
    baseUrl: string,
    token: string,
): Promise<MarketplaceStats | null> {
    try {
        const stats = buildStatsFromAvailable(await listAvailableGpus(baseUrl, token));
        return stats.count > 0 ? stats : null;
    } catch {
        return null;
    }
}
