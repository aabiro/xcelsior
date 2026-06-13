// Tests for marketplace-stats.ts + buildMarketplaceSlides (live Learn data).

import { describe, it, expect } from "vitest";
import { buildStatsFromListings } from "../marketplace-stats.js";
import { buildMarketplaceSlides } from "../learn-content.js";
import type { MarketplaceListing } from "../api-client.js";

const listing = (over: Partial<MarketplaceListing>): MarketplaceListing => ({
    host_id: "h", gpu_model: "RTX 4090", vram_gb: 24, price_per_hour: 1, owner: "o",
    active: true, total_jobs: 0, total_earned: 0, description: "", ...over,
});

describe("buildStatsFromListings", () => {
    const listings = [
        listing({ gpu_model: "H100", price_per_hour: 3.0, total_jobs: 10, active: true }),
        listing({ gpu_model: "H100", price_per_hour: 4.0, total_jobs: 5, active: false }),
        listing({ gpu_model: "RTX 4090", price_per_hour: 1.0, total_jobs: 20, active: true }),
    ];

    it("counts listings and active ones", () => {
        const s = buildStatsFromListings(listings);
        expect(s.count).toBe(3);
        expect(s.activeCount).toBe(2);
    });

    it("computes price min/avg/max", () => {
        const s = buildStatsFromListings(listings);
        expect(s.minPrice).toBe(1.0);
        expect(s.maxPrice).toBe(4.0);
        expect(s.avgPrice).toBeCloseTo(2.67, 1);
        expect(s.prices).toEqual([1.0, 3.0, 4.0]); // ascending
    });

    it("rolls up per-model with avg price and jobs, sorted by count", () => {
        const s = buildStatsFromListings(listings);
        expect(s.byModel[0].model).toBe("H100"); // 2 listings
        expect(s.byModel[0].count).toBe(2);
        expect(s.byModel[0].avgPrice).toBe(3.5);
        expect(s.byModel[0].totalJobs).toBe(15);
    });

    it("handles an empty list safely", () => {
        const s = buildStatsFromListings([]);
        expect(s.count).toBe(0);
        expect(s.avgPrice).toBe(0);
        expect(s.byModel).toEqual([]);
    });

    it("ignores non-finite/negative prices in the distribution", () => {
        const s = buildStatsFromListings([
            listing({ price_per_hour: NaN as unknown as number }),
            listing({ price_per_hour: -5 }),
            listing({ price_per_hour: 2 }),
        ]);
        expect(s.prices).toEqual([2]);
    });
});

describe("buildMarketplaceSlides", () => {
    it("returns [] for empty stats (caller falls back to concept cards)", () => {
        expect(buildMarketplaceSlides(buildStatsFromListings([]))).toEqual([]);
    });

    it("builds live, data-driven slides with charts", () => {
        const stats = buildStatsFromListings([
            listing({ gpu_model: "H100", price_per_hour: 3, total_jobs: 12 }),
            listing({ gpu_model: "A100", price_per_hour: 2, total_jobs: 30 }),
            listing({ gpu_model: "RTX 4090", price_per_hour: 1, total_jobs: 4 }),
        ]);
        const slides = buildMarketplaceSlides(stats);
        expect(slides.length).toBeGreaterThan(0);
        const joined = slides.map((s) => [s.heading, ...s.lines, ...(s.chart ?? [])].join(" ")).join("\n");
        expect(joined).toContain("Live marketplace");
        expect(joined).toContain("H100"); // real model in a chart
        expect(joined).toContain("3 GPUs listed");
        // every chart is non-empty and free of NaN/undefined
        for (const s of slides) {
            if (s.chart) expect(s.chart.join("")).not.toMatch(/NaN|undefined/);
        }
    });
});
