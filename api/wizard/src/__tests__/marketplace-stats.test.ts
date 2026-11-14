// Tests for marketplace-stats.ts + buildMarketplaceSlides (live Learn data).

import { describe, it, expect, vi } from "vitest";
import { buildStatsFromAvailable, fetchMarketplaceStats } from "../marketplace-stats.js";
import { buildMarketplaceSlides } from "../learn-content.js";
import type { AvailableGpu } from "../api-client.js";

const api = vi.hoisted(() => ({ listAvailableGpus: vi.fn() }));
vi.mock("../api-client.js", async (original) => ({
    ...await original<typeof import("../api-client.js")>(), ...api,
}));

const gpu = (over: Partial<AvailableGpu>): AvailableGpu => ({
    gpu_model: "RTX 4090", vram_gb: 24, region: "ca-on", province: "ON",
    count_available: 1, price_per_hour_cad: 1, service_tiers_available: true, ...over,
});

describe("buildStatsFromAvailable", () => {
    const feed = [
        gpu({ gpu_model: "H100", price_per_hour_cad: 3.0, count_available: 2 }),
        gpu({ gpu_model: "RTX 4090", price_per_hour_cad: 1.0, count_available: 1 }),
    ];

    it("counts free GPUs, not rows", () => {
        const s = buildStatsFromAvailable(feed);
        expect(s.count).toBe(3);
        expect(s.activeCount).toBe(3);
    });

    it("computes price min/avg/max over free GPUs", () => {
        const s = buildStatsFromAvailable(feed);
        expect(s.minPrice).toBe(1.0);
        expect(s.maxPrice).toBe(3.0);
        expect(s.avgPrice).toBeCloseTo(2.33, 1);
        expect(s.prices).toEqual([1.0, 3.0, 3.0]); // ascending, one per GPU
    });

    it("rolls up per model, sorted by free count", () => {
        const s = buildStatsFromAvailable(feed);
        expect(s.byModel[0]).toMatchObject({ model: "H100", count: 2, avgPrice: 3 });
        expect(s.byModel[0].totalJobs).toBe(0); // the feed has no job history
    });

    it("handles an empty fleet safely", () => {
        const s = buildStatsFromAvailable([]);
        expect(s.count).toBe(0);
        expect(s.avgPrice).toBe(0);
        expect(s.byModel).toEqual([]);
    });

    it("ignores unpriced or empty entries", () => {
        const s = buildStatsFromAvailable([
            gpu({ price_per_hour_cad: NaN }),
            gpu({ price_per_hour_cad: 0 }),
            gpu({ count_available: 0, price_per_hour_cad: 5 }),
            gpu({ price_per_hour_cad: 2 }),
        ]);
        expect(s.prices).toEqual([2]);
    });
});

describe("fetchMarketplaceStats", () => {
    it("reads the live availability feed", async () => {
        api.listAvailableGpus.mockResolvedValueOnce([gpu({ gpu_model: "RTX 3060", price_per_hour_cad: 0.35 })]);
        const s = await fetchMarketplaceStats("https://example.test", "tok");
        expect(s?.byModel[0].model).toBe("RTX 3060");
    });

    it("returns null for an empty fleet so the pane shows concept cards", async () => {
        api.listAvailableGpus.mockResolvedValueOnce([]);
        expect(await fetchMarketplaceStats("https://example.test", "tok")).toBeNull();
    });
});

describe("buildMarketplaceSlides", () => {
    it("returns [] for empty stats (caller falls back to concept cards)", () => {
        expect(buildMarketplaceSlides(buildStatsFromAvailable([]))).toEqual([]);
    });

    it("builds live, data-driven slides with charts", () => {
        const stats = buildStatsFromAvailable([
            gpu({ gpu_model: "H100", price_per_hour_cad: 3 }),
            gpu({ gpu_model: "A100", price_per_hour_cad: 2 }),
            gpu({ gpu_model: "RTX 4090", price_per_hour_cad: 1 }),
        ]);
        const slides = buildMarketplaceSlides(stats);
        expect(slides.length).toBeGreaterThan(0);
        const joined = slides.map((s) => [s.heading, ...s.lines, ...(s.chart ?? [])].join(" ")).join("\n");
        expect(joined).toContain("Live marketplace");
        expect(joined).toContain("H100"); // real model in a chart
        expect(joined).toContain("3 GPU(s) free right now");
        // every chart is non-empty and free of NaN/undefined
        for (const s of slides) {
            if (s.chart) expect(s.chart.join("")).not.toMatch(/NaN|undefined/);
        }
    });
});
