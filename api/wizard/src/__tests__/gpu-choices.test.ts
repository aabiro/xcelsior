// The renter's GPU list comes from the live feed, priced at the launch rate.

import { describe, expect, it } from "vitest";
import {
    buildGpuChoices, gpuChoiceId, gpuChoiceLabel, parseGpuChoiceId, sortGpuChoices, unpricedGpuCount,
} from "../gpu-choices.js";
import type { AvailableGpu } from "../api-client.js";

const gpu = (over: Partial<AvailableGpu>): AvailableGpu => ({
    gpu_model: "RTX 3060", vram_gb: 12, region: "ca-on", province: "ON",
    count_available: 1, price_per_hour_cad: 0.35, service_tiers_available: true, ...over,
});

describe("buildGpuChoices", () => {
    it("groups regions of one model and VRAM, summing free GPUs at the lowest rate", () => {
        const [choice, ...rest] = buildGpuChoices([
            gpu({ region: "ca-on", price_per_hour_cad: 0.40 }),
            gpu({ region: "us-east", province: "", count_available: 2, price_per_hour_cad: 0.35 }),
        ]);
        expect(rest).toEqual([]);
        expect(choice).toMatchObject({
            id: "RTX 3060|12", gpu_model: "RTX 3060", vram_gb: 12,
            price_per_hour: 0.35, count_available: 3, regions: ["ca-on", "us-east"],
        });
    });

    it("keeps two VRAM sizes of one model apart", () => {
        const ids = buildGpuChoices([gpu({ vram_gb: 12 }), gpu({ vram_gb: 24 })]).map((c) => c.id);
        expect(ids.sort()).toEqual(["RTX 3060|12", "RTX 3060|24"]);
    });

    it("leaves out GPUs a launch would be refused for", () => {
        const feed = [
            gpu({ gpu_model: "A", service_tiers_available: false }),
            gpu({ gpu_model: "B", price_per_hour_cad: 0 }),
            gpu({ gpu_model: "C", count_available: 0 }),
            gpu({ gpu_model: "D", vram_gb: 0 }),
            gpu({ gpu_model: "E" }),
        ];
        expect(buildGpuChoices(feed).map((c) => c.gpu_model)).toEqual(["E"]);
        expect(unpricedGpuCount(feed)).toBe(2);
    });

    it("offers nothing when nothing is free, so no stale or test listing can appear", () => {
        expect(buildGpuChoices([])).toEqual([]);
    });
});

describe("sortGpuChoices", () => {
    const choices = buildGpuChoices([
        gpu({ gpu_model: "RTX 4090", vram_gb: 24, price_per_hour_cad: 0.9 }),
        gpu({ gpu_model: "RTX 3060", vram_gb: 12, price_per_hour_cad: 0.35 }),
        gpu({ gpu_model: "A100", vram_gb: 80, price_per_hour_cad: 2.1 }),
    ]);

    it("cheapest first", () => {
        expect(sortGpuChoices(choices, "cheapest").map((c) => c.gpu_model)).toEqual(["RTX 3060", "RTX 4090", "A100"]);
    });

    it("most VRAM first for best", () => {
        expect(sortGpuChoices(choices, "best").map((c) => c.gpu_model)).toEqual(["A100", "RTX 4090", "RTX 3060"]);
    });

    it("by name otherwise", () => {
        expect(sortGpuChoices(choices, "specific").map((c) => c.gpu_model)).toEqual(["A100", "RTX 3060", "RTX 4090"]);
    });
});

describe("pick ids", () => {
    it("round-trip a model name with spaces", () => {
        expect(parseGpuChoiceId(gpuChoiceId("RTX 4070 Ti", 12))).toEqual({ gpu_model: "RTX 4070 Ti", vram_gb: 12 });
    });

    it("reject a legacy host-id pick", () => {
        expect(parseGpuChoiceId("host-1")).toBeNull();
        expect(parseGpuChoiceId("aaryn-tuf-rtx2060")).toBeNull();
        expect(parseGpuChoiceId(undefined)).toBeNull();
        expect(parseGpuChoiceId("RTX 3060|0")).toBeNull();
    });
});

it("labels a choice with its rate, free count and regions", () => {
    const [choice] = buildGpuChoices([gpu({ count_available: 2 })]);
    expect(gpuChoiceLabel(choice)).toBe("RTX 3060 · 12 GB · $0.35/hr · 2 free · ca-on");
});
