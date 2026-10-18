// Tests for buildWizardContext — rich page_context generation

import { describe, it, expect } from "vitest";
import {
    buildWizardContext,
    type AutoCheckResults,
    type ProviderSummaryData,
} from "../useWizardFlow.js";
import type { MarketplaceListing } from "../api-client.js";

function makeSummary(overrides: Partial<ProviderSummaryData> = {}): ProviderSummaryData {
    return {
        gpuModel: "NVIDIA RTX 4090",
        vramGb: 24,
        xcuScore: 950,
        tflops: 82.5,
        verified: true,
        verificationState: "verified",
        hostId: "host-abc",
        pricing: "recommended",
        costPerHour: 0.42,
        admitted: true,
        runtimeRecommendation: "Use docker compose",
        reputationPoints: 100,
        tier: "gold",
        ...overrides,
    };
}

function makeListing(overrides: Partial<MarketplaceListing> = {}): MarketplaceListing {
    return {
        host_id: "host-1",
        gpu_model: "RTX 4090",
        vram_gb: 24,
        price_per_hour: 0.50,
        owner: "datacenter-alpha",
        active: true,
        total_jobs: 10,
        total_earned: 100,
        description: "Fast GPU",
        ...overrides,
    };
}

describe("buildWizardContext", () => {
    it("always includes cli-wizard prefix with step id", () => {
        const ctx = buildWizardContext("docker-check", {}, {}, null, [], null);
        expect(ctx).toContain("cli-wizard:docker-check");
    });

    it("includes mode when set", () => {
        const ctx = buildWizardContext("pricing", { mode: "provide" }, {}, null, [], null);
        expect(ctx).toContain("mode=provide");
    });

    it("includes pricing for provider mode", () => {
        const ctx = buildWizardContext("pricing", { mode: "provide", pricing: "custom", "custom-rate": "0.75" }, {}, null, [], null);
        expect(ctx).toContain("pricing=custom");
        expect(ctx).toContain("custom_rate=$0.75/hr");
    });

    it("includes host_id for provider mode", () => {
        const ctx = buildWizardContext("host-register", { mode: "provide", "_host_id": "host-xyz" }, {}, null, [], null);
        expect(ctx).toContain("host_id=host-xyz");
    });

    it("excludes provider context for renter mode", () => {
        const ctx = buildWizardContext("workload", {
            mode: "rent",
            pricing: "recommended", // should be ignored for renter
        }, {}, null, [], null);
        expect(ctx).not.toContain("pricing=");
        expect(ctx).toContain("mode=rent");
    });

    it("includes failed_checks summary", () => {
        const checks: Record<string, AutoCheckResults> = {
            "docker-check": {
                items: [
                    { name: "Docker", ok: false, detail: "not found" },
                    { name: "runc", ok: true, detail: "v1.2" },
                ],
                allPassed: false,
            },
        };
        const ctx = buildWizardContext("docker-check", { mode: "provide" }, checks, null, [], null);
        expect(ctx).toContain("failed_checks=");
        // failed_checks is URL-encoded; verify the decoded value
        const decoded = decodeURIComponent(ctx);
        expect(decoded).toContain("Docker: not found");
    });

    it("omits failed_checks when all pass", () => {
        const checks: Record<string, AutoCheckResults> = {
            "docker-check": {
                items: [{ name: "Docker", ok: true, detail: "v27" }],
                allPassed: true,
            },
        };
        const ctx = buildWizardContext("docker-check", { mode: "provide" }, checks, null, [], null);
        expect(ctx).not.toContain("failed_checks");
    });

    it("includes provider summary GPU info", () => {
        const summary = makeSummary();
        const ctx = buildWizardContext("provider-summary", { mode: "provide" }, {}, summary, [], null);
        expect(ctx).toContain("gpu=NVIDIA RTX 4090");
        expect(ctx).toContain("vram=24GB");
        expect(ctx).toContain("xcu=950");
        expect(ctx).toContain("tflops=82.5");
        expect(ctx).toContain("tier=gold");
        expect(ctx).toContain("verified=true");
    });

    it("includes renter workload and gpu preference", () => {
        const ctx = buildWizardContext("gpu-pick", {
            mode: "rent",
            workload: "training",
            "gpu-preference": "cheapest",
        }, {}, null, [], null);
        expect(ctx).toContain("workload=training");
        expect(ctx).toContain("gpu_pref=cheapest");
    });

    it("includes picked GPU details", () => {
        const listings = [makeListing({ host_id: "host-1", gpu_model: "A100", vram_gb: 80, price_per_hour: 2.50 })];
        const ctx = buildWizardContext("confirm-launch", {
            mode: "rent",
            "gpu-pick": "host-1",
        }, {}, null, listings, null);
        expect(ctx).toContain("picked_gpu=A100/80GB/$2.5/hr");
    });

    it("omits picked_gpu when listing not found", () => {
        const ctx = buildWizardContext("confirm-launch", {
            mode: "rent",
            "gpu-pick": "nonexistent",
        }, {}, null, [], null);
        expect(ctx).not.toContain("picked_gpu=");
    });

    it("includes image when set", () => {
        const ctx = buildWizardContext("confirm-launch", {
            mode: "rent",
            "image-pick": "nvcr.io/nvidia/pytorch:24.01-py3",
        }, {}, null, [], null);
        expect(ctx).toContain("image=nvcr.io/nvidia/pytorch:24.01-py3");
    });

    it("includes browse_error when present", () => {
        const ctx = buildWizardContext("browse-gpus", {
            mode: "rent",
        }, {}, null, [], "Network timeout");
        expect(ctx).toContain("browse_error=Network timeout");
    });

    it("omits browse_error when null", () => {
        const ctx = buildWizardContext("browse-gpus", { mode: "rent" }, {}, null, [], null);
        expect(ctx).not.toContain("browse_error");
    });

    it("handles both mode — includes provider and renter context", () => {
        const summary = makeSummary({ gpuModel: "A100", vramGb: 80 });
        const listings = [makeListing()];
        const ctx = buildWizardContext("provider-summary", {
            mode: "both",
            pricing: "recommended",
            workload: "inference",
            "gpu-preference": "most-vram",
        }, {}, summary, listings, null);
        // Provider context
        expect(ctx).toContain("pricing=recommended");
        expect(ctx).toContain("gpu=A100");
        expect(ctx).toContain("vram=80GB");
        // Renter context
        expect(ctx).toContain("workload=inference");
        expect(ctx).toContain("gpu_pref=most-vram");
    });

    it("multiple failed checks across steps", () => {
        const checks: Record<string, AutoCheckResults> = {
            "docker-check": {
                items: [{ name: "Docker", ok: false, detail: "not running" }],
                allPassed: false,
            },
            "version-check": {
                items: [{ name: "CUDA", ok: false, detail: "too old" }],
                allPassed: false,
            },
        };
        const ctx = buildWizardContext("verification", { mode: "provide" }, checks, null, [], null);
        // failed_checks is URL-encoded; verify the decoded value
        const decoded = decodeURIComponent(ctx);
        expect(decoded).toContain("Docker: not running");
        expect(decoded).toContain("CUDA: too old");
    });

    it("parts are pipe-delimited", () => {
        const ctx = buildWizardContext("pricing", {
            mode: "provide",
            pricing: "recommended",
        }, {}, null, [], null);
        const parts = ctx.split(" | ");
        expect(parts.length).toBeGreaterThanOrEqual(3);
        expect(parts[0]).toBe("cli-wizard:pricing");
    });

    it("produces minimal output with no answers", () => {
        const ctx = buildWizardContext("mode", {}, {}, null, [], null);
        expect(ctx).toBe("cli-wizard:mode");
    });
});
