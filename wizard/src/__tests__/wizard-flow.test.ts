// Tests for wizard-flow.ts — step definitions, conditions, navigation

import { describe, it, expect } from "vitest";
import { WIZARD_STEPS, getNextStep, IMAGE_TEMPLATES, WORKLOAD_IMAGE_MAP } from "../wizard-flow.js";

describe("wizard-flow", () => {
    it("first step is mode select", () => {
        expect(WIZARD_STEPS[0].id).toBe("mode");
        expect(WIZARD_STEPS[0].type).toBe("select");
        expect(WIZARD_STEPS[0].options).toHaveLength(3);
    });

    it("last step is done", () => {
        const last = WIZARD_STEPS[WIZARD_STEPS.length - 1];
        expect(last.type).toBe("done");
        expect(last.id).toBe("done");
    });

    it("every step has a unique id", () => {
        const ids = WIZARD_STEPS.map((s) => s.id);
        expect(new Set(ids).size).toBe(ids.length);
    });

    it("every step has a non-empty prompt", () => {
        for (const step of WIZARD_STEPS) {
            expect(step.prompt.length).toBeGreaterThan(0);
        }
    });

    it("IMAGE_TEMPLATES has at least 5 entries", () => {
        expect(IMAGE_TEMPLATES.length).toBeGreaterThanOrEqual(5);
        for (const t of IMAGE_TEMPLATES) {
            expect(t.label.length).toBeGreaterThan(0);
            expect(t.value.length).toBeGreaterThan(0);
        }
    });

    it("WORKLOAD_IMAGE_MAP covers common workloads", () => {
        expect(WORKLOAD_IMAGE_MAP["training"]).toBeDefined();
        expect(WORKLOAD_IMAGE_MAP["inference"]).toBeDefined();
        expect(WORKLOAD_IMAGE_MAP["other"]).toBeDefined();
    });

    describe("getNextStep — renter flow", () => {
        it("includes device-auth after mode for renters", () => {
            const next = getNextStep(0, { mode: "rent" });
            expect(WIZARD_STEPS[next].id).toBe("device-auth");
        });

        it("skips provider-only steps", () => {
            // Walk entire renter flow
            const visited: string[] = [];
            let idx = 0;
            const answers: Record<string, string> = {
                mode: "rent",
                "device-auth": "authorized",
                "api-key": "test-token",
                workload: "training",
                "gpu-preference": "cheapest",
                "browse-gpus": "done",
                "gpu-pick": "host-1",
                "image-pick": "nvcr.io/nvidia/pytorch:24.01-py3",
                "confirm-launch": "yes",
                "wallet-check": "passed",
                "_wallet_insufficient": "false",
                "launch-instance": "passed",
            };

            while (idx < WIZARD_STEPS.length && WIZARD_STEPS[idx].type !== "done") {
                visited.push(WIZARD_STEPS[idx].id);
                idx = getNextStep(idx, answers);
                if (idx === -1) break;
            }

            // Should NOT include provider-only steps
            expect(visited).not.toContain("confirm-setup");
            // Should include renter steps
            expect(visited).toContain("workload");
            expect(visited).toContain("gpu-preference");
            expect(visited).toContain("browse-gpus");
            expect(visited).toContain("gpu-pick");
            expect(visited).toContain("device-auth");
        });
    });

    describe("getNextStep — provider flow", () => {
        it("includes docker-check for providers", () => {
            const next = getNextStep(0, { mode: "provide" });
            expect(WIZARD_STEPS[next].id).toBe("docker-check");
        });

        it("skips renter-only steps", () => {
            const visited: string[] = [];
            let idx = 0;
            const answers: Record<string, string> = { mode: "provide", pricing: "recommended" };

            while (idx < WIZARD_STEPS.length && WIZARD_STEPS[idx].type !== "done") {
                visited.push(WIZARD_STEPS[idx].id);
                idx = getNextStep(idx, answers);
                if (idx === -1) break;
            }

            expect(visited).not.toContain("workload");
            expect(visited).not.toContain("gpu-preference");
            expect(visited).not.toContain("browse-gpus");
            expect(visited).not.toContain("gpu-pick");
            expect(visited).toContain("docker-check");
            expect(visited).toContain("gpu-detect");
            expect(visited).toContain("pricing");
            // confirm-setup only shows for "both" mode; provider-only confirms at provider-summary
            expect(visited).not.toContain("confirm-setup");
            expect(visited).toContain("provider-summary");
            // custom-rate skipped because pricing != "custom"
            expect(visited).not.toContain("custom-rate");
        });
    });

    describe("getNextStep — both mode", () => {
        it("includes both provider and renter steps", () => {
            const visited: string[] = [];
            let idx = 0;
            const answers: Record<string, string> = {
                mode: "both",
                pricing: "custom",
                "custom-rate": "0.50",
                "device-auth": "authorized",
                "api-key": "test-token",
                workload: "training",
                "gpu-preference": "cheapest",
                "browse-gpus": "done",
                "gpu-pick": "host-1",
                "image-pick": "nvcr.io/nvidia/pytorch:24.01-py3",
                "confirm-launch": "yes",
                "wallet-check": "passed",
                "_wallet_insufficient": "false",
                "launch-instance": "passed",
            };

            while (idx < WIZARD_STEPS.length && WIZARD_STEPS[idx].type !== "done") {
                visited.push(WIZARD_STEPS[idx].id);
                idx = getNextStep(idx, answers);
                if (idx === -1) break;
            }

            expect(visited).toContain("docker-check");
            expect(visited).toContain("gpu-detect");
            expect(visited).toContain("pricing");
            expect(visited).toContain("custom-rate"); // because pricing=custom
            expect(visited).toContain("workload");
            expect(visited).toContain("gpu-preference");
        });
    });

    describe("getNextStep — custom rate condition", () => {
        it("shows custom-rate only when pricing is custom", () => {
            const withCustom = { mode: "provide", pricing: "custom" };
            const withRecommended = { mode: "provide", pricing: "recommended" };

            // Find pricing step index
            const pricingIdx = WIZARD_STEPS.findIndex((s) => s.id === "pricing");

            const nextCustom = getNextStep(pricingIdx, withCustom);
            expect(WIZARD_STEPS[nextCustom].id).toBe("custom-rate");

            const nextRec = getNextStep(pricingIdx, withRecommended);
            expect(WIZARD_STEPS[nextRec].id).not.toBe("custom-rate");
        });
    });

    describe("validation", () => {
        it("custom-rate step has validate function", () => {
            const step = WIZARD_STEPS.find((s) => s.id === "custom-rate");
            expect(step?.validate).toBeDefined();
            expect(step!.validate!("0.50")).toBeNull();
            expect(step!.validate!("abc")).not.toBeNull();
            expect(step!.validate!("")).not.toBeNull();
            expect(step!.validate!("-1")).not.toBeNull();
        });
    });

    describe("payment gate condition", () => {
        it("payment-gate shows only when wallet is insufficient", () => {
            const walletStep = WIZARD_STEPS.find((s) => s.id === "wallet-check");
            expect(walletStep).toBeDefined();

            const paymentStep = WIZARD_STEPS.find((s) => s.id === "payment-gate");
            expect(paymentStep).toBeDefined();
            expect(paymentStep!.condition).toBeDefined();
            expect(paymentStep!.condition!({ mode: "rent", "_wallet_insufficient": "true" })).toBe(true);
            expect(paymentStep!.condition!({ mode: "rent", "_wallet_insufficient": "false" })).toBe(false);
        });
    });
});
