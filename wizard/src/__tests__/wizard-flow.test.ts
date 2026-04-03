// Tests for wizard-flow.ts — step definitions, conditions, navigation

import { describe, it, expect } from "vitest";
import { WIZARD_STEPS, getNextStep } from "../wizard-flow.js";

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

    describe("getNextStep — renter flow", () => {
        const renterAnswers = { mode: "rent" };

        it("skips docker-check for renters", () => {
            // From mode (index 0), next should skip docker-check
            const next = getNextStep(0, renterAnswers);
            expect(WIZARD_STEPS[next].id).toBe("api-url");
        });

        it("skips provider-only steps", () => {
            // Walk entire renter flow
            const visited: string[] = [];
            let idx = 0;
            const answers: Record<string, string> = { mode: "rent" };

            while (idx < WIZARD_STEPS.length && WIZARD_STEPS[idx].type !== "done") {
                visited.push(WIZARD_STEPS[idx].id);
                idx = getNextStep(idx, answers);
                if (idx === -1) break;
            }

            // Should NOT include provider-only steps
            expect(visited).not.toContain("docker-check");
            expect(visited).not.toContain("gpu-detect");
            expect(visited).not.toContain("pricing");
            expect(visited).not.toContain("custom-rate");

            // Should include renter steps
            expect(visited).toContain("workload");
            expect(visited).toContain("gpu-preference");
        });
    });

    describe("getNextStep — provider flow", () => {
        const providerAnswers = { mode: "provide" };

        it("includes docker-check for providers", () => {
            const next = getNextStep(0, providerAnswers);
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
            expect(visited).toContain("docker-check");
            expect(visited).toContain("gpu-detect");
            expect(visited).toContain("pricing");
            // custom-rate skipped because pricing != "custom"
            expect(visited).not.toContain("custom-rate");
        });
    });

    describe("getNextStep — both mode", () => {
        it("includes both provider and renter steps", () => {
            const visited: string[] = [];
            let idx = 0;
            const answers: Record<string, string> = { mode: "both", pricing: "custom" };

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
});
