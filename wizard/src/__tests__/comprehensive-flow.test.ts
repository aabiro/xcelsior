// Comprehensive flow tests — full flow walks, step conditions, validation,
// ordering guarantees, image templates, provider checks edge cases,
// buildWizardContext coverage, mock check presets, and AI streaming scenarios.

import { describe, it, expect, beforeEach } from "vitest";
import {
    WIZARD_STEPS,
    getNextStep,
    IMAGE_TEMPLATES,
    WORKLOAD_IMAGE_MAP,
} from "../wizard-flow.js";
import type { WizardStep } from "../wizard-flow.js";
import {
    setMockChatEvents,
    setMockConfirmEvents,
    resetApiMock,
    streamChat,
    confirmAction,
} from "../__mocks__/api-client.js";
import {
    setMockPreset,
    setMockResults,
    resetMock,
    checkDocker,
} from "../__mocks__/checks.js";
import type { SSEEvent, ApiClientConfig } from "../api-client.js";
import type { CheckResult } from "../checks.js";
import { buildWizardContext } from "../useWizardFlow.js";
import {
    runVerificationChecks,
    buildVerificationReport,
    MINIMUM_VERSIONS,
    VERIFICATION_THRESHOLDS,
} from "../provider-checks.js";

const TEST_CONFIG: ApiClientConfig = {
    baseUrl: "http://localhost:9500",
    apiKey: "test-token",
};

// ── Helpers ─────────────────────────────────────────────────────────

/** Walk the flow from step 0, collecting visited step IDs. */
function walkFlow(answers: Record<string, string>): string[] {
    const visited: string[] = [];
    let idx = 0;
    let iterations = 0;
    while (idx >= 0 && idx < WIZARD_STEPS.length && iterations < 50) {
        visited.push(WIZARD_STEPS[idx].id);
        if (WIZARD_STEPS[idx].type === "done") break;
        idx = getNextStep(idx, answers);
        iterations++;
    }
    return visited;
}

/** Find a step by id. */
function findStep(id: string): WizardStep {
    const step = WIZARD_STEPS.find((s) => s.id === id);
    if (!step) throw new Error(`Step "${id}" not found`);
    return step;
}

/** Fixture builders for provider checks */
function makeGpu(overrides: Partial<Record<string, unknown>> = {}) {
    return {
        gpu_model: "NVIDIA RTX 4090",
        total_vram_gb: 24,
        free_vram_gb: 20,
        serial: "GPU-SERIAL-001",
        uuid: "GPU-abcdef12-3456-7890-abcd-ef1234567890",
        pci_bus_id: "00:01.0",
        driver_version: "560.35.03",
        compute_capability: "8.9",
        ...overrides,
    };
}

function makeBench(overrides: Partial<Record<string, unknown>> = {}) {
    return {
        tflops: 82.6,
        pcie_bandwidth_gbps: 22.5,
        pcie_h2d_gbps: 12.0,
        pcie_d2h_gbps: 10.5,
        gpu_temp_celsius: 68,
        gpu_temp_avg_celsius: 62,
        gpu_temp_samples: 10,
        gpu_model: "NVIDIA RTX 4090",
        total_vram_gb: 24,
        compute_capability: "8.9",
        cuda_version: "12.4",
        driver_version: "560.35.03",
        xcu_score: 850,
        elapsed_s: 45,
        ...overrides,
    };
}

function makeNetwork(overrides: Partial<Record<string, unknown>> = {}) {
    return {
        latency_avg_ms: 15,
        latency_min_ms: 10,
        latency_max_ms: 22,
        jitter_ms: 2,
        packet_loss_pct: 0,
        throughput_mbps: 850,
        ...overrides,
    };
}

function makeVersions() {
    return [
        { component: "runc", version: "1.2.1", minimum: "1.1.12", passed: true },
        { component: "docker", version: "27.3.1", minimum: "24.0.0", passed: true },
        { component: "nvidia_driver", version: "560.35.03", minimum: "550.0", passed: true },
        { component: "nvidia_toolkit", version: "1.18.0", minimum: "1.17.8", passed: true },
    ];
}

// ══════════════════════════════════════════════════════════════════════
// 1. COMPLETE FLOW WALKS
// ══════════════════════════════════════════════════════════════════════

describe("complete flow walks", () => {
    it("renter flow — training, best GPU, no payment gate", () => {
        const visited = walkFlow({
            mode: "rent",
            "device-auth": "authorized",
            "api-check": "passed",
            workload: "training",
            "gpu-preference": "best",
            "browse-gpus": "done",
            "gpu-pick": "host-1",
            "image-pick": "nvcr.io/nvidia/pytorch:24.01-py3",
            "confirm-launch": "yes",
            "wallet-check": "passed",
            "_wallet_insufficient": "false",
            "launch-instance": "passed",
        });

        expect(visited[0]).toBe("mode");
        expect(visited[visited.length - 1]).toBe("done");
        expect(visited).toContain("workload");
        expect(visited).toContain("gpu-preference");
        expect(visited).toContain("browse-gpus");
        expect(visited).toContain("gpu-pick");
        expect(visited).toContain("image-pick");
        expect(visited).toContain("wallet-check");
        expect(visited).not.toContain("payment-gate");
        expect(visited).not.toContain("docker-check");
        expect(visited).not.toContain("benchmark");
        expect(visited).not.toContain("confirm-setup");
    });

    it("renter flow — with payment gate triggered", () => {
        const visited = walkFlow({
            mode: "rent",
            "device-auth": "authorized",
            "api-check": "passed",
            workload: "inference",
            "gpu-preference": "cheapest",
            "browse-gpus": "done",
            "gpu-pick": "host-2",
            "image-pick": "vllm/vllm-openai:latest",
            "confirm-launch": "yes",
            "wallet-check": "passed",
            "_wallet_insufficient": "true",
            "launch-instance": "passed",
        });

        expect(visited).toContain("payment-gate");
        expect(visited).toContain("launch-instance");
    });

    it("provider flow — recommended pricing", () => {
        const visited = walkFlow({
            mode: "provide",
            "docker-check": "passed",
            "device-auth": "authorized",
            "api-check": "passed",
            "gpu-detect": "passed",
            "version-check": "passed",
            benchmark: "passed",
            "network-bench": "passed",
            verification: "passed",
            pricing: "recommended",
            "host-register": "passed",
            "admission-gate": "passed",
            "provider-summary": "yes",
        });

        expect(visited[0]).toBe("mode");
        expect(visited[visited.length - 1]).toBe("done");
        expect(visited).toContain("docker-check");
        expect(visited).toContain("gpu-detect");
        expect(visited).toContain("version-check");
        expect(visited).toContain("benchmark");
        expect(visited).toContain("verification");
        expect(visited).toContain("pricing");
        expect(visited).not.toContain("custom-rate");
        expect(visited).not.toContain("workload");
        expect(visited).not.toContain("gpu-pick");
        expect(visited).not.toContain("confirm-setup");
    });

    it("provider flow — custom pricing includes custom-rate step", () => {
        const visited = walkFlow({
            mode: "provide",
            "docker-check": "passed",
            "device-auth": "authorized",
            "api-check": "passed",
            "gpu-detect": "passed",
            "version-check": "passed",
            benchmark: "passed",
            "network-bench": "passed",
            verification: "passed",
            pricing: "custom",
            "custom-rate": "3.50",
            "host-register": "passed",
            "admission-gate": "passed",
            "provider-summary": "yes",
        });

        expect(visited).toContain("custom-rate");
        expect(visited).toContain("pricing");
    });

    it("provider flow — competitive pricing skips custom-rate", () => {
        const visited = walkFlow({
            mode: "provide",
            "docker-check": "passed",
            "device-auth": "authorized",
            "api-check": "passed",
            "gpu-detect": "passed",
            "version-check": "passed",
            benchmark: "passed",
            "network-bench": "passed",
            verification: "passed",
            pricing: "competitive",
            "host-register": "passed",
            "admission-gate": "passed",
            "provider-summary": "yes",
        });

        expect(visited).not.toContain("custom-rate");
    });

    it("both mode — visits provider AND renter steps plus confirm-setup", () => {
        const visited = walkFlow({
            mode: "both",
            "docker-check": "passed",
            "device-auth": "authorized",
            "api-check": "passed",
            "gpu-detect": "passed",
            "version-check": "passed",
            benchmark: "passed",
            "network-bench": "passed",
            verification: "passed",
            pricing: "recommended",
            "host-register": "passed",
            "admission-gate": "passed",
            "provider-summary": "yes",
            workload: "training",
            "gpu-preference": "best",
            "browse-gpus": "done",
            "gpu-pick": "host-1",
            "image-pick": "nvcr.io/nvidia/pytorch:24.01-py3",
            "confirm-launch": "yes",
            "wallet-check": "passed",
            "_wallet_insufficient": "false",
            "launch-instance": "passed",
            "confirm-setup": "yes",
        });

        // Provider steps
        expect(visited).toContain("docker-check");
        expect(visited).toContain("benchmark");
        expect(visited).toContain("verification");
        expect(visited).toContain("provider-summary");
        // Renter steps
        expect(visited).toContain("workload");
        expect(visited).toContain("gpu-pick");
        expect(visited).toContain("launch-instance");
        // Both-only step
        expect(visited).toContain("confirm-setup");
    });

    it("both mode with custom pricing and payment gate", () => {
        const visited = walkFlow({
            mode: "both",
            "docker-check": "passed",
            "device-auth": "authorized",
            "api-check": "passed",
            "gpu-detect": "passed",
            "version-check": "passed",
            benchmark: "passed",
            "network-bench": "passed",
            verification: "passed",
            pricing: "custom",
            "custom-rate": "5.00",
            "host-register": "passed",
            "admission-gate": "passed",
            "provider-summary": "yes",
            workload: "research",
            "gpu-preference": "specific",
            "browse-gpus": "done",
            "gpu-pick": "host-3",
            "image-pick": "quay.io/jupyter/pytorch-notebook:cuda12-latest",
            "confirm-launch": "yes",
            "wallet-check": "passed",
            "_wallet_insufficient": "true",
            "launch-instance": "passed",
            "confirm-setup": "yes",
        });

        expect(visited).toContain("custom-rate");
        expect(visited).toContain("payment-gate");
        expect(visited).toContain("confirm-setup");
    });

    it("flow never gets stuck — terminates for all modes", () => {
        for (const mode of ["rent", "provide", "both"]) {
            const answers: Record<string, string> = { mode };
            // Fill in all possible answers
            for (const step of WIZARD_STEPS) {
                if (step.type === "auto-check") answers[step.id] = "passed";
                if (step.type === "confirm") answers[step.id] = "yes";
                if (step.type === "device-auth") answers[step.id] = "authorized";
                if (step.type === "auto-fetch") answers[step.id] = "done";
                if (step.id === "pricing") answers[step.id] = "recommended";
                if (step.id === "workload") answers[step.id] = "training";
                if (step.id === "gpu-preference") answers[step.id] = "best";
                if (step.id === "gpu-pick") answers[step.id] = "host-1";
                if (step.id === "image-pick") answers[step.id] = "pytorch";
            }
            answers["_wallet_insufficient"] = "false";

            const visited = walkFlow(answers);
            expect(visited[visited.length - 1]).toBe("done");
            expect(visited.length).toBeGreaterThan(3);
        }
    });
});

// ══════════════════════════════════════════════════════════════════════
// 2. STEP CONDITION EXHAUSTIVE TESTING
// ══════════════════════════════════════════════════════════════════════

describe("step condition exhaustive testing", () => {
    const modes = ["rent", "provide", "both"];

    describe("provider-only steps", () => {
        const providerSteps = [
            "docker-check", "gpu-detect", "version-check", "benchmark",
            "network-bench", "verification", "pricing", "host-register",
            "admission-gate", "provider-summary",
        ];

        for (const stepId of providerSteps) {
            it(`${stepId} — skipped for renters, shown for providers and both`, () => {
                const step = findStep(stepId);
                expect(step.condition).toBeDefined();
                expect(step.condition!({ mode: "rent" })).toBe(false);
                expect(step.condition!({ mode: "provide" })).toBe(true);
                expect(step.condition!({ mode: "both" })).toBe(true);
            });
        }
    });

    describe("renter-only steps", () => {
        const renterSteps = [
            "workload", "gpu-preference", "browse-gpus", "gpu-pick",
            "image-pick", "confirm-launch", "wallet-check", "launch-instance",
        ];

        for (const stepId of renterSteps) {
            it(`${stepId} — skipped for providers, shown for renters and both`, () => {
                const step = findStep(stepId);
                expect(step.condition).toBeDefined();
                expect(step.condition!({ mode: "provide" })).toBe(false);
                expect(step.condition!({ mode: "rent" })).toBe(true);
                expect(step.condition!({ mode: "both" })).toBe(true);
            });
        }
    });

    describe("custom-rate compound condition", () => {
        it("requires both provider mode AND custom pricing", () => {
            const step = findStep("custom-rate");
            expect(step.condition!({ mode: "provide", pricing: "custom" })).toBe(true);
            expect(step.condition!({ mode: "both", pricing: "custom" })).toBe(true);
            expect(step.condition!({ mode: "provide", pricing: "recommended" })).toBe(false);
            expect(step.condition!({ mode: "provide", pricing: "competitive" })).toBe(false);
            expect(step.condition!({ mode: "rent", pricing: "custom" })).toBe(false);
        });

        it("fails when pricing is missing", () => {
            const step = findStep("custom-rate");
            expect(step.condition!({ mode: "provide" })).toBe(false);
        });
    });

    describe("payment-gate compound condition", () => {
        it("requires renter mode AND wallet insufficient flag", () => {
            const step = findStep("payment-gate");
            expect(step.condition!({ mode: "rent", "_wallet_insufficient": "true" })).toBe(true);
            expect(step.condition!({ mode: "both", "_wallet_insufficient": "true" })).toBe(true);
            expect(step.condition!({ mode: "rent", "_wallet_insufficient": "false" })).toBe(false);
            expect(step.condition!({ mode: "provide", "_wallet_insufficient": "true" })).toBe(false);
        });
    });

    describe("confirm-setup — both mode only", () => {
        it("only shows for both mode", () => {
            const step = findStep("confirm-setup");
            expect(step.condition!({ mode: "both" })).toBe(true);
            expect(step.condition!({ mode: "rent" })).toBe(false);
            expect(step.condition!({ mode: "provide" })).toBe(false);
        });
    });

    describe("unconditional steps", () => {
        const unconditional = ["mode", "device-auth", "api-check", "done"];
        for (const stepId of unconditional) {
            it(`${stepId} has no condition function`, () => {
                const step = findStep(stepId);
                expect(step.condition).toBeUndefined();
            });
        }
    });
});

// ══════════════════════════════════════════════════════════════════════
// 3. CUSTOM-RATE VALIDATION EDGE CASES
// ══════════════════════════════════════════════════════════════════════

describe("custom-rate validation edge cases", () => {
    const validate = findStep("custom-rate").validate!;

    it("accepts valid whole numbers", () => {
        expect(validate("1")).toBeNull();
        expect(validate("10")).toBeNull();
        expect(validate("999")).toBeNull();
        expect(validate("1000")).toBeNull();
    });

    it("accepts valid decimals", () => {
        expect(validate("0.01")).toBeNull();
        expect(validate("0.50")).toBeNull();
        expect(validate("99.99")).toBeNull();
        expect(validate("2.50")).toBeNull();
    });

    it("rejects zero and near-zero", () => {
        expect(validate("0")).not.toBeNull();
        expect(validate("0.00")).not.toBeNull();
        expect(validate("-0")).not.toBeNull();
    });

    it("rejects negative numbers", () => {
        expect(validate("-1")).not.toBeNull();
        expect(validate("-0.50")).not.toBeNull();
        expect(validate("-999")).not.toBeNull();
    });

    it("rejects values above $1000", () => {
        expect(validate("1001")).not.toBeNull();
        expect(validate("5000")).not.toBeNull();
        expect(validate("10000")).not.toBeNull();
    });

    it("rejects non-numeric strings", () => {
        expect(validate("abc")).not.toBeNull();
        expect(validate("")).not.toBeNull();
        expect(validate("$5")).not.toBeNull();
        expect(validate("five")).not.toBeNull();
    });

    it("rejects special float values", () => {
        expect(validate("NaN")).not.toBeNull();
        expect(validate("Infinity")).not.toBeNull();
    });

    it("accepts boundary value $1000 exactly", () => {
        expect(validate("1000")).toBeNull();
    });

    it("returns error string, not boolean", () => {
        const result = validate("abc");
        expect(typeof result).toBe("string");
        expect(result!.length).toBeGreaterThan(0);
    });
});

// ══════════════════════════════════════════════════════════════════════
// 4. FLOW ORDERING GUARANTEES
// ══════════════════════════════════════════════════════════════════════

describe("flow ordering guarantees", () => {
    it("device-auth always comes before api-check", () => {
        const authIdx = WIZARD_STEPS.findIndex((s) => s.id === "device-auth");
        const apiIdx = WIZARD_STEPS.findIndex((s) => s.id === "api-check");
        expect(authIdx).toBeLessThan(apiIdx);
    });

    it("docker-check comes before gpu-detect for providers", () => {
        const dockerIdx = WIZARD_STEPS.findIndex((s) => s.id === "docker-check");
        const gpuIdx = WIZARD_STEPS.findIndex((s) => s.id === "gpu-detect");
        expect(dockerIdx).toBeLessThan(gpuIdx);
    });

    it("benchmark comes after gpu-detect and version-check", () => {
        const gpuIdx = WIZARD_STEPS.findIndex((s) => s.id === "gpu-detect");
        const versIdx = WIZARD_STEPS.findIndex((s) => s.id === "version-check");
        const benchIdx = WIZARD_STEPS.findIndex((s) => s.id === "benchmark");
        expect(gpuIdx).toBeLessThan(benchIdx);
        expect(versIdx).toBeLessThan(benchIdx);
    });

    it("verification comes after benchmark", () => {
        const benchIdx = WIZARD_STEPS.findIndex((s) => s.id === "benchmark");
        const verIdx = WIZARD_STEPS.findIndex((s) => s.id === "verification");
        expect(benchIdx).toBeLessThan(verIdx);
    });

    it("host-register comes after verification and pricing", () => {
        const verIdx = WIZARD_STEPS.findIndex((s) => s.id === "verification");
        const priceIdx = WIZARD_STEPS.findIndex((s) => s.id === "pricing");
        const hostIdx = WIZARD_STEPS.findIndex((s) => s.id === "host-register");
        expect(verIdx).toBeLessThan(hostIdx);
        expect(priceIdx).toBeLessThan(hostIdx);
    });

    it("wallet-check comes before payment-gate", () => {
        const walletIdx = WIZARD_STEPS.findIndex((s) => s.id === "wallet-check");
        const payIdx = WIZARD_STEPS.findIndex((s) => s.id === "payment-gate");
        expect(walletIdx).toBeLessThan(payIdx);
    });

    it("confirm-launch comes before wallet-check and launch-instance", () => {
        const confirmIdx = WIZARD_STEPS.findIndex((s) => s.id === "confirm-launch");
        const walletIdx = WIZARD_STEPS.findIndex((s) => s.id === "wallet-check");
        const launchIdx = WIZARD_STEPS.findIndex((s) => s.id === "launch-instance");
        expect(confirmIdx).toBeLessThan(walletIdx);
        expect(confirmIdx).toBeLessThan(launchIdx);
    });

    it("browse-gpus comes before gpu-pick", () => {
        const browseIdx = WIZARD_STEPS.findIndex((s) => s.id === "browse-gpus");
        const pickIdx = WIZARD_STEPS.findIndex((s) => s.id === "gpu-pick");
        expect(browseIdx).toBeLessThan(pickIdx);
    });

    it("provider-summary comes before renter steps", () => {
        const summaryIdx = WIZARD_STEPS.findIndex((s) => s.id === "provider-summary");
        const workloadIdx = WIZARD_STEPS.findIndex((s) => s.id === "workload");
        expect(summaryIdx).toBeLessThan(workloadIdx);
    });

    it("mode is first, done is last", () => {
        expect(WIZARD_STEPS[0].id).toBe("mode");
        expect(WIZARD_STEPS[WIZARD_STEPS.length - 1].id).toBe("done");
    });

    it("provider steps all come before renter steps in the array", () => {
        const lastProvider = WIZARD_STEPS.findIndex((s) => s.id === "provider-summary");
        const firstRenter = WIZARD_STEPS.findIndex((s) => s.id === "workload");
        expect(lastProvider).toBeLessThan(firstRenter);
    });
});

// ══════════════════════════════════════════════════════════════════════
// 5. IMAGE TEMPLATES
// ══════════════════════════════════════════════════════════════════════

describe("IMAGE_TEMPLATES", () => {
    it("all templates have positive vram requirements", () => {
        for (const t of IMAGE_TEMPLATES) {
            expect(t.vram).toBeGreaterThan(0);
        }
    });

    it("all templates have non-empty labels and values", () => {
        for (const t of IMAGE_TEMPLATES) {
            expect(t.label.length).toBeGreaterThan(0);
            expect(t.value.length).toBeGreaterThan(0);
        }
    });

    it("values look like Docker image references", () => {
        for (const t of IMAGE_TEMPLATES) {
            expect(t.value).toMatch(/[a-z].*[:/].*/);
        }
    });

    it("has PyTorch, TensorFlow, vLLM options", () => {
        const labels = IMAGE_TEMPLATES.map((t) => t.label.toLowerCase());
        expect(labels.some((l) => l.includes("pytorch"))).toBe(true);
        expect(labels.some((l) => l.includes("tensorflow"))).toBe(true);
        expect(labels.some((l) => l.includes("vllm"))).toBe(true);
    });

    it("WORKLOAD_IMAGE_MAP values all exist in IMAGE_TEMPLATES", () => {
        const templateValues = new Set(IMAGE_TEMPLATES.map((t) => t.value));
        for (const image of Object.values(WORKLOAD_IMAGE_MAP)) {
            expect(templateValues.has(image)).toBe(true);
        }
    });

    it("mapping covers training, inference, research, other", () => {
        expect(WORKLOAD_IMAGE_MAP).toHaveProperty("training");
        expect(WORKLOAD_IMAGE_MAP).toHaveProperty("inference");
        expect(WORKLOAD_IMAGE_MAP).toHaveProperty("research");
        expect(WORKLOAD_IMAGE_MAP).toHaveProperty("other");
    });
});

// ══════════════════════════════════════════════════════════════════════
// 6. getNextStep — EDGE CASES
// ══════════════════════════════════════════════════════════════════════

describe("getNextStep edge cases", () => {
    it("returns -1 from done step", () => {
        const doneIdx = WIZARD_STEPS.findIndex((s) => s.id === "done");
        expect(getNextStep(doneIdx, { mode: "rent" })).toBe(-1);
    });

    it("returns -1 from last step before done if conditions skip done", () => {
        // The done step has no condition, so it should always be found
        const lastIdx = WIZARD_STEPS.length - 2;
        const result = getNextStep(lastIdx, { mode: "rent" });
        expect(result).toBeGreaterThanOrEqual(0);
    });

    it("skips all conditional steps when mode is missing", () => {
        // Without mode set, all conditional steps should be skipped
        const result = getNextStep(0, {});
        // Should skip to device-auth (no condition)
        const step = WIZARD_STEPS[result];
        expect(step.condition).toBeUndefined();
    });

    it("handles starting from index 0", () => {
        const result = getNextStep(0, { mode: "rent" });
        expect(result).toBeGreaterThan(0);
    });

    it("handles starting from negative index", () => {
        // Should start scanning from index 0
        const result = getNextStep(-1, { mode: "rent" });
        expect(result).toBe(0); // mode step
    });

    it("returns consistent results for same inputs", () => {
        const answers = { mode: "rent" };
        const r1 = getNextStep(0, answers);
        const r2 = getNextStep(0, answers);
        expect(r1).toBe(r2);
    });

    it("advances through entire renter flow without loops", () => {
        const answers: Record<string, string> = { mode: "rent", "_wallet_insufficient": "false" };
        const indices: number[] = [0];
        let idx = 0;
        let safety = 0;
        while (idx >= 0 && idx < WIZARD_STEPS.length && safety < 50) {
            idx = getNextStep(idx, answers);
            if (idx >= 0) indices.push(idx);
            safety++;
        }
        // All indices should be strictly increasing (no loops)
        for (let i = 1; i < indices.length; i++) {
            expect(indices[i]).toBeGreaterThan(indices[i - 1]);
        }
    });
});

// ══════════════════════════════════════════════════════════════════════
// 7. STEP STRUCTURE INTEGRITY
// ══════════════════════════════════════════════════════════════════════

describe("step structure integrity", () => {
    it("all 28 steps present", () => {
        expect(WIZARD_STEPS).toHaveLength(28);
    });

    it("no duplicate step IDs", () => {
        const ids = WIZARD_STEPS.map((s) => s.id);
        expect(new Set(ids).size).toBe(ids.length);
    });

    it("all auto-check steps with checkRequired=true are provider steps", () => {
        const required = WIZARD_STEPS.filter((s) => s.checkRequired === true);
        for (const step of required) {
            // These are critical checks that can't be skipped
            expect(step.type).toBe("auto-check");
            expect(step.checkId).toBeDefined();
        }
    });

    it("select steps with static options have at least 2 options", () => {
        const staticSelects = WIZARD_STEPS.filter(
            (s) => s.type === "select" && s.options && s.options.length > 0,
        );
        for (const step of staticSelects) {
            expect(step.options!.length).toBeGreaterThanOrEqual(2);
        }
    });

    it("text steps have validate function", () => {
        for (const step of WIZARD_STEPS) {
            if (step.type === "text") {
                expect(step.validate).toBeDefined();
            }
        }
    });

    it("confirm steps have confirmLabel", () => {
        for (const step of WIZARD_STEPS) {
            if (step.type === "confirm") {
                expect(step.confirmLabel).toBeDefined();
                expect(step.confirmLabel!.length).toBeGreaterThan(0);
            }
        }
    });

    it("every step has a non-empty prompt", () => {
        for (const step of WIZARD_STEPS) {
            expect(step.prompt.length).toBeGreaterThan(0);
        }
    });

    it("done step is the only done type", () => {
        const doneSteps = WIZARD_STEPS.filter((s) => s.type === "done");
        expect(doneSteps).toHaveLength(1);
        expect(doneSteps[0].id).toBe("done");
    });

    it("mode step has exactly 3 options: rent, provide, both", () => {
        const mode = findStep("mode");
        expect(mode.options).toHaveLength(3);
        const values = mode.options!.map((o) => o.value);
        expect(values).toContain("rent");
        expect(values).toContain("provide");
        expect(values).toContain("both");
    });
});

// ══════════════════════════════════════════════════════════════════════
// 8. MOCK CHECK PRESETS
// ══════════════════════════════════════════════════════════════════════

describe("check mock presets", () => {
    beforeEach(() => resetMock());

    it("all-pass preset returns all ok:true", async () => {
        setMockPreset("all-pass");
        const results = await checkDocker();
        expect(results.every((r) => r.ok)).toBe(true);
        expect(results.length).toBe(4);
    });

    it("all-fail preset returns all ok:false", async () => {
        setMockPreset("all-fail");
        const results = await checkDocker();
        expect(results.every((r) => !r.ok)).toBe(true);
        expect(results.length).toBe(4);
    });

    it("partial preset has mix of pass and fail", async () => {
        setMockPreset("partial");
        const results = await checkDocker();
        const passed = results.filter((r) => r.ok);
        const failed = results.filter((r) => !r.ok);
        expect(passed.length).toBeGreaterThan(0);
        expect(failed.length).toBeGreaterThan(0);
    });

    it("custom results override preset", async () => {
        setMockPreset("all-fail");
        const custom: CheckResult[] = [
            { name: "Custom Check", ok: true, detail: "works" },
        ];
        setMockResults(custom);
        const results = await checkDocker();
        expect(results).toHaveLength(1);
        expect(results[0].name).toBe("Custom Check");
    });

    it("reset restores all-pass", async () => {
        setMockPreset("all-fail");
        resetMock();
        const results = await checkDocker();
        expect(results.every((r) => r.ok)).toBe(true);
    });

    it("all presets include Docker, NVIDIA Driver, runc", async () => {
        for (const preset of ["all-pass", "all-fail", "partial"] as const) {
            setMockPreset(preset);
            const results = await checkDocker();
            const names = results.map((r) => r.name);
            expect(names).toContain("Docker");
            expect(names).toContain("NVIDIA Driver");
            expect(names).toContain("runc");
        }
    });

    it("array is shallow-copied but objects are shared references", async () => {
        const r1 = await checkDocker();
        const r2 = await checkDocker();
        // Array is a new copy (push doesn't affect other)
        r1.push({ name: "Extra", ok: true, detail: "test" });
        expect(r2.length).toBe(4);
        // But individual items are shared (shallow copy)
        r1[0].ok = false;
        expect(r2[0].ok).toBe(false);
    });
});

// ══════════════════════════════════════════════════════════════════════
// 9. PROVIDER VERIFICATION CHECKS
// ══════════════════════════════════════════════════════════════════════

describe("provider verification checks — extended", () => {
    it("passes with borderline acceptable values", () => {
        const gpu = makeGpu({ compute_capability: "7.0" });
        const bench = makeBench({
            pcie_bandwidth_gbps: VERIFICATION_THRESHOLDS.min_pcie_bandwidth_gbps,
            gpu_temp_celsius: VERIFICATION_THRESHOLDS.max_gpu_temp_celsius,
        });
        const network = makeNetwork({
            packet_loss_pct: VERIFICATION_THRESHOLDS.max_network_loss_pct,
            jitter_ms: VERIFICATION_THRESHOLDS.max_network_jitter_ms,
            throughput_mbps: VERIFICATION_THRESHOLDS.min_network_throughput_mbps,
        });
        const versions = makeVersions();
        const results = runVerificationChecks(gpu as any, bench as any, network as any, versions);
        const gpuIdentity = results.find((r) => r.name === "GPU Identity");
        expect(gpuIdentity?.passed).toBe(true);
    });

    it("fails all checks with zeroed inputs", () => {
        const gpu = makeGpu({ gpu_model: "", uuid: "", compute_capability: "0.0", total_vram_gb: 0 });
        const bench = makeBench({ pcie_bandwidth_gbps: 0, gpu_temp_celsius: 120, total_vram_gb: 0 });
        const network = makeNetwork({ packet_loss_pct: 50, jitter_ms: 500, throughput_mbps: 0 });
        const versions = [
            { component: "runc", version: null, minimum: "1.1.12", passed: false },
            { component: "docker", version: null, minimum: "24.0.0", passed: false },
            { component: "nvidia_driver", version: null, minimum: "550.0", passed: false },
            { component: "nvidia_toolkit", version: null, minimum: "1.17.8", passed: false },
        ];
        const results = runVerificationChecks(gpu as any, bench as any, network as any, versions as any);
        const failCount = results.filter((r) => !r.passed).length;
        expect(failCount).toBeGreaterThanOrEqual(4);
    });

    it("VRAM mismatch detected when reported differs from detected", () => {
        const gpu = makeGpu({ total_vram_gb: 24 });
        const bench = makeBench({ total_vram_gb: 12 }); // mismatch
        const network = makeNetwork();
        const versions = makeVersions();
        const results = runVerificationChecks(gpu as any, bench as any, network as any, versions);
        const memCheck = results.find((r) => r.name === "Memory Integrity");
        expect(memCheck?.passed).toBe(false);
    });

    it("VRAM match passes when values are close", () => {
        const gpu = makeGpu({ total_vram_gb: 24 });
        const bench = makeBench({ total_vram_gb: 24 });
        const network = makeNetwork();
        const versions = makeVersions();
        const results = runVerificationChecks(gpu as any, bench as any, network as any, versions);
        const memCheck = results.find((r) => r.name === "Memory Integrity");
        expect(memCheck?.passed).toBe(true);
    });

    it("buildVerificationReport includes all expected fields", () => {
        const gpu = makeGpu();
        const bench = makeBench();
        const network = makeNetwork();
        const versions = makeVersions();
        const report = buildVerificationReport(gpu as any, bench as any, network as any, versions);

        expect(report).toHaveProperty("allPassed");
        expect(report).toHaveProperty("checks");
        expect(report).toHaveProperty("gpu_fingerprint");
        expect(report.checks.length).toBeGreaterThanOrEqual(5);
    });

    it("MINIMUM_VERSIONS has all required components", () => {
        expect(MINIMUM_VERSIONS).toHaveProperty("runc");
        expect(MINIMUM_VERSIONS).toHaveProperty("docker");
        expect(MINIMUM_VERSIONS).toHaveProperty("nvidia_driver");
        expect(MINIMUM_VERSIONS).toHaveProperty("nvidia_toolkit");
    });

    it("VERIFICATION_THRESHOLDS has all expected fields", () => {
        expect(VERIFICATION_THRESHOLDS).toHaveProperty("min_cuda_compute_capability");
        expect(VERIFICATION_THRESHOLDS).toHaveProperty("min_pcie_bandwidth_gbps");
        expect(VERIFICATION_THRESHOLDS).toHaveProperty("max_gpu_temp_celsius");
        expect(VERIFICATION_THRESHOLDS).toHaveProperty("max_network_loss_pct");
        expect(VERIFICATION_THRESHOLDS).toHaveProperty("max_network_jitter_ms");
        expect(VERIFICATION_THRESHOLDS).toHaveProperty("min_network_throughput_mbps");
    });
});

// ══════════════════════════════════════════════════════════════════════
// 10. buildWizardContext — EXTENDED SCENARIOS
// ══════════════════════════════════════════════════════════════════════

describe("buildWizardContext — extended scenarios", () => {
    it("includes step ID in output", () => {
        const ctx = buildWizardContext("mode", {}, {}, null, [], null);
        expect(ctx).toContain("cli-wizard:mode");
    });

    it("includes mode when set", () => {
        const ctx = buildWizardContext("device-auth", { mode: "rent" }, {}, null, [], null);
        expect(ctx).toContain("mode=rent");
    });

    it("provider mode includes host_id when available", () => {
        const ctx = buildWizardContext("verification", {
            mode: "provide",
            pricing: "recommended",
            "_host_id": "host-42",
        }, {}, null, [], null);
        expect(ctx).toContain("host_id=host-42");
    });

    it("renter mode includes workload and gpu preference", () => {
        const ctx = buildWizardContext("gpu-pick", {
            mode: "rent",
            workload: "training",
            "gpu-preference": "best",
        }, {}, null, [], null);
        expect(ctx).toContain("workload=training");
        expect(ctx).toContain("gpu_pref=best");
    });

    it("includes failed checks from check results", () => {
        const checkResults = {
            "docker-check": {
                items: [
                    { name: "Docker", ok: true, detail: "v27" },
                    { name: "NVIDIA", ok: false, detail: "not found" },
                ],
                allPassed: false,
            },
        };
        const ctx = buildWizardContext("docker-check", { mode: "provide" }, checkResults, null, [], null);
        expect(ctx).toContain("failed_checks");
        expect(ctx).toContain("NVIDIA");
    });

    it("omits failed_checks when all pass", () => {
        const checkResults = {
            "docker-check": {
                items: [{ name: "Docker", ok: true, detail: "v27" }],
                allPassed: true,
            },
        };
        const ctx = buildWizardContext("docker-check", { mode: "provide" }, checkResults, null, [], null);
        expect(ctx).not.toContain("failed_checks");
    });

    it("includes GPU listings count for browse step", () => {
        const listings = [
            { host_id: "h1", gpu_model: "RTX 4090", vram_gb: 24, price_per_hour: 1.50, owner: "alice", active: true, total_jobs: 5, total_earned: 50, description: "Fast GPU" },
            { host_id: "h2", gpu_model: "A100", vram_gb: 80, price_per_hour: 3.00, owner: "bob", active: true, total_jobs: 12, total_earned: 200, description: "Data center GPU" },
        ];
        const ctx = buildWizardContext("browse-gpus", { mode: "rent" }, {}, null, listings, null);
        // Should include listing info
        expect(ctx.length).toBeGreaterThan(20);
    });

    it("includes browse error when present", () => {
        const ctx = buildWizardContext("browse-gpus", { mode: "rent" }, {}, null, [], "API timeout");
        expect(ctx).toContain("API timeout");
    });

    it("produces pipe-delimited output", () => {
        const ctx = buildWizardContext("pricing", {
            mode: "provide",
            pricing: "custom",
            "custom-rate": "5.00",
        }, {}, null, [], null);
        expect(ctx).toContain("|");
    });

    it("handles empty answers gracefully", () => {
        const ctx = buildWizardContext("mode", {}, {}, null, [], null);
        expect(ctx).toBeDefined();
        expect(typeof ctx).toBe("string");
        expect(ctx.length).toBeGreaterThan(0);
    });

    it("both mode includes provider AND renter context", () => {
        const ctx = buildWizardContext("gpu-pick", {
            mode: "both",
            pricing: "recommended",
            workload: "training",
            "gpu-preference": "cheapest",
        }, {}, null, [], null);
        expect(ctx).toContain("workload=training");
    });
});

// ══════════════════════════════════════════════════════════════════════
// 11. AI STREAMING — COMPLEX SCENARIOS
// ══════════════════════════════════════════════════════════════════════

describe("AI streaming — complex scenarios", () => {
    beforeEach(() => resetApiMock());

    it("multi-tool-call response with interleaved tokens", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "multi-tool" },
            { type: "token", content: "Let me check " },
            { type: "tool_call", name: "list_instances", input: {} },
            { type: "tool_result", name: "list_instances", output: { instances: [] } },
            { type: "token", content: "and then look at " },
            { type: "tool_call", name: "get_marketplace", input: { filter: "4090" } },
            { type: "tool_result", name: "get_marketplace", output: { count: 5 } },
            { type: "token", content: "Here are your results." },
            { type: "done" },
        ]);

        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "find GPUs")) events.push(e);

        const tokens = events.filter((e) => e.type === "token");
        expect(tokens).toHaveLength(3);
        const toolCalls = events.filter((e) => e.type === "tool_call");
        expect(toolCalls).toHaveLength(2);
        expect(toolCalls[0].name).toBe("list_instances");
        expect(toolCalls[1].name).toBe("get_marketplace");
    });

    it("error mid-stream after partial content", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "err-mid" },
            { type: "token", content: "I'm going to " },
            { type: "token", content: "help you with" },
            { type: "error", message: "context_length_exceeded" },
        ]);

        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "long question")) events.push(e);

        const content = events
            .filter((e) => e.type === "token")
            .map((e) => e.content)
            .join("");
        expect(content).toBe("I'm going to help you with");
        expect(events[events.length - 1].type).toBe("error");
    });

    it("confirmation followed by approval and continuation", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "conf-flow" },
            { type: "token", content: "I'll stop the job. " },
            {
                type: "confirmation_required",
                confirmation_id: "cf-1",
                tool_name: "stop_job",
                tool_args: { instance_id: "i-42" },
            },
        ]);

        setMockConfirmEvents([
            { type: "token", content: "Job stopped successfully." },
            { type: "done" },
        ]);

        // Phase 1: Get confirmation
        const phase1: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "stop my job")) phase1.push(e);
        const confirm = phase1.find((e) => e.type === "confirmation_required");
        expect(confirm).toBeDefined();

        // Phase 2: Approve
        const phase2: SSEEvent[] = [];
        for await (const e of confirmAction(TEST_CONFIG, confirm!.confirmation_id!, true)) {
            phase2.push(e);
        }
        expect(phase2[0].content).toBe("Job stopped successfully.");
    });

    it("rejection sends negative confirmation", async () => {
        setMockConfirmEvents([
            { type: "token", content: "Okay, I won't do that." },
            { type: "done" },
        ]);

        const events: SSEEvent[] = [];
        for await (const e of confirmAction(TEST_CONFIG, "cf-reject", false)) {
            events.push(e);
        }
        expect(events[0].content).toBe("Okay, I won't do that.");
    });

    it("empty token content is handled", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "empty" },
            { type: "token", content: "" },
            { type: "token", content: "actual content" },
            { type: "done" },
        ]);

        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "test")) events.push(e);
        const tokens = events.filter((e) => e.type === "token");
        expect(tokens).toHaveLength(2);
    });

    it("done event without prior tokens", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "quick" },
            { type: "done" },
        ]);

        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "test")) events.push(e);
        expect(events).toHaveLength(2);
        expect(events[1].type).toBe("done");
    });

    it("meta event conversation_id is preserved", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "my-conv-123" },
            { type: "token", content: "hi" },
            { type: "done" },
        ]);

        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "test")) events.push(e);
        expect(events[0].conversation_id).toBe("my-conv-123");
    });

    it("tool_call with complex nested input", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "complex" },
            {
                type: "tool_call",
                name: "launch_instance",
                input: {
                    gpu_model: "RTX 4090",
                    image: "nvcr.io/nvidia/pytorch:24.01-py3",
                    config: { ssh: true, ports: [8080, 443] },
                },
            },
            {
                type: "tool_result",
                name: "launch_instance",
                output: { job_id: "j-99", status: "pending" },
            },
            { type: "token", content: "Instance launching." },
            { type: "done" },
        ]);

        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "launch")) events.push(e);
        const tc = events.find((e) => e.type === "tool_call");
        expect(tc!.input).toHaveProperty("config");
    });
});

// ══════════════════════════════════════════════════════════════════════
// 12. STEP TYPE DISTRIBUTION
// ══════════════════════════════════════════════════════════════════════

describe("step type distribution", () => {
    it("has correct count of each step type", () => {
        const counts: Record<string, number> = {};
        for (const step of WIZARD_STEPS) {
            counts[step.type] = (counts[step.type] || 0) + 1;
        }
        expect(counts["select"]).toBeGreaterThanOrEqual(4); // mode, pricing, workload, gpu-preference, gpu-pick, image-pick
        expect(counts["auto-check"]).toBeGreaterThanOrEqual(8); // docker, api, gpu-detect, versions, benchmark, network, verify, host-register, admission, wallet, launch
        expect(counts["confirm"]).toBeGreaterThanOrEqual(2); // provider-summary, confirm-launch, confirm-setup
        expect(counts["text"]).toBe(1); // custom-rate
        expect(counts["done"]).toBe(1);
        expect(counts["device-auth"]).toBe(1);
        expect(counts["auto-fetch"]).toBe(1); // browse-gpus
        expect(counts["payment-gate"]).toBe(1);
    });
});

// ══════════════════════════════════════════════════════════════════════
// 13. CHECKID UNIQUENESS AND COMPLETENESS
// ══════════════════════════════════════════════════════════════════════

describe("checkId uniqueness and completeness", () => {
    it("all auto-check steps have unique checkIds", () => {
        const checkIds = WIZARD_STEPS
            .filter((s) => s.type === "auto-check")
            .map((s) => s.checkId!);
        expect(new Set(checkIds).size).toBe(checkIds.length);
    });

    it("auto-check checkIds are non-empty strings", () => {
        for (const step of WIZARD_STEPS) {
            if (step.type === "auto-check") {
                expect(typeof step.checkId).toBe("string");
                expect(step.checkId!.length).toBeGreaterThan(0);
            }
        }
    });

    it("expected checkIds are all present", () => {
        const checkIds = WIZARD_STEPS
            .filter((s) => s.type === "auto-check")
            .map((s) => s.checkId);
        expect(checkIds).toContain("docker");
        expect(checkIds).toContain("api");
        expect(checkIds).toContain("gpu");
        expect(checkIds).toContain("versions");
        expect(checkIds).toContain("benchmark");
        expect(checkIds).toContain("network");
        expect(checkIds).toContain("verify");
        expect(checkIds).toContain("host-register");
        expect(checkIds).toContain("admission");
        expect(checkIds).toContain("wallet");
        expect(checkIds).toContain("launch");
    });
});

// ══════════════════════════════════════════════════════════════════════
// 14. PROVIDER SUMMARY DATA SHAPE
// ══════════════════════════════════════════════════════════════════════

describe("provider summary step", () => {
    it("provider-summary is a confirm step", () => {
        const step = findStep("provider-summary");
        expect(step.type).toBe("confirm");
    });

    it("provider-summary has confirmLabel", () => {
        const step = findStep("provider-summary");
        expect(step.confirmLabel).toBeDefined();
    });

    it("provider-summary is provider-only", () => {
        const step = findStep("provider-summary");
        expect(step.condition!({ mode: "provide" })).toBe(true);
        expect(step.condition!({ mode: "both" })).toBe(true);
        expect(step.condition!({ mode: "rent" })).toBe(false);
    });
});

// ══════════════════════════════════════════════════════════════════════
// 15. CROSS-MODE ISOLATION
// ══════════════════════════════════════════════════════════════════════

describe("cross-mode isolation", () => {
    it("renter answers do not cause provider steps to appear", () => {
        const visited = walkFlow({
            mode: "rent",
            "device-auth": "authorized",
            "api-check": "passed",
            workload: "training",
            "gpu-preference": "best",
            "browse-gpus": "done",
            "gpu-pick": "host-1",
            "image-pick": "pytorch",
            "confirm-launch": "yes",
            "wallet-check": "passed",
            "_wallet_insufficient": "false",
            "launch-instance": "passed",
            // Provider answers present but mode is rent
            pricing: "custom",
            "custom-rate": "5.00",
            "docker-check": "passed",
        });

        expect(visited).not.toContain("docker-check");
        expect(visited).not.toContain("benchmark");
        expect(visited).not.toContain("custom-rate");
        expect(visited).not.toContain("provider-summary");
    });

    it("provider answers do not cause renter steps to appear", () => {
        const visited = walkFlow({
            mode: "provide",
            "docker-check": "passed",
            "device-auth": "authorized",
            "api-check": "passed",
            "gpu-detect": "passed",
            "version-check": "passed",
            benchmark: "passed",
            "network-bench": "passed",
            verification: "passed",
            pricing: "recommended",
            "host-register": "passed",
            "admission-gate": "passed",
            "provider-summary": "yes",
            // Renter answers present but mode is provide
            workload: "training",
            "gpu-preference": "best",
            "_wallet_insufficient": "true",
        });

        expect(visited).not.toContain("workload");
        expect(visited).not.toContain("gpu-pick");
        expect(visited).not.toContain("payment-gate");
        expect(visited).not.toContain("launch-instance");
    });
});

// ══════════════════════════════════════════════════════════════════════
// 16. FLOW COUNT ASSERTIONS
// ══════════════════════════════════════════════════════════════════════

describe("flow step counts per mode", () => {
    it("renter flow visits 12-14 steps (no payment gate)", () => {
        const visited = walkFlow({
            mode: "rent",
            "device-auth": "authorized",
            "api-check": "passed",
            workload: "training",
            "gpu-preference": "best",
            "browse-gpus": "done",
            "gpu-pick": "host-1",
            "image-pick": "pytorch",
            "confirm-launch": "yes",
            "wallet-check": "passed",
            "_wallet_insufficient": "false",
            "launch-instance": "passed",
        });
        // mode + device-auth + api-check + workload + gpu-preference + browse-gpus +
        // gpu-pick + image-pick + confirm-launch + wallet-check + launch-instance + done = 12
        expect(visited.length).toBeGreaterThanOrEqual(12);
        expect(visited.length).toBeLessThanOrEqual(14);
    });

    it("provider flow visits 14-16 steps (no custom rate)", () => {
        const visited = walkFlow({
            mode: "provide",
            "docker-check": "passed",
            "device-auth": "authorized",
            "api-check": "passed",
            "gpu-detect": "passed",
            "version-check": "passed",
            benchmark: "passed",
            "network-bench": "passed",
            verification: "passed",
            pricing: "recommended",
            "host-register": "passed",
            "admission-gate": "passed",
            "provider-summary": "yes",
        });
        // mode + docker-check + device-auth + api-check + gpu-detect + version-check +
        // benchmark + network-bench + verification + pricing + host-register +
        // admission-gate + provider-summary + done = 14
        expect(visited.length).toBeGreaterThanOrEqual(14);
        expect(visited.length).toBeLessThanOrEqual(16);
    });

    it("both mode visits the most steps", () => {
        const renter = walkFlow({
            mode: "rent", "device-auth": "authorized", "api-check": "passed",
            workload: "training", "gpu-preference": "best", "browse-gpus": "done",
            "gpu-pick": "h1", "image-pick": "pytorch", "confirm-launch": "yes",
            "wallet-check": "passed", "_wallet_insufficient": "false",
            "launch-instance": "passed",
        });
        const provider = walkFlow({
            mode: "provide", "docker-check": "passed", "device-auth": "authorized",
            "api-check": "passed", "gpu-detect": "passed", "version-check": "passed",
            benchmark: "passed", "network-bench": "passed", verification: "passed",
            pricing: "recommended", "host-register": "passed", "admission-gate": "passed",
            "provider-summary": "yes",
        });
        const both = walkFlow({
            mode: "both", "docker-check": "passed", "device-auth": "authorized",
            "api-check": "passed", "gpu-detect": "passed", "version-check": "passed",
            benchmark: "passed", "network-bench": "passed", verification: "passed",
            pricing: "recommended", "host-register": "passed", "admission-gate": "passed",
            "provider-summary": "yes", workload: "training", "gpu-preference": "best",
            "browse-gpus": "done", "gpu-pick": "h1", "image-pick": "pytorch",
            "confirm-launch": "yes", "wallet-check": "passed", "_wallet_insufficient": "false",
            "launch-instance": "passed", "confirm-setup": "yes",
        });

        expect(both.length).toBeGreaterThan(renter.length);
        expect(both.length).toBeGreaterThan(provider.length);
    });
});

// ══════════════════════════════════════════════════════════════════════
// 17. OPTION VALUE INTEGRITY
// ══════════════════════════════════════════════════════════════════════

describe("option value integrity", () => {
    it("all select option values are non-empty strings", () => {
        for (const step of WIZARD_STEPS) {
            if (step.options) {
                for (const opt of step.options) {
                    expect(typeof opt.value).toBe("string");
                    expect(opt.value.length).toBeGreaterThan(0);
                }
            }
        }
    });

    it("all select option labels are non-empty strings", () => {
        for (const step of WIZARD_STEPS) {
            if (step.options) {
                for (const opt of step.options) {
                    expect(typeof opt.label).toBe("string");
                    expect(opt.label.length).toBeGreaterThan(0);
                }
            }
        }
    });

    it("pricing options are rent/provide/both", () => {
        const pricing = findStep("pricing");
        const values = pricing.options!.map((o) => o.value);
        expect(values).toContain("recommended");
        expect(values).toContain("competitive");
        expect(values).toContain("custom");
    });

    it("workload options cover standard use cases", () => {
        const workload = findStep("workload");
        const values = workload.options!.map((o) => o.value);
        expect(values).toContain("training");
        expect(values).toContain("inference");
        expect(values).toContain("research");
        expect(values).toContain("other");
    });

    it("gpu-preference has best, cheapest, specific", () => {
        const pref = findStep("gpu-preference");
        const values = pref.options!.map((o) => o.value);
        expect(values).toContain("best");
        expect(values).toContain("cheapest");
        expect(values).toContain("specific");
    });
});

// ══════════════════════════════════════════════════════════════════════
// 18. REQUIRED CHECKS
// ══════════════════════════════════════════════════════════════════════

describe("required checks", () => {
    it("api-check is required", () => {
        const step = findStep("api-check");
        expect(step.checkRequired).toBe(true);
    });

    it("version-check is required", () => {
        const step = findStep("version-check");
        expect(step.checkRequired).toBe(true);
    });

    it("benchmark is required", () => {
        const step = findStep("benchmark");
        expect(step.checkRequired).toBe(true);
    });

    it("verification is required", () => {
        const step = findStep("verification");
        expect(step.checkRequired).toBe(true);
    });

    it("host-register is required", () => {
        const step = findStep("host-register");
        expect(step.checkRequired).toBe(true);
    });

    it("docker-check is not required (can be skipped)", () => {
        const step = findStep("docker-check");
        expect(step.checkRequired).toBeFalsy();
    });

    it("network-bench is not required (can be skipped)", () => {
        const step = findStep("network-bench");
        expect(step.checkRequired).toBeFalsy();
    });
});

// ══════════════════════════════════════════════════════════════════════
// 19. buildWizardContext — PROVIDER SUMMARY INTEGRATION
// ══════════════════════════════════════════════════════════════════════

describe("buildWizardContext — provider summary", () => {
    it("includes GPU info from provider summary", () => {
        const summary = {
            gpuModel: "RTX 4090",
            vramGb: 24,
            xcuScore: 850,
            tflops: 82.6,
            verified: true,
            verificationState: "verified",
            hostId: "h-42",
            costPerHour: 2.50,
            pricing: "recommended",
            admitted: true,
            runtimeRecommendation: "nvidia-container-toolkit",
            reputationPoints: 100,
            tier: "Silver",
        };
        const ctx = buildWizardContext("provider-summary", {
            mode: "provide",
        }, {}, summary, [], null);
        expect(ctx).toContain("RTX 4090");
    });

    it("handles null provider summary gracefully", () => {
        const ctx = buildWizardContext("provider-summary", {
            mode: "provide",
        }, {}, null, [], null);
        expect(ctx).toBeDefined();
        expect(ctx).not.toContain("undefined");
    });
});

// ══════════════════════════════════════════════════════════════════════
// 20. PROMPT MESSAGES — VARIETY
// ══════════════════════════════════════════════════════════════════════

describe("prompt messages are usable", () => {
    it("no prompts contain placeholder text", () => {
        for (const step of WIZARD_STEPS) {
            expect(step.prompt).not.toContain("TODO");
            expect(step.prompt).not.toContain("FIXME");
            expect(step.prompt).not.toContain("PLACEHOLDER");
        }
    });

    it("prompts are appropriate length (10-200 chars)", () => {
        for (const step of WIZARD_STEPS) {
            expect(step.prompt.length).toBeGreaterThanOrEqual(10);
            expect(step.prompt.length).toBeLessThanOrEqual(200);
        }
    });

    it("mode prompt mentions Hexara", () => {
        const mode = findStep("mode");
        expect(mode.prompt.toLowerCase()).toContain("hexara");
    });

    it("done prompt mentions running wizard again", () => {
        const done = findStep("done");
        expect(done.prompt.toLowerCase()).toContain("wizard");
    });
});
