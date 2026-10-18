// Edge case and regression tests — bugs fixed during audit pass 2

import { describe, it, expect, beforeEach } from "vitest";
import { WIZARD_STEPS, getNextStep, IMAGE_TEMPLATES, WORKLOAD_IMAGE_MAP } from "../wizard-flow.js";
import {
    setMockChatEvents,
    setMockConfirmEvents,
    resetApiMock,
    streamChat,
    confirmAction,
} from "../__mocks__/api-client.js";
import type { SSEEvent, ApiClientConfig } from "../api-client.js";

const TEST_CONFIG: ApiClientConfig = {
    baseUrl: "http://localhost:9500",
    apiKey: "test-token",
};

// ── Navigation Regressions ──────────────────────────────────────────

describe("wizard navigation regressions", () => {
    it("payment-gate condition handles missing _wallet_insufficient", () => {
        const paymentStep = WIZARD_STEPS.find((s) => s.id === "payment-gate");
        // When key is missing entirely, condition should be false (not crash)
        expect(paymentStep!.condition!({ mode: "rent" })).toBe(false);
    });

    it("payment-gate condition with empty string", () => {
        const paymentStep = WIZARD_STEPS.find((s) => s.id === "payment-gate");
        expect(paymentStep!.condition!({ mode: "rent", "_wallet_insufficient": "" })).toBe(false);
    });

    it("custom-rate validates decimal values", () => {
        const step = WIZARD_STEPS.find((s) => s.id === "custom-rate");
        expect(step!.validate!("0.01")).toBeNull();
        expect(step!.validate!("0.99")).toBeNull();
        expect(step!.validate!("100.50")).toBeNull();
    });

    it("custom-rate rejects zero", () => {
        const step = WIZARD_STEPS.find((s) => s.id === "custom-rate");
        expect(step!.validate!("0")).not.toBeNull();
        expect(step!.validate!("0.00")).not.toBeNull();
    });

    it("provider flow visits all required check steps", () => {
        const answers: Record<string, string> = {
            mode: "provide",
            pricing: "recommended",
            "docker-check": "passed",
            "device-auth": "authorized",
            "api-key": "test-token",
            "api-check": "passed",
            "gpu-detect": "passed",
            "version-check": "passed",
            "benchmark": "passed",
            "network-bench": "passed",
            "verification": "passed",
            "host-register": "passed",
            "admission-gate": "passed",
            "provider-summary": "confirmed",
        };

        const visited: string[] = [];
        let idx = 0;
        while (idx < WIZARD_STEPS.length && WIZARD_STEPS[idx].type !== "done") {
            visited.push(WIZARD_STEPS[idx].id);
            idx = getNextStep(idx, answers);
            if (idx === -1) break;
        }

        // Provider MUST visit these infrastructure steps
        expect(visited).toContain("docker-check");
        expect(visited).toContain("gpu-detect");
        expect(visited).toContain("version-check");
        expect(visited).toContain("benchmark");
        expect(visited).toContain("network-bench");
        expect(visited).toContain("verification");
        expect(visited).toContain("host-register");
    });

    it("renter flow visits wallet-check", () => {
        const answers: Record<string, string> = {
            mode: "rent",
            "device-auth": "authorized",
            "api-key": "test-token",
            workload: "training",
            "gpu-preference": "cheapest",
            "browse-gpus": "done",
            "gpu-pick": "host-1",
            "image-pick": "nvcr.io/nvidia/pytorch:24.01-py3",
            "_wallet_insufficient": "false",
            "wallet-check": "passed",
            "confirm-launch": "yes",
            "launch-instance": "passed",
        };

        const visited: string[] = [];
        let idx = 0;
        while (idx < WIZARD_STEPS.length && WIZARD_STEPS[idx].type !== "done") {
            visited.push(WIZARD_STEPS[idx].id);
            idx = getNextStep(idx, answers);
            if (idx === -1) break;
        }

        expect(visited).toContain("wallet-check");
        expect(visited).toContain("confirm-launch");
    });

    it("every step has valid type", () => {
        const validTypes = ["select", "text", "auto-check", "info", "confirm", "done", "device-auth", "auto-fetch", "payment-gate"];
        for (const step of WIZARD_STEPS) {
            expect(validTypes).toContain(step.type);
        }
    });

    it("auto-check steps have checkId", () => {
        for (const step of WIZARD_STEPS) {
            if (step.type === "auto-check") {
                expect(step.checkId).toBeDefined();
                expect(step.checkId!.length).toBeGreaterThan(0);
            }
        }
    });

    it("select steps have options array", () => {
        for (const step of WIZARD_STEPS) {
            if (step.type === "select") {
                expect(step.options).toBeDefined();
                // Some selects (gpu-pick, image-pick) are dynamically populated at runtime
                expect(Array.isArray(step.options)).toBe(true);
            }
        }
    });
});

// ── Workload / Image Map Regressions ────────────────────────────────

describe("workload image mapping", () => {
    it("every workload maps to a valid image template", () => {
        for (const [, image] of Object.entries(WORKLOAD_IMAGE_MAP)) {
            const found = IMAGE_TEMPLATES.find((t) => t.value === image);
            expect(found).toBeDefined();
        }
    });

    it("other workload has a fallback image", () => {
        expect(WORKLOAD_IMAGE_MAP["other"]).toBeDefined();
    });

    it("image templates have unique values", () => {
        const values = IMAGE_TEMPLATES.map((t) => t.value);
        expect(new Set(values).size).toBe(values.length);
    });
});

// ── Mock Infrastructure Edge Cases ──────────────────────────────────

describe("mock infrastructure edge cases", () => {
    beforeEach(() => resetApiMock());

    it("default streamChat includes meta + tokens + done", async () => {
        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "hi")) events.push(e);
        expect(events[0].type).toBe("meta");
        expect(events[events.length - 1].type).toBe("done");
        const tokens = events.filter((e) => e.type === "token");
        expect(tokens.length).toBeGreaterThan(0);
    });

    it("default confirmAction returns success", async () => {
        const events: SSEEvent[] = [];
        for await (const e of confirmAction(TEST_CONFIG, "cf-1", true)) events.push(e);
        expect(events.some((e) => e.type === "token")).toBe(true);
        expect(events[events.length - 1].type).toBe("done");
    });

    it("custom events override defaults completely", async () => {
        setMockChatEvents([
            { type: "error", message: "server down" },
        ]);
        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "hi")) events.push(e);
        expect(events).toHaveLength(1);
        expect(events[0].type).toBe("error");
    });

    it("resetApiMock restores defaults", async () => {
        setMockChatEvents([{ type: "done" }]);
        resetApiMock();
        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "hi")) events.push(e);
        expect(events.length).toBeGreaterThan(1);
    });

    it("confirm events can be set independently of chat events", async () => {
        setMockConfirmEvents([
            { type: "token", content: "custom confirm" },
            { type: "done" },
        ]);
        // Chat events should still be default
        const chatEvents: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "hi")) chatEvents.push(e);
        expect(chatEvents[0].type).toBe("meta");

        // Confirm events should be custom
        const confirmEvents: SSEEvent[] = [];
        for await (const e of confirmAction(TEST_CONFIG, "cf-1", true)) confirmEvents.push(e);
        expect(confirmEvents[0].content).toBe("custom confirm");
    });

    it("large SSE event sequence", async () => {
        // Simulate a long response with many tokens
        const events: SSEEvent[] = [{ type: "meta", conversation_id: "long" }];
        for (let i = 0; i < 50; i++) {
            events.push({ type: "token", content: `word${i} ` });
        }
        events.push({ type: "done" });
        setMockChatEvents(events);

        const collected: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "generate")) collected.push(e);
        expect(collected).toHaveLength(52); // meta + 50 tokens + done
        const allContent = collected
            .filter((e) => e.type === "token")
            .map((e) => e.content)
            .join("");
        expect(allContent).toContain("word0");
        expect(allContent).toContain("word49");
    });
});

// ── AI Write Tool Scenarios ─────────────────────────────────────────

describe("AI write tool scenarios", () => {
    beforeEach(() => resetApiMock());

    it("stop_job confirmation flow", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "stop-1" },
            { type: "token", content: "I'll stop your job. " },
            { type: "confirmation_required", confirmation_id: "cf-stop", tool_name: "stop_job", tool_args: { instance_id: "inst-42" } },
        ]);

        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "stop my job")) events.push(e);
        const confirm = events.find((e) => e.type === "confirmation_required");
        expect(confirm).toBeDefined();
        expect(confirm!.tool_name).toBe("stop_job");
        expect(confirm!.tool_args).toEqual({ instance_id: "inst-42" });
    });

    it("create_api_key confirmation flow", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "key-1" },
            { type: "confirmation_required", confirmation_id: "cf-key", tool_name: "create_api_key", tool_args: { name: "my-key" } },
        ]);

        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "create key")) events.push(e);
        expect(events.some((e) => e.type === "confirmation_required")).toBe(true);
    });

    it("revoke_api_key confirmation flow", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "rev-1" },
            { type: "token", content: "This will revoke key abc." },
            { type: "confirmation_required", confirmation_id: "cf-rev", tool_name: "revoke_api_key", tool_args: { key_id: "abc" } },
        ]);

        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "revoke key abc")) events.push(e);
        const confirm = events.find((e) => e.type === "confirmation_required");
        expect(confirm!.tool_name).toBe("revoke_api_key");
    });
});

// ── API Client Config ───────────────────────────────────────────────

describe("ApiClientConfig", () => {
    beforeEach(() => resetApiMock());

    it("streamChat accepts config without pageContext", async () => {
        const config: ApiClientConfig = { baseUrl: "http://test", apiKey: "key" };
        // Should not throw
        const events: SSEEvent[] = [];
        for await (const e of streamChat(config, "hi")) events.push(e);
        expect(events.length).toBeGreaterThan(0);
    });

    it("streamChat accepts config with pageContext", async () => {
        const config: ApiClientConfig = {
            baseUrl: "http://test",
            apiKey: "key",
            pageContext: "cli-wizard:pricing | mode=provide | pricing=custom",
        };
        const events: SSEEvent[] = [];
        for await (const e of streamChat(config, "help")) events.push(e);
        expect(events.length).toBeGreaterThan(0);
    });

    it("streamChat accepts optional conversationId", async () => {
        const events: SSEEvent[] = [];
        for await (const e of streamChat(TEST_CONFIG, "test", "conv-abc")) events.push(e);
        expect(events.length).toBeGreaterThan(0);
    });
});
