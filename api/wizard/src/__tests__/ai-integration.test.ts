// Tests for AI integration — SSE event handling, tool calls, confirmations,
// mock streamChat/confirmAction protocol, and page context enrichment.

import { describe, it, expect, beforeEach } from "vitest";
import {
    streamChat,
    confirmAction,
    setMockChatEvents,
    setMockConfirmEvents,
    resetApiMock,
} from "../__mocks__/api-client.js";
import type { SSEEvent, ApiClientConfig } from "../api-client.js";
import type { PendingConfirmation, AiToolCall } from "../useWizardFlow.js";

const TEST_CONFIG: ApiClientConfig = {
    baseUrl: "http://localhost:9500",
    apiKey: "test-token-123",
    pageContext: "cli-wizard:docker-check | mode=provide",
};

// Helper to collect all events from an async generator
async function collectEvents(gen: AsyncGenerator<SSEEvent>): Promise<SSEEvent[]> {
    const events: SSEEvent[] = [];
    for await (const event of gen) {
        events.push(event);
    }
    return events;
}

// Simulates the askAi SSE processing loop (extracted from useWizardFlow.ts)
async function processAiStream(
    config: ApiClientConfig,
    message: string,
    conversationId?: string,
): Promise<{
    content: string;
    toolCalls: AiToolCall[];
    pendingConfirmation: PendingConfirmation | null;
    conversationId: string | null;
    errors: string[];
}> {
    let content = "";
    const toolCalls: AiToolCall[] = [];
    let pendingConfirmation: PendingConfirmation | null = null;
    let convId: string | null = conversationId ?? null;
    const errors: string[] = [];

    for await (const event of streamChat(config, message, conversationId)) {
        switch (event.type) {
            case "meta":
                if (event.conversation_id) convId = event.conversation_id;
                break;
            case "token":
                content += event.content ?? "";
                break;
            case "tool_call":
                if (event.name) {
                    toolCalls.push({ name: event.name, input: event.input ?? {} });
                }
                break;
            case "tool_result":
                if (event.name) {
                    const existing = toolCalls.find((tc) => tc.name === event.name && !tc.output);
                    if (existing) existing.output = event.output ?? {};
                }
                break;
            case "confirmation_required":
                if (event.confirmation_id && event.tool_name) {
                    pendingConfirmation = {
                        confirmationId: event.confirmation_id,
                        toolName: event.tool_name,
                        toolArgs: event.tool_args ?? {},
                    };
                }
                break;
            case "error":
                errors.push(event.message ?? "unknown error");
                break;
            case "done":
                break;
        }
    }

    return { content, toolCalls, pendingConfirmation, conversationId: convId, errors };
}

// Simulates the confirmAi processing loop
async function processConfirmation(
    config: ApiClientConfig,
    confirmationId: string,
    approved: boolean,
    existingContent: string,
): Promise<{ content: string; errors: string[] }> {
    let content = existingContent;
    const errors: string[] = [];

    for await (const event of confirmAction(config, confirmationId, approved)) {
        if (event.type === "token") {
            content += event.content ?? "";
        } else if (event.type === "error") {
            errors.push(event.message ?? "unknown");
        }
    }

    return { content, errors };
}

describe("AI SSE event protocol", () => {
    beforeEach(() => resetApiMock());

    it("processes meta → tokens → done sequence", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "conv-42" },
            { type: "token", content: "Hello " },
            { type: "token", content: "world!" },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "Hi");
        expect(result.content).toBe("Hello world!");
        expect(result.conversationId).toBe("conv-42");
        expect(result.toolCalls).toHaveLength(0);
        expect(result.pendingConfirmation).toBeNull();
        expect(result.errors).toHaveLength(0);
    });

    it("handles empty content tokens gracefully", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "token" }, // no content field
            { type: "token", content: "" },
            { type: "token", content: "actual content" },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "test");
        expect(result.content).toBe("actual content");
    });

    it("preserves existing conversation id when meta has none", async () => {
        setMockChatEvents([
            { type: "meta" }, // no conversation_id
            { type: "token", content: "reply" },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "test", "existing-conv");
        expect(result.conversationId).toBe("existing-conv");
    });

    it("updates conversation id from meta event", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "new-conv" },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "test", "old-conv");
        expect(result.conversationId).toBe("new-conv");
    });
});

describe("AI tool_call events", () => {
    beforeEach(() => resetApiMock());

    it("captures single tool call with result", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "token", content: "Let me search... " },
            { type: "tool_call", name: "search_marketplace", input: { gpu_model: "A100" } },
            { type: "tool_result", name: "search_marketplace", output: { listings: [{ gpu: "A100", price: 2.50 }] } },
            { type: "token", content: "Found an A100 at $2.50/hr." },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "Find me an A100");
        expect(result.content).toBe("Let me search... Found an A100 at $2.50/hr.");
        expect(result.toolCalls).toHaveLength(1);
        expect(result.toolCalls[0].name).toBe("search_marketplace");
        expect(result.toolCalls[0].input).toEqual({ gpu_model: "A100" });
        expect(result.toolCalls[0].output).toEqual({ listings: [{ gpu: "A100", price: 2.50 }] });
    });

    it("captures multiple tool calls", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "tool_call", name: "search_marketplace", input: { gpu_model: "A100" } },
            { type: "tool_result", name: "search_marketplace", output: { results: [] } },
            { type: "tool_call", name: "recommend_gpu", input: { workload: "training" } },
            { type: "tool_result", name: "recommend_gpu", output: { recommendation: "RTX 4090" } },
            { type: "token", content: "I recommend an RTX 4090 for training." },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "What GPU?");
        expect(result.toolCalls).toHaveLength(2);
        expect(result.toolCalls[0].name).toBe("search_marketplace");
        expect(result.toolCalls[1].name).toBe("recommend_gpu");
        expect(result.toolCalls[1].output).toEqual({ recommendation: "RTX 4090" });
    });

    it("ignores tool_call without name", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "tool_call", input: { foo: "bar" } }, // no name
            { type: "token", content: "ok" },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "test");
        expect(result.toolCalls).toHaveLength(0);
    });

    it("handles tool_result without matching tool_call", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "tool_result", name: "orphan_tool", output: { data: 42 } },
            { type: "token", content: "ok" },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "test");
        expect(result.toolCalls).toHaveLength(0);
        expect(result.content).toBe("ok");
    });

    it("matches tool_result to first unresolved tool_call of same name", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "tool_call", name: "estimate_cost", input: { hours: 1 } },
            { type: "tool_call", name: "estimate_cost", input: { hours: 10 } },
            { type: "tool_result", name: "estimate_cost", output: { cost: 2.50 } },
            { type: "tool_result", name: "estimate_cost", output: { cost: 25.00 } },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "estimate costs");
        expect(result.toolCalls).toHaveLength(2);
        expect(result.toolCalls[0].output).toEqual({ cost: 2.50 });
        expect(result.toolCalls[1].output).toEqual({ cost: 25.00 });
    });
});

describe("AI confirmation_required events", () => {
    beforeEach(() => resetApiMock());

    it("captures confirmation for write action", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "token", content: "I'll launch a job for you. " },
            { type: "confirmation_required", confirmation_id: "cf-1", tool_name: "launch_job", tool_args: { host_id: "host-1", image: "pytorch" } },
        ]);

        const result = await processAiStream(TEST_CONFIG, "Launch a job on host-1");
        expect(result.content).toBe("I'll launch a job for you. ");
        expect(result.pendingConfirmation).not.toBeNull();
        expect(result.pendingConfirmation!.confirmationId).toBe("cf-1");
        expect(result.pendingConfirmation!.toolName).toBe("launch_job");
        expect(result.pendingConfirmation!.toolArgs).toEqual({ host_id: "host-1", image: "pytorch" });
    });

    it("ignores confirmation without id", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "confirmation_required", tool_name: "stop_job" }, // no confirmation_id
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "stop");
        expect(result.pendingConfirmation).toBeNull();
    });

    it("ignores confirmation without tool_name", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "confirmation_required", confirmation_id: "cf-2" }, // no tool_name
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "test");
        expect(result.pendingConfirmation).toBeNull();
    });

    it("handles confirmation with empty tool_args", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "confirmation_required", confirmation_id: "cf-3", tool_name: "create_api_key" },
            // no tool_args field
        ]);

        const result = await processAiStream(TEST_CONFIG, "create api key");
        expect(result.pendingConfirmation).not.toBeNull();
        expect(result.pendingConfirmation!.toolArgs).toEqual({});
    });
});

describe("AI error events", () => {
    beforeEach(() => resetApiMock());

    it("captures single error", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "error", message: "Rate limit exceeded" },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "hello");
        expect(result.errors).toHaveLength(1);
        expect(result.errors[0]).toBe("Rate limit exceeded");
    });

    it("accumulates content before error", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "token", content: "Starting... " },
            { type: "error", message: "Connection lost" },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "hello");
        expect(result.content).toBe("Starting... ");
        expect(result.errors).toContain("Connection lost");
    });

    it("handles error with no message", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "c1" },
            { type: "error" },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "hello");
        expect(result.errors).toHaveLength(1);
        expect(result.errors[0]).toBe("unknown error");
    });
});

describe("confirmAction flow", () => {
    beforeEach(() => resetApiMock());

    it("approval returns continuation tokens", async () => {
        setMockConfirmEvents([
            { type: "token", content: "Job launched successfully! " },
            { type: "token", content: "Instance ID: inst-42." },
            { type: "done" },
        ]);

        const result = await processConfirmation(
            TEST_CONFIG,
            "cf-1",
            true,
            "I'll launch the job. ",
        );
        expect(result.content).toBe("I'll launch the job. Job launched successfully! Instance ID: inst-42.");
        expect(result.errors).toHaveLength(0);
    });

    it("rejection returns cancellation message", async () => {
        setMockConfirmEvents([
            { type: "token", content: "Action cancelled by user." },
            { type: "done" },
        ]);

        const result = await processConfirmation(
            TEST_CONFIG,
            "cf-1",
            false,
            "I'll stop the job. ",
        );
        expect(result.content).toContain("Action cancelled by user.");
    });

    it("handles error during confirmation", async () => {
        setMockConfirmEvents([
            { type: "error", message: "Confirmation expired" },
            { type: "done" },
        ]);

        const result = await processConfirmation(
            TEST_CONFIG,
            "cf-expired",
            true,
            "",
        );
        expect(result.errors).toHaveLength(1);
        expect(result.errors[0]).toBe("Confirmation expired");
    });

    it("empty confirmation stream", async () => {
        setMockConfirmEvents([
            { type: "done" },
        ]);

        const result = await processConfirmation(
            TEST_CONFIG,
            "cf-1",
            true,
            "original",
        );
        expect(result.content).toBe("original");
        expect(result.errors).toHaveLength(0);
    });
});

describe("full AI interaction sequences", () => {
    beforeEach(() => resetApiMock());

    it("search → recommend → answer flow", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "full-1" },
            { type: "tool_call", name: "search_marketplace", input: { gpu_model: "RTX 4090" } },
            {
                type: "tool_result", name: "search_marketplace", output: {
                    total: 3, listings: [
                        { gpu_model: "RTX 4090", price: 0.45 },
                        { gpu_model: "RTX 4090", price: 0.50 },
                        { gpu_model: "RTX 4090", price: 0.55 },
                    ]
                }
            },
            { type: "tool_call", name: "estimate_cost", input: { gpu: "RTX 4090", hours: 8 } },
            { type: "tool_result", name: "estimate_cost", output: { estimated_cost: 3.60 } },
            { type: "token", content: "Found 3 RTX 4090s. " },
            { type: "token", content: "Cheapest is $0.45/hr. " },
            { type: "token", content: "For 8 hours, estimated cost is $3.60 CAD." },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "What RTX 4090s are available?");
        expect(result.toolCalls).toHaveLength(2);
        expect(result.content).toContain("$0.45/hr");
        expect(result.content).toContain("$3.60 CAD");
        expect(result.conversationId).toBe("full-1");
    });

    it("tool call → confirmation → approve flow", async () => {
        // Step 1: AI proposes a write action
        setMockChatEvents([
            { type: "meta", conversation_id: "full-2" },
            { type: "token", content: "I'll launch a training job for you. " },
            { type: "tool_call", name: "launch_job", input: { host_id: "host-1", image: "pytorch" } },
            { type: "confirmation_required", confirmation_id: "cf-launch", tool_name: "launch_job", tool_args: { host_id: "host-1", image: "pytorch" } },
        ]);

        const askResult = await processAiStream(TEST_CONFIG, "Launch training");
        expect(askResult.pendingConfirmation).not.toBeNull();
        expect(askResult.pendingConfirmation!.toolName).toBe("launch_job");
        expect(askResult.toolCalls).toHaveLength(1);

        // Step 2: User approves
        setMockConfirmEvents([
            { type: "token", content: "Job launched! Instance ID: inst-99." },
            { type: "done" },
        ]);

        const confirmResult = await processConfirmation(
            TEST_CONFIG,
            askResult.pendingConfirmation!.confirmationId,
            true,
            askResult.content,
        );
        expect(confirmResult.content).toContain("Job launched!");
        expect(confirmResult.content).toContain("inst-99");
    });

    it("handles only-error response gracefully", async () => {
        setMockChatEvents([
            { type: "error", message: "Unauthorized: invalid token" },
            { type: "done" },
        ]);

        const result = await processAiStream(TEST_CONFIG, "help");
        expect(result.content).toBe("");
        expect(result.errors).toContain("Unauthorized: invalid token");
        expect(result.conversationId).toBeNull();
    });

    it("handles done-only response", async () => {
        setMockChatEvents([{ type: "done" }]);

        const result = await processAiStream(TEST_CONFIG, "hello");
        expect(result.content).toBe("");
        expect(result.toolCalls).toHaveLength(0);
        expect(result.pendingConfirmation).toBeNull();
    });

    it("conversation continuity — second message uses previous conversation id", async () => {
        // First message
        setMockChatEvents([
            { type: "meta", conversation_id: "conv-abc" },
            { type: "token", content: "hi" },
            { type: "done" },
        ]);
        const first = await processAiStream(TEST_CONFIG, "hello");
        expect(first.conversationId).toBe("conv-abc");

        // Second message — server may return same or new conv id
        setMockChatEvents([
            { type: "meta", conversation_id: "conv-abc" },
            { type: "token", content: "still here" },
            { type: "done" },
        ]);
        const second = await processAiStream(TEST_CONFIG, "follow up", first.conversationId!);
        expect(second.conversationId).toBe("conv-abc");
    });
});

describe("PendingConfirmation type", () => {
    it("has required fields", () => {
        const pc: PendingConfirmation = {
            confirmationId: "cf-123",
            toolName: "launch_job",
            toolArgs: { host_id: "h1" },
        };
        expect(pc.confirmationId).toBe("cf-123");
        expect(pc.toolName).toBe("launch_job");
        expect(pc.toolArgs).toEqual({ host_id: "h1" });
    });
});

describe("AiToolCall type", () => {
    it("tracks name, input, and optional output", () => {
        const tc: AiToolCall = { name: "search_marketplace", input: { gpu: "A100" } };
        expect(tc.output).toBeUndefined();

        tc.output = { results: [{ id: 1 }] };
        expect(tc.output).toEqual({ results: [{ id: 1 }] });
    });
});
