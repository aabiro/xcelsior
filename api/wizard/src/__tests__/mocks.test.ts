// Tests for mock modules — ensure mocks work correctly before using in component tests

import { describe, it, expect, beforeEach } from "vitest";
import {
    checkDocker,
    setMockPreset,
    setMockResults,
    resetMock,
} from "../__mocks__/checks.js";
import {
    streamChat,
    setMockChatEvents,
    resetApiMock,
} from "../__mocks__/api-client.js";

describe("checks mock", () => {
    beforeEach(() => resetMock());

    it("returns all-pass by default", async () => {
        const results = await checkDocker();
        expect(results).toHaveLength(4);
        expect(results.every((r) => r.ok)).toBe(true);
    });

    it("returns all-fail when preset is set", async () => {
        setMockPreset("all-fail");
        const results = await checkDocker();
        expect(results.every((r) => !r.ok)).toBe(true);
    });

    it("returns partial failures", async () => {
        setMockPreset("partial");
        const results = await checkDocker();
        const passing = results.filter((r) => r.ok);
        const failing = results.filter((r) => !r.ok);
        expect(passing.length).toBeGreaterThan(0);
        expect(failing.length).toBeGreaterThan(0);
    });

    it("accepts custom results", async () => {
        setMockResults([{ name: "Custom", ok: true, detail: "test" }]);
        const results = await checkDocker();
        expect(results).toHaveLength(1);
        expect(results[0].name).toBe("Custom");
    });
});

describe("api-client mock", () => {
    beforeEach(() => resetApiMock());

    it("streams default chat events", async () => {
        const events = [];
        for await (const event of streamChat(
            { baseUrl: "http://test", apiKey: "test" },
            "hello",
        )) {
            events.push(event);
        }
        expect(events.length).toBeGreaterThanOrEqual(3);
        expect(events[0].type).toBe("meta");
        expect(events[events.length - 1].type).toBe("done");
    });

    it("streams custom chat events", async () => {
        setMockChatEvents([
            { type: "token", content: "custom response" },
            { type: "done" },
        ]);
        const events = [];
        for await (const event of streamChat(
            { baseUrl: "http://test", apiKey: "test" },
            "hello",
        )) {
            events.push(event);
        }
        expect(events).toHaveLength(2);
        expect(events[0].content).toBe("custom response");
    });

    it("yields tool_call and confirmation events", async () => {
        setMockChatEvents([
            { type: "meta", conversation_id: "conv-1" },
            { type: "token", content: "Let me check... " },
            { type: "tool_call", name: "search_marketplace", input: { gpu: "A100" } },
            { type: "tool_result", name: "search_marketplace", output: { results: [] } },
            { type: "token", content: "No A100s available right now." },
            { type: "done" },
        ]);

        const events = [];
        for await (const event of streamChat(
            { baseUrl: "http://test", apiKey: "test" },
            "find me an A100",
        )) {
            events.push(event);
        }

        expect(events.find((e) => e.type === "tool_call")).toBeDefined();
        expect(events.find((e) => e.type === "tool_result")).toBeDefined();
    });
});
