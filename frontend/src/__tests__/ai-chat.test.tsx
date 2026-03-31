/**
 * Frontend tests for the useAiChat hook:
 * - SSE parsing, state management, CRUD operations
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useAiChat } from "@/hooks/useAiChat";

// ── Helpers ──────────────────────────────────────────────────────────

function createMockSSEResponse(events: Array<{ type: string; [key: string]: unknown }>) {
  const text = events.map((e) => `data: ${JSON.stringify(e)}\n`).join("\n") + "\n";
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(text));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

// ─────────────────────────────────────────────────────────────────────
describe("useAiChat hook", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    // Default: no stored conversation to restore
    vi.spyOn(global, "fetch").mockResolvedValue(new Response("", { status: 404 }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("starts with empty state", () => {
    const { result } = renderHook(() => useAiChat());
    expect(result.current.messages).toEqual([]);
    expect(result.current.isStreaming).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.conversations).toEqual([]);
    expect(result.current.suggestions).toEqual([]);
  });

  it("restores conversation_id from localStorage and loads messages", async () => {
    const convId = "conv-test-123";
    localStorage.setItem("xcelsior-ai-conv-id", convId);

    const mockMessages = [
      { message_id: "m1", role: "user", content: "Hello", created_at: 1000 },
      { message_id: "m2", role: "assistant", content: "Hi there", created_at: 1001 },
    ];

    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ messages: mockMessages }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const { result } = renderHook(() => useAiChat());

    await waitFor(() => {
      expect(result.current.messages.length).toBe(2);
    });

    expect(result.current.messages[0].content).toBe("Hello");
    expect(result.current.messages[0].role).toBe("user");
    expect(result.current.messages[1].content).toBe("Hi there");
    expect(result.current.messages[1].role).toBe("assistant");
  });

  it("sends a message and processes SSE stream", async () => {
    const convId = "conv-new";
    const sseEvents = [
      { type: "meta", conversation_id: convId },
      { type: "token", content: "Hello " },
      { type: "token", content: "world!" },
    ];

    vi.spyOn(global, "fetch").mockResolvedValue(createMockSSEResponse(sseEvents));

    const { result } = renderHook(() => useAiChat());

    await act(async () => {
      await result.current.sendMessage("Test message");
    });

    expect(result.current.messages.length).toBeGreaterThanOrEqual(2);
    const userMsg = result.current.messages.find((m) => m.role === "user");
    expect(userMsg?.content).toBe("Test message");

    const assistantMsg = result.current.messages.find((m) => m.role === "assistant");
    expect(assistantMsg?.content).toBe("Hello world!");

    expect(localStorage.getItem("xcelsior-ai-conv-id")).toBe(convId);
  });

  it("handles tool_call events in SSE stream", async () => {
    const sseEvents = [
      { type: "meta", conversation_id: "c1" },
      { type: "tool_call", name: "get_billing_summary", input: { period: "month" } },
      { type: "tool_result", name: "get_billing_summary", output: { balance: 50.0 } },
      { type: "token", content: "Your balance is $50." },
    ];

    vi.spyOn(global, "fetch").mockResolvedValue(createMockSSEResponse(sseEvents));

    const { result } = renderHook(() => useAiChat());

    await act(async () => {
      await result.current.sendMessage("What's my balance?");
    });

    const toolCall = result.current.messages.find((m) => m.role === "tool_call");
    expect(toolCall).toBeDefined();
    expect(toolCall?.toolName).toBe("get_billing_summary");
    expect(toolCall?.toolInput).toEqual({ period: "month" });

    const toolResult = result.current.messages.find((m) => m.role === "tool_result");
    expect(toolResult).toBeDefined();
    expect(toolResult?.toolOutput).toEqual({ balance: 50.0 });
  });

  it("handles confirmation_required events", async () => {
    const sseEvents = [
      { type: "meta", conversation_id: "c1" },
      {
        type: "confirmation_required",
        confirmation_id: "confirm-abc",
        tool_name: "launch_job",
        tool_args: { gpu: "A100", count: 2 },
      },
    ];

    vi.spyOn(global, "fetch").mockResolvedValue(createMockSSEResponse(sseEvents));

    const { result } = renderHook(() => useAiChat());

    await act(async () => {
      await result.current.sendMessage("Launch a job");
    });

    const confirmation = result.current.messages.find((m) => m.role === "confirmation");
    expect(confirmation).toBeDefined();
    expect(confirmation?.confirmationId).toBe("confirm-abc");
    expect(confirmation?.toolName).toBe("launch_job");
    expect(confirmation?.toolInput).toEqual({ gpu: "A100", count: 2 });
  });

  it("handles SSE error events", async () => {
    const sseEvents = [
      { type: "meta", conversation_id: "c1" },
      { type: "error", message: "Rate limit exceeded" },
    ];

    vi.spyOn(global, "fetch").mockResolvedValue(createMockSSEResponse(sseEvents));

    const { result } = renderHook(() => useAiChat());

    await act(async () => {
      await result.current.sendMessage("test");
    });

    expect(result.current.error).toBe("Rate limit exceeded");
  });

  it("handles HTTP error from chat endpoint", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Unauthorized" }), { status: 401 }),
    );

    const { result } = renderHook(() => useAiChat());

    await act(async () => {
      await result.current.sendMessage("test");
    });

    expect(result.current.error).toBe("Unauthorized");
    expect(result.current.isStreaming).toBe(false);
  });

  it("newConversation clears state", () => {
    localStorage.setItem("xcelsior-ai-conv-id", "old-conv");

    const { result } = renderHook(() => useAiChat());

    act(() => {
      result.current.newConversation();
    });

    expect(result.current.messages).toEqual([]);
    expect(result.current.error).toBeNull();
    expect(result.current.isStreaming).toBe(false);
    expect(localStorage.getItem("xcelsior-ai-conv-id")).toBeNull();
  });

  it("loadConversations fetches list from API", async () => {
    const mockConversations = [
      { conversation_id: "c1", title: "First chat", updated_at: 1000, message_count: 5 },
      { conversation_id: "c2", title: "Second chat", updated_at: 2000, message_count: 3 },
    ];

    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ conversations: mockConversations }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const { result } = renderHook(() => useAiChat());

    await act(async () => {
      await result.current.loadConversations();
    });

    expect(result.current.conversations).toHaveLength(2);
    expect(result.current.conversations[0].title).toBe("First chat");
  });

  it("loadSuggestions fetches from API", async () => {
    const mockSuggestions = [
      { label: "Rent GPUs", prompt: "I want to rent GPUs" },
      { label: "Check billing", prompt: "Show my billing" },
    ];

    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ suggestions: mockSuggestions }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const { result } = renderHook(() => useAiChat());

    await act(async () => {
      await result.current.loadSuggestions();
    });

    expect(result.current.suggestions).toHaveLength(2);
    expect(result.current.suggestions[0].label).toBe("Rent GPUs");
  });

  it("deleteConversation removes from list and clears if active", async () => {
    const { result } = renderHook(() => useAiChat());

    // Load conversation list
    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          conversations: [{ conversation_id: "c1", title: "Test", updated_at: 1000, message_count: 1 }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    await act(async () => {
      await result.current.loadConversations();
    });
    expect(result.current.conversations).toHaveLength(1);

    // Delete
    vi.spyOn(global, "fetch").mockResolvedValue(new Response("", { status: 200 }));
    await act(async () => {
      await result.current.deleteConversation("c1");
    });
    expect(result.current.conversations).toHaveLength(0);
  });

  it("does not send empty messages", async () => {
    const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValue(new Response("", { status: 404 }));

    const { result } = renderHook(() => useAiChat());

    await act(async () => {
      await result.current.sendMessage("");
    });

    expect(fetchSpy).not.toHaveBeenCalledWith("/api/ai/chat", expect.anything());
  });

  it("does not send whitespace-only messages", async () => {
    const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValue(new Response("", { status: 404 }));

    const { result } = renderHook(() => useAiChat());

    await act(async () => {
      await result.current.sendMessage("   ");
    });

    expect(fetchSpy).not.toHaveBeenCalledWith("/api/ai/chat", expect.anything());
  });
});

