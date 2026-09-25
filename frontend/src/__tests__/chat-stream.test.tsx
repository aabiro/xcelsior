import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useChatStream } from "@/hooks/useChatStream";

function streamResponse() {
  const events = [
    { type: "meta", conversation_id: "owned-conversation" },
    { type: "token", content: "A response" },
    { type: "done", message_id: "123" },
  ];
  return new Response(
    events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    { headers: { "Content-Type": "text/event-stream" } },
  );
}

describe("support chat persistence", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("keeps the persisted message ID so feedback can identify the response", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(streamResponse());
    const { result } = renderHook(() => useChatStream());
    await act(async () => { await result.current.sendMessage("Hello"); });
    expect(result.current.messages[1]).toMatchObject({
      content: "A response", messageId: "123",
    });
  });

  it("continues the selected history conversation when sending the next message", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockResolvedValue(streamResponse());
    const { result } = renderHook(() => useChatStream());
    act(() => result.current.restoreConversation("selected-conversation", [
      { id: "123", messageId: "123", role: "assistant", content: "Earlier response", timestamp: 1 },
    ]));
    expect(localStorage.getItem("xcelsior-chat-conv-id")).toBe("selected-conversation");
    await act(async () => { await result.current.sendMessage("Continue"); });
    expect(JSON.parse(fetchMock.mock.calls[0][1]?.body as string).conversation_id).toBe("selected-conversation");
  });

  it("forgets a missing or inaccessible saved conversation before the next send", async () => {
    localStorage.setItem("xcelsior-chat-conv-id", "expired-conversation");
    const fetchMock = vi.spyOn(global, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Conversation not found" }), { status: 404 }))
      .mockResolvedValueOnce(streamResponse());
    const { result } = renderHook(() => useChatStream());
    await act(async () => { await result.current.sendMessage("Hello"); });
    expect(localStorage.getItem("xcelsior-chat-conv-id")).toBeNull();
    await act(async () => { await result.current.sendMessage("Start again"); });
    expect(JSON.parse(fetchMock.mock.calls[1][1]?.body as string).conversation_id).toBeNull();
  });

  it("ignores a cleared request that finishes after a new request starts", async () => {
    let finishOld!: (response: Response) => void;
    let finishNew!: (response: Response) => void;
    vi.spyOn(global, "fetch")
      .mockReturnValueOnce(new Promise<Response>((resolve) => { finishOld = resolve; }))
      .mockReturnValueOnce(new Promise<Response>((resolve) => { finishNew = resolve; }));
    const { result } = renderHook(() => useChatStream());
    let oldSend!: Promise<void>;
    act(() => { oldSend = result.current.sendMessage("Old request"); });
    act(() => result.current.clearChat());
    let newSend!: Promise<void>;
    act(() => { newSend = result.current.sendMessage("New request"); });
    await act(async () => {
      finishOld(streamResponse());
      await oldSend;
    });
    expect(result.current.isStreaming).toBe(true);
    expect(result.current.messages.map((message) => message.content)).toEqual(["New request", ""]);
    await act(async () => {
      finishNew(streamResponse());
      await newSend;
    });
    expect(result.current.isStreaming).toBe(false);
    expect(result.current.messages[1].messageId).toBe("123");
  });
});
