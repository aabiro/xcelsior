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

function historyResponse(content: string) {
  return Response.json({ ok: true, messages: [
    { message_id: "history-message", role: "assistant", content, timestamp: 123 },
  ] });
}

describe("support chat persistence", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("keeps the persisted message ID so feedback can identify the response", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(streamResponse());
    const { result } = renderHook(() => useChatStream("user-a"));
    await act(async () => { await result.current.sendMessage("Hello"); });
    expect(result.current.messages[1]).toMatchObject({
      content: "A response", messageId: "123",
    });
  });

  it("continues the selected history conversation when sending the next message", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockResolvedValue(streamResponse());
    const { result } = renderHook(() => useChatStream("user-a"));
    act(() => result.current.restoreConversation("selected-conversation", [
      { id: "123", messageId: "123", role: "assistant", content: "Earlier response", timestamp: 1 },
    ]));
    expect(localStorage.getItem("xcelsior-chat-conv-id:user-a")).toBe("selected-conversation");
    await act(async () => { await result.current.sendMessage("Continue"); });
    expect(JSON.parse(fetchMock.mock.calls[0][1]?.body as string).conversation_id).toBe("selected-conversation");
  });

  it("forgets a missing or inaccessible saved conversation before the next send", async () => {
    localStorage.setItem("xcelsior-chat-conv-id:user-a", "expired-conversation");
    const fetchMock = vi.spyOn(global, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Conversation not found" }), { status: 404 }))
      .mockResolvedValueOnce(streamResponse());
    const { result } = renderHook(() => useChatStream("user-a"));
    await act(async () => { await result.current.sendMessage("Hello"); });
    expect(localStorage.getItem("xcelsior-chat-conv-id:user-a")).toBeNull();
    await act(async () => { await result.current.sendMessage("Start again"); });
    expect(JSON.parse(fetchMock.mock.calls[1][1]?.body as string).conversation_id).toBeNull();
  });

  it("ignores a cleared request that finishes after a new request starts", async () => {
    let finishOld!: (response: Response) => void;
    let finishNew!: (response: Response) => void;
    vi.spyOn(global, "fetch")
      .mockReturnValueOnce(new Promise<Response>((resolve) => { finishOld = resolve; }))
      .mockReturnValueOnce(new Promise<Response>((resolve) => { finishNew = resolve; }));
    const { result } = renderHook(() => useChatStream("user-a"));
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

  it("clears private messages and cancels the old user's response on account change", async () => {
    let finish!: (response: Response) => void;
    const fetchMock = vi.spyOn(global, "fetch")
      .mockReturnValueOnce(new Promise<Response>((resolve) => { finish = resolve; }));
    const { result, rerender } = renderHook(({ owner }) => useChatStream(owner), {
      initialProps: { owner: "user-a" as string | null },
    });
    act(() => result.current.restoreConversation("private-a", [{
      id: "a", role: "assistant", content: "Private account details", timestamp: 1,
    }]));
    let sending!: Promise<void>;
    act(() => { sending = result.current.sendMessage("Continue my private conversation"); });
    rerender({ owner: "user-b" });
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true);
    expect(result.current.messages).toEqual([]);
    expect(result.current.conversationId).toBeNull();
    await act(async () => { finish(streamResponse()); await sending; });
    expect(result.current.messages).toEqual([]);
    expect(localStorage.getItem("xcelsior-chat-conv-id:user-b")).toBeNull();
    expect(localStorage.getItem("xcelsior-chat-conv-id:user-a")).toBe("private-a");
    rerender({ owner: null });
    expect(result.current.conversationId).toBeNull();
  });

  it("restores only this account's saved ID and ignores the old unowned key", () => {
    localStorage.setItem("xcelsior-chat-conv-id", "unknown-owner");
    localStorage.setItem("xcelsior-chat-conv-id:user-a", "private-a");
    localStorage.setItem("xcelsior-chat-conv-id:user-b", "private-b");
    const { result, rerender } = renderHook(({ owner }) => useChatStream(owner), {
      initialProps: { owner: "user-a" },
    });
    expect(result.current.conversationId).toBe("private-a");
    rerender({ owner: "user-b" });
    expect(result.current.conversationId).toBe("private-b");
  });

  it("aborts a pending stream when the widget unmounts", async () => {
    let finish!: (response: Response) => void;
    const fetchMock = vi.spyOn(global, "fetch")
      .mockReturnValueOnce(new Promise<Response>((resolve) => { finish = resolve; }));
    const { result, unmount } = renderHook(() => useChatStream("user-a"));
    let sending!: Promise<void>;
    act(() => { sending = result.current.sendMessage("Hello"); });
    unmount();
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true);
    await act(async () => { finish(streamResponse()); await sending; });
    expect(localStorage.getItem("xcelsior-chat-conv-id:user-a")).toBeNull();
  });

  it("admits only one send when two events arrive before React renders", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockImplementation(async () => streamResponse());
    const { result } = renderHook(() => useChatStream("user-a"));
    await act(async () => {
      await Promise.all([result.current.sendMessage("First"), result.current.sendMessage("Second")]);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.current.messages).toHaveLength(2);
  });

  it("keeps the latest history selection when an older request resolves first", async () => {
    let finishOld!: (response: Response) => void;
    let finishNew!: (response: Response) => void;
    const fetchMock = vi.spyOn(global, "fetch")
      .mockReturnValueOnce(new Promise<Response>((resolve) => { finishOld = resolve; }))
      .mockReturnValueOnce(new Promise<Response>((resolve) => { finishNew = resolve; }));
    const { result } = renderHook(() => useChatStream("user-a"));
    let oldLoad!: Promise<void>;
    let newLoad!: Promise<void>;
    act(() => { oldLoad = result.current.loadConversation("old-history"); });
    act(() => { newLoad = result.current.loadConversation("selected-history"); });
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true);
    await act(async () => { finishOld(historyResponse("Stale history")); await oldLoad; });
    expect(result.current.messages).toEqual([]);
    expect(result.current.loadingHistory).toBe(true);
    await act(async () => { finishNew(historyResponse("Selected history")); await newLoad; });
    expect(result.current.loadingHistory).toBe(false);
    expect(result.current.messages[0]).toMatchObject({
      content: "Selected history", messageId: "history-message", timestamp: 123000,
    });
    expect(result.current.conversationId).toBe("selected-history");
  });

  it("does not restore cleared history over a new streaming conversation", async () => {
    let finishHistory!: (response: Response) => void;
    vi.spyOn(global, "fetch")
      .mockReturnValueOnce(new Promise<Response>((resolve) => { finishHistory = resolve; }))
      .mockResolvedValueOnce(streamResponse());
    const { result } = renderHook(() => useChatStream("user-a"));
    let history!: Promise<void>;
    act(() => { history = result.current.loadConversation("old-history"); });
    act(() => result.current.clearChat());
    await act(async () => { await result.current.sendMessage("New conversation"); });
    await act(async () => { finishHistory(historyResponse("Old conversation")); await history; });
    expect(result.current.conversationId).toBe("owned-conversation");
    expect(result.current.messages.map((message) => message.content)).toEqual([
      "New conversation", "A response",
    ]);
  });

  it("does not send while history is loading", async () => {
    let finish!: (response: Response) => void;
    const fetchMock = vi.spyOn(global, "fetch")
      .mockReturnValueOnce(new Promise<Response>((resolve) => { finish = resolve; }));
    const { result } = renderHook(() => useChatStream("user-a"));
    let history!: Promise<void>;
    act(() => { history = result.current.loadConversation("selected"); });
    await act(async () => { await result.current.sendMessage("Wait for context"); });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await act(async () => { finish(historyResponse("Context")); await history; });
  });

  it("ignores old history after switching accounts, even while JSON is being read", async () => {
    let finishJson!: (value: unknown) => void;
    const response = new Response();
    vi.spyOn(response, "json").mockReturnValue(new Promise((resolve) => { finishJson = resolve; }));
    vi.spyOn(global, "fetch").mockResolvedValue(response);
    const { result, rerender } = renderHook(({ owner }) => useChatStream(owner), {
      initialProps: { owner: "user-a" },
    });
    let history!: Promise<void>;
    await act(async () => { history = result.current.loadConversation("private-a"); });
    rerender({ owner: "user-b" });
    await act(async () => {
      finishJson({ ok: true, messages: [{ message_id: "secret", role: "assistant", content: "Private", timestamp: 1 }] });
      await history;
    });
    expect(result.current.messages).toEqual([]);
    expect(result.current.conversationId).toBeNull();
    expect(localStorage.getItem("xcelsior-chat-conv-id:user-b")).toBeNull();
  });

  it("reports history failures and forgets inaccessible saved IDs", async () => {
    localStorage.setItem("xcelsior-chat-conv-id:user-a", "expired");
    vi.spyOn(global, "fetch").mockResolvedValue(new Response("", { status: 404 }));
    const { result } = renderHook(() => useChatStream("user-a"));
    await act(async () => { await result.current.loadConversation("expired"); });
    expect(result.current.error).toBe("Conversation not found");
    expect(result.current.loadingHistory).toBe(false);
    expect(result.current.conversationId).toBeNull();
    expect(localStorage.getItem("xcelsior-chat-conv-id:user-a")).toBeNull();
  });
  it("refreshes an expired session before restoring saved history", async () => {
    const fetchMock = vi.spyOn(global, "fetch")
      .mockResolvedValueOnce(Response.json({ detail: "Expired" }, { status: 401 }))
      .mockResolvedValueOnce(Response.json({ ok: true }))
      .mockResolvedValueOnce(historyResponse("Recovered conversation"));
    const { result } = renderHook(() => useChatStream("user-a"));
    await act(async () => { await result.current.loadConversation("saved"); });
    expect(result.current.error).toBeNull();
    expect(result.current.messages[0]?.content).toBe("Recovered conversation");
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/api/chat/history/saved", "/api/auth/refresh", "/api/chat/history/saved",
    ]);
  });

});
