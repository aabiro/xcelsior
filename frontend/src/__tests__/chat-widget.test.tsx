import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { LazyMotion, domAnimation } from "framer-motion";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const auth = vi.hoisted(() => ({ useAuth: vi.fn() }));
vi.mock("@/lib/auth", () => auth);
vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));

import { ChatWidget } from "@/components/ChatWidget";

function widget() {
  return <LazyMotion features={domAnimation}><ChatWidget externalOpen embedded /></LazyMotion>;
}

function history(content: string) {
  return Response.json({ ok: true, messages: [
    { message_id: "saved-message", role: "assistant", content, timestamp: 1 },
  ] });
}

describe("support widget account and history lifecycle", () => {
  beforeEach(() => {
    localStorage.clear();
    auth.useAuth.mockReturnValue({ user: { user_id: "a" }, loading: false });
  });

  afterEach(() => vi.restoreAllMocks());

  it("removes private content and unsent input immediately on account change", async () => {
    localStorage.setItem("xcelsior-chat-conv-id:a", "history-a");
    vi.spyOn(global, "fetch").mockResolvedValue(history("Account A private content"));
    const { rerender } = render(widget());
    await screen.findByText("Account A private content");
    fireEvent.change(screen.getByPlaceholderText("chat.placeholder"), { target: { value: "Private draft" } });
    auth.useAuth.mockReturnValue({ user: { user_id: "b" }, loading: false });
    rerender(widget());
    expect(screen.queryByText("Account A private content")).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText("chat.placeholder")).toHaveValue("");
    expect(screen.getByText("chat.greeting")).toBeInTheDocument();
  });

  it("cancels a saved history load when cleared and does not restore its late response", async () => {
    localStorage.setItem("xcelsior-chat-conv-id:a", "history-a");
    let finish!: (response: Response) => void;
    const fetchMock = vi.spyOn(global, "fetch")
      .mockReturnValue(new Promise<Response>((resolve) => { finish = resolve; }));
    render(widget());
    await screen.findByText("chat.loading_history");
    expect(screen.getByPlaceholderText("chat.placeholder")).toBeDisabled();
    fireEvent.click(screen.getByTitle("chat.clear"));
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true);
    await act(async () => { finish(history("Old private content")); });
    expect(screen.queryByText("Old private content")).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText("chat.placeholder")).toBeEnabled();
    expect(localStorage.getItem("xcelsior-chat-conv-id:a")).toBeNull();
  });

  it("selects history through the drawer and sends to that saved conversation", async () => {
    const fetchMock = vi.spyOn(global, "fetch")
      .mockResolvedValueOnce(Response.json({ conversations: [
        { conversation_id: "selected", preview: "Choose this conversation", updated_at: 1 },
      ] }))
      .mockResolvedValueOnce(history("Saved context"))
      .mockResolvedValueOnce(new Response('data: {"type":"token","content":"Continued"}\n\n'));
    render(widget());
    fireEvent.click(screen.getByTitle("chat.history"));
    fireEvent.click(await screen.findByText("Choose this conversation"));
    await screen.findByText("Saved context");
    fireEvent.change(screen.getByPlaceholderText("chat.placeholder"), { target: { value: "Continue" } });
    fireEvent.click(screen.getByRole("button", { name: "chat.send" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    expect(JSON.parse(fetchMock.mock.calls[2][1]?.body as string)).toEqual({
      message: "Continue", conversation_id: "selected",
    });
    await screen.findByText("Continued");
  });
  it("refreshes an expired session when opening the history drawer", async () => {
    const fetchMock = vi.spyOn(global, "fetch")
      .mockResolvedValueOnce(Response.json({ detail: "Expired" }, { status: 401 }))
      .mockResolvedValueOnce(Response.json({ ok: true }))
      .mockResolvedValueOnce(Response.json({ conversations: [
        { conversation_id: "saved", preview: "Recovered history", updated_at: 1 },
      ] }));
    render(widget());
    fireEvent.click(screen.getByTitle("chat.history"));
    await screen.findByText("Recovered history");
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/api/chat/conversations", "/api/auth/refresh", "/api/chat/conversations",
    ]);
  });

  it("refreshes an expired session before recording feedback", async () => {
    localStorage.setItem("xcelsior-chat-conv-id:a", "history-a");
    const fetchMock = vi.spyOn(global, "fetch")
      .mockResolvedValueOnce(history("Saved answer"))
      .mockResolvedValueOnce(Response.json({ detail: "Expired" }, { status: 401 }))
      .mockResolvedValueOnce(Response.json({ ok: true }))
      .mockResolvedValueOnce(Response.json({ ok: true }));
    render(widget());
    fireEvent.click(await screen.findByRole("button", { name: "Helpful" }));
    await screen.findByText("chat.feedback_thanks");
    expect(fetchMock.mock.calls.slice(1).map(([url]) => url)).toEqual([
      "/api/chat/feedback", "/api/auth/refresh", "/api/chat/feedback",
    ]);
    expect(fetchMock.mock.calls[3][1]?.body).toBe(fetchMock.mock.calls[1][1]?.body);
  });

});
