"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { apiFetch, ApiError } from "@/lib/api";

export interface ChatMessage {
  id: string;
  messageId?: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

interface UseChatStreamReturn {
  messages: ChatMessage[];
  isStreaming: boolean;
  loadingHistory: boolean;
  error: string | null;
  conversationId: string | null;
  sendMessage: (message: string) => Promise<void>;
  clearChat: () => void;
  restoreConversation: (conversationId: string, messages: ChatMessage[]) => void;
  loadConversation: (conversationId: string) => Promise<void>;
  setMessages: (msgs: ChatMessage[]) => void;
}

const CONV_STORAGE_KEY = "xcelsior-chat-conv-id";

export function useChatStream(ownerId: string | null): UseChatStreamReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const conversationIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const storageKey = ownerId ? `${CONV_STORAGE_KEY}:${encodeURIComponent(ownerId)}` : null;

  const rememberConversation = useCallback((id: string | null) => {
    conversationIdRef.current = id;
    setConversationId(id);
    try {
      if (storageKey) {
        if (id) localStorage.setItem(storageKey, id);
        else localStorage.removeItem(storageKey);
      }
    } catch { /* Storage unavailable */ }
  }, [storageKey]);

  // Browser storage and pending requests belong to the authenticated account.
  // The legacy unscoped ID has no known owner and is intentionally not restored.
  useEffect(() => {
    setMessages([]);
    setError(null);
    setIsStreaming(false);
    setLoadingHistory(false);
    let stored: string | null = null;
    try {
      if (storageKey) stored = localStorage.getItem(storageKey);
    } catch { /* Storage unavailable */ }
    conversationIdRef.current = stored;
    setConversationId(stored);
    return () => {
      abortRef.current?.abort();
      abortRef.current = null;
    };
  }, [storageKey]);

  const sendMessage = useCallback(async (message: string) => {
    // The ref closes the gap before React commits isStreaming, and prevents a
    // send from racing an outstanding history selection.
    if (!message.trim() || abortRef.current) return;

    setError(null);

    // Add user message
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: message.trim(),
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);

    // Create placeholder for assistant response
    const assistantId = crypto.randomUUID();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, assistantMsg]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: message.trim(),
          conversation_id: conversationIdRef.current,
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        if (res.status === 404 && !controller.signal.aborted) {
          rememberConversation(null);
        }
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || body?.error?.message || `Error ${res.status}`);
      }
      if (controller.signal.aborted) return;

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response stream");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (controller.signal.aborted) return;
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === "meta" && data.conversation_id) {
              rememberConversation(data.conversation_id);
            } else if (data.type === "token" && data.content) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: m.content + data.content }
                    : m
                )
              );
            } else if (data.type === "done" && data.message_id) {
              setMessages((prev) => prev.map((m) =>
                m.id === assistantId ? { ...m, messageId: String(data.message_id) } : m
              ));
            } else if (data.type === "error") {
              setError(data.message || "An error occurred");
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }
    } catch (err) {
      if (controller.signal.aborted || (err as Error).name === "AbortError") return;
      const msg = (err as Error).message || "Failed to send message";
      setError(msg);
      // Remove empty assistant message on error
      setMessages((prev) => prev.filter((m) => m.id !== assistantId || m.content));
    } finally {
      if (abortRef.current === controller) {
        setIsStreaming(false);
        abortRef.current = null;
      }
    }
  }, [rememberConversation]);

  const clearChat = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setMessages([]);
    setError(null);
    setIsStreaming(false);
    setLoadingHistory(false);
    rememberConversation(null);
  }, [rememberConversation]);

  const restoreConversation = useCallback((conversationId: string, restored: ChatMessage[]) => {
    clearChat();
    rememberConversation(conversationId);
    setMessages(restored);
  }, [clearChat, rememberConversation]);

  const loadConversation = useCallback(async (id: string) => {
    clearChat();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoadingHistory(true);
    try {
      const data = await apiFetch<{ ok: boolean; messages: {
        message_id: string; role: ChatMessage["role"]; content: string; timestamp: number;
      }[] }>(`/api/chat/history/${encodeURIComponent(id)}`, {
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      if (!data.ok || !Array.isArray(data.messages)) {
        throw new Error("Unable to load conversation. Please try again.");
      }
      setMessages(data.messages.map((message: {
        message_id: string; role: ChatMessage["role"]; content: string; timestamp: number;
      }, index: number) => ({
        id: `history-${id}-${index}`,
        messageId: String(message.message_id),
        role: message.role,
        content: message.content,
        timestamp: message.timestamp * 1000,
      })));
      rememberConversation(id);
    } catch (err) {
      if (!controller.signal.aborted) {
        setError(err instanceof ApiError
          ? err.status === 404 ? "Conversation not found" : "Unable to load conversation. Please try again."
          : (err as Error).message || "Unable to load conversation. Please try again.");
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setLoadingHistory(false);
      }
    }
  }, [clearChat, rememberConversation]);

  return {
    messages,
    isStreaming,
    loadingHistory,
    error,
    conversationId,
    sendMessage,
    clearChat,
    restoreConversation,
    loadConversation,
    setMessages,
  };
}
