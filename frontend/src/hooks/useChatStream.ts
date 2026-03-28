"use client";

import { useState, useCallback, useRef, useEffect } from "react";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

interface UseChatStreamReturn {
  messages: ChatMessage[];
  isStreaming: boolean;
  error: string | null;
  conversationId: string | null;
  sendMessage: (message: string) => Promise<void>;
  clearChat: () => void;
  setMessages: (msgs: ChatMessage[]) => void;
}

const CONV_STORAGE_KEY = "xcelsior-chat-conv-id";

export function useChatStream(): UseChatStreamReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const conversationIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Restore conversation_id from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(CONV_STORAGE_KEY);
      if (stored) conversationIdRef.current = stored;
    } catch {
      // SSR or storage unavailable
    }
  }, []);

  const sendMessage = useCallback(async (message: string) => {
    if (!message.trim() || isStreaming) return;

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
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || body?.error?.message || `Error ${res.status}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response stream");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === "meta" && data.conversation_id) {
              conversationIdRef.current = data.conversation_id;
              try {
                localStorage.setItem(CONV_STORAGE_KEY, data.conversation_id);
              } catch {
                // Storage unavailable
              }
            } else if (data.type === "token" && data.content) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: m.content + data.content }
                    : m
                )
              );
            } else if (data.type === "error") {
              setError(data.message || "An error occurred");
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      const msg = (err as Error).message || "Failed to send message";
      setError(msg);
      // Remove empty assistant message on error
      setMessages((prev) => prev.filter((m) => m.id !== assistantId || m.content));
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  }, [isStreaming]);

  const clearChat = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setError(null);
    setIsStreaming(false);
    conversationIdRef.current = null;
    try {
      localStorage.removeItem(CONV_STORAGE_KEY);
    } catch {
      // Storage unavailable
    }
  }, []);

  return {
    messages,
    isStreaming,
    error,
    conversationId: conversationIdRef.current,
    sendMessage,
    clearChat,
    setMessages,
  };
}
