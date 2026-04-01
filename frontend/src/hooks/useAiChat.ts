"use client";

import { useState, useCallback, useRef, useEffect } from "react";

export interface AiMessage {
  id: string;
  role: "user" | "assistant" | "tool_call" | "tool_result" | "confirmation";
  content: string;
  toolName?: string;
  toolInput?: Record<string, unknown>;
  toolOutput?: Record<string, unknown>;
  confirmationId?: string;
  timestamp: number;
}

export interface AiConversation {
  conversation_id: string;
  title: string;
  updated_at: number;
  message_count: number;
}

interface UseAiChatReturn {
  messages: AiMessage[];
  isStreaming: boolean;
  error: string | null;
  conversationId: string | null;
  conversations: AiConversation[];
  suggestions: { label: string; prompt: string }[];
  sendMessage: (message: string) => Promise<void>;
  confirmAction: (confirmationId: string, approved: boolean) => Promise<void>;
  newConversation: () => void;
  loadConversation: (conversationId: string) => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<void>;
  loadConversations: () => Promise<void>;
  loadSuggestions: () => Promise<void>;
}

const AI_CONV_KEY = "xcelsior-ai-conv-id";

export function useAiChat(): UseAiChatReturn {
  const [messages, setMessages] = useState<AiMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversations, setConversations] = useState<AiConversation[]>([]);
  const [suggestions, setSuggestions] = useState<{ label: string; prompt: string }[]>([]);
  const conversationIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Restore conversation_id from localStorage on mount and auto-load messages
  useEffect(() => {
    try {
      const stored = localStorage.getItem(AI_CONV_KEY);
      if (stored) {
        conversationIdRef.current = stored;
        // Auto-load messages from server for the stored conversation
        fetch(`/api/ai/conversations/${stored}`, { credentials: "include" })
          .then((res) => (res.ok ? res.json() : null))
          .then((data) => {
            if (!data?.messages?.length) return;
            const loaded: AiMessage[] = data.messages.map((m: Record<string, unknown>) => ({
              id: (m.message_id as string) || crypto.randomUUID(),
              role: m.role as AiMessage["role"],
              content: (m.content as string) || "",
              toolName: (m.tool_name as string) || undefined,
              toolInput: m.tool_input ? (m.tool_input as Record<string, unknown>) : undefined,
              toolOutput: m.tool_output ? (m.tool_output as Record<string, unknown>) : undefined,
              timestamp: ((m.created_at as number) || 0) * 1000,
            }));
            setMessages(loaded);
          })
          .catch(() => { /* non-fatal — will start fresh */ });
      }
    } catch {
      // SSR or storage unavailable
    }
  }, []);

  const processSSEStream = useCallback(
    async (
      res: Response,
      assistantId: string,
      onMeta?: (convId: string) => void,
    ) => {
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
              try { localStorage.setItem(AI_CONV_KEY, data.conversation_id); } catch {}
              onMeta?.(data.conversation_id);
            } else if (data.type === "token" && data.content) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: m.content + data.content }
                    : m,
                ),
              );
            } else if (data.type === "tool_call") {
              setMessages((prev) => [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "tool_call",
                  content: "",
                  toolName: data.name,
                  toolInput: data.input,
                  timestamp: Date.now(),
                },
              ]);
            } else if (data.type === "tool_result") {
              setMessages((prev) => [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "tool_result",
                  content: "",
                  toolName: data.name,
                  toolOutput: data.output,
                  timestamp: Date.now(),
                },
              ]);
            } else if (data.type === "confirmation_required") {
              setMessages((prev) => [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "confirmation",
                  content: "",
                  toolName: data.tool_name,
                  toolInput: data.tool_args,
                  confirmationId: data.confirmation_id,
                  timestamp: Date.now(),
                },
              ]);
            } else if (data.type === "error") {
              setError(data.message || "An error occurred");
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }
    },
    [],
  );

  const sendMessage = useCallback(
    async (message: string) => {
      if (!message.trim() || isStreaming) return;

      setError(null);

      const userMsg: AiMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: message.trim(),
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);

      const assistantId = crypto.randomUUID();
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: "assistant", content: "", timestamp: Date.now() },
      ]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch("/api/ai/chat", {
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

        await processSSEStream(res, assistantId);
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        const msg = (err as Error).message || "Failed to send message";
        setError(msg);
        setMessages((prev) => prev.filter((m) => m.id !== assistantId || m.content));
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
        // Refresh conversation list so sidebar updates
        try {
          const r = await fetch("/api/ai/conversations", { credentials: "include" });
          if (r.ok) {
            const d = await r.json();
            setConversations(Array.isArray(d.conversations) ? d.conversations : []);
          }
        } catch { /* ignore */ }
      }
    },
    [isStreaming, processSSEStream],
  );

  const confirmAction = useCallback(
    async (confirmationId: string, approved: boolean) => {
      setIsStreaming(true);
      setError(null);

      const assistantId = crypto.randomUUID();
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: "assistant", content: "", timestamp: Date.now() },
      ]);

      try {
        const res = await fetch("/api/ai/confirm", {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ confirmation_id: confirmationId, approved }),
        });

        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body?.detail || `Error ${res.status}`);
        }

        await processSSEStream(res, assistantId);
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setError((err as Error).message || "Failed to confirm action");
      } finally {
        setIsStreaming(false);
      }
    },
    [processSSEStream],
  );

  const newConversation = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setError(null);
    setIsStreaming(false);
    conversationIdRef.current = null;
    try { localStorage.removeItem(AI_CONV_KEY); } catch {}
  }, []);

  const loadConversation = useCallback(async (conversationId: string) => {
    try {
      const res = await fetch(`/api/ai/conversations/${conversationId}`, {
        credentials: "include",
      });
      if (!res.ok) throw new Error("Failed to load conversation");
      const data = await res.json();
      conversationIdRef.current = conversationId;
      try { localStorage.setItem(AI_CONV_KEY, conversationId); } catch {}

      const loaded: AiMessage[] = (Array.isArray(data.messages) ? data.messages : []).map((m: Record<string, unknown>) => ({
        id: (m.message_id as string) || crypto.randomUUID(),
        role: m.role as AiMessage["role"],
        content: (m.content as string) || "",
        toolName: (m.tool_name as string) || undefined,
        toolInput: m.tool_input ? (m.tool_input as Record<string, unknown>) : undefined,
        toolOutput: m.tool_output ? (m.tool_output as Record<string, unknown>) : undefined,
        timestamp: ((m.created_at as number) || 0) * 1000,
      }));
      setMessages(loaded);
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  const deleteConversation = useCallback(async (conversationId: string) => {
    try {
      await fetch(`/api/ai/conversations/${conversationId}`, {
        method: "DELETE",
        credentials: "include",
      });
      if (conversationIdRef.current === conversationId) {
        newConversation();
      }
      setConversations((prev) => prev.filter((c) => c.conversation_id !== conversationId));
    } catch {
      // Ignore delete errors
    }
  }, [newConversation]);

  const loadConversations = useCallback(async () => {
    try {
      const res = await fetch("/api/ai/conversations", { credentials: "include" });
      if (!res.ok) return;
      const data = await res.json();
      setConversations(Array.isArray(data.conversations) ? data.conversations : []);
    } catch {
      // Ignore fetch errors
    }
  }, []);

  const loadSuggestions = useCallback(async () => {
    try {
      const res = await fetch("/api/ai/suggestions", { credentials: "include" });
      if (!res.ok) return;
      const data = await res.json();
      setSuggestions(Array.isArray(data.suggestions) ? data.suggestions : []);
    } catch {
      // Ignore fetch errors
    }
  }, []);

  return {
    messages,
    isStreaming,
    error,
    conversationId: conversationIdRef.current,
    conversations,
    suggestions,
    sendMessage,
    confirmAction,
    newConversation,
    loadConversation,
    deleteConversation,
    loadConversations,
    loadSuggestions,
  };
}
