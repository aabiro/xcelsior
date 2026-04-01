"use client";

import { useEffect, useRef, useState, useMemo, FormEvent, useCallback } from "react";
import {
  Send, Plus, Loader2, Sparkles, Trash2, ChevronLeft, History,
} from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useAiChat, AiMessage, AiConversation } from "@/hooks/useAiChat";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import {
  ToolCallBubble,
  ToolResultBubble,
  ConfirmationCard,
  MessageBubble,
  getStreamingState,
} from "@/components/ai/chat-messages";

// ── Sidebar: Conversation History ────────────────────────────────────
function ConversationSidebar({
  conversations,
  activeId,
  onSelect,
  onDelete,
  onNew,
  collapsed,
}: {
  conversations: AiConversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNew: () => void;
  collapsed: boolean;
}) {
  const { t } = useLocale();
  if (collapsed) return null;

  return (
    <div className="w-64 shrink-0 border-r border-border/60 bg-surface/50 flex flex-col h-full">
      <div className="p-3 border-b border-border/60">
        <button
          onClick={onNew}
          className="flex w-full items-center gap-2 rounded-lg bg-accent-cyan/10 border border-accent-cyan/20 px-3 py-2 text-sm font-medium text-accent-cyan hover:bg-accent-cyan/20 transition-colors"
        >
          <Plus className="h-4 w-4" />
          {t("ai.new_chat")}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
        {conversations.length === 0 && (
          <p className="text-xs text-text-muted px-2 py-4 text-center">{t("ai.no_conversations")}</p>
        )}
        {conversations.map((c) => (
          <div
            key={c.conversation_id}
            className={cn(
              "group flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm cursor-pointer transition-colors",
              activeId === c.conversation_id
                ? "bg-accent-cyan/8 text-accent-cyan"
                : "text-text-secondary hover:bg-surface-hover hover:text-text-primary",
            )}
            onClick={() => onSelect(c.conversation_id)}
          >
            <History className="h-3.5 w-3.5 shrink-0" />
            <span className="truncate flex-1">{c.title || "New conversation"}</span>
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(c.conversation_id); }}
              className="opacity-0 group-hover:opacity-100 text-text-muted hover:text-red-400 transition-opacity"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────
export default function AiAssistantPage() {
  const { t } = useLocale();
  const {
    messages, isStreaming, error, conversationId,
    conversations, suggestions,
    sendMessage, confirmAction, newConversation,
    loadConversation, deleteConversation,
    loadConversations, loadSuggestions,
  } = useAiChat();

  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Load conversations and suggestions on mount
  useEffect(() => {
    loadConversations();
    loadSuggestions();
  }, [loadConversations, loadSuggestions]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Autofocus input
  useEffect(() => {
    inputRef.current?.focus();
  }, [conversationId]);

  const handleSubmit = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      if (!input.trim() || isStreaming) return;
      const msg = input;
      setInput("");
      sendMessage(msg);
    },
    [input, isStreaming, sendMessage],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e as unknown as FormEvent);
      }
    },
    [handleSubmit],
  );

  const handleSuggestion = useCallback(
    (prompt: string) => {
      sendMessage(prompt);
    },
    [sendMessage],
  );

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-[calc(100vh-3.5rem)] bg-background overflow-hidden">
      {/* Sidebar */}
      <ConversationSidebar
        conversations={conversations}
        activeId={conversationId}
        onSelect={loadConversation}
        onDelete={deleteConversation}
        onNew={newConversation}
        collapsed={!sidebarOpen}
      />

      {/* Main chat area */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <div className="flex items-center gap-3 border-b border-border/60 px-4 py-2.5">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="text-text-muted hover:text-text-primary transition-colors"
            title={sidebarOpen ? "Hide history" : "Show history"}
          >
            <ChevronLeft className={cn("h-4 w-4 transition-transform", !sidebarOpen && "rotate-180")} />
          </button>
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-cyan/10">
              <Sparkles className="h-4 w-4 text-accent-cyan" />
            </div>
            <h1 className="text-sm font-semibold">{t("ai.title")}</h1>
          </div>
          <span className="rounded bg-accent-cyan/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-accent-cyan">
            Beta
          </span>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-4">
          {isEmpty ? (
            <div className="flex flex-col items-center justify-center h-full max-w-lg mx-auto text-center">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent-cyan/10 mb-4">
                <Sparkles className="h-7 w-7 text-accent-cyan" />
              </div>
              <h2 className="text-lg font-semibold mb-2">{t("ai.welcome_title")}</h2>
              <p className="text-sm text-text-secondary mb-6">{t("ai.welcome_description")}</p>

              {suggestions.length > 0 && (
                <div className="flex flex-wrap gap-2 justify-center">
                  {suggestions.slice(0, 6).map((s, i) => (
                    <button
                      key={i}
                      onClick={() => handleSuggestion(s.prompt)}
                      className="rounded-full border border-border/60 bg-surface px-3.5 py-1.5 text-xs text-text-secondary hover:border-accent-cyan/30 hover:text-accent-cyan hover:bg-accent-cyan/5 transition-colors"
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <>
              <AnimatePresence initial={false}>
                {messages.map((msg) => {
                  const { streamingMsgId, executingToolIds } = getStreamingState(messages, isStreaming);
                  return (
                    <motion.div
                      key={msg.id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.15 }}
                    >
                      {msg.role === "tool_call" ? (
                        <ToolCallBubble msg={msg} isExecuting={executingToolIds.has(msg.id)} />
                      ) : msg.role === "tool_result" ? (
                        <ToolResultBubble msg={msg} />
                      ) : msg.role === "confirmation" ? (
                        <ConfirmationCard msg={msg} onConfirm={confirmAction} />
                      ) : (
                        <MessageBubble msg={msg} isLastStreaming={msg.id === streamingMsgId} />
                      )}
                    </motion.div>
                  );
                })}
              </AnimatePresence>
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="mx-4 mb-2 rounded-lg bg-red-500/10 border border-red-500/20 px-3 py-2 text-xs text-red-400">
            {error}
          </div>
        )}

        {/* Input */}
        <form onSubmit={handleSubmit} className="border-t border-border/60 px-4 py-3">
          <div className="relative flex items-end gap-2 max-w-3xl mx-auto">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t("ai.placeholder")}
              rows={1}
              className="flex-1 resize-none rounded-xl border border-border/60 bg-surface px-4 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:border-accent-cyan/40 focus:outline-none focus:ring-1 focus:ring-accent-cyan/20 max-h-32"
              style={{ minHeight: "2.5rem" }}
              disabled={isStreaming}
            />
            <button
              type="submit"
              disabled={isStreaming || !input.trim()}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent-cyan text-white hover:bg-accent-cyan/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {isStreaming ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </button>
          </div>
          <p className="text-center text-[10px] text-text-muted mt-2">
            {t("ai.disclaimer")}
          </p>
        </form>
      </div>
    </div>
  );
}
