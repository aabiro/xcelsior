"use client";

import { useEffect, useRef, useState, FormEvent, useCallback } from "react";
import { Plus, Sparkles, X, ExternalLink, PanelRightClose } from "lucide-react";
import Link from "next/link";
import { AnimatePresence, motion } from "framer-motion";
import { useAiChat } from "@/hooks/useAiChat";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import {
  ToolCallBubble,
  ToolResultBubble,
  ConfirmationCard,
  MessageBubble,
  EmptyState,
  ChatInput,
  getStreamingState,
} from "@/components/ai/chat-messages";

// ── AiPanel: Premium side panel ──────────────────────────────────────
export function AiPanel({ onClose }: { onClose: () => void }) {
  const { t } = useLocale();
  const {
    messages, isStreaming, error,
    suggestions,
    sendMessage, confirmAction, newConversation,
    loadSuggestions,
  } = useAiChat();

  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { loadSuggestions(); }, [loadSuggestions]);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);
  useEffect(() => { inputRef.current?.focus(); }, []);

  const handleSubmit = useCallback(() => {
    if (!input.trim() || isStreaming) return;
    const msg = input;
    setInput("");
    sendMessage(msg);
  }, [input, isStreaming, sendMessage]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const isEmpty = messages.length === 0;
  const { streamingMsgId, executingToolIds } = getStreamingState(messages, isStreaming);

  return (
    <div className="flex flex-col h-full bg-background/95 backdrop-blur-md ai-panel-border">
      {/* Header — glass style */}
      <div className="flex items-center justify-between border-b border-border/30 px-3 py-2.5 bg-surface/30 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-gradient-to-br from-accent-cyan/20 to-accent-violet/10">
            <Sparkles className="h-3 w-3 text-accent-cyan" />
          </div>
          <span className="text-xs font-semibold">{t("ai.title")}</span>
          <span className="rounded-full bg-accent-cyan/8 px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-widest text-accent-cyan/70">
            Beta
          </span>
        </div>
        <div className="flex items-center gap-0.5">
          <button
            onClick={newConversation}
            className="rounded-md p-1.5 text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors"
            title={t("ai.new_chat")}
          >
            <Plus className="h-3.5 w-3.5" />
          </button>
          <Link
            href="/dashboard/ai"
            onClick={onClose}
            className="rounded-md p-1.5 text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors"
            title={t("ai.open_full")}
          >
            <ExternalLink className="h-3.5 w-3.5" />
          </Link>
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-text-muted hover:bg-red-500/10 hover:text-red-400 transition-colors"
            title={t("ai.close_panel")}
          >
            <PanelRightClose className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 py-3">
        {isEmpty ? (
          <EmptyState
            title={t("ai.welcome_title")}
            description={t("ai.panel_description")}
            suggestions={suggestions}
            onSuggestion={sendMessage}
            compact
          />
        ) : (
          <>
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.15, ease: "easeOut" }}
                >
                  {msg.role === "tool_call" ? (
                    <ToolCallBubble msg={msg} isExecuting={executingToolIds.has(msg.id)} compact />
                  ) : msg.role === "tool_result" ? (
                    <ToolResultBubble msg={msg} compact />
                  ) : msg.role === "confirmation" ? (
                    <ConfirmationCard msg={msg} onConfirm={confirmAction} compact />
                  ) : (
                    <MessageBubble msg={msg} isLastStreaming={msg.id === streamingMsgId} compact />
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Error */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="mx-3 mb-2 rounded-lg bg-red-500/8 border border-red-500/15 px-2.5 py-1.5 text-[11px] text-red-400"
        >
          {error}
        </motion.div>
      )}

      {/* Input */}
      <ChatInput
        value={input}
        onChange={setInput}
        onSubmit={handleSubmit}
        onKeyDown={handleKeyDown}
        isStreaming={isStreaming}
        placeholder={t("ai.placeholder")}
        disclaimer={t("ai.disclaimer")}
        inputRef={inputRef}
        compact
      />
    </div>
  );
}
