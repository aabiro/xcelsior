"use client";

import { useEffect, useRef, useState, useMemo, FormEvent, useCallback } from "react";
import { Send, Plus, Loader2, Sparkles, X, ExternalLink } from "lucide-react";
import Link from "next/link";
import { AnimatePresence, motion } from "framer-motion";
import { useAiChat } from "@/hooks/useAiChat";
import { useLocale } from "@/lib/locale";
import {
  ToolCallBubble,
  ToolResultBubble,
  ConfirmationCard,
  MessageBubble,
  getStreamingState,
} from "@/components/ai/chat-messages";

// ── AiPanel: Embeddable context panel for layout sidebar ─────────────
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

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/60 px-3 py-2">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-md bg-accent-cyan/10">
            <Sparkles className="h-3.5 w-3.5 text-accent-cyan" />
          </div>
          <span className="text-xs font-semibold">{t("ai.title")}</span>
          <span className="rounded bg-accent-cyan/10 px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-accent-cyan">
            Beta
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={newConversation}
            className="rounded-md p-1 text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors"
            title={t("ai.new_chat")}
          >
            <Plus className="h-3.5 w-3.5" />
          </button>
          <Link
            href="/dashboard/ai"
            className="rounded-md p-1 text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors"
            title={t("ai.open_full")}
          >
            <ExternalLink className="h-3.5 w-3.5" />
          </Link>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors"
            title={t("ai.close_panel")}
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 py-3">
        {isEmpty ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-cyan/10 mb-3">
              <Sparkles className="h-5 w-5 text-accent-cyan" />
            </div>
            <h3 className="text-sm font-semibold mb-1">{t("ai.welcome_title")}</h3>
            <p className="text-[11px] text-text-secondary mb-4 leading-relaxed">{t("ai.panel_description")}</p>
            {suggestions.length > 0 && (
              <div className="flex flex-wrap gap-1.5 justify-center">
                {suggestions.slice(0, 4).map((s, i) => (
                  <button
                    key={i}
                    onClick={() => sendMessage(s.prompt)}
                    className="rounded-full border border-border/60 bg-surface px-2.5 py-1 text-[10px] text-text-secondary hover:border-accent-cyan/30 hover:text-accent-cyan hover:bg-accent-cyan/5 transition-colors"
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
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.12 }}
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
                );
              })}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mx-3 mb-2 rounded-lg bg-red-500/10 px-2.5 py-1.5 text-[11px] text-red-400">
          {error}
        </div>
      )}

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t border-border/60 px-3 py-2">
        <div className="flex items-end gap-1.5">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t("ai.placeholder")}
            rows={1}
            className="flex-1 resize-none rounded-lg border border-border/60 bg-surface px-3 py-2 text-xs text-text-primary placeholder:text-text-muted focus:border-accent-cyan/40 focus:outline-none focus:ring-1 focus:ring-accent-cyan/20 max-h-20"
            style={{ minHeight: "2rem" }}
            disabled={isStreaming}
          />
          <button
            type="submit"
            disabled={isStreaming || !input.trim()}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent-cyan text-white hover:bg-accent-cyan/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {isStreaming ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Send className="h-3.5 w-3.5" />
            )}
          </button>
        </div>
        <p className="text-center text-[9px] text-text-muted mt-1">
          {t("ai.disclaimer")}
        </p>
      </form>
    </div>
  );
}
