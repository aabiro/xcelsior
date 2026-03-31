"use client";

import { useEffect, useRef, useState, FormEvent, useCallback } from "react";
import {
  Send, Plus, Loader2, Sparkles, Wrench, CheckCircle2,
  XCircle, AlertTriangle, X, ExternalLink,
} from "lucide-react";
import Link from "next/link";
import { AnimatePresence, motion } from "framer-motion";
import { useAiChat, AiMessage } from "@/hooks/useAiChat";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";

// ── Markdown formatter ───────────────────────────────────────────────
function formatMarkdown(text: string): string {
  return text
    .replace(
      /```(\w*)\n?([\s\S]*?)```/g,
      '<pre class="bg-navy/50 rounded-lg p-3 my-2 text-xs overflow-x-auto border border-border/40"><code>$2</code></pre>',
    )
    .replace(/`([^`]+)`/g, '<code class="bg-navy/50 rounded px-1.5 py-0.5 text-xs font-mono">$1</code>')
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener" class="text-accent-cyan underline hover:text-accent-cyan/80">$1</a>',
    )
    .replace(/\n/g, "<br />");
}

function ToolCallBubble({ msg }: { msg: AiMessage }) {
  const { t } = useLocale();
  const toolLabel = msg.toolName?.replace(/_/g, " ") || "tool";
  return (
    <div className="flex justify-start mb-2">
      <div className="flex items-center gap-2 rounded-lg bg-accent-cyan/5 border border-accent-cyan/20 px-3 py-1.5 text-xs text-accent-cyan">
        <Wrench className="h-3 w-3" />
        <span>{t("ai.tool_calling")}: <strong>{toolLabel}</strong></span>
      </div>
    </div>
  );
}

function ToolResultBubble({ msg }: { msg: AiMessage }) {
  return (
    <div className="flex justify-start mb-2">
      <div className="flex items-center gap-2 rounded-lg bg-green-500/5 border border-green-500/20 px-3 py-1.5 text-xs text-green-400">
        <CheckCircle2 className="h-3 w-3" />
        <span>{msg.toolName?.replace(/_/g, " ")} — result received</span>
      </div>
    </div>
  );
}

function ConfirmationCard({ msg, onConfirm }: {
  msg: AiMessage;
  onConfirm: (confirmationId: string, approved: boolean) => void;
}) {
  const { t } = useLocale();
  const [decided, setDecided] = useState(false);
  const toolLabel = msg.toolName?.replace(/_/g, " ") || "action";

  const handleDecision = (approved: boolean) => {
    if (decided || !msg.confirmationId) return;
    setDecided(true);
    onConfirm(msg.confirmationId, approved);
  };

  return (
    <div className="flex justify-start mb-3">
      <div className="max-w-[95%] rounded-xl border border-amber-500/30 bg-amber-500/5 p-3">
        <div className="flex items-center gap-2 text-amber-400 mb-2">
          <AlertTriangle className="h-3.5 w-3.5" />
          <span className="text-xs font-medium">{t("ai.confirmation_required")}</span>
        </div>
        <p className="text-xs text-text-secondary mb-1">
          {t("ai.wants_to")}: <strong className="text-text-primary">{toolLabel}</strong>
        </p>
        {!decided && (
          <div className="flex gap-2 mt-2">
            <button
              onClick={() => handleDecision(true)}
              className="flex items-center gap-1 rounded-lg bg-green-500/10 border border-green-500/30 px-2.5 py-1 text-[11px] font-medium text-green-400 hover:bg-green-500/20 transition-colors"
            >
              <CheckCircle2 className="h-3 w-3" />
              {t("ai.approve")}
            </button>
            <button
              onClick={() => handleDecision(false)}
              className="flex items-center gap-1 rounded-lg bg-red-500/10 border border-red-500/30 px-2.5 py-1 text-[11px] font-medium text-red-400 hover:bg-red-500/20 transition-colors"
            >
              <XCircle className="h-3 w-3" />
              {t("ai.reject")}
            </button>
          </div>
        )}
        {decided && (
          <p className="text-[11px] text-text-muted mt-1 italic">{t("ai.action_submitted")}</p>
        )}
      </div>
    </div>
  );
}

function MessageBubble({ msg }: { msg: AiMessage }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-2`}>
      <div
        className={cn(
          "max-w-[90%] rounded-2xl px-3 py-2 text-xs leading-relaxed",
          isUser
            ? "bg-accent-cyan text-white rounded-br-md"
            : "bg-surface-hover text-text-primary rounded-bl-md",
        )}
      >
        {isUser ? (
          <p>{msg.content}</p>
        ) : msg.content ? (
          <div
            className="prose-xs prose-invert [&_pre]:my-1 [&_code]:text-[10px]"
            dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.content) }}
          />
        ) : (
          <span className="inline-flex gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-accent-cyan/60 animate-bounce [animation-delay:0ms]" />
            <span className="h-1.5 w-1.5 rounded-full bg-accent-cyan/60 animate-bounce [animation-delay:150ms]" />
            <span className="h-1.5 w-1.5 rounded-full bg-accent-cyan/60 animate-bounce [animation-delay:300ms]" />
          </span>
        )}
      </div>
    </div>
  );
}

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
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.12 }}
                >
                  {msg.role === "tool_call" ? (
                    <ToolCallBubble msg={msg} />
                  ) : msg.role === "tool_result" ? (
                    <ToolResultBubble msg={msg} />
                  ) : msg.role === "confirmation" ? (
                    <ConfirmationCard msg={msg} onConfirm={confirmAction} />
                  ) : (
                    <MessageBubble msg={msg} />
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
