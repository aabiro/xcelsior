"use client";

import { useState } from "react";
import {
  Wrench, CheckCircle2, XCircle, AlertTriangle,
  Loader2, ChevronDown, Sparkles, User, Send,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import type { AiMessage } from "@/hooks/useAiChat";
import { useTypewriterText } from "@/hooks/useTypewriterText";

// ── Markdown formatter ───────────────────────────────────────────────

export function formatMarkdown(text: string): string {
  return text
    .replace(
      /```(\w*)\n?([\s\S]*?)```/g,
      '<pre class="bg-navy/60 rounded-lg p-3 my-2.5 text-xs overflow-x-auto border border-border/30 backdrop-blur-sm"><code>$2</code></pre>',
    )
    .replace(
      /`([^`]+)`/g,
      '<code class="bg-navy/40 rounded px-1.5 py-0.5 text-xs font-mono border border-border/20">$1</code>',
    )
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-accent-cyan underline decoration-accent-cyan/30 hover:decoration-accent-cyan/80 transition-colors">$1</a>',
    )
    .replace(/\n/g, "<br />");
}

// ── AI Avatar ────────────────────────────────────────────────────────

function AiAvatar({ compact }: { compact?: boolean }) {
  const size = compact ? "h-5 w-5" : "h-6 w-6";
  const icon = compact ? "h-2.5 w-2.5" : "h-3 w-3";
  return (
    <div className={cn(
      "shrink-0 rounded-lg bg-gradient-to-br from-accent-cyan/20 to-accent-violet/20 flex items-center justify-center ring-1 ring-accent-cyan/20",
      size,
    )}>
      <Sparkles className={cn("text-accent-cyan", icon)} />
    </div>
  );
}

function UserAvatar({ compact }: { compact?: boolean }) {
  const size = compact ? "h-5 w-5" : "h-6 w-6";
  const icon = compact ? "h-2.5 w-2.5" : "h-3 w-3";
  return (
    <div className={cn(
      "shrink-0 rounded-lg bg-accent-cyan/10 flex items-center justify-center",
      size,
    )}>
      <User className={cn("text-accent-cyan/70", icon)} />
    </div>
  );
}

// ── Tool Call Bubble ─────────────────────────────────────────────────

export function ToolCallBubble({
  msg,
  isExecuting,
  compact,
}: {
  msg: AiMessage;
  isExecuting?: boolean;
  compact?: boolean;
}) {
  const { t } = useLocale();
  const toolLabel = msg.toolName?.replace(/_/g, " ") || "tool";

  return (
    <div className={cn("flex items-start gap-2", compact ? "mb-1.5 pl-7" : "mb-2 pl-8")}>
      <div
        className={cn(
          "flex items-center gap-2 rounded-lg border px-2.5 py-1 text-[11px] transition-all duration-300",
          isExecuting
            ? "bg-accent-cyan/8 border-accent-cyan/25 text-accent-cyan shadow-[0_0_12px_rgba(0,212,255,0.06)]"
            : "bg-surface/50 border-border/40 text-text-muted",
        )}
      >
        {isExecuting ? (
          <Loader2 className="h-3 w-3 animate-spin" />
        ) : (
          <Wrench className="h-3 w-3" />
        )}
        <span>
          {isExecuting ? t("ai.tool_calling") : "Used"}:{" "}
          <span className="font-medium text-text-secondary">{toolLabel}</span>
        </span>
      </div>
    </div>
  );
}

// ── Tool Result Bubble (expandable) ──────────────────────────────────

export function ToolResultBubble({
  msg,
  compact,
}: {
  msg: AiMessage;
  compact?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const hasData = msg.toolOutput && Object.keys(msg.toolOutput).length > 0;
  const hasError = hasData && "error" in (msg.toolOutput ?? {});

  return (
    <div className={cn("flex items-start gap-2", compact ? "mb-1.5 pl-7" : "mb-2 pl-8")}>
      <div
        className={cn(
          "rounded-lg border overflow-hidden max-w-[90%] transition-all duration-200",
          hasError
            ? "bg-red-500/5 border-red-500/15"
            : "bg-emerald/5 border-emerald/15",
        )}
      >
        <button
          onClick={() => hasData && setExpanded(!expanded)}
          className={cn(
            "flex items-center gap-2 px-2.5 py-1 text-[11px] w-full text-left transition-colors",
            hasError ? "text-red-400/80" : "text-emerald/80",
            hasData && "hover:bg-white/[0.02] cursor-pointer",
          )}
          disabled={!hasData}
        >
          <CheckCircle2 className="h-3 w-3 shrink-0" />
          <span className="truncate">
            {msg.toolName?.replace(/_/g, " ")} —{" "}
            {hasError ? "error" : "done"}
          </span>
          {hasData && (
            <ChevronDown
              className={cn(
                "h-3 w-3 shrink-0 ml-auto transition-transform duration-200",
                expanded && "rotate-180",
              )}
            />
          )}
        </button>
        <AnimatePresence>
          {expanded && hasData && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="border-t border-inherit overflow-hidden"
            >
              <pre
                className={cn(
                  "px-3 py-2 text-[10px] max-h-48 overflow-auto font-mono leading-relaxed",
                  hasError ? "text-red-300/60" : "text-text-muted/80",
                )}
              >
                {JSON.stringify(msg.toolOutput, null, 2)}
              </pre>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// ── Confirmation Card ────────────────────────────────────────────────

export function ConfirmationCard({
  msg,
  onConfirm,
  compact,
}: {
  msg: AiMessage;
  onConfirm: (confirmationId: string, approved: boolean) => void;
  compact?: boolean;
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
    <div className={cn("flex items-start gap-2", compact ? "mb-2 pl-7" : "mb-3 pl-8")}>
      <div
        className={cn(
          "rounded-xl border border-amber-500/20 bg-amber-500/[0.03] backdrop-blur-sm",
          compact ? "max-w-[95%] p-3" : "max-w-[85%] p-4",
        )}
      >
        <div className="flex items-center gap-2 text-amber-400 mb-2">
          <AlertTriangle className={compact ? "h-3.5 w-3.5" : "h-4 w-4"} />
          <span className={cn("font-medium", compact ? "text-xs" : "text-sm")}>
            {t("ai.confirmation_required")}
          </span>
        </div>
        <p className={cn("text-text-secondary mb-1", compact ? "text-xs" : "text-sm")}>
          {t("ai.wants_to")}:{" "}
          <strong className="text-text-primary">{toolLabel}</strong>
        </p>
        {!compact && msg.toolInput && Object.keys(msg.toolInput).length > 0 && (
          <pre className="bg-navy/50 rounded-lg p-2.5 my-2 text-xs border border-border/30 overflow-x-auto">
            {JSON.stringify(msg.toolInput, null, 2)}
          </pre>
        )}
        {!decided && (
          <div className={cn("flex gap-2", compact ? "mt-2" : "mt-3")}>
            <button
              onClick={() => handleDecision(true)}
              className={cn(
                "flex items-center gap-1.5 rounded-lg bg-emerald/10 border border-emerald/25 font-medium text-emerald hover:bg-emerald/20 transition-all duration-200",
                compact ? "px-2.5 py-1 text-[11px]" : "px-3 py-1.5 text-xs",
              )}
            >
              <CheckCircle2 className="h-3 w-3" />
              {t("ai.approve")}
            </button>
            <button
              onClick={() => handleDecision(false)}
              className={cn(
                "flex items-center gap-1.5 rounded-lg bg-red-500/10 border border-red-500/25 font-medium text-red-400 hover:bg-red-500/20 transition-all duration-200",
                compact ? "px-2.5 py-1 text-[11px]" : "px-3 py-1.5 text-xs",
              )}
            >
              <XCircle className="h-3 w-3" />
              {t("ai.reject")}
            </button>
          </div>
        )}
        {decided && (
          <p className={cn("text-text-muted italic", compact ? "text-[11px] mt-1" : "text-xs mt-2")}>
            {t("ai.action_submitted")}
          </p>
        )}
      </div>
    </div>
  );
}

// ── Message Bubble (with avatar + typewriter cursor) ─────────────────

export function MessageBubble({
  msg,
  isLastStreaming,
  compact,
}: {
  msg: AiMessage;
  isLastStreaming?: boolean;
  compact?: boolean;
}) {
  const isUser = msg.role === "user";
  const dotSize = compact ? "h-1 w-1" : "h-1.5 w-1.5";
  const { displayedText, isTyping } = useTypewriterText(msg.content, {
    animate: !isUser && Boolean(isLastStreaming),
    resetKey: msg.id,
  });
  const showStreamingCursor = !isUser && (Boolean(isLastStreaming) || isTyping);

  return (
    <div
      className={cn(
        "flex gap-2",
        isUser ? "flex-row-reverse" : "flex-row",
        compact ? "mb-2.5" : "mb-4",
      )}
    >
      {/* Avatar */}
      {isUser ? <UserAvatar compact={compact} /> : <AiAvatar compact={compact} />}

      {/* Bubble */}
      <div
        className={cn(
          "rounded-2xl leading-relaxed min-w-0",
          compact ? "max-w-[85%] px-3 py-2 text-xs" : "max-w-[80%] px-4 py-2.5 text-sm",
          isUser
            ? "bg-accent-cyan text-white rounded-br-md"
            : "bg-surface-hover/80 text-text-primary rounded-bl-md border border-border/30",
        )}
      >
        {isUser ? (
          <p>{msg.content}</p>
        ) : msg.content ? (
          <div className="relative">
            <div
              className={cn(
                "prose-invert [&_pre]:my-1 [&_a]:text-accent-cyan",
                compact ? "prose-xs [&_code]:text-[10px]" : "prose-sm [&_code]:text-xs",
              )}
              dangerouslySetInnerHTML={{ __html: formatMarkdown(displayedText) }}
            />
            {showStreamingCursor && (
              <span
                className="inline-block w-[2px] h-[1em] bg-accent-cyan rounded-full ml-0.5 align-text-bottom animate-cursor-blink"
                aria-hidden="true"
              />
            )}
          </div>
        ) : (
          /* Typing indicator — three pulsing dots */
          <span className="inline-flex items-center gap-1 py-0.5">
            <span className={cn("rounded-full bg-accent-cyan/50 animate-ai-pulse", dotSize)} />
            <span className={cn("rounded-full bg-accent-cyan/50 animate-ai-pulse [animation-delay:200ms]", dotSize)} />
            <span className={cn("rounded-full bg-accent-cyan/50 animate-ai-pulse [animation-delay:400ms]", dotSize)} />
          </span>
        )}
      </div>
    </div>
  );
}

// ── Helper: detect which messages are in an executing state ──────────

export function getStreamingState(messages: AiMessage[], isStreaming: boolean) {
  const streamingMsgId = isStreaming
    ? messages.filter((m) => m.role === "assistant").pop()?.id ?? null
    : null;

  const executingToolIds = new Set<string>();
  if (isStreaming) {
    const resultNames = new Set(
      messages.filter((m) => m.role === "tool_result").map((m) => m.toolName),
    );
    for (const m of messages) {
      if (m.role === "tool_call" && m.toolName && !resultNames.has(m.toolName)) {
        executingToolIds.add(m.id);
      }
    }
  }

  return { streamingMsgId, executingToolIds };
}

// ── Empty State ──────────────────────────────────────────────────────

export function EmptyState({
  title,
  description,
  suggestions,
  onSuggestion,
  compact,
}: {
  title: string;
  description: string;
  suggestions: { label: string; prompt: string }[];
  onSuggestion: (prompt: string) => void;
  compact?: boolean;
}) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      {/* Animated icon */}
      <div className={cn(
        "relative flex items-center justify-center mb-4",
        compact ? "h-12 w-12" : "h-16 w-16",
      )}>
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-accent-cyan/20 via-accent-violet/10 to-accent-cyan/5 animate-ai-glow" />
        <div className={cn(
          "relative rounded-2xl bg-gradient-to-br from-accent-cyan/15 to-accent-violet/10 flex items-center justify-center backdrop-blur-sm",
          compact ? "h-10 w-10" : "h-14 w-14",
        )}>
          <Sparkles className={cn("text-accent-cyan", compact ? "h-5 w-5" : "h-7 w-7")} />
        </div>
      </div>

      <h3 className={cn("font-semibold mb-1.5", compact ? "text-sm" : "text-lg")}>{title}</h3>
      <p className={cn("text-text-secondary mb-5 max-w-md leading-relaxed", compact ? "text-[11px]" : "text-sm")}>{description}</p>

      {suggestions.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
          className="flex flex-wrap gap-2 justify-center max-w-lg"
        >
          {suggestions.slice(0, compact ? 4 : 6).map((s) => (
            <button
              key={`${s.label}-${s.prompt}`}
              onClick={() => onSuggestion(s.prompt)}
              className={cn(
                "rounded-full border border-border/50 bg-surface/60 text-text-secondary backdrop-blur-sm",
                "hover:border-accent-cyan/30 hover:text-accent-cyan hover:bg-accent-cyan/5 hover:shadow-[0_0_12px_rgba(0,212,255,0.06)]",
                "transition-all duration-200",
                compact ? "px-2.5 py-1 text-[10px]" : "px-3.5 py-1.5 text-xs",
              )}
            >
              {s.label}
            </button>
          ))}
        </motion.div>
      )}
    </div>
  );
}

// ── Chat Input Bar ───────────────────────────────────────────────────

export function ChatInput({
  value,
  onChange,
  onSubmit,
  onKeyDown,
  isStreaming,
  placeholder,
  disclaimer,
  compact,
  inputRef,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  isStreaming: boolean;
  placeholder: string;
  disclaimer: string;
  compact?: boolean;
  inputRef?: React.RefObject<HTMLTextAreaElement | null>;
}) {
  return (
    <div className={cn(
      "border-t border-border/40",
      compact ? "px-3 py-2" : "px-4 py-3",
    )}>
      <div className={cn(
        "relative flex items-end gap-2",
        !compact && "max-w-3xl mx-auto",
      )}>
        <div className={cn(
          "flex-1 relative rounded-xl border border-border/50 bg-surface/80 backdrop-blur-sm",
          "focus-within:border-accent-cyan/40 focus-within:ring-1 focus-within:ring-accent-cyan/15 focus-within:shadow-[0_0_20px_rgba(0,212,255,0.04)]",
          "transition-all duration-200",
        )}>
          <textarea
            ref={inputRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={placeholder}
            rows={1}
            className={cn(
              "w-full resize-none bg-transparent text-text-primary placeholder:text-text-muted/60 focus:outline-none",
              compact
                ? "px-3 py-2 text-xs max-h-20"
                : "px-4 py-2.5 text-sm max-h-32",
            )}
            style={{ minHeight: compact ? "2rem" : "2.5rem" }}
            disabled={isStreaming}
          />
        </div>
        <button
          onClick={onSubmit}
          disabled={isStreaming || !value.trim()}
          className={cn(
            "shrink-0 flex items-center justify-center rounded-xl bg-accent-cyan text-white",
            "hover:bg-accent-cyan/90 hover:shadow-[0_0_16px_rgba(0,212,255,0.2)]",
            "disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none",
            "transition-all duration-200",
            compact ? "h-8 w-8" : "h-10 w-10",
          )}
        >
          {isStreaming ? (
            <Loader2 className={cn("animate-spin", compact ? "h-3.5 w-3.5" : "h-4 w-4")} />
          ) : (
            <Send className={cn(compact ? "h-3.5 w-3.5" : "h-4 w-4")} />
          )}
        </button>
      </div>
      <p className={cn(
        "text-center text-text-muted/60 mt-1.5",
        compact ? "text-[8px]" : "text-[10px]",
      )}>
        {disclaimer}
      </p>
    </div>
  );
}
