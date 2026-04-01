"use client";

import { useState } from "react";
import {
  Wrench, CheckCircle2, XCircle, AlertTriangle,
  Loader2, ChevronDown,
} from "lucide-react";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import type { AiMessage } from "@/hooks/useAiChat";

// ── Markdown formatter ───────────────────────────────────────────────

export function formatMarkdown(text: string): string {
  return text
    .replace(
      /```(\w*)\n?([\s\S]*?)```/g,
      '<pre class="bg-navy/50 rounded-lg p-3 my-2 text-xs overflow-x-auto border border-border/40"><code>$2</code></pre>',
    )
    .replace(
      /`([^`]+)`/g,
      '<code class="bg-navy/50 rounded px-1.5 py-0.5 text-xs font-mono">$1</code>',
    )
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-accent-cyan underline hover:text-accent-cyan/80">$1</a>',
    )
    .replace(/\n/g, "<br />");
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
    <div className="flex justify-start mb-2">
      <div
        className={cn(
          "flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs",
          isExecuting
            ? "bg-accent-cyan/8 border-accent-cyan/30 text-accent-cyan"
            : "bg-accent-cyan/5 border-accent-cyan/20 text-accent-cyan/80",
        )}
      >
        {isExecuting ? (
          <Loader2 className="h-3 w-3 animate-spin" />
        ) : (
          <Wrench className="h-3 w-3" />
        )}
        <span>
          {isExecuting ? t("ai.tool_calling") : "Used"}:{" "}
          <strong>{toolLabel}</strong>
        </span>
        {!isExecuting && msg.toolInput && Object.keys(msg.toolInput).length > 0 && (
          <code className="ml-1 text-[10px] text-text-muted truncate max-w-[200px]">
            {JSON.stringify(msg.toolInput).slice(0, 60)}
            {JSON.stringify(msg.toolInput).length > 60 ? "…" : ""}
          </code>
        )}
      </div>
    </div>
  );
}

// ── Tool Result Bubble (expandable) ──────────────────────────────────

export function ToolResultBubble({
  msg,
}: {
  msg: AiMessage;
  compact?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const hasData = msg.toolOutput && Object.keys(msg.toolOutput).length > 0;
  const hasError = hasData && "error" in (msg.toolOutput ?? {});

  return (
    <div className="flex justify-start mb-2">
      <div
        className={cn(
          "rounded-lg border overflow-hidden max-w-[95%]",
          hasError
            ? "bg-red-500/5 border-red-500/20"
            : "bg-green-500/5 border-green-500/20",
        )}
      >
        <button
          onClick={() => hasData && setExpanded(!expanded)}
          className={cn(
            "flex items-center gap-2 px-3 py-1.5 text-xs w-full text-left",
            hasError ? "text-red-400" : "text-green-400",
            hasData && "hover:bg-white/[0.02] transition-colors cursor-pointer",
          )}
          disabled={!hasData}
        >
          <CheckCircle2 className="h-3 w-3 shrink-0" />
          <span className="truncate">
            {msg.toolName?.replace(/_/g, " ")} —{" "}
            {hasError ? "error" : "result received"}
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
        {expanded && hasData && (
          <div className="border-t border-inherit">
            <pre
              className={cn(
                "px-3 py-2 text-[10px] max-h-48 overflow-auto font-mono leading-relaxed",
                hasError ? "text-red-300/70" : "text-text-muted",
              )}
            >
              {JSON.stringify(msg.toolOutput, null, 2)}
            </pre>
          </div>
        )}
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
    <div className="flex justify-start mb-3">
      <div
        className={cn(
          "rounded-xl border border-amber-500/30 bg-amber-500/5",
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
          <pre className="bg-navy/50 rounded p-2 my-2 text-xs border border-border/40 overflow-x-auto">
            {JSON.stringify(msg.toolInput, null, 2)}
          </pre>
        )}
        {!decided && (
          <div className={cn("flex gap-2", compact ? "mt-2" : "mt-3")}>
            <button
              onClick={() => handleDecision(true)}
              className={cn(
                "flex items-center gap-1.5 rounded-lg bg-green-500/10 border border-green-500/30 font-medium text-green-400 hover:bg-green-500/20 transition-colors",
                compact ? "px-2.5 py-1 text-[11px]" : "px-3 py-1.5 text-xs",
              )}
            >
              <CheckCircle2 className="h-3 w-3" />
              {t("ai.approve")}
            </button>
            <button
              onClick={() => handleDecision(false)}
              className={cn(
                "flex items-center gap-1.5 rounded-lg bg-red-500/10 border border-red-500/30 font-medium text-red-400 hover:bg-red-500/20 transition-colors",
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

// ── Message Bubble (with typewriter cursor) ──────────────────────────

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
  const dotSize = compact ? "h-1.5 w-1.5" : "h-2 w-2";

  return (
    <div
      className={cn(
        "flex",
        isUser ? "justify-end" : "justify-start",
        compact ? "mb-2" : "mb-3",
      )}
    >
      <div
        className={cn(
          "rounded-2xl leading-relaxed",
          compact ? "max-w-[90%] px-3 py-2 text-xs" : "max-w-[85%] px-4 py-2.5 text-sm",
          isUser
            ? "bg-accent-cyan text-white rounded-br-md"
            : "bg-surface-hover text-text-primary rounded-bl-md",
        )}
      >
        {isUser ? (
          <p>{msg.content}</p>
        ) : msg.content ? (
          <div className="relative">
            <div
              className={cn(
                "prose-invert [&_pre]:my-1",
                compact ? "prose-xs [&_code]:text-[10px]" : "prose-sm [&_code]:text-xs",
              )}
              dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.content) }}
            />
            {isLastStreaming && (
              <span
                className="inline-block w-[2px] h-[1em] bg-accent-cyan rounded-full ml-0.5 align-text-bottom animate-cursor-blink"
                aria-hidden="true"
              />
            )}
          </div>
        ) : (
          <span className="inline-flex gap-1">
            <span className={cn("rounded-full bg-accent-cyan/60 animate-bounce [animation-delay:0ms]", dotSize)} />
            <span className={cn("rounded-full bg-accent-cyan/60 animate-bounce [animation-delay:150ms]", dotSize)} />
            <span className={cn("rounded-full bg-accent-cyan/60 animate-bounce [animation-delay:300ms]", dotSize)} />
          </span>
        )}
      </div>
    </div>
  );
}

// ── Helper: detect which messages are in an executing state ──────────

export function getStreamingState(messages: AiMessage[], isStreaming: boolean) {
  // Find the last assistant message ID (the one being streamed)
  const streamingMsgId = isStreaming
    ? messages.filter((m) => m.role === "assistant").pop()?.id ?? null
    : null;

  // Find tool_calls that don't yet have a matching tool_result
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
