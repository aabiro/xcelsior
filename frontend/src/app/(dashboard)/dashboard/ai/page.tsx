"use client";

import { useEffect, useRef, useState, FormEvent, useCallback } from "react";
import {
  Send, Plus, Loader2, Sparkles, Trash2, ChevronLeft,
  History, Wrench, CheckCircle2, XCircle, AlertTriangle,
} from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useAiChat, AiMessage, AiConversation } from "@/hooks/useAiChat";
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

// ── Tool Call Indicator ──────────────────────────────────────────────
function ToolCallBubble({ msg }: { msg: AiMessage }) {
  const { t } = useLocale();
  const toolLabel = msg.toolName?.replace(/_/g, " ") || "tool";
  return (
    <div className="flex justify-start mb-2">
      <div className="flex items-center gap-2 rounded-lg bg-accent-cyan/5 border border-accent-cyan/20 px-3 py-1.5 text-xs text-accent-cyan">
        <Wrench className="h-3 w-3" />
        <span>{t("ai.tool_calling")}: <strong>{toolLabel}</strong></span>
        {msg.toolInput && Object.keys(msg.toolInput).length > 0 && (
          <code className="ml-1 text-[10px] text-text-muted">
            {JSON.stringify(msg.toolInput).slice(0, 80)}
          </code>
        )}
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

// ── Confirmation Card ────────────────────────────────────────────────
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
      <div className="max-w-[85%] rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
        <div className="flex items-center gap-2 text-amber-400 mb-2">
          <AlertTriangle className="h-4 w-4" />
          <span className="text-sm font-medium">{t("ai.confirmation_required")}</span>
        </div>
        <p className="text-sm text-text-secondary mb-1">
          {t("ai.wants_to")}: <strong className="text-text-primary">{toolLabel}</strong>
        </p>
        {msg.toolInput && Object.keys(msg.toolInput).length > 0 && (
          <pre className="bg-navy/50 rounded p-2 my-2 text-xs border border-border/40 overflow-x-auto">
            {JSON.stringify(msg.toolInput, null, 2)}
          </pre>
        )}
        {!decided && (
          <div className="flex gap-2 mt-3">
            <button
              onClick={() => handleDecision(true)}
              className="flex items-center gap-1.5 rounded-lg bg-green-500/10 border border-green-500/30 px-3 py-1.5 text-xs font-medium text-green-400 hover:bg-green-500/20 transition-colors"
            >
              <CheckCircle2 className="h-3.5 w-3.5" />
              {t("ai.approve")}
            </button>
            <button
              onClick={() => handleDecision(false)}
              className="flex items-center gap-1.5 rounded-lg bg-red-500/10 border border-red-500/30 px-3 py-1.5 text-xs font-medium text-red-400 hover:bg-red-500/20 transition-colors"
            >
              <XCircle className="h-3.5 w-3.5" />
              {t("ai.reject")}
            </button>
          </div>
        )}
        {decided && (
          <p className="text-xs text-text-muted mt-2 italic">{t("ai.action_submitted")}</p>
        )}
      </div>
    </div>
  );
}

// ── Message Bubble ───────────────────────────────────────────────────
function MessageBubble({ msg }: { msg: AiMessage }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
          isUser
            ? "bg-accent-cyan text-white rounded-br-md"
            : "bg-surface-hover text-text-primary rounded-bl-md",
        )}
      >
        {isUser ? (
          <p>{msg.content}</p>
        ) : msg.content ? (
          <div
            className="prose-sm prose-invert [&_pre]:my-1 [&_code]:text-xs"
            dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.content) }}
          />
        ) : (
          <span className="inline-flex gap-1">
            <span className="h-2 w-2 rounded-full bg-accent-cyan/60 animate-bounce [animation-delay:0ms]" />
            <span className="h-2 w-2 rounded-full bg-accent-cyan/60 animate-bounce [animation-delay:150ms]" />
            <span className="h-2 w-2 rounded-full bg-accent-cyan/60 animate-bounce [animation-delay:300ms]" />
          </span>
        )}
      </div>
    </div>
  );
}

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
                {messages.map((msg) => (
                  <motion.div
                    key={msg.id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.15 }}
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
