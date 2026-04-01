"use client";

import { useEffect, useRef, useState, FormEvent, useCallback } from "react";
import {
  Plus, Sparkles, Trash2, ChevronLeft, History,
  PanelRight, MessageSquare,
} from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { useAiChat, AiMessage, AiConversation } from "@/hooks/useAiChat";
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

const AI_PANEL_KEY = "xcelsior-ai-panel-open";

// ── Sidebar: Conversation History (animated) ─────────────────────────
function ConversationSidebar({
  conversations,
  activeId,
  onSelect,
  onDelete,
  onNew,
  open,
  onToggle,
}: {
  conversations: AiConversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNew: () => void;
  open: boolean;
  onToggle: () => void;
}) {
  const { t } = useLocale();

  return (
    <>
      {/* Toggle button (always visible) */}
      <button
        onClick={onToggle}
        className={cn(
          "absolute top-3 z-10 rounded-lg p-1.5 text-text-muted hover:text-text-primary hover:bg-surface-hover transition-all duration-200",
          open ? "left-[15.5rem]" : "left-3",
        )}
        title={open ? "Hide history" : "Show history"}
      >
        <ChevronLeft className={cn("h-4 w-4 transition-transform duration-200", !open && "rotate-180")} />
      </button>

      {/* Sidebar drawer */}
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 256, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ type: "spring", damping: 28, stiffness: 320 }}
            className="shrink-0 border-r border-border/30 bg-surface/30 backdrop-blur-sm flex flex-col h-full overflow-hidden"
          >
            <div className="p-3 border-b border-border/30">
              <button
                onClick={onNew}
                className="flex w-full items-center gap-2 rounded-xl bg-accent-cyan/8 border border-accent-cyan/15 px-3 py-2 text-sm font-medium text-accent-cyan hover:bg-accent-cyan/15 hover:border-accent-cyan/25 transition-all duration-200"
              >
                <Plus className="h-4 w-4" />
                {t("ai.new_chat")}
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
              {conversations.length === 0 && (
                <div className="flex flex-col items-center gap-2 py-8 text-text-muted">
                  <MessageSquare className="h-5 w-5 opacity-40" />
                  <p className="text-xs">{t("ai.no_conversations")}</p>
                </div>
              )}
              {conversations.map((c) => (
                <motion.div
                  key={c.conversation_id}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  className={cn(
                    "group flex items-center gap-2 rounded-xl px-2.5 py-2 text-sm cursor-pointer transition-all duration-150",
                    activeId === c.conversation_id
                      ? "bg-accent-cyan/8 text-accent-cyan border border-accent-cyan/15"
                      : "text-text-secondary hover:bg-surface-hover hover:text-text-primary border border-transparent",
                  )}
                  onClick={() => onSelect(c.conversation_id)}
                >
                  <History className="h-3.5 w-3.5 shrink-0 opacity-60" />
                  <span className="truncate flex-1 text-xs">{c.title || "New conversation"}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); onDelete(c.conversation_id); }}
                    className="opacity-0 group-hover:opacity-100 text-text-muted hover:text-red-400 transition-all duration-150"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

// ── Main Page ────────────────────────────────────────────────────────
export default function AiAssistantPage() {
  const { t } = useLocale();
  const router = useRouter();
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

  useEffect(() => {
    loadConversations();
    loadSuggestions();
  }, [loadConversations, loadSuggestions]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [conversationId]);

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

  const pinToPanel = useCallback(() => {
    try { localStorage.setItem(AI_PANEL_KEY, "true"); } catch {}
    window.dispatchEvent(new CustomEvent("xcelsior-open-ai-panel"));
    router.push("/dashboard");
  }, [router]);

  const isEmpty = messages.length === 0;
  const { streamingMsgId, executingToolIds } = getStreamingState(messages, isStreaming);

  return (
    <div className="flex h-[calc(100vh-3.5rem)] bg-background overflow-hidden relative">
      {/* Conversation sidebar */}
      <ConversationSidebar
        conversations={conversations}
        activeId={conversationId}
        onSelect={loadConversation}
        onDelete={deleteConversation}
        onNew={newConversation}
        open={sidebarOpen}
        onToggle={() => setSidebarOpen(v => !v)}
      />

      {/* Main chat area */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border/30 px-4 py-2.5 bg-surface/20 backdrop-blur-sm">
          <div className={cn("flex items-center gap-2.5", sidebarOpen ? "ml-0" : "ml-8")}>
            <div className="flex h-7 w-7 items-center justify-center rounded-xl bg-gradient-to-br from-accent-cyan/20 to-accent-violet/10">
              <Sparkles className="h-3.5 w-3.5 text-accent-cyan" />
            </div>
            <h1 className="text-sm font-semibold">{t("ai.title")}</h1>
            <span className="rounded-full bg-accent-cyan/8 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-widest text-accent-cyan/70">
              Beta
            </span>
          </div>

          {/* Pin to right panel button */}
          <button
            onClick={pinToPanel}
            className={cn(
              "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs text-text-muted",
              "border border-border/30 bg-surface/50 backdrop-blur-sm",
              "hover:bg-accent-cyan/5 hover:text-accent-cyan hover:border-accent-cyan/20",
              "transition-all duration-200",
            )}
            title="Pin to side panel"
          >
            <PanelRight className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Pin to side</span>
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-4">
            {isEmpty ? (
              <div className="flex items-center justify-center h-[calc(100vh-12rem)]">
                <EmptyState
                  title={t("ai.welcome_title")}
                  description={t("ai.welcome_description")}
                  suggestions={suggestions}
                  onSuggestion={sendMessage}
                />
              </div>
            ) : (
              <>
                <AnimatePresence initial={false}>
                  {messages.map((msg) => (
                    <motion.div
                      key={msg.id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.18, ease: "easeOut" }}
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
                  ))}
                </AnimatePresence>
                <div ref={messagesEndRef} />
              </>
            )}
          </div>
        </div>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 4 }}
              className="mx-4 mb-2 max-w-3xl mx-auto rounded-xl bg-red-500/8 border border-red-500/15 px-3 py-2 text-xs text-red-400"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

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
        />
      </div>
    </div>
  );
}
