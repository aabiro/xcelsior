"use client";

import { useState, useRef, useEffect, FormEvent, useCallback } from "react";
import {
  MessageCircle,
  X,
  Send,
  Trash2,
  ThumbsUp,
  ThumbsDown,
  Loader2,
  Volume2,
  VolumeX,
  User,
  History,
  ChevronLeft,
  Sparkles,
} from "lucide-react";
import { AnimatePresence, motion, useMotionValue, useTransform, PanInfo } from "framer-motion";
import { useChatStream, ChatMessage } from "@/hooks/useChatStream";
import { useTypewriterText } from "@/hooks/useTypewriterText";
import { useLocale } from "@/lib/locale";
import { useAuth } from "@/lib/auth";

// ── Notification Sound (inline base64 tiny blip) ─────────────────────
const NOTIFICATION_SOUND_URI =
  "data:audio/wav;base64,UklGRl4AAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YToAAAAYADAARABSAFoAXABYAE4AQAA0ACQAFgAIAP7/9v/w/+7/8P/0//j//P8AAAQACAAIAAQA/v/4/+z/5P/c/9j/2P/c/+T/7P/0//z/";

function playNotificationSound() {
  try {
    const audio = new Audio(NOTIFICATION_SOUND_URI);
    audio.volume = 0.3;
    void audio.play();
  } catch {
    // Audio not available
  }
}

// ── Markdown formatter ───────────────────────────────────────────────
function formatMarkdown(text: string): string {
  return text
    .replace(
      /```(\w*)\n?([\s\S]*?)```/g,
      '<pre class="bg-navy/50 rounded p-2 my-1 text-xs overflow-x-auto"><code>$2</code></pre>'
    )
    .replace(/`([^`]+)`/g, '<code class="bg-navy/50 rounded px-1 py-0.5 text-xs">$1</code>')
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener" class="text-accent-red underline">$1</a>'
    )
    .replace(/\n/g, "<br />");
}

// ── Message Bubble ───────────────────────────────────────────────────
function MessageBubble({
  msg,
  isLastStreaming,
  onFeedback,
  feedbackGiven,
}: {
  msg: ChatMessage;
  isLastStreaming?: boolean;
  onFeedback?: (id: string, vote: "up" | "down") => void;
  feedbackGiven?: "up" | "down" | null;
}) {
  const { t } = useLocale();
  const isUser = msg.role === "user";
  const { displayedText, isTyping } = useTypewriterText(msg.content, {
    animate: !isUser && Boolean(isLastStreaming),
    resetKey: msg.id,
  });
  const showStreamingCursor = !isUser && (Boolean(isLastStreaming) || isTyping);

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3 group`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "bg-accent-red text-white rounded-br-md"
            : "bg-surface-hover text-text-primary rounded-bl-md"
        }`}
      >
        {isUser ? (
          <p>{msg.content}</p>
        ) : msg.content ? (
          <>
            <div
              className="prose-sm prose-invert [&_pre]:my-1 [&_code]:text-xs"
              dangerouslySetInnerHTML={{ __html: formatMarkdown(displayedText) }}
            />
            {showStreamingCursor && (
              <span
                className="inline-block w-[2px] h-[1em] bg-accent-red rounded-full ml-0.5 align-text-bottom animate-cursor-blink"
                aria-hidden="true"
              />
            )}
            {/* Thumbs up/down feedback */}
            {onFeedback && (
              <div className="flex items-center gap-1 mt-1.5">
                <button
                  onClick={() => onFeedback(msg.id, "up")}
                  className={`rounded p-0.5 transition-colors ${
                    feedbackGiven === "up"
                      ? "text-green-400"
                      : "text-text-muted hover:text-green-400"
                  }`}
                  aria-label="Helpful"
                >
                  <ThumbsUp className="h-3 w-3" />
                </button>
                <button
                  onClick={() => onFeedback(msg.id, "down")}
                  className={`rounded p-0.5 transition-colors ${
                    feedbackGiven === "down"
                      ? "text-red-400"
                      : "text-text-muted hover:text-red-400"
                  }`}
                  aria-label="Not helpful"
                >
                  <ThumbsDown className="h-3 w-3" />
                </button>
                {feedbackGiven && (
                  <span className="text-[10px] text-text-muted ml-1">
                    {t("chat.feedback_thanks")}
                  </span>
                )}
              </div>
            )}
          </>
        ) : (
          <div className="flex items-center gap-1.5 text-text-muted">
            <Loader2 className="h-3 w-3 animate-spin" />
            <span className="text-xs">{t("chat.thinking")}</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Conversation History Drawer ──────────────────────────────────────
function HistoryDrawer({
  onClose,
  onSelect,
}: {
  onClose: () => void;
  onSelect: (conversationId: string) => void;
}) {
  const { t } = useLocale();
  const [conversations, setConversations] = useState<
    { conversation_id: string; preview: string; updated_at: number }[]
  >([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/chat/conversations", { credentials: "include" });
        if (res.ok) {
          const data = await res.json();
          setConversations(Array.isArray(data.conversations) ? data.conversations : []);
        }
      } catch {
        // No-op
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 border-b border-border px-4 py-3 bg-navy">
        <button
          onClick={onClose}
          className="rounded-lg p-1.5 text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <p className="text-sm font-semibold text-text-primary">{t("chat.history")}</p>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {loading ? (
          <div className="flex justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-text-muted" />
          </div>
        ) : conversations.length === 0 ? (
          <p className="text-xs text-text-muted text-center py-8">{t("chat.no_history")}</p>
        ) : (
          conversations.map((c) => (
            <button
              key={c.conversation_id}
              onClick={() => onSelect(c.conversation_id)}
              className="w-full text-left rounded-lg px-3 py-2.5 mb-1 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
            >
              <p className="truncate">{c.preview || "New conversation"}</p>
              <p className="text-[10px] text-text-muted mt-0.5">
                {new Date(c.updated_at * 1000).toLocaleDateString()}
              </p>
            </button>
          ))
        )}
      </div>
    </div>
  );
}

// ── Main Widget ──────────────────────────────────────────────────────
export function ChatWidget({ onOpenAiPanel }: { onOpenAiPanel?: () => void }) {
  const { t } = useLocale();
  const { user } = useAuth();
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [hasUnread, setHasUnread] = useState(false);
  const [feedback, setFeedback] = useState<Record<string, "up" | "down">>({});
  const historyLoadedRef = useRef(false);
  const prevMsgCountRef = useRef(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const { messages, isStreaming, error, sendMessage, clearChat, setMessages } = useChatStream();
  const lastStreamingAssistantId = isStreaming
    ? [...messages].reverse().find((msg) => msg.role === "assistant")?.id ?? null
    : null;

  // Mobile swipe-to-dismiss
  const dragY = useMotionValue(0);
  const panelOpacity = useTransform(dragY, [0, 200], [1, 0.3]);

  // Sound prefs from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem("xcelsior-chat-sound");
      if (stored === "false") setSoundEnabled(false);
    } catch { /* noop */ }
  }, []);

  const toggleSound = useCallback(() => {
    setSoundEnabled((prev) => {
      const next = !prev;
      try { localStorage.setItem("xcelsior-chat-sound", String(next)); } catch { /* noop */ }
      return next;
    });
  }, []);

  // Unread indicator — track new assistant messages when chat is closed
  useEffect(() => {
    if (!open && messages.length > prevMsgCountRef.current) {
      const lastMsg = messages[messages.length - 1];
      if (lastMsg?.role === "assistant" && lastMsg.content) {
        setHasUnread(true);
        if (soundEnabled) playNotificationSound();
      }
    }
    prevMsgCountRef.current = messages.length;
  }, [messages, open, soundEnabled]);

  // Clear unread when opening
  useEffect(() => {
    if (open) setHasUnread(false);
  }, [open]);

  // Fetch conversation history from server when panel opens
  const fetchHistory = useCallback(async () => {
    const convId =
      typeof window !== "undefined" ? localStorage.getItem("xcelsior-chat-conv-id") : null;
    if (!convId || historyLoadedRef.current || messages.length > 0) return;

    setLoadingHistory(true);
    try {
      const res = await fetch(`/api/chat/history/${encodeURIComponent(convId)}`, {
        credentials: "include",
      });
      if (res.ok) {
        const data = await res.json();
        if (data.ok && data.messages?.length) {
          const restored: ChatMessage[] = data.messages.map(
            (m: { role: string; content: string; timestamp: number }, i: number) => ({
              id: `hist-${i}`,
              role: m.role as "user" | "assistant",
              content: m.content,
              timestamp: m.timestamp * 1000,
            })
          );
          setMessages(restored);
        }
      } else if (res.status === 404) {
        localStorage.removeItem("xcelsior-chat-conv-id");
      }
    } catch {
      // Network error — no-op
    } finally {
      setLoadingHistory(false);
      historyLoadedRef.current = true;
    }
  }, [messages.length, setMessages]);

  useEffect(() => {
    if (open) void fetchHistory();
  }, [open, fetchHistory]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  // Focus input when opening
  useEffect(() => {
    if (open && !showHistory) {
      setTimeout(() => inputRef.current?.focus(), 200);
    }
  }, [open, showHistory]);

  // Keyboard shortcuts: Escape to close, Ctrl+/ to toggle
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Ctrl+/ or Cmd+/ to toggle chat
      if ((e.ctrlKey || e.metaKey) && e.key === "/") {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
      // Escape to close (only when chat is open)
      if (e.key === "Escape" && open) {
        e.preventDefault();
        setOpen(false);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;
    void sendMessage(input);
    setInput("");
  };



  const handleSuggestion = (text: string) => {
    void sendMessage(text);
  };

  const handleFeedback = useCallback((msgId: string, vote: "up" | "down") => {
    setFeedback((prev) => ({ ...prev, [msgId]: vote }));
    // Fire-and-forget feedback to server
    fetch("/api/chat/feedback", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message_id: msgId, vote }),
    }).catch((e) => console.error("Failed to send feedback", e));
  }, []);

  const handleTalkToHuman = () => {
    window.open("mailto:support@xcelsior.ca?subject=Chat%20Escalation", "_blank");
  };

  // Load a specific conversation from history drawer
  const handleSelectConversation = useCallback(
    async (conversationId: string) => {
      setShowHistory(false);
      setLoadingHistory(true);
      try {
        const res = await fetch(`/api/chat/history/${encodeURIComponent(conversationId)}`, {
          credentials: "include",
        });
        if (res.ok) {
          const data = await res.json();
          if (data.ok && data.messages?.length) {
            const restored: ChatMessage[] = data.messages.map(
              (m: { role: string; content: string; timestamp: number }, i: number) => ({
                id: `hist-sel-${i}`,
                role: m.role as "user" | "assistant",
                content: m.content,
                timestamp: m.timestamp * 1000,
              })
            );
            clearChat();
            setMessages(restored);
            try {
              localStorage.setItem("xcelsior-chat-conv-id", conversationId);
            } catch { /* noop */ }
          }
        }
      } catch {
        // No-op
      } finally {
        setLoadingHistory(false);
      }
    },
    [clearChat, setMessages]
  );

  // Mobile swipe-to-dismiss handler
  const handleDragEnd = (_: unknown, info: PanInfo) => {
    if (info.offset.y > 100) {
      setOpen(false);
    }
  };

  const suggestions = [
    t("chat.suggestion_gpu"),
    t("chat.suggestion_tiers"),
    t("chat.suggestion_billing"),
    t("chat.suggestion_job"),
  ];

  return (
    <>
      {/* Floating button */}
      <AnimatePresence>
        {!open && (
          <motion.button
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            transition={{ type: "spring", damping: 20, stiffness: 300 }}
            onClick={() => setOpen(true)}
            className="fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-accent-red text-white shadow-lg hover:bg-accent-red/90 transition-colors focus:outline-none focus:ring-2 focus:ring-accent-red focus:ring-offset-2"
            aria-label={t("chat.open")}
          >
            <MessageCircle className="h-6 w-6" />
            {/* Unread dot indicator */}
            {hasUnread && (
              <span className="absolute -top-0.5 -right-0.5 flex h-3.5 w-3.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-green-500" />
              </span>
            )}
          </motion.button>
        )}
      </AnimatePresence>

      {/* Chat panel */}
      <AnimatePresence>
        {open && (
          <motion.div
            ref={panelRef}
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            // Mobile: draggable to dismiss
            drag={typeof window !== "undefined" && window.innerWidth < 640 ? "y" : false}
            dragConstraints={{ top: 0 }}
            dragElastic={0.2}
            onDragEnd={handleDragEnd}
            style={{
              y: typeof window !== "undefined" && window.innerWidth < 640 ? dragY : undefined,
              opacity: typeof window !== "undefined" && window.innerWidth < 640 ? panelOpacity : undefined,
            }}
            className="fixed bottom-6 right-6 z-50 flex flex-col w-[360px] h-[500px] max-sm:inset-0 max-sm:w-full max-sm:h-full max-sm:bottom-0 max-sm:right-0 max-sm:rounded-none rounded-2xl border border-border bg-navy-light shadow-2xl overflow-hidden"
          >
            {/* Mobile swipe hint */}
            <div className="sm:hidden flex justify-center pt-2 pb-0">
              <div className="w-10 h-1 rounded-full bg-text-muted/30" />
            </div>

            {showHistory ? (
              <HistoryDrawer
                onClose={() => setShowHistory(false)}
                onSelect={handleSelectConversation}
              />
            ) : (
              <>
                {/* Header */}
                <div className="flex items-center justify-between border-b border-border px-4 py-3 bg-navy">
                  <div className="flex items-center gap-2">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-red">
                      <span className="text-sm font-bold text-white">X</span>
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-text-primary">{t("chat.title")}</p>
                      <p className="text-xs text-text-muted">{t("chat.subtitle")}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    {/* Sound toggle */}
                    <button
                      onClick={toggleSound}
                      className="rounded-lg p-1.5 text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors"
                      title={soundEnabled ? t("chat.sound_on") : t("chat.sound_off")}
                    >
                      {soundEnabled ? (
                        <Volume2 className="h-4 w-4" />
                      ) : (
                        <VolumeX className="h-4 w-4" />
                      )}
                    </button>
                    {/* History (authenticated users only) */}
                    {user && (
                      <button
                        onClick={() => setShowHistory(true)}
                        className="rounded-lg p-1.5 text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors"
                        title={t("chat.history")}
                      >
                        <History className="h-4 w-4" />
                      </button>
                    )}
                    {/* Open Xcel AI */}
                    {onOpenAiPanel && (
                      <button
                        onClick={() => { onOpenAiPanel(); setOpen(false); }}
                        className="rounded-lg p-1.5 text-text-muted hover:bg-accent-cyan/20 hover:text-accent-cyan transition-colors"
                        title={t("ai.open_panel")}
                      >
                        <Sparkles className="h-4 w-4" />
                      </button>
                    )}
                    {/* Clear */}
                    {messages.length > 0 && (
                      <button
                        onClick={() => {
                          clearChat();
                          historyLoadedRef.current = false;
                        }}
                        className="rounded-lg p-1.5 text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors"
                        title={t("chat.clear")}
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    )}
                    {/* Close */}
                    <button
                      onClick={() => setOpen(false)}
                      className="rounded-lg p-1.5 text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors"
                      aria-label={t("chat.close")}
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto px-4 py-3">
                  {loadingHistory ? (
                    <div className="flex flex-col items-center justify-center h-full text-center">
                      <Loader2 className="h-6 w-6 animate-spin text-accent-red mb-2" />
                      <p className="text-xs text-text-muted">{t("chat.loading_history")}</p>
                    </div>
                  ) : messages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-center">
                      {/* Pre-chat greeting */}
                      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-accent-red/10 mb-3">
                        <MessageCircle className="h-6 w-6 text-accent-red" />
                      </div>
                      <p className="text-sm font-medium text-text-primary mb-1">
                        {t("chat.greeting")}
                      </p>
                      <p className="text-xs text-text-muted mb-4">
                        {t("chat.greeting_sub")}
                      </p>
                      <p className="text-[10px] text-text-muted mt-4">{t("chat.shortcut_hint")}</p>
                    </div>
                  ) : (
                    <>
                      {messages.map((msg) => (
                        <MessageBubble
                          key={msg.id}
                          msg={msg}
                          isLastStreaming={msg.id === lastStreamingAssistantId}
                          onFeedback={msg.role === "assistant" ? handleFeedback : undefined}
                          feedbackGiven={feedback[msg.id] ?? null}
                        />
                      ))}
                      <div ref={messagesEndRef} />
                    </>
                  )}
                </div>

                {/* Error */}
                {error && (
                  <div className="mx-4 mb-2 rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-400">
                    {error}
                  </div>
                )}

                {/* Talk to a human + Input */}
                <div className="border-t border-border bg-navy">
                  {/* Talk to a human button */}
                  {messages.length >= 4 && (
                    <div className="px-4 pt-2">
                      <button
                        onClick={handleTalkToHuman}
                        className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-xs text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                      >
                        <User className="h-3 w-3" />
                        {t("chat.talk_to_human")}
                      </button>
                    </div>
                  )}
                  <form onSubmit={handleSubmit} className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <input
                        ref={inputRef}
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder={t("chat.placeholder")}
                        disabled={isStreaming}
                        className="flex-1 rounded-xl border border-border bg-surface-hover px-4 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:border-accent-red focus:outline-none focus:ring-1 focus:ring-accent-red disabled:opacity-50"
                        maxLength={2000}
                      />
                      <button
                        type="submit"
                        disabled={!input.trim() || isStreaming}
                        className="flex self-stretch aspect-square shrink-0 items-center justify-center rounded-xl bg-accent-red text-white hover:bg-accent-red/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                        aria-label={t("chat.send")}
                      >
                        {isStreaming ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Send className="h-4 w-4" />
                        )}
                      </button>
                    </div>
                    <p className="mt-1.5 text-center text-[10px] text-text-muted">
                      {t("chat.disclaimer")}
                    </p>
                  </form>
                  {/* Suggestion chips — below input, visible only in empty chat */}
                  {messages.length === 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
                      className="flex flex-wrap gap-2 justify-center px-4 pb-3"
                    >
                      {suggestions.map((s) => (
                        <button
                          key={s}
                          onClick={() => handleSuggestion(s)}
                          className="rounded-full border border-border px-3 py-1.5 text-xs text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                        >
                          {s}
                        </button>
                      ))}
                    </motion.div>
                  )}
                </div>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
