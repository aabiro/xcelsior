import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { AI_PANEL_SPRING } from "@/lib/ai-panel-transition";

const push = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push, replace: vi.fn(), back: vi.fn() }),
}));

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({ t: (key: string) => key }),
}));

vi.mock("@/hooks/useAiChat", () => ({
  useAiChat: () => ({
    messages: [],
    isStreaming: false,
    error: null,
    conversationId: null,
    conversations: [],
    suggestions: [],
    sendMessage: vi.fn(),
    confirmAction: vi.fn(),
    newConversation: vi.fn(),
    loadConversation: vi.fn(),
    deleteConversation: vi.fn(),
    loadConversations: vi.fn(),
    loadSuggestions: vi.fn(),
  }),
}));

vi.mock("@/components/ai/xcel-ai-onboarding", () => ({
  useAiOnboardingGate: () => ({ showOnboarding: false, dismissOnboarding: vi.fn() }),
  XcelAiOnboarding: () => null,
}));

vi.mock("@/components/ai/chat-messages", () => ({
  ToolCallBubble: () => null,
  ToolResultBubble: () => null,
  ConfirmationCard: () => null,
  MessageBubble: () => null,
  EmptyState: () => null,
  ChatInput: () => null,
  getStreamingState: () => ({ streamingMsgId: null, executingToolIds: new Set() }),
}));

vi.mock("framer-motion", () => ({
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>{children}</div>,
  },
}));

const AI_PANEL_KEY = "xcelsior-ai-panel-open";

describe("Xcel AI moveHere", () => {
  beforeEach(() => {
    push.mockClear();
    localStorage.clear();
    localStorage.setItem(AI_PANEL_KEY, "true");
  });

  it("closes the side panel and navigates to full-page chat", async () => {
    const closeHandler = vi.fn();
    window.addEventListener("xcelsior-close-ai-panel", closeHandler);

    const { default: AiAssistantPage } = await import("./page");
    render(<AiAssistantPage />);

    fireEvent.click(screen.getByRole("button", { name: /move it here/i }));

    expect(localStorage.getItem(AI_PANEL_KEY)).toBe("false");
    expect(closeHandler).toHaveBeenCalled();
    expect(push).toHaveBeenCalledWith("/dashboard/ai");

    window.removeEventListener("xcelsior-close-ai-panel", closeHandler);
  });

  it("wires shared spring transition between panel placeholder and full-page chat", async () => {
    const pageSource = readFileSync(resolve(__dirname, "page.tsx"), "utf8");
    const shellSource = readFileSync(
      resolve(__dirname, "../../dashboard-shell.tsx"),
      "utf8",
    );

    expect(pageSource).toContain('from "@/lib/ai-panel-transition"');
    expect(pageSource).toContain("AnimatePresence mode=\"wait\"");
    expect(pageSource).toContain("key=\"panel-active-placeholder\"");
    expect(pageSource).toContain("key=\"full-page-chat\"");
    expect(pageSource).toContain("AI_PANEL_SPRING");
    expect(pageSource).toContain("AI_PANEL_CROSSFADE");
    expect(shellSource).toContain("transition={AI_PANEL_SPRING}");
    expect(AI_PANEL_SPRING).toMatchObject({ damping: 28, stiffness: 320, mass: 0.85 });
  });
});