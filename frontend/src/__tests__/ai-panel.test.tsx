/**
 * Frontend tests for AiPanel component rendering and interactions.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { AiMessage, AiConversation } from "@/hooks/useAiChat";

// ── Module-level mock with mutable state ─────────────────────────────
const _m = {
  messages: [] as AiMessage[],
  isStreaming: false,
  error: null as string | null,
  conversationId: null as string | null,
  conversations: [] as AiConversation[],
  suggestions: [] as { label: string; prompt: string }[],
  sendMessage: vi.fn(),
  confirmAction: vi.fn(),
  newConversation: vi.fn(),
  loadConversation: vi.fn(),
  deleteConversation: vi.fn(),
  loadConversations: vi.fn(),
  loadSuggestions: vi.fn(),
};

vi.mock("@/hooks/useAiChat", () => ({
  useAiChat: () => _m,
}));

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({
    t: (key: string) => key,
    locale: "en",
    setLocale: vi.fn(),
  }),
}));

vi.mock("@/lib/utils", () => ({
  cn: (...args: unknown[]) => args.filter(Boolean).join(" "),
}));

vi.mock("framer-motion", () => ({
  AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
  motion: {
    div: ({ children }: Record<string, unknown>) => <div>{children as React.ReactNode}</div>,
    aside: ({ children }: Record<string, unknown>) => <aside>{children as React.ReactNode}</aside>,
  },
}));

vi.mock("next/link", () => ({
  default: ({ children, href, ...rest }: { children: React.ReactNode; href: string; [k: string]: unknown }) => (
    <a href={href} {...(rest as React.AnchorHTMLAttributes<HTMLAnchorElement>)}>{children}</a>
  ),
}));

import { AiPanel } from "@/components/AiPanel";

// ─────────────────────────────────────────────────────────────────────
describe("AiPanel component", () => {
  beforeEach(() => {
    _m.messages = [];
    _m.isStreaming = false;
    _m.error = null;
    _m.conversationId = null;
    _m.conversations = [];
    _m.suggestions = [];
    _m.sendMessage = vi.fn();
    _m.confirmAction = vi.fn();
    _m.newConversation = vi.fn();
    _m.loadConversation = vi.fn();
    _m.deleteConversation = vi.fn();
    _m.loadConversations = vi.fn();
    _m.loadSuggestions = vi.fn();
  });

  afterEach(() => { vi.clearAllMocks(); });

  function renderPanel() {
    const onClose = vi.fn();
    const result = render(<AiPanel onClose={onClose} />);
    return { ...result, onClose };
  }

  it("renders empty landing state with welcome title", () => {
    renderPanel();
    expect(screen.getByText("ai.welcome_title")).toBeInTheDocument();
    expect(screen.getByText("ai.panel_description")).toBeInTheDocument();
  });

  it("renders suggestions in empty state", () => {
    _m.suggestions = [
      { label: "Rent GPUs", prompt: "I want to rent" },
      { label: "Provide GPUs", prompt: "I want to provide" },
    ];
    renderPanel();
    expect(screen.getByText("Rent GPUs")).toBeInTheDocument();
    expect(screen.getByText("Provide GPUs")).toBeInTheDocument();
  });

  it("calls sendMessage when suggestion chip clicked", () => {
    _m.suggestions = [{ label: "Test chip", prompt: "test-prompt" }];
    renderPanel();
    fireEvent.click(screen.getByText("Test chip"));
    expect(_m.sendMessage).toHaveBeenCalledWith("test-prompt");
  });

  it("renders user and assistant messages", () => {
    _m.messages = [
      { id: "1", role: "user", content: "Hello AI", timestamp: 1000 },
      { id: "2", role: "assistant", content: "Hello human", timestamp: 1001 },
    ];
    renderPanel();
    expect(screen.getByText("Hello AI")).toBeInTheDocument();
    expect(screen.getByText((c) => c.includes("Hello human"))).toBeInTheDocument();
  });

  it("renders tool_call bubble", () => {
    _m.messages = [
      { id: "1", role: "tool_call", content: "", toolName: "get_billing_summary", timestamp: 1000 },
    ];
    renderPanel();
    expect(screen.getByText(/get billing summary/)).toBeInTheDocument();
  });

  it("renders tool_result bubble", () => {
    _m.messages = [
      { id: "1", role: "tool_result", content: "", toolName: "search_marketplace", timestamp: 1000 },
    ];
    renderPanel();
    expect(screen.getByText(/search marketplace/)).toBeInTheDocument();
  });

  it("renders confirmation card with approve/reject buttons", () => {
    _m.messages = [{
      id: "1", role: "confirmation", content: "", toolName: "launch_job",
      toolInput: { gpu: "A100" }, confirmationId: "conf-1", timestamp: 1000,
    }];
    renderPanel();
    expect(screen.getByText("ai.confirmation_required")).toBeInTheDocument();
    expect(screen.getByText("ai.approve")).toBeInTheDocument();
    expect(screen.getByText("ai.reject")).toBeInTheDocument();
  });

  it("calls confirmAction when approve clicked", () => {
    _m.messages = [{
      id: "1", role: "confirmation", content: "", toolName: "launch_job",
      confirmationId: "conf-1", timestamp: 1000,
    }];
    renderPanel();
    fireEvent.click(screen.getByText("ai.approve"));
    expect(_m.confirmAction).toHaveBeenCalledWith("conf-1", true);
  });

  it("calls confirmAction(false) when reject clicked", () => {
    _m.messages = [{
      id: "1", role: "confirmation", content: "", toolName: "stop_job",
      confirmationId: "conf-2", timestamp: 1000,
    }];
    renderPanel();
    fireEvent.click(screen.getByText("ai.reject"));
    expect(_m.confirmAction).toHaveBeenCalledWith("conf-2", false);
  });

  it("shows error message when error state is set", () => {
    _m.error = "Something went wrong";
    renderPanel();
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("disables input when streaming", () => {
    _m.isStreaming = true;
    renderPanel();
    expect(screen.getByPlaceholderText("ai.placeholder")).toBeDisabled();
  });

  it("calls onClose when close button clicked", () => {
    const { onClose } = renderPanel();
    fireEvent.click(screen.getByTitle("ai.close_panel"));
    expect(onClose).toHaveBeenCalled();
  });

  it("calls newConversation when new chat button clicked", () => {
    renderPanel();
    fireEvent.click(screen.getByTitle("ai.new_chat"));
    expect(_m.newConversation).toHaveBeenCalled();
  });

  it("has a link to full AI page", () => {
    renderPanel();
    expect(screen.getByTitle("ai.open_full")).toHaveAttribute("href", "/dashboard/ai");
  });

  it("renders Beta badge", () => {
    renderPanel();
    expect(screen.getByText("Beta")).toBeInTheDocument();
  });

  it("renders disclaimer text", () => {
    renderPanel();
    expect(screen.getByText("ai.disclaimer")).toBeInTheDocument();
  });

  it("shows loading dots for empty assistant message", () => {
    _m.messages = [{ id: "1", role: "assistant", content: "", timestamp: 1000 }];
    const { container } = renderPanel();
    expect(container.querySelectorAll(".animate-bounce").length).toBe(3);
  });
});
