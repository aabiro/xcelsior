import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useTypewriterText } from "@/hooks/useTypewriterText";

describe("useTypewriterText", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("reveals text progressively when animation is enabled", () => {
    const { result } = renderHook(() =>
      useTypewriterText("Hello world", { animate: true, resetKey: "msg-1" }),
    );

    expect(result.current.displayedText).toBe("");
    expect(result.current.isTyping).toBe(true);

    act(() => {
      vi.advanceTimersByTime(26);
    });

    expect(result.current.displayedText.length).toBeGreaterThan(0);
    expect(result.current.displayedText).not.toBe("Hello world");

    for (let i = 0; i < 20; i += 1) {
      act(() => {
        vi.advanceTimersByTime(26);
      });
    }

    expect(result.current.displayedText).toBe("Hello world");
    expect(result.current.isTyping).toBe(false);
  });

  it("shows full text immediately when animation is disabled", () => {
    const { result } = renderHook(() =>
      useTypewriterText("Immediate response", { animate: false, resetKey: "msg-2" }),
    );

    expect(result.current.displayedText).toBe("Immediate response");
    expect(result.current.isTyping).toBe(false);
  });
});
