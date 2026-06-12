// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { useHoldInteraction } from "@/hooks/useHoldInteraction";

function HoldProbe({ enabled = true }: { enabled?: boolean }) {
  const { progress, isHolding, isArmed, bind } = useHoldInteraction({
    enabled,
    durationMs: 200,
    moveTolerancePx: 10,
    onArmed: vi.fn(),
  });
  return (
    <button type="button" data-testid="hold-btn" data-progress={progress} data-holding={isHolding} data-armed={isArmed} {...bind}>
      hold
    </button>
  );
}

describe("useHoldInteraction", () => {
  beforeEach(() => {
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb: FrameRequestCallback) => {
      cb(performance.now() + 250);
      return 1;
    });
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("ignores non-primary buttons", () => {
    render(<HoldProbe />);
    const btn = screen.getByTestId("hold-btn");
    fireEvent.pointerDown(btn, { button: 2, pointerId: 1 });
    expect(btn.getAttribute("data-holding")).toBe("false");
  });

  it("does not arm when disabled", () => {
    render(<HoldProbe enabled={false} />);
    const btn = screen.getByTestId("hold-btn");
    fireEvent.pointerDown(btn, { button: 0, pointerId: 1, clientX: 0, clientY: 0 });
    expect(btn.getAttribute("data-armed")).toBe("false");
  });
});