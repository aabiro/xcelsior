import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { CodeBlock } from "@/components/ui/code-block";

const writeText = vi.fn();
Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });

afterEach(() => writeText.mockReset());

describe("CodeBlock", () => {
  it("is a label bar with Copy over coloured code, and no window dots", () => {
    const { container } = render(<CodeBlock filename="terminal" code="npm install @xcelsior-gpu/sdk" />);
    const card = container.querySelector(".code-card")!;
    expect(card.querySelector(".code-card-label")?.textContent).toBe("terminal");
    expect(screen.getByRole("button", { name: "Copy terminal" })).toBeTruthy();
    // The bar holds the label and the Copy button and nothing else.
    expect(card.querySelector(".code-card-bar")?.children).toHaveLength(2);
    expect(card.querySelector(".rounded-full")).toBeNull();
    expect(card.querySelector(".tok-keyword")?.textContent).toBe("npm");
  });

  it("says Copied once the code is on the clipboard, and tells the caller", async () => {
    writeText.mockResolvedValue(undefined);
    const onCopy = vi.fn();
    render(<CodeBlock filename="request.sh" code="curl -s https://example.test" onCopy={onCopy} />);
    fireEvent.click(screen.getByRole("button", { name: "Copy request.sh" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Copied" })).toBeTruthy());
    expect(writeText).toHaveBeenCalledWith("curl -s https://example.test");
    expect(onCopy).toHaveBeenCalledOnce();
  });

  it("does not claim a copy the clipboard refused", async () => {
    writeText.mockRejectedValue(new Error("denied"));
    const onCopy = vi.fn();
    render(<CodeBlock filename="config.json" code="{}" onCopy={onCopy} />);
    fireEvent.click(screen.getByRole("button", { name: "Copy config.json" }));
    await waitFor(() => expect(writeText).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: "Copied" })).toBeNull();
    expect(onCopy).not.toHaveBeenCalled();
  });
});
