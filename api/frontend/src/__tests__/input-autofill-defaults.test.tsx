import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Input, NumberInput, TextArea } from "@/components/ui/input";

// Search boxes and resource names had no autocomplete token, so phones offered
// saved names and addresses on them (owner, 2026-11-23). Fields that should be
// filled in pass their own token.
describe("shared fields", () => {
  it("turn autofill off unless the field asks for it", () => {
    render(<>
      <Input aria-label="search" />
      <Input aria-label="name" autoComplete="name" />
      <TextArea aria-label="notes" />
      <NumberInput aria-label="count" value={1} />
    </>);
    expect(screen.getByLabelText("search")).toHaveAttribute("autocomplete", "off");
    expect(screen.getByLabelText("name")).toHaveAttribute("autocomplete", "name");
    expect(screen.getByLabelText("notes")).toHaveAttribute("autocomplete", "off");
    expect(screen.getByLabelText("count")).toHaveAttribute("autocomplete", "off");
  });

  it("steps a number on a tap, not only on a mouse press", () => {
    const onChange = vi.fn();
    render(<NumberInput aria-label="count" value={2} onChange={onChange} min={0} max={5} />);
    fireEvent.focus(screen.getByLabelText("count"));
    fireEvent.click(screen.getByRole("button", { name: "Increase value" }));
    expect(onChange).toHaveBeenLastCalledWith(3);
    fireEvent.click(screen.getByRole("button", { name: "Decrease value" }));
    expect(onChange).toHaveBeenLastCalledWith(1);
  });
});
