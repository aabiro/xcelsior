import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PickerSelect } from "@/components/ui/picker-select";

// iOS draws a native select as a see-through glass menu; in the deploy sheet the
// page showed through it, and an empty one opened a "No Options" bubble (owner
// review, 2026-11-22). On a phone the list is drawn by the page, solid.

function phone(matches: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches, media: query, addEventListener: () => {}, removeEventListener: () => {},
  }));
}
afterEach(() => vi.unstubAllGlobals());

const native = (select: Parameters<React.ComponentProps<typeof PickerSelect>["renderNative"]>[0]) => <select {...select} />;
const options = [{ value: "ca-on", label: "Ontario" }, { value: "ca-qc", label: "Quebec" }];

describe("PickerSelect", () => {
  it("is the native select on a wide screen", () => {
    phone(false);
    render(<PickerSelect id="r" value="ca-on" options={options} onChange={vi.fn()} renderNative={native} />);
    expect(screen.getByRole("combobox")).toHaveValue("ca-on");
  });

  it("is a solid list on a phone, and picks on a tap", () => {
    phone(true);
    const onChange = vi.fn();
    render(<PickerSelect id="r" value="ca-on" options={options} onChange={onChange} renderNative={native} />);
    expect(screen.queryByRole("combobox")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Ontario" }));
    const list = screen.getByRole("listbox");
    expect(list.className).toContain("bg-[var(--popover-solid)]");
    fireEvent.click(screen.getByRole("option", { name: "Quebec" }));
    expect(onChange).toHaveBeenCalledWith("ca-qc");
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("does not open an empty list on a phone", () => {
    phone(true);
    render(<PickerSelect id="r" value="" options={[]} placeholder="Choose a GPU first" onChange={vi.fn()} renderNative={native} />);
    const trigger = screen.getByRole("button", { name: "Choose a GPU first" });
    expect(trigger).toBeDisabled();
  });
});
