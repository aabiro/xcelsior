import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AddressLookup } from "@/components/ui/province-selector";

// Chrome ignores autocomplete="off" for address autofill and drew its saved-address
// picker over our own results (2026-11-18). An unrecognized token is what it honours.
describe("address search", () => {
  it("opts out of browser and password-manager autofill", () => {
    render(<AddressLookup country="CA" onSelect={vi.fn()} />);
    const input = screen.getByLabelText("Search address");
    const token = input.getAttribute("autocomplete") ?? "";
    expect(token).not.toBe("off");
    expect(token).not.toMatch(/^(on|street-address|address-line\d|address-level\d|postal-code|country|name)$/);
    expect(token.length).toBeGreaterThan(0);
    expect(input).toHaveAttribute("data-1p-ignore");
    expect(input).toHaveAttribute("data-lpignore", "true");
  });

  // Safari ignores an unknown token and guesses from the field's name and type,
  // so it kept offering a saved contact address on iPhone (owner, 2026-11-23).
  it("gives Safari nothing to guess an address from", () => {
    render(<AddressLookup country="CA" onSelect={vi.fn()} />);
    const input = screen.getByLabelText("Search address");
    expect(input).toHaveAttribute("type", "search");
    const name = input.getAttribute("name") ?? "";
    expect(name).toMatch(/search/);
    expect(name).not.toMatch(/addr|street|postal|city/i);
    expect(input.getAttribute("autocomplete") ?? "").not.toMatch(/addr/i);
  });
});
