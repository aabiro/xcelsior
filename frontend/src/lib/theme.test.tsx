import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { ThemeProvider, useTheme } from "@/lib/theme";

const storage: Record<string, string> = {};
beforeEach(() => {
  Object.keys(storage).forEach((k) => delete storage[k]);
  vi.spyOn(Storage.prototype, "getItem").mockImplementation((k) => storage[k] ?? null);
  vi.spyOn(Storage.prototype, "setItem").mockImplementation((k, v) => {
    storage[k] = v;
  });
});

function renderThemeHook() {
  return renderHook(() => useTheme(), { wrapper: ThemeProvider });
}

describe("useTheme", () => {
  it("defaults to dark", () => {
    const { result } = renderThemeHook();
    expect(result.current.theme).toBe("dark");
  });

  it("toggles between dark and light", () => {
    const { result } = renderThemeHook();
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe("light");
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe("dark");
  });

  it("reads stored theme from localStorage", () => {
    storage["xcelsior-theme"] = "light";
    const { result } = renderThemeHook();
    expect(result.current.theme).toBe("light");
  });
});
