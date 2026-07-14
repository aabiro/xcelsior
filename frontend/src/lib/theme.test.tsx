import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { ThemeProvider, useTheme, applyTheme, syncThemeFromStorage } from "@/lib/theme";

const storage: Record<string, string> = {};

beforeEach(() => {
  Object.keys(storage).forEach((k) => delete storage[k]);
  vi.spyOn(Storage.prototype, "getItem").mockImplementation((k) => storage[k] ?? null);
  vi.spyOn(Storage.prototype, "setItem").mockImplementation((k, v) => {
    storage[k] = v;
  });
  document.documentElement.className = "";
  document.documentElement.removeAttribute("data-theme");
});

afterEach(() => {
  vi.restoreAllMocks();
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

describe("applyTheme", () => {
  it("sets html class, data-theme, color-scheme, and storage", () => {
    applyTheme("light");
    expect(document.documentElement.classList.contains("light")).toBe(true);
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(document.documentElement.style.colorScheme).toBe("light");
    expect(storage["xcelsior-theme"]).toBe("light");
  });

  it("syncThemeFromStorage re-applies after external localStorage change", () => {
    applyTheme("dark");
    storage["xcelsior-theme"] = "light";
    const theme = syncThemeFromStorage();
    expect(theme).toBe("light");
    expect(document.documentElement.classList.contains("light")).toBe(true);
    expect(document.documentElement.dataset.theme).toBe("light");
  });
});

describe("ThemeProvider re-sync", () => {
  it("reconciles when localStorage changes on window focus (marketing→dashboard handoff)", () => {
    const { result } = renderThemeHook();
    expect(result.current.theme).toBe("dark");

    storage["xcelsior-theme"] = "light";
    act(() => {
      window.dispatchEvent(new Event("focus"));
    });

    expect(result.current.theme).toBe("light");
    expect(document.documentElement.classList.contains("light")).toBe(true);
  });
});