import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { StrictMode } from "react";
import { renderHook, render, act } from "@testing-library/react";
import { ThemeProvider, useTheme, applyTheme, syncThemeFromStorage, type Theme } from "@/lib/theme";

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

describe("ThemeProvider hydration safety", () => {
  // Regression: the shells render `data-theme={theme}`. If the provider's first
  // render reads localStorage, the client's initial value ("light") disagrees
  // with the server's ("dark"); React keeps the server markup and the corrective
  // mount effect becomes a no-op, so the shell stays data-theme="dark" while
  // <html> is light. The first committed render MUST be the SSR-safe default and
  // then reconcile — that guarantees a real re-render that flips the attribute.
  it("first render is dark even when storage is light, then reconciles to light", () => {
    storage["xcelsior-theme"] = "light";
    const seen: Theme[] = [];
    function Probe() {
      seen.push(useTheme().theme);
      return null;
    }
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    expect(seen[0]).toBe("dark"); // SSR-safe initial render (no hydration mismatch)
    expect(seen[seen.length - 1]).toBe("light"); // reconciled after mount
    expect(document.documentElement.dataset.theme).toBe("light");
  });
});

describe("ThemeProvider re-sync", () => {
  it("does not keep reconciling document visibility after unmount", () => {
    const { unmount } = renderThemeHook();
    unmount();
    storage["xcelsior-theme"] = "light";
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("remains usable when browser storage is unavailable", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new DOMException("Storage blocked", "SecurityError"); });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new DOMException("Storage blocked", "SecurityError"); });
    const { result } = renderThemeHook();
    expect(result.current.theme).toBe("dark");
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");
  });

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

describe("ThemeProvider under StrictMode", () => {
  // Regression: a `[theme]` effect applied the placeholder "dark" on mount and
  // wrote it to storage; StrictMode's effect replay then re-read "dark", so a
  // stored light theme never survived a reload in development.
  it("keeps a stored light theme through the effect replay", () => {
    storage["xcelsior-theme"] = "light";
    const seen: Theme[] = [];
    function Probe() {
      seen.push(useTheme().theme);
      return null;
    }
    render(
      <StrictMode>
        <ThemeProvider>
          <Probe />
        </ThemeProvider>
      </StrictMode>,
    );
    expect(seen[seen.length - 1]).toBe("light");
    expect(storage["xcelsior-theme"]).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("toggle writes the new theme straight through", () => {
    const { result } = renderHook(() => useTheme(), {
      wrapper: ({ children }) => (
        <StrictMode>
          <ThemeProvider>{children}</ThemeProvider>
        </StrictMode>
      ),
    });
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe("light");
    expect(storage["xcelsior-theme"]).toBe("light");
    expect(document.documentElement.classList.contains("light")).toBe(true);
  });
});
