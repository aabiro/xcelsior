"use client";

import {
  createContext,
  useContext,
  useEffect,
  useLayoutEffect,
  useState,
  useCallback,
} from "react";

export type Theme = "dark" | "light";

// Layout effect on the client (runs before paint), plain effect on the server
// (useLayoutEffect warns during SSR and is a no-op there anyway).
const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

const STORAGE_KEY = "xcelsior-theme";

interface ThemeContextValue {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "dark",
  toggleTheme: () => {},
});

export function useTheme() {
  return useContext(ThemeContext);
}

function readStoredTheme(): Theme {
  if (typeof window === "undefined") return "dark";
  const stored = localStorage.getItem(STORAGE_KEY);
  return stored === "light" || stored === "dark" ? stored : "dark";
}

/** Single source of truth: html class, data-theme, color-scheme, and storage. */
export function applyTheme(theme: Theme) {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  root.classList.remove("dark", "light");
  root.classList.add(theme);
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* storage unavailable */
  }
}

/** Re-read storage and re-apply — used after marketing→dashboard navigation and tab focus. */
export function syncThemeFromStorage(): Theme {
  const stored = readStoredTheme();
  applyTheme(stored);
  return stored;
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  // Initialize to a STABLE value on both server and client. Do NOT read
  // localStorage in this initializer: on the server it returns "dark", on the
  // client it would return the stored theme — a mismatch that React resolves by
  // keeping the server-rendered markup. Worse, because the mount effect below
  // would then be setting the value the client already initialized to, it
  // produces no re-render, so the stale server value (e.g. data-theme="dark" on
  // the dashboard/marketing shells) is never corrected. That stranded the
  // dashboard in dark mode while <html> was already light — everything keyed on
  // the shell's data-theme stayed dark while the app bar (keyed on the <html>
  // class the inline script sets) went light. Keep this constant.
  const [theme, setTheme] = useState<Theme>("dark");

  // Reconcile to the stored theme before the browser paints the hydrated tree,
  // so the shells' data-theme flips in the same frame (no flash) and — crucially
  // — a real re-render occurs, updating the DOM attribute the shells render.
  useIsomorphicLayoutEffect(() => {
    const stored = readStoredTheme();
    applyTheme(stored);
    setTheme(stored);
  }, []);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    const reconcile = () => {
      const stored = readStoredTheme();
      setTheme((current) => (current !== stored ? stored : current));
      applyTheme(stored);
    };
    const onStorage = (event: StorageEvent) => {
      if (event.key !== STORAGE_KEY) return;
      const next = event.newValue === "light" || event.newValue === "dark" ? event.newValue : "dark";
      setTheme(next);
      applyTheme(next);
    };
    window.addEventListener("storage", onStorage);
    window.addEventListener("focus", reconcile);
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") reconcile();
    });
    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener("focus", reconcile);
    };
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}