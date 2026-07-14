"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
} from "react";

export type Theme = "dark" | "light";

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
  const [theme, setTheme] = useState<Theme>(() => readStoredTheme());

  useEffect(() => {
    const stored = syncThemeFromStorage();
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