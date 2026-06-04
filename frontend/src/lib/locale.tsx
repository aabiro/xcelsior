"use client";

import { createContext, useContext, useEffect, useState, useCallback } from "react";
import en from "@/lib/i18n/en";
import fr from "@/lib/i18n/fr";

export type Locale = "en" | "fr";

interface LocaleContextValue {
  locale: Locale;
  /** Locale safe for SSR + hydration (en until client has read storage). */
  displayLocale: Locale;
  toggleLocale: () => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
}

const dictionaries = { en, fr } as const;

const LocaleContext = createContext<LocaleContextValue>({
  locale: "en",
  displayLocale: "en",
  toggleLocale: () => {},
  t: (key: string) => key,
});

export function useLocale() {
  return useContext(LocaleContext);
}

export function LocaleProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocale] = useState<Locale>("en");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("xcelsior-locale") as Locale | null;
    if (stored === "en" || stored === "fr") {
      setLocale(stored);
    }
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    document.documentElement.lang = locale;
    localStorage.setItem("xcelsior-locale", locale);
  }, [locale, mounted]);

  const toggleLocale = useCallback(
    () => setLocale((l) => (l === "en" ? "fr" : "en")),
    [],
  );

  const t = useCallback(
    (key: string, vars?: Record<string, string | number>): string => {
      const activeLocale = mounted ? locale : "en";
      let str = dictionaries[activeLocale][key] ?? dictionaries.en[key] ?? key;
      if (vars) {
        for (const [k, v] of Object.entries(vars)) {
          str = str.replaceAll(`{${k}}`, String(v));
        }
      }
      return str;
    },
    [locale, mounted],
  );

  const displayLocale = mounted ? locale : "en";

  return (
    <LocaleContext.Provider value={{ locale, displayLocale, toggleLocale, t }}>
      {children}
    </LocaleContext.Provider>
  );
}
