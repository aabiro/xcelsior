"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  useCallback,
} from "react";
import { usePathname, useRouter } from "next/navigation";
import enPublic from "@/lib/i18n/en-public";
import enDashboard from "@/lib/i18n/en-dashboard";

export type Locale = "en" | "fr";

interface LocaleContextValue {
  locale: Locale;
  /** Locale safe for SSR + hydration (en until client has read storage). */
  displayLocale: Locale;
  toggleLocale: () => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
}

const LocaleContext = createContext<LocaleContextValue>({
  locale: "en",
  displayLocale: "en",
  toggleLocale: () => {},
  t: (key: string) => key,
});

export function useLocale() {
  return useContext(LocaleContext);
}

const LEGAL_PATHS = new Set(["/privacy", "/terms"]);

function isDashboardPath(pathname: string | null | undefined): boolean {
  return Boolean(pathname?.startsWith("/dashboard"));
}

export function LocaleProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocale] = useState<Locale>("en");
  const [mounted, setMounted] = useState(false);
  const [frPublic, setFrPublic] = useState<Record<string, string> | null>(null);
  const [frDashboard, setFrDashboard] = useState<Record<string, string> | null>(null);
  const pathname = usePathname();
  const router = useRouter();
  const onDashboard = isDashboardPath(pathname);

  useEffect(() => {
    const stored = localStorage.getItem("xcelsior-locale") as Locale | null;
    if (stored === "en" || stored === "fr") {
      setLocale(stored);
      document.cookie = `xcelsior-locale=${stored};path=/;max-age=31536000;SameSite=Lax`;
    }
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    document.documentElement.lang = locale;
    localStorage.setItem("xcelsior-locale", locale);
    document.cookie = `xcelsior-locale=${locale};path=/;max-age=31536000;SameSite=Lax`;
  }, [locale, mounted]);

  useEffect(() => {
    if (locale !== "fr" || frPublic) return;
    let cancelled = false;
    import("@/lib/i18n/fr-public").then((mod) => {
      if (!cancelled) setFrPublic(mod.default);
    });
    return () => {
      cancelled = true;
    };
  }, [locale, frPublic]);

  useEffect(() => {
    if (!onDashboard || locale !== "fr" || frDashboard) return;
    let cancelled = false;
    import("@/lib/i18n/fr-dashboard").then((mod) => {
      if (!cancelled) setFrDashboard(mod.default);
    });
    return () => {
      cancelled = true;
    };
  }, [onDashboard, locale, frDashboard]);

  const toggleLocale = useCallback(() => {
    const next: Locale = locale === "en" ? "fr" : "en";
    setLocale(next);
    document.cookie = `xcelsior-locale=${next};path=/;max-age=31536000;SameSite=Lax`;
    if (pathname && LEGAL_PATHS.has(pathname)) {
      router.refresh();
    }
  }, [locale, pathname, router]);

  const dictionary = useMemo(() => {
    const activeLocale = mounted ? locale : "en";
    const en = { ...enPublic, ...enDashboard };
    if (activeLocale !== "fr" || !frPublic) return en;
    return { ...en, ...frPublic, ...(frDashboard ?? {}) };
  }, [mounted, locale, enDashboard, frPublic, frDashboard]);

  const t = useCallback(
    (key: string, vars?: Record<string, string | number>): string => {
      let str = dictionary[key] ?? enPublic[key] ?? key;
      if (vars) {
        for (const [k, v] of Object.entries(vars)) {
          str = str.replaceAll(`{${k}}`, String(v));
        }
      }
      return str;
    },
    [dictionary],
  );

  const displayLocale = mounted ? locale : "en";

  return (
    <LocaleContext.Provider value={{ locale, displayLocale, toggleLocale, t }}>
      {children}
    </LocaleContext.Provider>
  );
}