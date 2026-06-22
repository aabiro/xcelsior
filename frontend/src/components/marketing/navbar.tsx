"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { MapPin, Menu, X } from "lucide-react";
import { AnimatePresence, m } from "@/components/marketing/motion";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { LocaleToggle } from "@/components/ui/locale-toggle";
import { useLocale } from "@/lib/locale";
import { useAuth } from "@/lib/auth";

const navKeys = [
  { href: "/features", key: "nav.features" },
  { href: "/pricing", key: "nav.pricing" },
  { href: "/gpu-availability", key: "nav.gpus" },
  { href: "/mcp", key: "nav.mcp" },
  { href: "/sovereignty", key: "nav.sovereignty" },
  { href: "/download", key: "nav.download" },
  { href: "https://docs.xcelsior.ca", key: "nav.docs", external: true },
];

export function Navbar() {
  const [open, setOpen] = useState(false);
  const { t } = useLocale();
  const { user, logout } = useAuth();
  const pathname = usePathname();

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  const isActive = (href: string) => {
    if (!pathname || href.startsWith("http")) return false;
    if (href === "/") return pathname === "/";
    return pathname === href || pathname.startsWith(href + "/");
  };

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-navy/80 backdrop-blur-lg">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-2 px-4 sm:px-6">
        <Link href="/" className="flex min-w-0 shrink items-center gap-1.5 sm:gap-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/xcelsior-logo-wordmark-iconbg.svg" alt="Xcelsior" className="hidden dark:block h-9 w-auto max-w-[7.5rem] sm:h-11 sm:max-w-none" width={160} height={44} fetchPriority="high" />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/xcelsior-logo-wordmark-iconbg-light.svg" alt="Xcelsior" className="block h-9 w-auto max-w-[7.5rem] sm:h-11 sm:max-w-none dark:hidden" width={160} height={44} fetchPriority="high" />
          <span className="shrink-0 rounded bg-red-600 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-white dark:bg-accent-red/20 dark:text-red-100">
            Beta
          </span>
        </Link>

        <nav className="hidden md:flex items-center gap-8">
          {navKeys.map((l) => {
            const active = isActive(l.href);
            const linkClass = active
              ? "text-sm text-text-primary font-semibold relative after:absolute after:inset-x-0 after:-bottom-1 after:h-[2px] after:bg-accent-red after:rounded-full"
              : "text-sm text-text-secondary hover:text-text-primary transition-colors";
            return l.external ? (
              <a key={l.href} href={l.href} className={linkClass}>
                {t(l.key)}
              </a>
            ) : (
              <Link key={l.href} href={l.href} className={linkClass}>
                {t(l.key)}
              </Link>
            );
          })}
        </nav>

        <div className="flex shrink-0 items-center gap-2 sm:gap-3">
          <span className="hidden lg:flex items-center gap-1 text-xs text-text-muted">
            <MapPin className="h-3 w-3" />
            {t("nav.canadian_owned")}
          </span>
          <div className="hidden md:flex items-center gap-2">
            <LocaleToggle />
            <ThemeToggle />
          </div>
          <div className="hidden sm:block h-5 w-px bg-border" />
          {user ? (
            <>
              <button
                type="button"
                onClick={() => { void logout(); }}
                className="hidden sm:inline-flex min-h-11 items-center whitespace-nowrap px-2 text-sm text-text-secondary hover:text-text-primary transition-colors"
              >
                {t("nav.sign_out")}
              </button>
              <Link
                href="/dashboard"
                className="hidden sm:inline-flex min-h-11 items-center whitespace-nowrap rounded-lg bg-accent-red px-4 text-sm font-medium text-white hover:bg-accent-red-hover transition-colors"
              >
                {t("nav.dashboard")}
              </Link>
            </>
          ) : (
            <>
              <Link
                href="/login"
                className="hidden sm:inline-flex min-h-11 items-center whitespace-nowrap px-2 text-sm text-text-secondary hover:text-text-primary transition-colors"
              >
                {t("nav.sign_in")}
              </Link>
              <Link
                href="/register"
                className="hidden sm:inline-flex min-h-11 items-center whitespace-nowrap rounded-lg bg-accent-red px-4 text-sm font-medium text-white hover:bg-accent-red-hover transition-colors"
              >
                {t("nav.get_started")}
              </Link>
            </>
          )}
          {/* Mobile hamburger */}
          <button
            type="button"
            className="md:hidden flex min-h-11 min-w-11 items-center justify-center rounded-lg text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            aria-controls="mobile-nav-menu"
            aria-label={open ? "Close menu" : "Open menu"}
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {open && (
          <m.nav
            id="mobile-nav-menu"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="md:hidden overflow-hidden border-t border-border bg-navy/95 backdrop-blur-lg"
          >
            <div className="mx-auto max-w-7xl px-6 py-4 flex flex-col gap-1">
              {navKeys.map((l) => {
                const active = isActive(l.href);
                const mobileClass = active
                  ? "flex min-h-11 items-center rounded-lg px-3 text-sm text-text-primary font-semibold bg-surface-hover border-l-2 border-accent-red"
                  : "flex min-h-11 items-center rounded-lg px-3 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors";
                return l.external ? (
                  <a
                    key={l.href}
                    href={l.href}
                    onClick={() => setOpen(false)}
                    className={mobileClass}
                  >
                    {t(l.key)}
                  </a>
                ) : (
                  <Link
                    key={l.href}
                    href={l.href}
                    onClick={() => setOpen(false)}
                    className={mobileClass}
                  >
                    {t(l.key)}
                  </Link>
                );
              })}
              <div className="mt-3 flex flex-col gap-2 border-t border-border pt-3">
                <div className="flex items-center gap-2 px-3 py-1">
                  <LocaleToggle />
                  <ThemeToggle />
                </div>
                {user ? (
                  <>
                    <Link
                      href="/dashboard"
                      onClick={() => setOpen(false)}
                      className="inline-flex min-h-11 items-center justify-center rounded-lg bg-accent-red px-4 text-sm font-medium text-white hover:bg-accent-red-hover transition-colors"
                    >
                      {t("nav.dashboard")}
                    </Link>
                    <button
                      type="button"
                      onClick={() => { setOpen(false); void logout(); }}
                      className="flex min-h-11 items-center rounded-lg px-3 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                    >
                      {t("nav.sign_out")}
                    </button>
                  </>
                ) : (
                  <>
                    <Link
                      href="/login"
                      onClick={() => setOpen(false)}
                      className="flex min-h-11 items-center rounded-lg px-3 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                    >
                      {t("nav.sign_in")}
                    </Link>
                    <Link
                      href="/register"
                      onClick={() => setOpen(false)}
                      className="inline-flex min-h-11 items-center justify-center rounded-lg bg-accent-red px-4 text-sm font-medium text-white hover:bg-accent-red-hover transition-colors"
                    >
                      {t("nav.get_started")}
                    </Link>
                  </>
                )}
              </div>
            </div>
          </m.nav>
        )}
      </AnimatePresence>
    </header>
  );
}
