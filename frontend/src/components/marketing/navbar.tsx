"use client";

import { useState } from "react";
import Link from "next/link";
import { MapPin, Menu, X } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { LocaleToggle } from "@/components/ui/locale-toggle";
import { useLocale } from "@/lib/locale";

const navKeys = [
  { href: "/features", key: "nav.features" },
  { href: "/pricing", key: "nav.pricing" },
  { href: "/gpu-availability", key: "nav.gpus" },
  { href: "/sovereignty", key: "nav.sovereignty" },
  { href: "https://docs.xcelsior.ca", key: "nav.docs", external: true },
];

export function Navbar() {
  const [open, setOpen] = useState(false);
  const { t } = useLocale();

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-navy/80 backdrop-blur-lg">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg" style={{ background: 'linear-gradient(135deg, #060a13 0%, #0d1a2a 100%)' }}>
            <svg viewBox="0 0 256 256" className="h-5 w-5" aria-hidden>
              <defs>
                <linearGradient id="nav-lg" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#00d4ff" />
                  <stop offset="60%" stopColor="#7c3aed" />
                  <stop offset="100%" stopColor="#dc2626" />
                </linearGradient>
              </defs>
              <path d="M60 56 L120 128 L60 200 L90 200 L128 152 L166 200 L196 200 L136 128 L196 56 L166 56 L128 104 L90 56Z" fill="url(#nav-lg)" />
              <path d="M118 36 L128 20 L138 36Z" fill="#00d4ff" opacity="0.9" />
            </svg>
          </div>
          <span className="text-xl font-bold text-text-primary">
            Xcelsior
          </span>
          <span className="rounded bg-accent-red/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-accent-red">
            Beta
          </span>
        </Link>

        <nav className="hidden md:flex items-center gap-8">
          {navKeys.map((l) =>
            l.external ? (
              <a
                key={l.href}
                href={l.href}
                className="text-sm text-text-secondary hover:text-text-primary transition-colors"
              >
                {t(l.key)}
              </a>
            ) : (
              <Link
                key={l.href}
                href={l.href}
                className="text-sm text-text-secondary hover:text-text-primary transition-colors"
              >
                {t(l.key)}
              </Link>
            )
          )}
        </nav>

        <div className="flex items-center gap-3">
          <span className="hidden lg:flex items-center gap-1 text-xs text-text-muted">
            <MapPin className="h-3 w-3" />
            {t("nav.canadian_owned")}
          </span>
          <LocaleToggle />
          <ThemeToggle />
          <div className="hidden sm:block h-5 w-px bg-border" />
          <Link
            href="/login"
            className="hidden sm:inline text-sm text-text-secondary hover:text-text-primary transition-colors"
          >
            {t("nav.sign_in")}
          </Link>
          <Link
            href="/register"
            className="hidden sm:inline-flex h-9 items-center rounded-lg bg-accent-red px-4 text-sm font-medium text-white hover:bg-accent-red-hover transition-colors"
          >
            {t("nav.get_started")}
          </Link>
          {/* Mobile hamburger */}
          <button
            className="md:hidden flex items-center justify-center h-10 w-10 rounded-lg text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
            onClick={() => setOpen(!open)}
            aria-label={open ? "Close menu" : "Open menu"}
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {open && (
          <motion.nav
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="md:hidden overflow-hidden border-t border-border bg-navy/95 backdrop-blur-lg"
          >
            <div className="mx-auto max-w-7xl px-6 py-4 flex flex-col gap-1">
              {navKeys.map((l) =>
                l.external ? (
                  <a
                    key={l.href}
                    href={l.href}
                    onClick={() => setOpen(false)}
                    className="rounded-lg px-3 py-2.5 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                  >
                    {t(l.key)}
                  </a>
                ) : (
                  <Link
                    key={l.href}
                    href={l.href}
                    onClick={() => setOpen(false)}
                    className="rounded-lg px-3 py-2.5 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                  >
                    {t(l.key)}
                  </Link>
                )
              )}
              <div className="mt-3 flex flex-col gap-2 border-t border-border pt-3">
                <div className="flex items-center gap-2 px-3 py-1">
                  <LocaleToggle />
                  <ThemeToggle />
                </div>
                <Link
                  href="/login"
                  onClick={() => setOpen(false)}
                  className="rounded-lg px-3 py-2.5 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                >
                  {t("nav.sign_in")}
                </Link>
                <Link
                  href="/register"
                  onClick={() => setOpen(false)}
                  className="inline-flex h-10 items-center justify-center rounded-lg bg-accent-red px-4 text-sm font-medium text-white hover:bg-accent-red-hover transition-colors"
                >
                  {t("nav.get_started")}
                </Link>
              </div>
            </div>
          </motion.nav>
        )}
      </AnimatePresence>
    </header>
  );
}
