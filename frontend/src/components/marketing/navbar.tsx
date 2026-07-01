"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { SITE_ASSETS } from "@/lib/brand-assets";
import { useLocale } from "@/lib/locale";
import { hasSessionHint } from "@/lib/session-hint";
import { useTheme } from "@/lib/theme";
import { useMounted } from "@/hooks/useMounted";

const navKeys = [
  { href: "/", label: "Home" },
  { href: "/features", key: "nav.features" },
  { href: "/pricing", key: "nav.pricing" },
  { href: "/gpu-availability", key: "nav.gpus" },
  { href: "/mcp", key: "nav.mcp" },
  { href: "/download", key: "nav.download" },
  { href: "https://docs.xcelsior.ca", key: "nav.docs", external: true },
];

function ThemePill() {
  const { theme, toggleTheme } = useTheme();
  const isLight = theme === "light";

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label="Toggle light / dark"
      title="Toggle light / dark"
      className="site-theme-toggle"
    >
      <svg viewBox="0 0 24 24" className="site-theme-sun" aria-hidden>
        <circle cx="12" cy="12" r="4.2" fill="none" stroke="var(--text-3)" strokeWidth="2" />
        <g stroke="var(--text-3)" strokeWidth="2" strokeLinecap="round">
          <path d="M12 2.6v2.2M12 19.2v2.2M2.6 12h2.2M19.2 12h2.2M5.1 5.1l1.6 1.6M17.3 17.3l1.6 1.6M18.9 5.1l-1.6 1.6M6.7 17.3l-1.6 1.6" />
        </g>
      </svg>
      <svg viewBox="0 0 24 24" className="site-theme-moon" aria-hidden>
        <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" fill="none" stroke="var(--text-3)" strokeWidth="2" strokeLinejoin="round" />
      </svg>
      <span className="site-theme-knob" style={{ transform: isLight ? "translateX(30px)" : "translateX(0px)" }} />
    </button>
  );
}

export function Navbar() {
  const [open, setOpen] = useState(false);
  const { t, displayLocale, toggleLocale } = useLocale();
  const { user, loading, logout } = useAuth();
  const pathname = usePathname();
  const mounted = useMounted();
  const signedIn = !!user || (loading && hasSessionHint());

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

  const navItems = navKeys.map((item) => {
    const label = item.label ?? t(item.key ?? "");
    const className = `site-nav-link ${isActive(item.href) ? "site-nav-link-active" : ""}`;
    return item.external ? (
      <a key={item.href} href={item.href} className={className}>
        {label}
      </a>
    ) : (
      <Link key={item.href} href={item.href} className={className} onClick={() => setOpen(false)}>
        {label}
      </Link>
    );
  });

  const actionItems = !mounted ? (
    <span style={{ width: 150, height: 44 }} aria-hidden />
  ) : signedIn ? (
    <>
      <button type="button" onClick={() => void logout()} className="site-ghost-link">
        {t("nav.sign_out")}
      </button>
      <Link href="/dashboard" className="site-button site-button-primary" style={{ padding: "10px 18px", fontSize: 12 }}>
        {t("nav.dashboard")}
      </Link>
    </>
  ) : (
    <>
      <Link href="/login" className="site-ghost-link">
        {t("nav.sign_in")}
      </Link>
      <Link href="/register" className="site-button site-button-primary" style={{ padding: "10px 18px", fontSize: 12 }}>
        {t("home.cta_start")}
      </Link>
    </>
  );

  return (
    <header className="site-nav">
      <div className="site-container">
        <div className="site-nav-inner">
          <Link href="/" className="site-brand" onClick={() => setOpen(false)}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={SITE_ASSETS.iconGradient} className="site-brand-icon" alt="Xcelsior" fetchPriority="high" />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={SITE_ASSETS.wordmarkLight} className="wm-light" style={{ height: 17 }} alt="Xcelsior" fetchPriority="high" />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={SITE_ASSETS.wordmarkDark} className="wm-dark" style={{ height: 17 }} alt="Xcelsior" fetchPriority="high" />
          </Link>

          <nav className="site-nav-links" aria-label="Main navigation">
            {navItems}
          </nav>

          <div className="site-nav-actions">
            <button type="button" onClick={toggleLocale} className="site-ghost-link" aria-label="Toggle language">
              {displayLocale.toUpperCase()}
            </button>
            <ThemePill />
            {actionItems}
            <button
              type="button"
              className="site-mobile-menu-button"
              onClick={() => setOpen((value) => !value)}
              aria-expanded={open}
              aria-controls="mobile-nav-menu"
              aria-label={open ? "Close menu" : "Open menu"}
            >
              {open ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>
      </div>

      <nav id="mobile-nav-menu" className="site-mobile-panel" data-open={open ? "true" : "false"} aria-label="Mobile navigation">
        <div className="site-mobile-panel-inner">
          {navItems}
          <button type="button" onClick={toggleLocale} className="site-ghost-link" aria-label="Toggle language">
            {displayLocale.toUpperCase()}
          </button>
          {!mounted ? null : signedIn ? (
            <>
              <Link href="/dashboard" onClick={() => setOpen(false)} className="site-button site-button-primary">
                {t("nav.dashboard")}
              </Link>
              <button
                type="button"
                onClick={() => {
                  setOpen(false);
                  void logout();
                }}
                className="site-ghost-link"
              >
                {t("nav.sign_out")}
              </button>
            </>
          ) : (
            <>
              <Link href="/login" onClick={() => setOpen(false)} className="site-ghost-link">
                {t("nav.sign_in")}
              </Link>
              <Link href="/register" onClick={() => setOpen(false)} className="site-button site-button-primary">
                {t("home.cta_start")}
              </Link>
            </>
          )}
        </div>
      </nav>
    </header>
  );
}
