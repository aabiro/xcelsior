import type { HTMLAttributes, ReactNode } from "react";
import { SITE_ASSETS } from "@/lib/brand-assets";
import { cn } from "@/lib/utils";

export function SiteAuthRouteShell({ children }: { children: ReactNode }) {
  return (
    <section className="site-auth-shell">
      <div className="site-grid-bg" aria-hidden />
      <div className="site-container">
        <div className="site-rails site-auth-rails">{children}</div>
      </div>
    </section>
  );
}

export function SiteAuthCard({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("site-auth-card", className)} {...props} />;
}

export function SiteAuthHeader({
  eyebrow,
  title,
  subtitle,
  className,
  centered = true,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  subtitle?: ReactNode;
  className?: string;
  centered?: boolean;
}) {
  return (
    <div className={cn("site-auth-header", centered && "site-auth-header-centered", className)}>
      <div className={cn("site-auth-brand", centered && "site-auth-brand-centered")}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={SITE_ASSETS.iconGradient} alt="" aria-hidden className="site-auth-brand-mark" />
        <div className="site-auth-brand-wordmark-wrap">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={SITE_ASSETS.wordmarkDark} alt="Xcelsior" className="site-auth-brand-wordmark site-theme-dark" />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={SITE_ASSETS.wordmarkLight} alt="" aria-hidden className="site-auth-brand-wordmark site-theme-light" />
        </div>
      </div>
      {eyebrow ? <div className="site-pill site-auth-pill">{eyebrow}</div> : null}
      <h1 className="site-auth-title">{title}</h1>
      {subtitle ? <p className="site-auth-copy">{subtitle}</p> : null}
    </div>
  );
}

export function SiteAuthAlert({
  children,
  tone = "error",
  className,
}: {
  children: ReactNode;
  tone?: "error" | "success" | "warn" | "info";
  className?: string;
}) {
  return (
    <div className={cn("site-auth-alert", className)} data-tone={tone}>
      {children}
    </div>
  );
}

export function SiteAuthDivider({ label }: { label: ReactNode }) {
  return (
    <div className="site-auth-divider" aria-hidden>
      <span>{label}</span>
    </div>
  );
}
