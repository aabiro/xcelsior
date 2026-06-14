"use client";

import type { LucideIcon } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export function SettingsPageFrame({
  title,
  subtitle,
  icon: Icon,
  children,
}: {
  title: string;
  subtitle: string;
  icon: LucideIcon;
  children: React.ReactNode;
}) {
  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-navy-light/80 via-surface/40 to-navy-light/60 p-5 sm:p-6">
        <div
          className="pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full bg-accent-cyan/10 blur-3xl"
          aria-hidden
        />
        <div
          className="pointer-events-none absolute -bottom-24 -left-16 h-48 w-48 rounded-full bg-accent-gold/8 blur-3xl"
          aria-hidden
        />
        <div className="relative flex items-start gap-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-accent-cyan/12 ring-1 ring-accent-cyan/25 shadow-[0_0_24px_rgba(0,212,255,0.12)]">
            <Icon className="h-6 w-6 text-accent-cyan" />
          </div>
          <div className="min-w-0">
            <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
            <p className="mt-1 text-sm text-text-secondary">{subtitle}</p>
          </div>
        </div>
      </div>
      {children}
    </div>
  );
}

export function SettingsLayout({
  nav,
  content,
}: {
  nav: React.ReactNode;
  content: React.ReactNode;
}) {
  return (
    <div className="grid gap-6 lg:grid-cols-[220px_minmax(0,1fr)] lg:items-start">
      <aside className="lg:sticky lg:top-6">{nav}</aside>
      <div className="min-w-0">{content}</div>
    </div>
  );
}

export function SettingsNav({
  tabs,
  activeId,
  onSelect,
  tabRefs,
  indicatorStyle,
}: {
  tabs: Array<{ id: string; label: string; icon: LucideIcon; color: string }>;
  activeId: string;
  onSelect: (id: string) => void;
  tabRefs: React.MutableRefObject<Record<string, HTMLButtonElement | null>>;
  indicatorStyle: { left: number; width: number };
}) {
  return (
    <>
      {/* Mobile / tablet: horizontal pills */}
      <div className="relative lg:hidden">
        <div className="flex gap-1 overflow-x-auto rounded-xl border border-border/60 bg-surface/60 p-1 backdrop-blur-sm">
          {tabs.map((tab) => {
            const active = activeId === tab.id;
            return (
              <button
                key={tab.id}
                ref={(el) => {
                  tabRefs.current[tab.id] = el;
                }}
                onClick={() => onSelect(tab.id)}
                className={cn(
                  "relative flex shrink-0 items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors whitespace-nowrap",
                  active ? "text-text-primary" : "text-text-muted hover:text-text-secondary",
                )}
              >
                <tab.icon className={cn("h-3.5 w-3.5", active ? tab.color : "")} />
                {tab.label}
              </button>
            );
          })}
          <motion.div
            className="absolute bottom-1 h-[calc(100%-8px)] rounded-lg border border-border/50 bg-surface shadow-sm -z-10"
            animate={{ left: indicatorStyle.left, width: indicatorStyle.width }}
            transition={{ type: "spring", stiffness: 380, damping: 30, mass: 0.8 }}
          />
        </div>
      </div>

      {/* Desktop: vertical nav */}
      <nav className="hidden lg:flex flex-col gap-1 rounded-2xl border border-border/60 bg-surface/50 p-2 backdrop-blur-sm">
        {tabs.map((tab) => {
          const active = activeId === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onSelect(tab.id)}
              className={cn(
                "flex items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm font-medium transition-all",
                active
                  ? "bg-gradient-to-r from-accent-cyan/12 to-transparent text-text-primary ring-1 ring-accent-cyan/20 shadow-[inset_2px_0_0_0_rgba(0,212,255,0.7)]"
                  : "text-text-muted hover:bg-surface-hover hover:text-text-secondary",
              )}
            >
              <span
                className={cn(
                  "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ring-1",
                  active ? "bg-accent-cyan/10 ring-accent-cyan/25" : "bg-navy-light/40 ring-border/40",
                )}
              >
                <tab.icon className={cn("h-4 w-4", active ? tab.color : "text-text-muted")} />
              </span>
              <span className="truncate">{tab.label}</span>
            </button>
          );
        })}
      </nav>
    </>
  );
}

export function SettingsTabPanel({ children }: { children: React.ReactNode }) {
  return <div className="space-y-5">{children}</div>;
}

export function SettingsSection({
  icon: Icon,
  title,
  description,
  accent = "cyan",
  badge,
  action,
  children,
  className,
  highlight,
}: {
  icon: LucideIcon;
  title: string;
  description?: string;
  accent?: "cyan" | "violet" | "gold" | "emerald" | "red";
  badge?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  highlight?: boolean;
}) {
  const accentMap = {
    cyan: {
      icon: "text-accent-cyan",
      bg: "bg-accent-cyan/10",
      ring: "ring-accent-cyan/20",
      border: "border-accent-cyan/15",
    },
    violet: {
      icon: "text-accent-violet",
      bg: "bg-accent-violet/10",
      ring: "ring-accent-violet/20",
      border: "border-accent-violet/15",
    },
    gold: {
      icon: "text-accent-gold",
      bg: "bg-accent-gold/10",
      ring: "ring-accent-gold/20",
      border: "border-accent-gold/15",
    },
    emerald: {
      icon: "text-emerald",
      bg: "bg-emerald/10",
      ring: "ring-emerald/20",
      border: "border-emerald/15",
    },
    red: {
      icon: "text-accent-red",
      bg: "bg-accent-red/10",
      ring: "ring-accent-red/20",
      border: "border-accent-red/15",
    },
  } as const;
  const tone = accentMap[accent];

  return (
    <section
      className={cn(
        "overflow-hidden rounded-2xl border bg-surface/80 backdrop-blur-sm",
        highlight ? "brand-top-accent border-border/80 shadow-lg shadow-black/10" : "border-border/60",
        className,
      )}
    >
      <div className={cn("border-b px-5 py-4", tone.border)}>
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-start gap-3">
            <div
              className={cn(
                "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ring-1",
                tone.bg,
                tone.ring,
              )}
            >
              <Icon className={cn("h-4 w-4", tone.icon)} />
            </div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-sm font-semibold text-text-primary">{title}</h3>
                {badge}
              </div>
              {description && (
                <p className="mt-0.5 text-xs text-text-muted leading-relaxed">{description}</p>
              )}
            </div>
          </div>
          {action}
        </div>
      </div>
      <div className="p-5">{children}</div>
    </section>
  );
}

export function SettingsToggleRow({
  title,
  description,
  enabled,
  onToggle,
  disabled,
}: {
  title: string;
  description?: string;
  enabled: boolean;
  onToggle: () => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-xl border border-border/40 bg-navy-light/20 px-4 py-3.5">
      <div className="min-w-0">
        <p className="text-sm font-medium">{title}</p>
        {description && <p className="mt-0.5 text-xs text-text-secondary">{description}</p>}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={enabled}
        disabled={disabled}
        onClick={onToggle}
        className={cn(
          "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors duration-200 disabled:opacity-50",
          enabled ? "bg-emerald" : "bg-border",
        )}
      >
        <span
          className={cn(
            "inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform duration-200",
            enabled ? "translate-x-6" : "translate-x-1",
          )}
        />
      </button>
    </div>
  );
}

export function SettingsLinkRow({
  href,
  icon: Icon,
  title,
  description,
  iconClassName = "text-accent-cyan",
}: {
  href: string;
  icon: LucideIcon;
  title: string;
  description?: string;
  iconClassName?: string;
}) {
  return (
    <Link
      href={href}
      className="flex items-center justify-between gap-3 rounded-xl border border-transparent px-1 py-3 transition-colors hover:border-border/40 hover:bg-surface-hover/60"
    >
      <div className="flex min-w-0 items-center gap-3">
        <Icon className={cn("h-4 w-4 shrink-0", iconClassName)} />
        <div className="min-w-0">
          <p className="text-sm font-medium">{title}</p>
          {description && <p className="text-xs text-text-muted truncate">{description}</p>}
        </div>
      </div>
      <span className="text-text-muted text-lg leading-none" aria-hidden>
        ›
      </span>
    </Link>
  );
}

export function SettingsEmptyState({
  icon: Icon,
  title,
  description,
}: {
  icon: LucideIcon;
  title: string;
  description?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border/60 bg-navy-light/15 px-6 py-10 text-center">
      <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-xl bg-surface-hover ring-1 ring-border/50">
        <Icon className="h-5 w-5 text-text-muted" />
      </div>
      <p className="text-sm font-medium text-text-secondary">{title}</p>
      {description && <p className="mt-1 max-w-sm text-xs text-text-muted">{description}</p>}
    </div>
  );
}