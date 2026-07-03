"use client";

import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import { ArrowLeft } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { CopyableText } from "./copyable-text";

type HeroAccent = "violet" | "cyan";

const HERO_STYLES: Record<HeroAccent, { border: string; gradient: string; icon: string; blur: string }> = {
  violet: {
    border: "border-accent-violet/20",
    gradient: "from-accent-violet/10 via-surface to-accent-cyan/5",
    icon: "bg-accent-violet/20 text-accent-violet",
    blur: "bg-accent-violet/10",
  },
  cyan: {
    border: "border-accent-cyan/20",
    gradient: "from-accent-cyan/8 via-surface to-accent-violet/5",
    icon: "bg-accent-cyan/20 text-accent-cyan",
    blur: "bg-accent-cyan/10",
  },
};

export function ServerlessHero({
  icon: Icon,
  badge,
  title,
  description,
  accent = "violet",
  compact = false,
  children,
}: {
  icon: LucideIcon;
  badge?: string;
  title: string;
  description?: string;
  accent?: HeroAccent;
  compact?: boolean;
  children?: React.ReactNode;
}) {
  const s = HERO_STYLES[accent];
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-2xl border bg-gradient-to-br",
        s.border,
        s.gradient,
        compact ? "p-5 sm:p-6" : "p-6 sm:p-8",
      )}
    >
      <div className={cn("absolute -right-16 -top-16 h-48 w-48 rounded-full blur-3xl", s.blur)} />
      <div className="relative flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <div className={cn("flex h-9 w-9 items-center justify-center rounded-xl", s.icon)}>
              <Icon className="h-5 w-5" />
            </div>
            {badge && (
              <Badge variant="info" className="text-[10px] uppercase tracking-widest">
                {badge}
              </Badge>
            )}
          </div>
          <h1 className={cn("font-bold tracking-tight", compact ? "text-xl sm:text-2xl" : "text-2xl sm:text-3xl")}>
            {title}
          </h1>
          {description && (
            <p className="mt-2 text-sm text-text-muted max-w-2xl">{description}</p>
          )}
        </div>
        {children && <div className="flex gap-2 shrink-0">{children}</div>}
      </div>
    </div>
  );
}

export function ServerlessSelect({
  className,
  children,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={cn(
        "w-full rounded-lg border border-border bg-background px-3 py-2.5 text-sm",
        "transition-colors focus:border-accent-violet/50 focus:outline-none focus:ring-2 focus:ring-accent-violet/20",
        "hover:border-border/80",
        className,
      )}
    >
      {children}
    </select>
  );
}

export function StepRail({
  steps,
  current,
  onStepClick,
  label,
}: {
  steps: readonly { id: string; labelKey: string }[];
  current: number;
  onStepClick?: (index: number) => void;
  label: (key: string) => string;
}) {
  const progress = steps.length > 1 ? (current / (steps.length - 1)) * 100 : 0;
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-xs text-text-muted">
        <span>
          {current + 1} / {steps.length}
        </span>
        <span className="font-medium text-accent-violet">{Math.round(progress)}%</span>
      </div>
      <div className="h-1 rounded-full bg-surface-hover overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-accent-violet to-accent-cyan transition-all duration-300"
          style={{ width: `${Math.max(8, progress)}%` }}
        />
      </div>
      <div className="flex gap-1 overflow-x-auto pb-1 scrollbar-none">
        {steps.map((s, i) => {
          const active = i === current;
          const done = i < current;
          const clickable = done && onStepClick;
          return (
            <button
              key={s.id}
              type="button"
              disabled={!clickable}
              onClick={() => clickable && onStepClick(i)}
              className={cn(
                "flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium whitespace-nowrap transition-all shrink-0",
                active && "bg-accent-violet/15 text-accent-violet border border-accent-violet/30 shadow-[0_0_20px_rgba(139,92,246,0.12)]",
                done && !active && "text-text-secondary hover:bg-surface-hover cursor-pointer border border-transparent",
                !active && !done && "text-text-muted border border-transparent opacity-60",
                !clickable && !active && "cursor-default",
              )}
            >
              <span
                className={cn(
                  "flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold",
                  active && "bg-accent-violet text-white",
                  done && !active && "bg-accent-emerald/20 text-accent-emerald",
                  !active && !done && "bg-surface-hover text-text-muted",
                )}
              >
                {done && !active ? "✓" : i + 1}
              </span>
              <span className="hidden sm:inline">{label(s.labelKey)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function ApiUrlCard({
  title,
  url,
  slug,
  invokePath,
}: {
  title: string;
  url: string;
  slug?: string;
  invokePath?: string;
}) {
  const fullUrl = typeof window !== "undefined" ? `${window.location.origin}${url.startsWith("/") ? url : `/${url}`}` : url;
  return (
    <div className="glow-card brand-top-accent rounded-xl border border-border bg-surface p-4 space-y-3">
      <p className="text-sm font-semibold">{title}</p>
      <div className="rounded-lg border border-border bg-surface px-3 py-2.5">
        <CopyableText text={fullUrl} className="font-mono text-sm text-text-primary" />
      </div>
      {(slug || invokePath) && (
        <div className="flex flex-wrap gap-3 text-xs text-text-muted">
          {slug && (
            <span>
              Slug: <span className="font-mono text-text-secondary">{slug}</span>
            </span>
          )}
          {invokePath && (
            <span>
              Path: <span className="font-mono text-text-secondary">{invokePath}</span>
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export function EngineBadge({ engine }: { engine?: string }) {
  if (!engine) return null;
  const label = engine === "vllm" ? "vLLM" : engine === "tgi" ? "TGI" : engine === "sglang" ? "SGLang" : engine;
  return (
    <Badge variant="info" className="text-[10px] uppercase tracking-wide">
      {label}
    </Badge>
  );
}

export function ServerlessBackLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="inline-flex items-center gap-1 text-xs text-text-muted hover:text-text-primary transition-colors"
    >
      <ArrowLeft className="h-3 w-3" />
      {children}
    </Link>
  );
}

export function ServerlessEmptyState({
  icon: Icon,
  title,
  description,
  children,
  accent = "violet",
}: {
  icon: LucideIcon;
  title: string;
  description?: string;
  children?: React.ReactNode;
  accent?: HeroAccent;
}) {
  const s = HERO_STYLES[accent];
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-2xl border border-dashed text-center",
        s.border,
        "bg-gradient-to-br from-surface via-surface to-transparent py-14 px-6",
      )}
    >
      <div className={cn("absolute -right-12 -top-12 h-32 w-32 rounded-full blur-3xl opacity-60", s.blur)} />
      <div className="relative space-y-3">
        <div className={cn("mx-auto flex h-12 w-12 items-center justify-center rounded-2xl", s.icon)}>
          <Icon className="h-6 w-6" />
        </div>
        <p className="text-sm font-medium">{title}</p>
        {description && <p className="text-xs text-text-muted max-w-sm mx-auto">{description}</p>}
        {children}
      </div>
    </div>
  );
}

export function ServerlessSegmentedTabs<T extends string>({
  tabs,
  value,
  onChange,
  label,
}: {
  tabs: readonly { id: T; icon: LucideIcon; labelKey: string }[];
  value: T;
  onChange: (id: T) => void;
  label: (key: string) => string;
}) {
  return (
    <div className="flex gap-1 overflow-x-auto rounded-xl border border-border bg-surface-hover/40 p-1 scrollbar-none">
      {tabs.map((item) => {
        const active = value === item.id;
        return (
          <button
            key={item.id}
            type="button"
            onClick={() => onChange(item.id)}
            className={cn(
              "flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium whitespace-nowrap transition-all shrink-0",
              active
                ? "bg-accent-violet/15 text-accent-violet border border-accent-violet/25 shadow-[0_0_16px_rgba(139,92,246,0.1)]"
                : "text-text-muted hover:text-text-primary border border-transparent",
            )}
          >
            <item.icon className="h-3.5 w-3.5" />
            {label(item.labelKey)}
          </button>
        );
      })}
    </div>
  );
}

export function ServerlessPanel({
  className,
  children,
  accent = true,
}: {
  className?: string;
  children: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <div
      className={cn(
        "glow-card rounded-xl border border-border bg-surface",
        accent && "brand-top-accent",
        className,
      )}
    >
      {children}
    </div>
  );
}