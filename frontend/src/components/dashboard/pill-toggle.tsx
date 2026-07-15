"use client";

import { cn } from "@/lib/utils";

export type PillOption = { id: string; label: string };

/**
 * Rounded segmented control. A larger, `rounded-full` sibling of the inline
 * pill pattern used on the analytics page — the active option gets a raised
 * light pill, inactive options are muted.
 */
export function PillToggle({
  options,
  value,
  onChange,
  className,
  size = "md",
}: {
  options: PillOption[];
  value: string;
  onChange: (id: string) => void;
  className?: string;
  size?: "md" | "lg";
}) {
  return (
    <div
      role="tablist"
      className={cn(
        "inline-flex items-center gap-1 rounded-full border border-border/60 bg-surface/60 p-1 backdrop-blur-sm",
        className,
      )}
    >
      {options.map((opt) => {
        const active = opt.id === value;
        return (
          <button
            key={opt.id}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(opt.id)}
            className={cn(
              "rounded-full font-semibold transition-all duration-200",
              size === "lg" ? "px-6 py-2 text-sm" : "px-4 py-1.5 text-xs",
              active
                ? "bg-card text-text-primary shadow-sm ring-1 ring-accent-cyan/30"
                : "text-text-muted hover:text-text-primary hover:bg-surface-hover",
            )}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
