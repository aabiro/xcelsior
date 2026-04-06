"use client";

import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";

export function LocaleToggle({ className }: { className?: string }) {
  const { locale, toggleLocale } = useLocale();

  return (
    <button
      onClick={toggleLocale}
      className={cn(
        "flex h-10 items-center gap-1.5 rounded-lg px-2.5 text-base font-semibold text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors",
        className
      )}
      title={locale === "en" ? "Passer au français" : "Switch to English"}
      aria-label={locale === "en" ? "Passer au français" : "Switch to English"}
    >
      <span className={cn("transition-opacity", locale === "en" ? "opacity-100" : "opacity-50")}>
        EN
      </span>
      <span className="text-border">|</span>
      <span className={cn("transition-opacity", locale === "fr" ? "opacity-100" : "opacity-50")}>
        FR
      </span>
    </button>
  );
}
