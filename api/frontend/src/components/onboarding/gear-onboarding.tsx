"use client";

import Link from "next/link";
import { AlertCircle, CheckCircle2, Circle, Loader2, Rocket } from "lucide-react";
import { cn } from "@/lib/utils";
import { AUTO_DETECTED_KEYS, ONBOARDING_STEPS } from "./onboarding-steps";
import { useOnboardingState } from "./use-onboarding-state";

export function GearOnboarding({
  t,
  onNavigate,
  user,
  pathname,
  className,
}: {
  t: (key: string, vars?: Record<string, string | number>) => string;
  onNavigate?: () => void;
  user: { name?: string; email?: string; role?: string } | null;
  pathname: string;
  className?: string;
}) {
  const { completed, toggle, status, retry } = useOnboardingState(user, pathname);
  const doneCount = ONBOARDING_STEPS.filter((s) => completed[s.key]).length;
  const allDone = doneCount === ONBOARDING_STEPS.length;
  const progressPct = Math.round((doneCount / ONBOARDING_STEPS.length) * 100);

  return (
    <div className={cn("p-3", className)}>
      <div className="flex items-center gap-2 mb-2">
        <Rocket className="h-4 w-4 text-accent-gold" aria-hidden />
        <span className="text-sm font-semibold">{t("gear.onboarding")}</span>
      </div>
      <p className="text-xs text-text-muted mb-3 leading-relaxed">{t("gear.onboarding_desc")}</p>

      {status === "loading" && (
        <div
          className="flex items-center gap-2 text-xs text-text-muted mb-3"
          role="status"
          aria-live="polite"
        >
          <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" aria-hidden />
          {t("gear.loading")}
        </div>
      )}

      {status === "error" && (
        <div
          className="flex items-center justify-between gap-2 rounded-lg border border-accent-red/30 bg-accent-red/5 px-2.5 py-2 text-xs text-accent-red mb-3"
          role="alert"
        >
          <span className="flex items-center gap-1.5 min-w-0">
            <AlertCircle className="h-3.5 w-3.5 shrink-0" aria-hidden />
            <span className="truncate">{t("gear.load_error")}</span>
          </span>
          <button
            type="button"
            onClick={retry}
            className="shrink-0 rounded-md px-2 py-0.5 font-medium hover:bg-accent-red/10 transition-colors"
          >
            {t("gear.retry")}
          </button>
        </div>
      )}

      <div className="mb-3">
        <div className="flex items-center justify-between text-xs text-text-muted mb-1">
          <span id="gear-onboarding-progress-label">
            {allDone
              ? t("gear.all_done")
              : t("gear.progress", { done: doneCount, total: ONBOARDING_STEPS.length })}
          </span>
          <span className="font-mono" aria-hidden>
            {progressPct}%
          </span>
        </div>
        <div
          className="h-1.5 w-full rounded-full bg-surface-hover overflow-hidden"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={progressPct}
          aria-labelledby="gear-onboarding-progress-label"
        >
          <div
            className="h-full rounded-full bg-gradient-to-r from-accent-cyan to-accent-violet transition-all duration-300"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      <ol className="space-y-1 list-none" aria-busy={status === "loading"}>
        {ONBOARDING_STEPS.map((step) => {
          const done = !!completed[step.key];
          const isAuto = AUTO_DETECTED_KEYS.has(step.key);
          return (
            <li key={step.key} className="flex items-start gap-2 group">
              {isAuto ? (
                <span className="mt-0.5 shrink-0" aria-hidden>
                  {done ? (
                    <CheckCircle2 className="h-4 w-4 text-emerald" />
                  ) : (
                    <Circle className="h-4 w-4 text-text-muted" />
                  )}
                </span>
              ) : (
                <button
                  type="button"
                  onClick={() => toggle(step.key)}
                  className="mt-0.5 shrink-0"
                  aria-label={done ? t("gear.mark_incomplete") : t("gear.mark_complete")}
                  aria-pressed={done}
                >
                  {done ? (
                    <CheckCircle2 className="h-4 w-4 text-emerald" />
                  ) : (
                    <Circle className="h-4 w-4 text-text-muted group-hover:text-accent-cyan transition-colors" />
                  )}
                </button>
              )}
              <div className="flex-1 min-w-0">
                <Link
                  href={step.href}
                  onClick={onNavigate}
                  className={cn(
                    "text-sm leading-tight transition-colors hover:text-accent-cyan",
                    done ? "text-text-muted line-through" : "text-text-primary",
                  )}
                >
                  {t(step.labelKey)}
                </Link>
                <p className="text-[11px] text-text-muted leading-snug mt-0.5">{t(step.descKey)}</p>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}