"use client";

import { Cpu } from "lucide-react";
import { useLocale } from "@/lib/locale";
import type { DeployGpuOption } from "./gpu-options";

type Translate = (key: string, vars?: Record<string, string | number>) => string;

/** "RTX 3060 · 12 GB · 1 free" or "RTX 2060 · 6 GB · in use". */
export function deployGpuLabel(option: DeployGpuOption, t: Translate): string {
  const vram = option.vramGb ? ` · ${Math.round(option.vramGb)} GB` : "";
  const state =
    option.available > 0
      ? t("dash.serverless.gpu_free_count", { count: option.available })
      : t("dash.serverless.gpu_in_use");
  return `${option.model}${vram} · ${state}`;
}

/** Said in place of a GPU list when the fleet lists none, or the list did not load. */
export function DeployGpuEmpty({ failed = false, onRetry }: { failed?: boolean; onRetry?: () => void }) {
  const { t } = useLocale();
  return (
    <div
      role="status"
      className="flex items-start gap-3 rounded-xl border border-border bg-surface-hover/60 px-4 py-3 text-sm"
    >
      <Cpu className="mt-0.5 h-4 w-4 shrink-0 text-text-muted" aria-hidden />
      <div className="space-y-1">
        <p className="font-medium text-text-primary">
          {failed ? t("dash.serverless.gpus_load_failed") : t("dash.serverless.no_gpus_title")}
        </p>
        {!failed && <p className="text-text-muted">{t("dash.serverless.no_gpus_desc")}</p>}
        {onRetry && (
          <button type="button" onClick={onRetry} className="text-accent-cyan underline hover:no-underline">
            {t("dash.serverless.gpus_retry")}
          </button>
        )}
      </div>
    </div>
  );
}
