"use client";

import { Badge } from "@/components/ui/badge";
import { Server, Loader2 } from "lucide-react";
import type { ServerlessWorker } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import { ServerlessEmptyState } from "./serverless-ui";

const STATE_COLORS: Record<string, string> = {
  ready: "active",
  idle: "info",
  booting: "warning",
  draining: "warning",
  exited: "default",
};

interface WorkersPanelProps {
  workers: ServerlessWorker[];
  loading?: boolean;
}

export function WorkersPanel({ workers, loading }: WorkersPanelProps) {
  const { t } = useLocale();

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
      </div>
    );
  }

  if (workers.length === 0) {
    return (
      <ServerlessEmptyState
        icon={Server}
        title={t("dash.serverless.workers_empty")}
        description={t("dash.serverless.workers_empty_desc")}
      />
    );
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {workers.map((w) => (
        <div
          key={w.worker_id}
          className={cn(
            "glow-card rounded-xl border bg-surface p-4 flex items-start justify-between gap-3 transition-colors",
            w.state === "ready" || w.state === "idle"
              ? "border-accent-emerald/20 hover:border-accent-emerald/35"
              : "border-border hover:border-accent-violet/20",
          )}
        >
          <div className="min-w-0">
            <p className="font-mono text-xs text-text-muted truncate">{w.worker_id}</p>
            <div className="flex items-center gap-2 mt-2">
              <Badge variant={(STATE_COLORS[w.state?.toLowerCase() ?? ""] ?? "default") as "active" | "info" | "warning" | "default"}>
                {w.state}
              </Badge>
              {(w.current_concurrency ?? 0) > 0 && (
                <span className="text-xs text-text-muted">
                  {w.current_concurrency} {t("dash.serverless.concurrent")}
                </span>
              )}
            </div>
          </div>
          <div
            className={cn(
              "h-2.5 w-2.5 rounded-full shrink-0 mt-1",
              w.state === "ready" || w.state === "idle" ? "bg-emerald animate-pulse" : "bg-amber-400",
            )}
          />
        </div>
      ))}
    </div>
  );
}