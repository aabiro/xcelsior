"use client";

import { Badge } from "@/components/ui/badge";
import { ScrollText, Loader2 } from "lucide-react";
import type { ServerlessJob } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { CopyableText } from "./copyable-text";
import { ServerlessEmptyState } from "./serverless-ui";

const STATUS_VARIANT: Record<string, "active" | "warning" | "default"> = {
  COMPLETED: "active",
  FAILED: "warning",
  CANCELLED: "default",
  IN_QUEUE: "default",
  IN_PROGRESS: "default",
};

function fmtTime(ts?: number) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString();
}

interface LogsPanelProps {
  jobs: ServerlessJob[];
  loading?: boolean;
}

export function LogsPanel({ jobs, loading }: LogsPanelProps) {
  const { t } = useLocale();

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
      </div>
    );
  }

  if (jobs.length === 0) {
    return (
      <ServerlessEmptyState icon={ScrollText} title={t("dash.serverless.jobs_empty")} />
    );
  }

  return (
    <div className="space-y-2">
      {jobs.map((job) => (
        <div
          key={job.job_id}
          className="glow-card rounded-xl border border-border bg-surface px-4 py-3 hover:border-accent-violet/20 transition-colors"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CopyableText text={job.job_id} />
            <Badge variant={STATUS_VARIANT[job.status] ?? "default"}>{job.status}</Badge>
          </div>
          <div className="mt-2 grid gap-1 sm:grid-cols-3 text-xs text-text-muted">
            <span>{t("dash.serverless.job_queued")}: {fmtTime(job.queued_at)}</span>
            <span>{t("dash.serverless.job_finished")}: {fmtTime(job.finished_at)}</span>
            <span>
              {job.gpu_seconds != null && `${job.gpu_seconds}s GPU`}
              {job.cost_cad != null && ` · $${job.cost_cad.toFixed(4)}`}
            </span>
          </div>
          {job.error != null && (
            <pre className="mt-2 text-xs text-red-400 bg-red-500/5 rounded-lg p-2 overflow-x-auto">
              {typeof job.error === "string" ? job.error : JSON.stringify(job.error, null, 2)}
            </pre>
          )}
        </div>
      ))}
    </div>
  );
}