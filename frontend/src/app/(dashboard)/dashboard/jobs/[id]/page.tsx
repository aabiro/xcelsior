"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge, Badge } from "@/components/ui/badge";
import { LogViewer } from "@/components/ui/log-viewer";
import {
  ArrowLeft, Clock, Cpu, DollarSign, Server, RotateCcw, XCircle, Terminal,
} from "lucide-react";
import { fetchJob, cancelJob, requeueJob } from "@/lib/api";
import type { Job } from "@/lib/api";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";

const STATUS_STEPS = ["queued", "assigned", "running", "completed"] as const;

export default function JobDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { t } = useLocale();
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const [confirmCancel, setConfirmCancel] = useState(false);

  const load = () => {
    setLoading(true);
    fetchJob(id)
      .then((r) => setJob(r.job))
      .catch(() => toast.error("Failed to load job"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, [id]);

  async function handleCancel() {
    setConfirmCancel(false);
    try {
      await cancelJob(id);
      toast.success("Job cancelled");
      load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Cancel failed");
    }
  }

  async function handleRequeue() {
    try {
      const res = await requeueJob(id);
      toast.success("Job requeued");
      const newId = res.job?.job_id;
      if (newId && newId !== id) {
        router.push(`/dashboard/jobs/${newId}`);
      } else {
        load();
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Requeue failed");
    }
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-48 rounded bg-surface skeleton-pulse" />
        <div className="h-40 rounded-xl bg-surface skeleton-pulse" />
        <div className="h-64 rounded-xl bg-surface skeleton-pulse" />
      </div>
    );
  }

  if (!job) {
    return (
      <Card className="p-12 text-center">
        <h2 className="text-xl font-semibold mb-2">{t("dash.jobs.not_found")}</h2>
        <p className="text-text-secondary mb-4">{t("dash.jobs.not_found_desc")}</p>
        <Link href="/dashboard/jobs"><Button variant="outline">{t("dash.jobs.back")}</Button></Link>
      </Card>
    );
  }

  const isActive = job.status === "queued" || job.status === "running" || job.status === "assigned";
  const isFailed = job.status === "failed";

  // Status timeline step
  const currentStepIdx = STATUS_STEPS.indexOf(
    (job.status === "failed" || job.status === "cancelled") ? "running" : (job.status as typeof STATUS_STEPS[number]),
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link href="/dashboard/jobs">
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold">{job.name || job.job_id}</h1>
            <p className="text-sm font-mono text-text-muted">{job.job_id}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={job.status} />
          {isActive && (
            <Button variant="outline" size="sm" onClick={() => setConfirmCancel(true)} className="text-accent-red border-accent-red/30 hover:bg-accent-red/10">
              <XCircle className="h-3.5 w-3.5" /> {t("dash.jobs.cancel")}
            </Button>
          )}
          {isFailed && (
            <Button variant="outline" size="sm" onClick={handleRequeue}>
              <RotateCcw className="h-3.5 w-3.5" /> {t("dash.jobs.requeue")}
            </Button>
          )}
        </div>
      </div>

      {/* Status Timeline */}
      <Card>
        <h2 className="text-sm font-semibold text-text-secondary mb-4">{t("dash.jobs.timeline")}</h2>
        <div className="flex items-center gap-1">
          {STATUS_STEPS.map((step, i) => {
            const reached = i <= currentStepIdx;
            const isCurrent = i === currentStepIdx;
            return (
              <div key={step} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <div
                    className={`h-3 w-3 rounded-full border-2 ${
                      reached
                        ? isCurrent && (job.status === "failed" || job.status === "cancelled")
                          ? "border-accent-red bg-accent-red"
                          : "border-emerald bg-emerald"
                        : "border-border bg-navy"
                    }`}
                  />
                  <span className={`mt-1.5 text-xs capitalize ${reached ? "text-text-primary" : "text-text-muted"}`}>
                    {step}
                  </span>
                </div>
                {i < STATUS_STEPS.length - 1 && (
                  <div className={`h-0.5 flex-1 ${i < currentStepIdx ? "bg-emerald" : "bg-border"}`} />
                )}
              </div>
            );
          })}
        </div>
        {(job.status === "failed" || job.status === "cancelled") && (
          <div className="mt-3 flex items-center gap-2">
            <Badge variant={job.status === "failed" ? "failed" : "cancelled"}>
              {job.status}
            </Badge>
            <span className="text-xs text-text-muted">{t("dash.jobs.during_exec", { status: job.status })}</span>
          </div>
        )}
      </Card>

      {/* Details Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-ice-blue/10">
            <Cpu className="h-5 w-5 text-ice-blue" />
          </div>
          <div>
            <p className="text-xs text-text-muted">{t("dash.jobs.label_gpu")}</p>
            <p className="font-medium">{job.gpu_type || job.gpu_model || t("dash.jobs.auto")}</p>
          </div>
        </Card>

        <Card className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald/10">
            <Clock className="h-5 w-5 text-emerald" />
          </div>
          <div>
            <p className="text-xs text-text-muted">{t("dash.jobs.label_duration")}</p>
            <p className="font-medium font-mono">
              {job.duration_sec
                ? `${(job.duration_sec / 3600).toFixed(1)}h`
                : job.elapsed_sec
                  ? `${(job.elapsed_sec / 60).toFixed(0)}m`
                  : "—"}
            </p>
          </div>
        </Card>

        <Card className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent-gold/10">
            <DollarSign className="h-5 w-5 text-accent-gold" />
          </div>
          <div>
            <p className="text-xs text-text-muted">{t("dash.jobs.label_cost")}</p>
            <p className="font-medium font-mono">
              {job.cost_cad != null ? `$${job.cost_cad.toFixed(2)} CAD` : "—"}
            </p>
          </div>
        </Card>

        <Card className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent-red/10">
            <Server className="h-5 w-5 text-accent-red" />
          </div>
          <div>
            <p className="text-xs text-text-muted">{t("dash.jobs.label_host")}</p>
            <p className="font-medium">
              {job.host_id ? (
                <Link href={`/dashboard/hosts/${job.host_id}`} className="text-ice-blue hover:underline">
                  {job.host_id.slice(0, 12)}
                </Link>
              ) : (
                t("dash.jobs.unassigned")
              )}
            </p>
          </div>
        </Card>
      </div>

      {/* Metadata */}
      <Card>
        <h2 className="text-sm font-semibold text-text-secondary mb-3">{t("dash.jobs.details")}</h2>
        <dl className="grid gap-y-2 gap-x-6 text-sm sm:grid-cols-2">
          <div className="flex justify-between sm:block">
            <dt className="text-text-muted">{t("dash.jobs.docker_image")}</dt>
            <dd className="font-mono">{job.docker_image || "—"}</dd>
          </div>
          <div className="flex justify-between sm:block">
            <dt className="text-text-muted">{t("dash.jobs.pricing_tier")}</dt>
            <dd className="capitalize">{job.tier || "on-demand"}</dd>
          </div>
          <div className="flex justify-between sm:block">
            <dt className="text-text-muted">{t("dash.jobs.submitted")}</dt>
            <dd>{(job.submitted_at || job.created_at) ? new Date(job.submitted_at || job.created_at!).toLocaleString() : "—"}</dd>
          </div>
          <div className="flex justify-between sm:block">
            <dt className="text-text-muted">{t("dash.jobs.job_id")}</dt>
            <dd className="font-mono text-text-muted">{job.job_id}</dd>
          </div>
        </dl>
      </Card>

      {/* Logs */}
      <Card>
        <div className="flex items-center gap-2 mb-3">
          <Terminal className="h-4 w-4 text-text-muted" />
          <h2 className="text-sm font-semibold text-text-secondary">{t("dash.jobs.logs")}</h2>
        </div>
        <LogViewer
          jobId={id}
          live={job.status === "running" || job.status === "assigned"}
        />
      </Card>

      <ConfirmDialog
        open={confirmCancel}
        title={t("dash.jobs.cancel_title")}
        description={t("dash.jobs.cancel_desc")}
        confirmLabel={t("dash.jobs.cancel")}
        cancelLabel={t("common.cancel")}
        variant="danger"
        onConfirm={handleCancel}
        onCancel={() => setConfirmCancel(false)}
      />
    </div>
  );
}
