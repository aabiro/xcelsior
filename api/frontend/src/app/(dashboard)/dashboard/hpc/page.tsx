"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input, Label, Select, NumberInput } from "@/components/ui/input";
import { Cpu, RefreshCw, Play, XCircle, Loader2, Server, Clock, FileCode, Eye } from "lucide-react";
import * as api from "@/lib/api";
import type { SlurmMapping, SlurmProfileDetail } from "@/lib/api";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";

interface ProfileCard extends SlurmProfileDetail {
  id: string;
}

/** A row in the jobs list: the mapping as the Slurm reconciler last observed it. */
interface JobRow extends SlurmMapping {
  status: string;
  reason?: string | null;
}

// slurm_job_mappings.observed_state and the adapter's job states → one label.
const STATUS_LABEL: Record<string, string> = {
  pending: "pending",
  queued: "pending",
  running: "running",
  completed: "completed",
  failed: "failed",
  cancelled: "cancelled",
  preempted: "preempted",
  held: "held",
  suspended: "suspended",
  unknown: "unknown",
};

const WALLTIME = /^(\d+-)?\d{1,2}:\d{2}:\d{2}$|^\d+$/;

function formatEpoch(epoch: number | null): string {
  return epoch ? new Date(epoch * 1000).toLocaleString() : "—";
}

export default function HpcPage() {
  const { user } = useAuth();
  const router = useRouter();
  // The /api/slurm/* routes are admin-only; providers used to reach this page
  // and get nothing but 403s.
  const hasHpcAccess = !!user?.is_admin;

  useEffect(() => {
    if (user && !hasHpcAccess) router.replace("/dashboard");
  }, [user, hasHpcAccess, router]);

  if (!user || !hasHpcAccess) return null;

  return <HpcContent />;
}

function HpcContent() {
  const { t } = useLocale();
  const [profiles, setProfiles] = useState<ProfileCard[]>([]);
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"submit" | "jobs">("submit");

  // Submit form
  const [selectedProfile, setSelectedProfile] = useState("");
  const [jobName, setJobName] = useState("");
  const [command, setCommand] = useState("nvidia-smi");
  const [nodes, setNodes] = useState("1");
  const [gpus, setGpus] = useState("1");
  const [walltime, setWalltime] = useState("01:00:00");
  const [submitting, setSubmitting] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    Promise.allSettled([api.fetchSlurmProfiles(), api.fetchSlurmInstances()]).then(([p, j]) => {
      if (p.status === "fulfilled") {
        const details = p.value.details || {};
        const cards: ProfileCard[] = Object.entries(details).map(([id, d]) => ({ id, ...d }));
        setProfiles(cards);
        setSelectedProfile((cur) => cur || cards.find((c) => !c.retired)?.id || "");
      }
      if (j.status === "fulfilled") {
        const rows = Array.isArray(j.value.jobs) ? j.value.jobs : [];
        setJobs(rows.map((m) => ({
          ...m,
          status: STATUS_LABEL[m.observed_state || "unknown"] || "unknown",
          reason: m.state_reason,
        })));
      }
      setLoading(false);
    });
  }, []);

  useEffect(() => { load(); }, [load]);

  const buildRequest = (dryRun: boolean): api.SlurmSubmitRequest | null => {
    if (!jobName.trim()) { toast.error("Enter a job name"); return null; }
    if (!selectedProfile) { toast.error("Select a cluster profile"); return null; }
    if (walltime && !WALLTIME.test(walltime.trim())) {
      toast.error("Wall time must look like HH:MM:SS, D-HH:MM:SS or minutes");
      return null;
    }
    return {
      name: jobName.trim(),
      profile: selectedProfile,
      priority: 0,
      num_gpus: Math.max(0, Number(gpus)) * Math.max(1, Number(nodes)),
      walltime: walltime.trim() || undefined,
      command: command.trim() || undefined,
      dry_run: dryRun,
    };
  };

  const handlePreview = async () => {
    const req = buildRequest(true);
    if (!req) return;
    try {
      const res = await api.submitSlurmInstance(req);
      setPreview(res.script || "");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Preview failed");
    }
  };

  const handleSubmit = async () => {
    const req = buildRequest(false);
    if (!req) return;
    setSubmitting(true);
    try {
      const res = await api.submitSlurmInstance(req);
      toast.success(`Submitted: Slurm job ${res.slurm_job_id} on ${res.cluster_id || selectedProfile}`);
      setJobName("");
      setPreview(null);
      setTab("jobs");
      load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to submit job");
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancel = async (job: JobRow) => {
    try {
      await api.cancelSlurmJob(job.slurm_job_id, job.cluster_id);
      toast.success(`Cancel sent for Slurm job ${job.slurm_job_id}`);
      await handleRefreshStatus(job);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to cancel job");
    }
  };

  const handleRefreshStatus = async (job: JobRow) => {
    try {
      const res = await api.fetchSlurmJobStatus(job.slurm_job_id, job.cluster_id);
      // The reconciler's copy, not a live poll: refreshing costs the cluster nothing.
      setJobs((prev) => prev.map((j) => j.job_id === job.job_id
        ? {
            ...j,
            status: STATUS_LABEL[res.xcelsior_state] || "unknown",
            slurm_state: res.slurm_state,
            reason: res.reason,
            last_observed_at: res.last_observed_at ?? j.last_observed_at,
          }
        : j));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to refresh status");
    }
  };

  const tabs = [
    { id: "submit" as const, label: t("dash.hpc.tab_submit"), icon: Play },
    { id: "jobs" as const, label: t("dash.hpc.tab_jobs"), icon: Clock, count: jobs.filter((j) => j.status === "running" || j.status === "pending").length },
  ];
  const selected = profiles.find((p) => p.id === selectedProfile);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.hpc.title")}</h1>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
        </Button>
      </div>

      {/* Cluster Profiles */}
      {profiles.length > 0 && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {profiles.map((p) => (
            <Card
              key={p.id}
              className={`p-4 transition-all ${p.retired ? "opacity-60" : "cursor-pointer"} ${
                selectedProfile === p.id ? "ring-2 ring-ice-blue border-ice-blue" : "hover:border-text-muted"
              }`}
              onClick={() => { if (!p.retired) setSelectedProfile(p.id); }}
            >
              <div className="flex items-center gap-2 mb-2">
                <Server className="h-4 w-4 text-ice-blue" />
                <span className="text-sm font-bold">{p.name}</span>
              </div>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-text-muted">
                <span>Partition: {p.partition_gpu ?? "assigned by cluster"}</span>
                {p.max_walltime && <span>Max: {p.max_walltime}</span>}
              </div>
              <div className="mt-2 flex flex-wrap gap-1">
                {p.gpu_type && (
                  <Badge variant="info" className="text-[10px]">
                    {p.gpu_type}{p.gpus_per_node ? ` × ${p.gpus_per_node}/node` : ""}
                  </Badge>
                )}
                {p.retired && <Badge variant="default" className="text-[10px]">retired</Badge>}
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Tab bar */}
      <div className="flex gap-1 rounded-lg bg-surface p-1">
        {tabs.map((tb) => (
          <button
            key={tb.id}
            onClick={() => setTab(tb.id)}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              tab === tb.id ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary"
            }`}
          >
            <tb.icon className="h-3.5 w-3.5" /> {tb.label}
            {tb.count ? <span className="ml-1 bg-ice-blue text-white text-[10px] px-1.5 py-0.5 rounded-full">{tb.count}</span> : null}
          </button>
        ))}
      </div>

      {/* ── Submit Job Tab ── */}
      {tab === "submit" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><FileCode className="h-4 w-4" /> {t("dash.hpc.submit_title")}</CardTitle>
            <CardDescription>
              {selected ? `Submitting to ${selected.name}` : "Select a cluster profile above"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label className="text-xs">Job Name</Label>
                <Input placeholder="my-training-run" value={jobName} onChange={(e) => setJobName(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Cluster Profile</Label>
                <Select value={selectedProfile} onChange={(e) => setSelectedProfile(e.target.value)}>
                  {profiles.filter((p) => !p.retired).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                </Select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-1.5">
                <Label className="text-xs">Nodes</Label>
                <NumberInput min={1} value={nodes} onChange={(v) => setNodes(String(v))} />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">GPUs per Node (0 = CPU only)</Label>
                <NumberInput min={0} value={gpus} onChange={(v) => setGpus(String(v))} />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Wall Time</Label>
                <Input placeholder="HH:MM:SS" value={walltime} onChange={(e) => setWalltime(e.target.value)} />
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs">Command (the adapter writes the batch script around it)</Label>
              <textarea
                value={command}
                onChange={(e) => setCommand(e.target.value)}
                rows={4}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 font-mono text-xs text-text-primary focus:ring-2 focus:ring-ice-blue/50 focus:border-ice-blue outline-none resize-y"
                spellCheck={false}
              />
            </div>

            {preview !== null && (
              <pre className="max-h-72 overflow-auto rounded-lg border border-border bg-background p-3 font-mono text-[11px] text-text-secondary">
                {preview}
              </pre>
            )}

            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={handlePreview} disabled={!selectedProfile || !jobName.trim()}>
                <Eye className="h-4 w-4" /> Preview script
              </Button>
              <Button onClick={handleSubmit} disabled={submitting || !selectedProfile || !jobName.trim()}>
                {submitting ? <><Loader2 className="h-4 w-4 animate-spin" /> Submitting…</> : <><Play className="h-4 w-4" /> Submit Job</>}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Jobs Tab ── */}
      {tab === "jobs" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Clock className="h-4 w-4" /> HPC Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex justify-center py-8"><Loader2 className="h-6 w-6 animate-spin text-text-muted" /></div>
            ) : jobs.length === 0 ? (
              <div className="text-center py-8">
                <Cpu className="mx-auto h-8 w-8 text-text-muted mb-2" />
                <p className="text-sm text-text-muted">No HPC jobs submitted yet</p>
              </div>
            ) : (
              <div className="space-y-3">
                {jobs.map((job) => (
                  <div key={job.job_id} className="rounded-lg border border-border p-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <p className="text-sm font-medium">Slurm job {job.slurm_job_id}</p>
                          <Badge variant={
                            job.status === "completed" ? "completed" :
                            job.status === "running" ? "active" :
                            job.status === "failed" ? "failed" :
                            job.status === "pending" ? "warning" : "default"
                          }>
                            {job.slurm_state ? `${job.status} (${job.slurm_state})` : job.status}
                          </Badge>
                        </div>
                        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-text-muted">
                          <span>Xcelsior: {job.job_id}</span>
                          <span>Cluster: {job.cluster_id || "unknown"}</span>
                          <span>Submitted: {formatEpoch(job.submitted_at)}</span>
                          {job.terminal_at && <span>Ended: {formatEpoch(job.terminal_at)}</span>}
                          {job.last_observed_at && <span>Observed: {formatEpoch(job.last_observed_at)}</span>}
                          {job.reason && <span>Reason: {job.reason}</span>}
                        </div>
                      </div>
                      <div className="flex gap-1">
                        <Button variant="ghost" size="icon" onClick={() => handleRefreshStatus(job)} title="Refresh status">
                          <RefreshCw className="h-3.5 w-3.5" />
                        </Button>
                        {(job.status === "running" || job.status === "pending") && (
                          <Button
                            variant="ghost" size="icon"
                            onClick={() => handleCancel(job)}
                            className="text-accent-red hover:text-accent-red"
                            title="Cancel job"
                          >
                            <XCircle className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
