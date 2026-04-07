"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input, Label, Select } from "@/components/ui/input";
import {
  Cpu, RefreshCw, Play, XCircle, Loader2, Server, Clock, CheckCircle,
  AlertTriangle, FileCode,
} from "lucide-react";
import * as api from "@/lib/api";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";

interface SlurmProfile {
  id: string;
  name: string;
  description: string;
  partition: string;
  max_nodes: number;
  gpu_type?: string;
}

interface SlurmJob {
  job_id: string;
  name: string;
  status: string;
  partition: string;
  nodes: number;
  submitted_at: string;
  started_at?: string;
  completed_at?: string;
}

const STATUS_COLORS: Record<string, string> = {
  pending: "text-accent-gold",
  running: "text-ice-blue",
  completed: "text-emerald",
  failed: "text-accent-red",
  cancelled: "text-text-muted",
};

export default function HpcPage() {
  const { t } = useLocale();
  const { user } = useAuth();
  const router = useRouter();
  const hasHpcAccess = !!user?.is_admin || user?.role === "provider";

  useEffect(() => {
    if (user && !hasHpcAccess) router.replace("/dashboard");
  }, [user, hasHpcAccess, router]);

  if (!user || !hasHpcAccess) return null;

  return <HpcContent />;
}

function HpcContent() {
  const { t } = useLocale();
  const [profiles, setProfiles] = useState<SlurmProfile[]>([]);
  const [jobs, setJobs] = useState<SlurmJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"submit" | "jobs">("submit");

  // Submit form
  const [selectedProfile, setSelectedProfile] = useState("");
  const [jobName, setJobName] = useState("");
  const [script, setScript] = useState("#!/bin/bash\n#SBATCH --job-name=my-job\n#SBATCH --nodes=1\n#SBATCH --gpus=1\n\npython train.py");
  const [nodes, setNodes] = useState("1");
  const [gpus, setGpus] = useState("1");
  const [walltime, setWalltime] = useState("01:00:00");
  const [submitting, setSubmitting] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    Promise.allSettled([
      api.fetchSlurmProfiles(),
      fetch("/api/slurm/instances", { credentials: "include" }).then((r) => r.ok ? r.json() : Promise.reject()),
    ]).then(([p, j]) => {
      if (p.status === "fulfilled") {
        const raw = p.value.profiles || {};
        const profs: SlurmProfile[] = Object.entries(raw).map(([key, v]) => ({
          id: key,
          name: key,
          description: v.description,
          partition: v.partitions?.[0] || "default",
          max_nodes: 16,
          gpu_type: v.gpus?.[0],
        }));
        setProfiles(profs);
        if (profs.length > 0 && !selectedProfile) setSelectedProfile(profs[0].id);
      }
      if (j.status === "fulfilled") setJobs(Array.isArray(j.value.jobs) ? j.value.jobs : []);
      setLoading(false);
    });
  }, [selectedProfile]);

  useEffect(() => { load(); }, [load]);

  const handleSubmit = async () => {
    if (!jobName.trim()) { toast.error("Enter a job name"); return; }
    if (!selectedProfile) { toast.error("Select a cluster profile"); return; }
    setSubmitting(true);
    try {
      const res = await api.submitSlurmInstance({
        name: jobName.trim(),

        priority: "normal",
        profile: selectedProfile,
        num_gpus: Number(gpus) * Number(nodes),
      });
      toast.success(`Instance submitted: ${res.slurm_job_id || res.instance_id || "submitted"}`);
      setJobName("");
      setTab("jobs");
      load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to submit job");
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancel = async (jobId: string) => {
    try {
      await api.cancelSlurmJob(jobId);
      setJobs((prev) => prev.map((j) => j.job_id === jobId ? { ...j, status: "cancelled" } : j));
      toast.success("Job cancelled");
    } catch { toast.error("Failed to cancel job"); }
  };

  const handleRefreshStatus = async (jobId: string) => {
    try {
      const res = await api.fetchSlurmJobStatus(jobId);
      setJobs((prev) => prev.map((j) => j.job_id === jobId ? { ...j, ...res } : j));
    } catch { toast.error("Failed to refresh status"); }
  };

  const tabs = [
    { id: "submit" as const, label: t("dash.hpc.tab_submit"), icon: Play },
    { id: "jobs" as const, label: t("dash.hpc.tab_jobs"), icon: Clock, count: jobs.filter((j) => j.status === "running" || j.status === "pending").length },
  ];

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
              className={`p-4 cursor-pointer transition-all ${
                selectedProfile === p.id ? "ring-2 ring-ice-blue border-ice-blue" : "hover:border-text-muted"
              }`}
              onClick={() => setSelectedProfile(p.id)}
            >
              <div className="flex items-center gap-2 mb-2">
                <Server className="h-4 w-4 text-ice-blue" />
                <span className="text-sm font-bold">{p.name}</span>
              </div>
              <p className="text-xs text-text-secondary mb-2">{p.description}</p>
              <div className="flex items-center gap-3 text-xs text-text-muted">
                <span>Partition: {p.partition}</span>
                <span>Max: {p.max_nodes} nodes</span>
              </div>
              {p.gpu_type && <Badge variant="info" className="mt-2 text-[10px]">{p.gpu_type}</Badge>}
            </Card>
          ))}
        </div>
      )}

      {/* Tab bar */}
      <div className="flex gap-1 rounded-lg bg-surface p-1">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              tab === t.id ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary"
            }`}
          >
            <t.icon className="h-3.5 w-3.5" /> {t.label}
            {t.count ? <span className="ml-1 bg-ice-blue text-white text-[10px] px-1.5 py-0.5 rounded-full">{t.count}</span> : null}
          </button>
        ))}
      </div>

      {/* ── Submit Job Tab ── */}
      {tab === "submit" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><FileCode className="h-4 w-4" /> {t("dash.hpc.submit_title")}</CardTitle>
            <CardDescription>
              {selectedProfile
                ? `Submitting to ${profiles.find((p) => p.id === selectedProfile)?.name || "cluster"}`
                : "Select a cluster profile above"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label className="text-xs">Job Name</Label>
                <Input
                  placeholder="my-training-run"
                  value={jobName}
                  onChange={(e) => setJobName(e.target.value)}
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Cluster Profile</Label>
                <Select value={selectedProfile} onChange={(e) => setSelectedProfile(e.target.value)}>
                  {profiles.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                </Select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-1.5">
                <Label className="text-xs">Nodes</Label>
                <Input type="number" min={1} value={nodes} onChange={(e) => setNodes(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">GPUs per Node</Label>
                <Input type="number" min={0} value={gpus} onChange={(e) => setGpus(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Wall Time</Label>
                <Input placeholder="HH:MM:SS" value={walltime} onChange={(e) => setWalltime(e.target.value)} />
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs">Job Script</Label>
              <textarea
                value={script}
                onChange={(e) => setScript(e.target.value)}
                rows={10}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 font-mono text-xs text-text-primary focus:ring-2 focus:ring-ice-blue/50 focus:border-ice-blue outline-none resize-y"
                spellCheck={false}
              />
            </div>

            <div className="flex justify-end">
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
                          <p className="text-sm font-medium">{job.name || job.job_id}</p>
                          <Badge variant={
                            job.status === "completed" ? "completed" :
                            job.status === "running" ? "active" :
                            job.status === "failed" ? "failed" :
                            job.status === "pending" ? "warning" : "default"
                          }>
                            {job.status}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-3 text-xs text-text-muted">
                          <span>ID: {job.job_id}</span>
                          <span>Partition: {job.partition}</span>
                          <span>Nodes: {job.nodes}</span>
                          <span>Submitted: {new Date(job.submitted_at).toLocaleString()}</span>
                        </div>
                      </div>
                      <div className="flex gap-1">
                        <Button
                          variant="ghost" size="icon"
                          onClick={() => handleRefreshStatus(job.job_id)}
                          title="Refresh status"
                        >
                          <RefreshCw className="h-3.5 w-3.5" />
                        </Button>
                        {(job.status === "running" || job.status === "pending") && (
                          <Button
                            variant="ghost" size="icon"
                            onClick={() => handleCancel(job.job_id)}
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
