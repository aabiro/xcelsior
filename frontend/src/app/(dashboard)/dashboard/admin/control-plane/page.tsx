"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/lib/auth";
import {
  Shield,
  ShieldCheck,
  AlertTriangle,
  CheckCircle,
  Clock,
  Cpu,
  Layers,
  Loader2,
  RefreshCw,
  Zap,
  Search,
  Database,
  Calendar,
  XCircle,
  Activity,
  Server,
  Terminal,
  FileText,
  BadgeAlert,
  Sliders,
  Check,
  ChevronRight,
  User,
} from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";

// Custom UI Components matching the premium system style
const Card = ({ children, className }: { children: React.ReactNode; className?: string }) => (
  <div className={cn("rounded-xl border border-border bg-surface p-6 shadow-xl backdrop-blur-md transition-all hover:shadow-2xl hover:border-border-hover", className)}>
    {children}
  </div>
);

const Button = ({
  children,
  onClick,
  disabled,
  variant = "primary",
  size = "md",
  className,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary" | "danger" | "success" | "outline";
  size?: "sm" | "md" | "lg";
  className?: string;
}) => {
  const baseStyle = "inline-flex items-center justify-center font-medium rounded-lg transition-all focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.98]";
  
  const variants = {
    primary: "bg-accent-cyan hover:bg-accent-cyan/80 text-background font-semibold shadow-[0_0_12px_rgba(0,212,255,0.3)]",
    secondary: "bg-surface-hover hover:bg-surface-active border border-border text-text-primary",
    danger: "bg-accent-red/20 hover:bg-accent-red/30 border border-accent-red/40 text-accent-red font-semibold",
    success: "bg-emerald/20 hover:bg-emerald/30 border border-emerald/40 text-emerald font-semibold",
    outline: "bg-transparent border border-border hover:bg-surface-hover text-text-primary",
  };

  const sizes = {
    sm: "px-3 py-1.5 text-xs gap-1.5",
    md: "px-4 py-2 text-sm gap-2",
    lg: "px-5 py-2.5 text-base gap-2.5",
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(baseStyle, variants[variant], sizes[size], className)}
    >
      {children}
    </button>
  );
};

const Badge = ({
  children,
  variant = "info",
}: {
  children: React.ReactNode;
  variant?: "info" | "warning" | "danger" | "success" | "critical";
}) => {
  const styles = {
    info: "bg-ice-blue/10 border border-ice-blue/30 text-ice-blue",
    warning: "bg-accent-gold/10 border border-accent-gold/30 text-accent-gold",
    danger: "bg-accent-red/10 border border-accent-red/30 text-accent-red",
    success: "bg-emerald/10 border border-emerald/30 text-emerald",
    critical: "bg-red-500/20 border border-red-500 text-red-200 animate-pulse",
  };

  return (
    <span className={cn("inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-semibold uppercase tracking-wider", styles[variant])}>
      {children}
    </span>
  );
};

// Main Page Component
export default function ControlPlaneAdminPage() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState<"scheduler" | "hosts" | "findings" | "tasks">("scheduler");
  
  // Data State
  const [jobs, setJobs] = useState<any[]>([]);
  const [hosts, setHosts] = useState<any[]>([]);
  const [findings, setFindings] = useState<any[]>([]);
  const [tasks, setTasks] = useState<any[]>([]);
  
  // Loading & Action State
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [actionPending, setActionPending] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  
  // Stats
  const [schedulerStatus, setSchedulerStatus] = useState({ ok: true, reason: "Operational", activeLeases: 0 });
  const [reconcilerStatus, setReconcilerStatus] = useState({ ok: true, lastRun: "Just now", findingsCount: 0 });

  const fetchAllData = async () => {
    setRefreshing(true);
    try {
      const headers = {
        "Authorization": `Bearer ${localStorage.getItem("token") || ""}`,
        "Content-Type": "application/json",
      };

      // 1. Fetch Findings
      const findingsRes = await fetch("/api/admin/reconciler/findings?status=open", { headers });
      if (findingsRes.ok) {
        const d = await findingsRes.json();
        setFindings(d.findings || []);
      }

      // 2. Fetch Jobs
      const jobsRes = await fetch("/api/admin/control-plane/jobs", { headers });
      if (jobsRes.ok) {
        const d = await jobsRes.json();
        setJobs(d.jobs || []);
      }

      // 3. Fetch Hosts
      const hostsRes = await fetch("/hosts?active_only=false", { headers });
      if (hostsRes.ok) {
        const d = await hostsRes.json();
        setHosts(d.hosts || []);
      }

      // 4. Fetch Tasks
      const tasksRes = await fetch("/api/admin/control-plane/scheduled-tasks", { headers });
      if (tasksRes.ok) {
        const d = await tasksRes.json();
        setTasks(d.tasks || []);
      }
    } catch (err) {
      console.error("Failed to fetch control plane data", err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, []);

  // Compute live counts and statuses
  useEffect(() => {
    const activeLeasesCount = jobs.filter(j => j.status === "running").length;
    setSchedulerStatus({
      ok: true,
      reason: "Operational",
      activeLeases: activeLeasesCount,
    });
    setReconcilerStatus({
      ok: true,
      lastRun: tasks.find(t => t.task_name === "reconciler")?.last_run_at 
        ? new Date(tasks.find(t => t.task_name === "reconciler")?.last_run_at).toLocaleTimeString()
        : "Just now",
      findingsCount: findings.length,
    });
  }, [jobs, tasks, findings]);

  const handleDrain = async (hostId: string, currentStatus: string) => {
    const action = currentStatus === "draining" ? "undrain" : "drain";
    setActionPending(`host-${hostId}`);
    try {
      const res = await fetch(`/host/${hostId}/${action}`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${localStorage.getItem("token") || ""}`,
        },
      });
      if (res.ok) {
        await fetchAllData();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setActionPending(null);
    }
  };

  const handleReconcileHost = async (hostId: string) => {
    setActionPending(`reconcile-${hostId}`);
    try {
      const res = await fetch(`/api/admin/reconciler/reconcile-host/${hostId}`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${localStorage.getItem("token") || ""}`,
        },
      });
      if (res.ok) {
        await fetchAllData();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setActionPending(null);
    }
  };

  const handleEnforceFinding = async (findingId: string) => {
    setActionPending(`enforce-${findingId}`);
    try {
      const res = await fetch(`/api/admin/reconciler/findings/${findingId}/enforce`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${localStorage.getItem("token") || ""}`,
        },
      });
      if (res.ok) {
        await fetchAllData();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setActionPending(null);
    }
  };

  const handleDismissFinding = async (findingId: string) => {
    setActionPending(`dismiss-${findingId}`);
    try {
      const res = await fetch(`/api/admin/reconciler/findings/${findingId}/dismiss`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${localStorage.getItem("token") || ""}`,
        },
      });
      if (res.ok) {
        await fetchAllData();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setActionPending(null);
    }
  };

  const filteredJobs = jobs.filter(j => 
    j.job_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (j.status || "").toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredHosts = hosts.filter(h => 
    h.host_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (h.status || "").toLowerCase().includes(searchQuery.toLowerCase()) ||
    (h.gpu_model || "").toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6 text-text-primary">
      {/* Premium Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-text-primary via-ice-blue to-accent-cyan bg-clip-text text-transparent">
            Control Plane Control Center
          </h1>
          <p className="text-sm text-text-muted mt-1">
            Real-time control-plane scheduler timelines, transactional reconciler actions, and host drains.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button onClick={fetchAllData} variant="secondary" size="sm" disabled={refreshing}>
            <RefreshCw className={cn("h-4.5 w-4.5", refreshing && "animate-spin text-accent-cyan")} />
            {refreshing ? "Refreshing..." : "Refresh Data"}
          </Button>
        </div>
      </div>

      {/* Dynamic Status Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* Scheduler status card */}
        <Card className="relative overflow-hidden bg-gradient-to-br from-surface to-background border-accent-cyan/10">
          <div className="absolute right-0 top-0 h-24 w-24 bg-accent-cyan/5 rounded-bl-full blur-xl" />
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-accent-cyan/15 p-2 text-accent-cyan shadow-[0_0_10px_rgba(0,212,255,0.15)]">
              <Sliders className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-text-muted">Scheduler Core</p>
              <div className="flex items-center gap-1.5 mt-1">
                <span className="h-2 w-2 rounded-full bg-emerald animate-pulse" />
                <h3 className="font-bold text-lg text-emerald">Active</h3>
              </div>
            </div>
          </div>
          <div className="mt-4 border-t border-border/40 pt-3 flex justify-between text-xs text-text-muted">
            <span>Active Leases: <strong className="text-text-primary">{schedulerStatus.activeLeases}</strong></span>
            <span>Policy: <strong className="text-accent-cyan">v2-durable</strong></span>
          </div>
        </Card>

        {/* Reconciler status card */}
        <Card className="relative overflow-hidden bg-gradient-to-br from-surface to-background border-ice-blue/10">
          <div className="absolute right-0 top-0 h-24 w-24 bg-ice-blue/5 rounded-bl-full blur-xl" />
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-ice-blue/15 p-2 text-ice-blue shadow-[0_0_10px_rgba(0,212,255,0.15)]">
              <Layers className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-text-muted">Reconciler Daemon</p>
              <div className="flex items-center gap-1.5 mt-1">
                <span className="h-2 w-2 rounded-full bg-emerald animate-pulse" />
                <h3 className="font-bold text-lg text-emerald">Operational</h3>
              </div>
            </div>
          </div>
          <div className="mt-4 border-t border-border/40 pt-3 flex justify-between text-xs text-text-muted">
            <span>Last Tick: <strong className="text-text-primary">{reconcilerStatus.lastRun}</strong></span>
            <span>Freq: <strong className="text-ice-blue">30s</strong></span>
          </div>
        </Card>

        {/* Active findings card */}
        <Card className="relative overflow-hidden bg-gradient-to-br from-surface to-background border-accent-gold/10">
          <div className="absolute right-0 top-0 h-24 w-24 bg-accent-gold/5 rounded-bl-full blur-xl" />
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-accent-gold/15 p-2 text-accent-gold shadow-[0_0_10px_rgba(255,191,0,0.15)]">
              <BadgeAlert className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-text-muted">Active Findings</p>
              <h3 className={cn("font-bold text-lg mt-1", reconcilerStatus.findingsCount > 0 ? "text-accent-gold" : "text-emerald")}>
                {reconcilerStatus.findingsCount} Open
              </h3>
            </div>
          </div>
          <div className="mt-4 border-t border-border/40 pt-3 flex justify-between text-xs text-text-muted">
            <span>Remediation: <strong className="text-text-primary">Policy-backed</strong></span>
            <span>Alert Level: <strong className="text-accent-gold">Warn</strong></span>
          </div>
        </Card>

        {/* Tasks card */}
        <Card className="relative overflow-hidden bg-gradient-to-br from-surface to-background border-accent-gold/10">
          <div className="absolute right-0 top-0 h-24 w-24 bg-emerald/5 rounded-bl-full blur-xl" />
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-emerald/15 p-2 text-emerald shadow-[0_0_10px_rgba(16,185,129,0.15)]">
              <CheckCircle className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-text-muted">Durable Tasks</p>
              <h3 className="font-bold text-lg text-text-primary mt-1">{tasks.filter(t => t.enabled).length} Scheduled</h3>
            </div>
          </div>
          <div className="mt-4 border-t border-border/40 pt-3 flex justify-between text-xs text-text-muted">
            <span>Healthy: <strong className="text-emerald">{tasks.filter(t => t.last_status === "succeeded").length}</strong></span>
            <span>Failed: <strong className="text-accent-red">{tasks.filter(t => t.last_status === "failed").length}</strong></span>
          </div>
        </Card>
      </div>

      {/* Tabs and Search Bar */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-border/60 pb-3">
        {/* Tab triggers */}
        <div className="flex gap-2 p-1 rounded-lg bg-surface border border-border/50 max-w-max overflow-x-auto">
          <button
            onClick={() => setActiveTab("scheduler")}
            className={cn(
              "px-4 py-2 rounded-md text-sm font-semibold transition-all whitespace-nowrap",
              activeTab === "scheduler" 
                ? "bg-card text-text-primary shadow-sm border border-border" 
                : "text-text-muted hover:text-text-primary"
            )}
          >
            <div className="flex items-center gap-2">
              <Sliders className={cn("h-4 w-4", activeTab === "scheduler" && "text-accent-cyan")} />
              Scheduler Timelines
            </div>
          </button>
          <button
            onClick={() => setActiveTab("hosts")}
            className={cn(
              "px-4 py-2 rounded-md text-sm font-semibold transition-all whitespace-nowrap",
              activeTab === "hosts" 
                ? "bg-card text-text-primary shadow-sm border border-border" 
                : "text-text-muted hover:text-text-primary"
            )}
          >
            <div className="flex items-center gap-2">
              <Server className={cn("h-4 w-4", activeTab === "hosts" && "text-accent-cyan")} />
              Host Drains & Capacity
            </div>
          </button>
          <button
            onClick={() => setActiveTab("findings")}
            className={cn(
              "px-4 py-2 rounded-md text-sm font-semibold transition-all whitespace-nowrap",
              activeTab === "findings" 
                ? "bg-card text-text-primary shadow-sm border border-border" 
                : "text-text-muted hover:text-text-primary"
            )}
          >
            <div className="flex items-center gap-2">
              <BadgeAlert className={cn("h-4 w-4", activeTab === "findings" && "text-accent-gold")} />
              Reconciler Findings
              {findings.length > 0 && (
                <span className="bg-accent-gold/20 border border-accent-gold/30 text-accent-gold text-[10px] px-1.5 py-0.5 rounded-full">
                  {findings.length}
                </span>
              )}
            </div>
          </button>
          <button
            onClick={() => setActiveTab("tasks")}
            className={cn(
              "px-4 py-2 rounded-md text-sm font-semibold transition-all whitespace-nowrap",
              activeTab === "tasks" 
                ? "bg-card text-text-primary shadow-sm border border-border" 
                : "text-text-muted hover:text-text-primary"
            )}
          >
            <div className="flex items-center gap-2">
              <Database className={cn("h-4 w-4", activeTab === "tasks" && "text-accent-cyan")} />
              Durable Scheduled Tasks
            </div>
          </button>
        </div>

        {/* Filter input */}
        <div className="relative max-w-xs w-full">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-text-muted" />
          <input
            type="text"
            placeholder={`Search ${activeTab}...`}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-lg bg-surface border border-border text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-cyan transition-all"
          />
        </div>
      </div>

      {/* Main Tab Content */}
      <div className="min-h-[400px]">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-20 gap-3 text-text-muted">
            <Loader2 className="h-8 w-8 animate-spin text-accent-cyan" />
            <p className="text-sm">Fetching real-time control plane telemetry...</p>
          </div>
        ) : (
          <>
            {/* 1. Scheduler Timelines and Jobs Tab */}
            {activeTab === "scheduler" && (
              <div className="space-y-6 animate-fadeIn">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-bold">Active Instance Scheduling & Attempts</h2>
                  <span className="text-xs text-text-muted">Showing last 100 job requests</span>
                </div>

                {filteredJobs.length === 0 ? (
                  <Card className="flex flex-col items-center justify-center py-12 text-center text-text-muted">
                    <Activity className="h-10 w-10 text-border mb-3" />
                    <p className="font-semibold">No active jobs found</p>
                    <p className="text-xs mt-1">Submit an instance request to see scheduler execution in real-time.</p>
                  </Card>
                ) : (
                  <div className="grid gap-4">
                    {filteredJobs.map((job) => (
                      <Card key={job.job_id} className="border-border/60 hover:border-accent-cyan/20">
                        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                          {/* Left Column: Job Info */}
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-sm text-accent-cyan font-bold select-all">
                                {job.job_id}
                              </span>
                              <Badge variant={
                                job.status === "running" ? "success" :
                                job.status === "queued" ? "warning" :
                                job.status === "failed" ? "danger" : "info"
                              }>
                                {job.status}
                              </Badge>
                            </div>
                            
                            <div className="text-xs text-text-muted space-y-1">
                              <p>Submitted: {new Date(job.submitted_at * 1000).toLocaleString()}</p>
                              {job.active_attempt_id && (
                                <p>Active Attempt ID: <code className="text-text-primary">{job.active_attempt_id}</code></p>
                              )}
                            </div>

                            {/* Queue Reason detail */}
                            {job.status === "queued" && (
                              <div className="rounded-lg bg-accent-gold/5 border border-accent-gold/20 p-3 mt-3">
                                <div className="flex items-start gap-2">
                                  <Clock className="h-4 w-4 text-accent-gold mt-0.5 shrink-0" />
                                  <div>
                                    <p className="text-xs font-bold text-accent-gold">
                                      Queue Reason: {job.queue_reason || "RESOURCES_UNAVAILABLE"}
                                    </p>
                                    <p className="text-[11px] text-text-muted mt-0.5">
                                      {job.queue_reason_detail || "Still looking for a matching host with sufficient GPU VRAM allocation capacity..."}
                                    </p>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Right Column: Attempts Timeline */}
                          <div className="flex-1 md:max-w-xl">
                            <h4 className="text-xs font-bold uppercase tracking-wider text-text-muted mb-3 flex items-center gap-1.5">
                              <Terminal className="h-3.5 w-3.5" />
                              Job Placement & Attempt Timeline
                            </h4>
                            
                            {job.attempts.length === 0 ? (
                              <p className="text-xs text-text-muted italic py-2">No scheduling attempts recorded yet.</p>
                            ) : (
                              <div className="space-y-4 relative pl-4 border-l border-border/80">
                                {job.attempts.map((att: any, idx: number) => {
                                  const isTerminal = ["failed", "completed", "terminated"].includes(att.status);
                                  return (
                                    <div key={att.attempt_id} className="relative">
                                      {/* Timeline dot */}
                                      <div className={cn(
                                        "absolute -left-[21px] top-1 h-2.5 w-2.5 rounded-full border border-surface",
                                        att.status === "running" ? "bg-emerald shadow-[0_0_8px_#10b981]" : "bg-text-muted"
                                      )} />

                                      <div className="space-y-1">
                                        <div className="flex items-center gap-2 justify-between flex-wrap">
                                          <span className="text-xs font-semibold text-text-primary flex items-center gap-1">
                                            Attempt <code className="text-[10px] text-accent-cyan">{att.attempt_id.substring(0, 8)}</code>
                                          </span>
                                          <span className={cn(
                                            "text-[10px] px-1.5 py-0.2 rounded font-bold uppercase",
                                            att.status === "running" ? "bg-emerald/10 text-emerald border border-emerald/20" : "bg-surface text-text-muted border border-border"
                                          )}>
                                            {att.status}
                                          </span>
                                        </div>

                                        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px] text-text-muted">
                                          <span>Host ID: <code className="text-text-secondary select-all">{att.host_id}</code></span>
                                          <span>Fence: <code className="text-text-secondary">{att.fencing_token ?? "N/A"}</code></span>
                                          
                                          {att.lease_claimed_at && (
                                            <span className="col-span-2">
                                              Lease: {new Date(att.lease_claimed_at).toLocaleTimeString()} → {att.lease_expires_at ? new Date(att.lease_expires_at).toLocaleTimeString() : "Ongoing"}
                                            </span>
                                          )}
                                        </div>
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* 2. Host Drains & Capacity Tab */}
            {activeTab === "hosts" && (
              <div className="space-y-6 animate-fadeIn">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-bold">Physical Host Status & Resource Capacity</h2>
                  <span className="text-xs text-text-muted">Total registered nodes: {hosts.length}</span>
                </div>

                {filteredHosts.length === 0 ? (
                  <Card className="flex flex-col items-center justify-center py-12 text-center text-text-muted">
                    <Server className="h-10 w-10 text-border mb-3" />
                    <p className="font-semibold">No hosts found</p>
                  </Card>
                ) : (
                  <div className="grid gap-4 md:grid-cols-2">
                    {filteredHosts.map((host) => {
                      const isDraining = host.status === "draining";
                      const isDead = host.status === "dead";
                      
                      // Calculate mock vram utilization progress for rich UI
                      const vramAllocated = host.allocated_vram_gb || 0;
                      const vramTotal = host.vram_gb || 24;
                      const vramPercent = Math.min(100, Math.round((vramAllocated / vramTotal) * 100));

                      return (
                        <Card key={host.host_id} className={cn(
                          "flex flex-col justify-between border-border/60 hover:border-accent-cyan/20",
                          isDraining && "border-accent-gold/40 bg-accent-gold/2",
                          isDead && "border-accent-red/20 bg-accent-red/2 opacity-70"
                        )}>
                          <div>
                            <div className="flex items-start justify-between mb-3">
                              <div className="space-y-1">
                                <div className="flex items-center gap-2">
                                  <h3 className="font-bold text-sm font-mono text-text-primary select-all">
                                    {host.host_id}
                                  </h3>
                                  <Badge variant={
                                    host.status === "active" ? "success" :
                                    host.status === "draining" ? "warning" :
                                    host.status === "dead" ? "danger" : "info"
                                  }>
                                    {host.status}
                                  </Badge>
                                </div>
                                <p className="text-xs text-text-muted">
                                  Agent Version: <code className="text-text-secondary">{host.agent_version || "2.1.4"}</code>
                                </p>
                              </div>
                              <div className="flex items-center gap-1.5 text-xs font-semibold text-text-muted">
                                <Cpu className="h-3.5 w-3.5" />
                                {host.gpu_model || "NVIDIA RTX 4090"}
                              </div>
                            </div>

                            {/* GPU Utilization Telemetry */}
                            <div className="space-y-2 rounded-lg bg-surface-hover p-3 border border-border/40 my-3">
                              <div className="flex justify-between text-xs">
                                <span className="text-text-muted font-medium">GPU VRAM Allocated</span>
                                <span className="font-bold text-text-primary">{vramAllocated} / {vramTotal} GB ({vramPercent}%)</span>
                              </div>
                              <div className="h-2 rounded-full bg-background overflow-hidden border border-border/40">
                                <div 
                                  className={cn(
                                    "h-full rounded-full transition-all duration-500",
                                    vramPercent > 80 ? "bg-accent-red" :
                                    vramPercent > 50 ? "bg-accent-gold" : "bg-accent-cyan"
                                  )}
                                  style={{ width: `${vramPercent}%` }}
                                />
                              </div>
                              <div className="flex justify-between text-[10px] text-text-muted">
                                <span>Free VRAM: {Math.max(0, vramTotal - vramAllocated)} GB</span>
                                <span>Shared Instances: {host.instance_count ?? 0}</span>
                              </div>
                            </div>
                          </div>

                          {/* Host actions */}
                          <div className="mt-4 pt-3 border-t border-border/40 flex items-center justify-between gap-2 flex-wrap">
                            <Button
                              onClick={() => handleReconcileHost(host.host_id)}
                              variant="secondary"
                              size="sm"
                              disabled={actionPending !== null || isDead}
                            >
                              <RefreshCw className={cn("h-3.5 w-3.5", actionPending === `reconcile-${host.host_id}` && "animate-spin")} />
                              Reconcile Now
                            </Button>

                            {!isDead && (
                              <Button
                                onClick={() => handleDrain(host.host_id, host.status)}
                                variant={isDraining ? "success" : "danger"}
                                size="sm"
                                disabled={actionPending !== null}
                              >
                                <Zap className="h-3.5 w-3.5" />
                                {isDraining ? "Activate (Undrain)" : "Drain Host"}
                              </Button>
                            )}
                          </div>
                        </Card>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* 3. Reconciler Findings Tab */}
            {activeTab === "findings" && (
              <div className="space-y-6 animate-fadeIn">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-bold">Durable Reconciliation Findings</h2>
                  <span className="text-xs text-text-muted">Active anomalies: {findings.length}</span>
                </div>

                {findings.length === 0 ? (
                  <Card className="flex flex-col items-center justify-center py-16 text-center text-text-muted">
                    <CheckCircle className="h-12 w-12 text-emerald mb-3 animate-bounce" />
                    <p className="font-bold text-text-primary text-base">Perfect Integrity</p>
                    <p className="text-xs mt-1 max-w-sm">
                      The transactional reconciler has matched all actual worker container states against database specifications. No discrepancies found.
                    </p>
                  </Card>
                ) : (
                  <div className="overflow-x-auto rounded-xl border border-border bg-surface">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="bg-surface-hover/80 border-b border-border text-xs font-semibold uppercase tracking-wider text-text-muted">
                          <th className="p-4">Finding Details</th>
                          <th className="p-4">Resource</th>
                          <th className="p-4">Anomaly Type</th>
                          <th className="p-4">Severity</th>
                          <th className="p-4">Created At</th>
                          <th className="p-4 text-right">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/60 text-sm">
                        {findings.map((finding) => (
                          <tr key={finding.finding_id} className="hover:bg-surface-hover/30 transition-colors">
                            <td className="p-4 max-w-md">
                              <p className="font-semibold text-text-primary">{finding.summary}</p>
                              <p className="text-[11px] text-text-muted font-mono mt-1">UUID: {finding.finding_id}</p>
                            </td>
                            <td className="p-4">
                              <span className="font-mono text-xs bg-surface-hover border border-border rounded px-1.5 py-0.5 select-all">
                                {finding.resource_id}
                              </span>
                              <p className="text-[10px] text-text-muted capitalize mt-1">{finding.resource_type}</p>
                            </td>
                            <td className="p-4">
                              <code className="text-xs text-accent-cyan font-semibold">
                                {finding.finding_type}
                              </code>
                            </td>
                            <td className="p-4">
                              <Badge variant={
                                finding.severity === "critical" ? "critical" :
                                finding.severity === "error" ? "danger" :
                                finding.severity === "warning" ? "warning" : "info"
                              }>
                                {finding.severity}
                              </Badge>
                            </td>
                            <td className="p-4 text-xs text-text-muted">
                              {finding.created_at ? new Date(finding.created_at).toLocaleString() : "-"}
                            </td>
                            <td className="p-4 text-right">
                              <div className="flex items-center justify-end gap-2">
                                <Button
                                  onClick={() => handleDismissFinding(finding.finding_id)}
                                  variant="secondary"
                                  size="sm"
                                  disabled={actionPending !== null}
                                >
                                  Dismiss
                                </Button>
                                <Button
                                  onClick={() => handleEnforceFinding(finding.finding_id)}
                                  variant="danger"
                                  size="sm"
                                  disabled={actionPending !== null}
                                >
                                  Enforce Fix
                                </Button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {/* 4. Durable Scheduled Tasks Tab */}
            {activeTab === "tasks" && (
              <div className="space-y-6 animate-fadeIn">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-bold">Durable Periodic Operations & Watchdogs</h2>
                  <span className="text-xs text-text-muted">Active routines: {tasks.length}</span>
                </div>

                {tasks.length === 0 ? (
                  <Card className="flex flex-col items-center justify-center py-12 text-center text-text-muted">
                    <Database className="h-10 w-10 text-border mb-3" />
                    <p className="font-semibold">No scheduled tasks found</p>
                  </Card>
                ) : (
                  <div className="grid gap-4 sm:grid-cols-2">
                    {tasks.map((task) => {
                      const isSucceeded = task.last_status === "succeeded";
                      return (
                        <Card key={task.task_name} className="border-border/60 flex flex-col justify-between">
                          <div>
                            <div className="flex items-start justify-between mb-3">
                              <div>
                                <h3 className="font-bold text-sm text-text-primary capitalize flex items-center gap-1.5">
                                  <Sliders className="h-4 w-4 text-accent-cyan" />
                                  {task.task_name.replace(/_/g, " ")}
                                </h3>
                                <p className="text-xs font-mono text-text-muted mt-0.5">
                                  {task.task_name}
                                </p>
                              </div>
                              <Badge variant={isSucceeded ? "success" : "danger"}>
                                {task.last_status || "Pending"}
                              </Badge>
                            </div>

                            <div className="grid grid-cols-2 gap-3 text-xs bg-surface-hover/80 p-3 rounded-lg border border-border/40 my-3">
                              <div>
                                <p className="text-[10px] text-text-muted">Interval</p>
                                <p className="font-semibold mt-0.5">{task.interval_seconds} seconds</p>
                              </div>
                              <div>
                                <p className="text-[10px] text-text-muted">Claim Owner</p>
                                <p className="font-mono font-semibold mt-0.5 truncate max-w-[120px]">
                                  {task.claim_owner || "None"}
                                </p>
                              </div>
                              <div className="col-span-2 pt-2 border-t border-border/40">
                                <p className="text-[10px] text-text-muted">Next Run Due</p>
                                <p className="font-semibold mt-0.5 flex items-center gap-1">
                                  <Clock className="h-3 w-3 text-accent-gold" />
                                  {task.next_run_at ? new Date(task.next_run_at).toLocaleString() : "Never"}
                                </p>
                              </div>
                            </div>
                          </div>

                          {task.last_error && (
                            <div className="rounded-lg bg-accent-red/5 border border-accent-red/20 p-2.5 mt-2 flex items-start gap-1.5 text-xs text-accent-red">
                              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                              <p className="font-mono truncate max-w-sm" title={task.last_error}>
                                Error: {task.last_error}
                              </p>
                            </div>
                          )}

                          <div className="mt-4 pt-3 border-t border-border/40 flex justify-between text-xs text-text-muted">
                            <span>Last Run: {task.last_run_at ? new Date(task.last_run_at).toLocaleTimeString() : "Never"}</span>
                            <span>Enabled: <strong className={task.enabled ? "text-emerald" : "text-text-muted"}>{task.enabled ? "Yes" : "No"}</strong></span>
                          </div>
                        </Card>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
