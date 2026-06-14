"use client";

import { useEffect, useState, useCallback } from "react";
import NextLink from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input, Select } from "@/components/ui/input";
import { Dialog } from "@/components/ui/dialog";
import { Pagination, usePagination } from "@/components/ui/pagination";
import {
  HardDrive, RefreshCw, Plus, Trash2, Loader2, Link, Unlink, Copy, Check, ChevronDown, ChevronUp, Globe, Lock,
  Rocket, ExternalLink, Search, ArrowUpDown, ArrowUp, ArrowDown, DollarSign, Database, Pencil, X, Camera, AlertTriangle,
} from "lucide-react";
import { FadeIn, StaggerList, StaggerItem } from "@/components/ui/motion";
import { useAuth } from "@/lib/auth";
import * as api from "@/lib/api";
import type { Volume, Instance, GpuAvailability, Host, VolumeSnapshot } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { LaunchInstanceModal } from "@/components/instances/launch-instance-modal";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { useLocale } from "@/lib/locale";
import { getTeamContext } from "@/lib/team-context";
import { TeamContextBanner } from "@/components/team/team-context-banner";

const PRICE_PER_GB = 0.03;

type SortKey = "name" | "size_gb" | "status" | "region" | "monthly_cost";
type SortDir = "asc" | "desc";

function CopyableId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(id).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <button
      onClick={copy}
      className="inline-flex items-center gap-1 font-mono text-xs text-text-muted hover:text-text-primary transition-colors cursor-pointer"
      title="Copy volume ID"
    >
      {id}
      {copied ? <Check className="h-3 w-3 text-green-500" /> : <Copy className="h-3 w-3" />}
    </button>
  );
}

function volumeActionError(err: unknown, viewerMessage: string): string {
  const msg = (err instanceof Error && err.message.trim()) || "Request failed";
  return /team viewers cannot/i.test(msg) ? viewerMessage : msg;
}

export default function VolumesPage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const team = getTeamContext(user);
  const canWrite = team.canWriteInstances;
  const [volumes, setVolumes] = useState<Volume[]>([]);
  const [instances, setInstances] = useState<Instance[]>([]);
  const [gpus, setGpus] = useState<GpuAvailability[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newSize, setNewSize] = useState(50);
  const [newRegion, setNewRegion] = useState("ca-east");
  const [newEncrypted, setNewEncrypted] = useState(true);
  const [attachingId, setAttachingId] = useState<string | null>(null);
  const [selectedInstance, setSelectedInstance] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [search, setSearch] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [page, setPage] = useState(1);
  const [launchVolumeId, setLaunchVolumeId] = useState<string | null>(null);
  const [hosts, setHosts] = useState<Host[]>([]);
  const [snapshotsVolId, setSnapshotsVolId] = useState<string | null>(null);
  const [snapshots, setSnapshots] = useState<VolumeSnapshot[]>([]);
  const [snapshotsLoading, setSnapshotsLoading] = useState(false);
  const [snapshotLabel, setSnapshotLabel] = useState("");
  const [snapshotCreating, setSnapshotCreating] = useState(false);

  const load = useCallback((opts?: { refresh?: boolean }) => {
    if (opts?.refresh) setLoading(true);
    Promise.all([
      api.listVolumes().then((res) => setVolumes(res.volumes || [])),
      api.fetchInstances().then((res) => setInstances(res.instances || [])).catch(() => {}),
      api.fetchAvailableGPUs().then((res) => setGpus(res.gpus || [])).catch(() => {}),
      api.fetchHosts().then((res) => setHosts(res.hosts || [])).catch(() => {}),
    ])
      .catch(() => toast.error(t("dash.volumes.load_failed")))
      .finally(() => setLoading(false));
  }, [t]);

  useEffect(() => {
    let active = true;
    Promise.all([
      api.listVolumes().then((res) => { if (active) setVolumes(res.volumes || []); }),
      api.fetchInstances().then((res) => { if (active) setInstances(res.instances || []); }).catch(() => {}),
      api.fetchAvailableGPUs().then((res) => { if (active) setGpus(res.gpus || []); }).catch(() => {}),
      api.fetchHosts().then((res) => { if (active) setHosts(res.hosts || []); }).catch(() => {}),
    ])
      .catch(() => { if (active) toast.error(t("dash.volumes.load_failed")); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [t]);

  useEffect(() => {
    const onTeamChanged = () => { load(); };
    window.addEventListener("xcelsior-team-changed", onTeamChanged);
    return () => window.removeEventListener("xcelsior-team-changed", onTeamChanged);
  }, [load]);

  // Live updates — re-fetch on volume/instance lifecycle changes
  useEventStream({
    eventTypes: ["volume_created", "volume_deleted", "volume_attached", "volume_detached", "volume_renamed", "job_status"],
    onEvent: () => { load(); },
  });

  const handleCreate = async () => {
    if (!newName.trim()) return toast.error("Volume name required");
    setCreating(true);
    try {
      const res = await api.createVolume({ name: newName.trim(), size_gb: newSize, region: newRegion, encrypted: newEncrypted });
      if (res.volume?.status === "error") {
        toast.error(t("dash.volumes.error_provision_failed"));
      } else {
        toast.success(t("dash.volumes.create_success"));
      }
      setNewName("");
      setShowCreate(false);
      load();
    } catch (e: unknown) {
      toast.error(volumeActionError(e, t("dash.volumes.viewer_blocked")));
    } finally {
      setCreating(false);
    }
  };

  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [confirmSnapshotId, setConfirmSnapshotId] = useState<string | null>(null);

  const handleDelete = async (volumeId: string) => {
    setConfirmDeleteId(null);
    try {
      await api.deleteVolume(volumeId);
      toast.success("Volume deleted");
      load();
    } catch (e: unknown) {
      toast.error(volumeActionError(e, t("dash.volumes.viewer_blocked")));
    }
  };

  const hostRegionById = useCallback((hostId?: string) => {
    if (!hostId) return "";
    const host = hosts.find((h) => h.host_id === hostId);
    return (host?.region || host?.province || "").trim().toLowerCase();
  }, [hosts]);

  const attachRegionMismatch = useCallback((volumeId: string, instanceId: string) => {
    const vol = volumes.find((v) => v.volume_id === volumeId);
    const inst = instances.find((i) => i.job_id === instanceId);
    const volRegion = (vol?.region || "").trim().toLowerCase();
    const hostRegion = hostRegionById(inst?.host_id);
    return Boolean(volRegion && hostRegion && volRegion !== hostRegion);
  }, [volumes, instances, hostRegionById]);

  const handleAttach = async (volumeId: string) => {
    if (!selectedInstance) return toast.error("Select an instance first");
    try {
      const res = await api.attachVolume(volumeId, selectedInstance);
      if (res.region_warning) {
        toast.warning(res.region_warning);
      }
      toast.success("Volume attached");
      setAttachingId(null);
      setSelectedInstance("");
      load();
    } catch (e: unknown) {
      toast.error(volumeActionError(e, t("dash.volumes.viewer_blocked")));
    }
  };

  const openSnapshots = async (volumeId: string) => {
    setSnapshotsVolId(volumeId);
    setSnapshotLabel("");
    setSnapshotsLoading(true);
    try {
      const res = await api.listVolumeSnapshots(volumeId);
      setSnapshots(res.snapshots || []);
    } catch {
      toast.error(t("dash.volumes.load_failed"));
      setSnapshotsVolId(null);
    } finally {
      setSnapshotsLoading(false);
    }
  };

  const handleCreateSnapshot = async () => {
    if (!snapshotsVolId) return;
    setSnapshotCreating(true);
    try {
      await api.createVolumeSnapshot(snapshotsVolId, snapshotLabel.trim());
      toast.success(t("dash.volumes.snapshot_created"));
      setSnapshotLabel("");
      const res = await api.listVolumeSnapshots(snapshotsVolId);
      setSnapshots(res.snapshots || []);
    } catch (e: unknown) {
      toast.error(volumeActionError(e, t("dash.volumes.viewer_blocked")));
    } finally {
      setSnapshotCreating(false);
    }
  };

  const handleDeleteSnapshot = async (snapshotId: string) => {
    setConfirmSnapshotId(null);
    if (!snapshotsVolId) return;
    try {
      await api.deleteVolumeSnapshot(snapshotsVolId, snapshotId);
      toast.success(t("dash.volumes.snapshot_deleted"));
      setSnapshots((prev) => prev.filter((s) => s.snapshot_id !== snapshotId));
    } catch (e: unknown) {
      toast.error(volumeActionError(e, t("dash.volumes.viewer_blocked")));
    }
  };

  const handleRetry = async (volumeId: string) => {
    const vol = volumes.find((v) => v.volume_id === volumeId);
    if (vol && (vol.status === "provisioning" || vol.status === "creating")) {
      toast.info(t("dash.volumes.retry_wait_provisioning"), { id: `retry-${volumeId}`, duration: 3500 });
      return;
    }
    const toastId = `retry-${volumeId}`;
    try {
      const res = await api.retryVolume(volumeId);
      if (res.volume?.status === "available") {
        toast.success(t("dash.volumes.retry_success"), { id: toastId, duration: 3500 });
      } else {
        toast.error(t("dash.volumes.retry_failed"), { id: toastId, duration: 4000 });
      }
      load();
    } catch (e: unknown) {
      const msg = volumeActionError(e, t("dash.volumes.viewer_blocked"));
      if (/provisioning/i.test(msg) && /not 'error'/i.test(msg)) {
        toast.info(t("dash.volumes.retry_wait_provisioning"), { id: toastId, duration: 3500 });
        load();
        return;
      }
      toast.error(msg, { id: toastId, duration: 4500 });
    }
  };

  const handleRename = async (volumeId: string) => {
    const trimmed = renameValue.trim();
    if (!trimmed) return;
    try {
      await api.renameVolume(volumeId, trimmed);
      toast.success("Volume renamed");
      setRenamingId(null);
      setRenameValue("");
      load();
    } catch (e: unknown) {
      toast.error(volumeActionError(e, t("dash.volumes.viewer_blocked")));
    }
  };

  const totalStorage = volumes.reduce((sum, v) => sum + (v.size_gb || 0), 0);
  const attachedCount = volumes.filter((v) => v.status === "attached").length;
  const estimatedCost = (newSize * PRICE_PER_GB).toFixed(2);
  const runningInstances = instances.filter((i) => i.status === "running");
  const regions = [...new Set([
    ...gpus.map((g) => g.region).filter(Boolean),
    ...volumes.map((v) => v.region).filter(Boolean),
  ])];
  if (regions.length === 0) regions.push("ca-east");
  const totalMonthlyCost = volumes.reduce((sum, v) => sum + (v.monthly_cost_cad ?? v.size_gb * PRICE_PER_GB), 0);

  const statusColor = (s: string): "default" | "active" | "warning" | "failed" => {
    if (s === "available") return "default";
    if (s === "attached") return "active";
    if (s === "error") return "failed";
    return "warning";
  };

  // Filter & sort
  const filtered = volumes
    .filter((v) => {
      if (statusFilter !== "all" && v.status !== statusFilter) return false;
      if (search && !v.name?.toLowerCase().includes(search.toLowerCase()) && !v.volume_id?.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    })
    .sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case "name": cmp = (a.name || "").localeCompare(b.name || ""); break;
        case "size_gb": cmp = (a.size_gb || 0) - (b.size_gb || 0); break;
        case "status": cmp = (a.status || "").localeCompare(b.status || ""); break;
        case "region": cmp = (a.region || "").localeCompare(b.region || ""); break;
        case "monthly_cost": cmp = (a.monthly_cost_cad ?? a.size_gb * PRICE_PER_GB) - (b.monthly_cost_cad ?? b.size_gb * PRICE_PER_GB); break;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });

  const { paginate, totalPages } = usePagination(filtered, 10);
  const pageItems = paginate(page);

  useEffect(() => { setPage(1); }, [search, statusFilter, sortKey, sortDir]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir(sortDir === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("asc"); }
  }

  function SortIcon({ col }: { col: SortKey }) {
    if (sortKey !== col) return <ArrowUpDown className="h-3 w-3 opacity-40" />;
    return sortDir === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />;
  }

  return (
    <div className="space-y-6">
      {/* Hero Header */}
      <FadeIn>
      <div className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-surface via-surface to-accent-violet/5 p-6 md:p-8">
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-accent-violet/10 blur-3xl" />
        <div className="absolute -bottom-10 -left-10 h-48 w-48 rounded-full bg-accent-cyan/10 blur-3xl" />
        <div className="relative z-10 flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-accent-violet/10 ring-1 ring-accent-violet/20">
                <HardDrive className="h-6 w-6 text-accent-violet" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Persistent Volumes</h1>
                <p className="text-sm text-text-secondary">
                  Encrypted persistent storage for your workloads
                </p>
              </div>
            </div>
            <p className="text-sm text-text-muted max-w-lg mt-3">
              Create and manage persistent block storage volumes. Attach them to running instances for persistent data that survives instance restarts.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button variant="outline" className="h-10" onClick={() => load({ refresh: true })}>
              <RefreshCw className="h-4 w-4" /> Refresh
            </Button>
            <Button
              className="h-10 bg-accent-violet hover:bg-accent-violet/90 text-white"
              onClick={() => setShowCreate(true)}
              disabled={!canWrite}
            >
              <Plus className="h-4 w-4" /> Create Volume
            </Button>
          </div>
        </div>
      </div>
      </FadeIn>

      <TeamContextBanner team={team} variant="volumes" />

      {/* Stats Row */}
      <StaggerList className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StaggerItem><StatCard label="Total Volumes" value={volumes.length} icon={HardDrive} glow="violet" /></StaggerItem>
        <StaggerItem><StatCard label="Total Storage" value={`${totalStorage} GB`} icon={Database} glow="cyan" /></StaggerItem>
        <StaggerItem><StatCard label="Attached" value={attachedCount} icon={Link} glow="emerald" /></StaggerItem>
        <StaggerItem>
          <StatCard
            label={team.isTeamMember ? t("dash.volumes.monthly_cost_team") : t("dash.volumes.monthly_cost_personal")}
            value={`$${totalMonthlyCost.toFixed(2)}`}
            icon={DollarSign}
            glow="gold"
          />
        </StaggerItem>
      </StaggerList>

      {/* Create Volume Modal */}
      <Dialog
        open={showCreate}
        onClose={() => setShowCreate(false)}
        title="Create Volume"
        description="Provision persistent storage. Billed in real-time from your credits."
        maxWidth="max-w-lg"
      >
        <div className="space-y-4">
          <div>
            <label className="text-xs font-medium text-text-secondary mb-1.5 block">Volume Name</label>
            <Input
              placeholder="my-dataset"
              value={newName}
              maxLength={128}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewName(e.target.value)}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-medium text-text-secondary mb-1.5 block">Size</label>
              <div className="relative group">
                <Input
                  type="number"
                  min={1}
                  max={2000}
                  placeholder="Size"
                  value={newSize}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewSize(Math.max(1, Number(e.target.value) || 1))}
                  className="pr-16"
                />
                <span className="absolute right-9 top-1/2 -translate-y-1/2 text-xs text-text-muted pointer-events-none">GB</span>
                <div className="absolute right-1 top-1/2 -translate-y-1/2 flex flex-col">
                  <button
                    type="button"
                    tabIndex={-1}
                    onClick={() => setNewSize(Math.min(2000, newSize + 10))}
                    className="flex items-center justify-center h-4 w-6 rounded-t text-text-muted hover:text-accent-cyan hover:bg-accent-cyan/10 transition-colors"
                  >
                    <ChevronUp className="h-3 w-3" />
                  </button>
                  <button
                    type="button"
                    tabIndex={-1}
                    onClick={() => setNewSize(Math.max(1, newSize - 10))}
                    className="flex items-center justify-center h-4 w-6 rounded-b text-text-muted hover:text-accent-cyan hover:bg-accent-cyan/10 transition-colors"
                  >
                    <ChevronDown className="h-3 w-3" />
                  </button>
                </div>
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-text-secondary mb-1.5 block">Region</label>
              <div className="relative">
                <select
                  value={newRegion}
                  onChange={(e) => setNewRegion(e.target.value)}
                  className="h-9 w-full rounded-md border border-border bg-background px-3 pr-8 text-sm appearance-none"
                >
                  {regions.map((r) => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-text-muted pointer-events-none" />
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setNewEncrypted(!newEncrypted)}
            className={cn(
              "flex items-center gap-3 w-full rounded-lg border p-3 text-sm transition-colors text-left",
              newEncrypted
                ? "border-accent-violet/50 bg-accent-violet/5"
                : "border-border/60 bg-surface-hover/40 hover:border-border",
            )}
          >
            <Lock className={cn("h-4 w-4 shrink-0", newEncrypted ? "text-accent-violet" : "text-text-muted")} />
            <div className="flex-1 min-w-0">
              <span className={cn("font-medium", newEncrypted ? "text-text-primary" : "text-text-secondary")}>
                LUKS2 Encryption
              </span>
              <p className="text-xs text-text-muted mt-0.5">AES-256 at rest &middot; per-volume key &middot; cryptographic erasure on delete</p>
            </div>
            <div className={cn(
              "h-5 w-9 rounded-full relative transition-colors shrink-0",
              newEncrypted ? "bg-accent-violet" : "bg-border",
            )}>
              <div className={cn(
                "absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform shadow-sm",
                newEncrypted ? "translate-x-4" : "translate-x-0.5",
              )} />
            </div>
          </button>
          <div className="rounded-lg border border-border/60 bg-surface-hover/40 p-3 space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="text-text-muted">${PRICE_PER_GB}/GB/month &middot; {newSize} GB</span>
              <span className="font-semibold text-accent-violet">${estimatedCost} CAD/mo</span>
            </div>
            <p className="text-xs text-text-muted">
              {team.isTeamMember
                ? t("dash.volumes.create_cost_team", { cost: estimatedCost })
                : t("dash.volumes.create_cost_personal", { cost: estimatedCost })}
            </p>
          </div>
          <div className="flex justify-end gap-3 pt-2">
            <Button variant="outline" onClick={() => setShowCreate(false)}>Cancel</Button>
            <Button
              className="bg-accent-violet hover:bg-accent-violet/90 text-white"
              onClick={handleCreate}
              disabled={creating}
            >
              {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
              Create Volume
            </Button>
          </div>
        </div>
      </Dialog>

      {/* Snapshots Modal */}
      <Dialog
        open={snapshotsVolId !== null}
        onClose={() => { setSnapshotsVolId(null); setSnapshots([]); }}
        title={t("dash.volumes.snapshots")}
        description="Instant copy-on-write snapshots stored on NFS."
        maxWidth="max-w-lg"
      >
        <div className="space-y-4">
          {canWrite && (
            <div className="flex gap-2">
              <Input
                placeholder={t("dash.volumes.snapshot_label")}
                value={snapshotLabel}
                maxLength={128}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSnapshotLabel(e.target.value)}
                className="flex-1"
              />
              <Button
                className="bg-accent-violet hover:bg-accent-violet/90 text-white shrink-0"
                onClick={handleCreateSnapshot}
                disabled={snapshotCreating}
              >
                {snapshotCreating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Camera className="h-4 w-4" />}
                {t("dash.volumes.snapshot_create")}
              </Button>
            </div>
          )}
          {snapshotsLoading ? (
            <div className="flex justify-center py-8"><Loader2 className="h-6 w-6 animate-spin text-text-muted" /></div>
          ) : snapshots.length === 0 ? (
            <p className="text-sm text-text-muted text-center py-6">{t("dash.volumes.snapshots_empty")}</p>
          ) : (
            <ul className="divide-y divide-border/60 rounded-lg border border-border/60 overflow-hidden">
              {snapshots.map((snap) => (
                <li key={snap.snapshot_id} className="flex items-center justify-between gap-3 px-4 py-3 text-sm">
                  <div className="min-w-0">
                    <p className="font-medium truncate">{snap.label || snap.snapshot_id}</p>
                    <p className="text-xs text-text-muted font-mono truncate">{snap.snapshot_id}</p>
                  </div>
                  {canWrite && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 text-text-muted hover:text-red-500 shrink-0"
                      onClick={() => setConfirmSnapshotId(snap.snapshot_id)}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </Dialog>

      {/* Filters */}
      <FadeIn delay={0.12}>
      <Card className="border-border/60">
        <CardContent className="py-4 px-5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted pointer-events-none" />
              <Input
                className="pl-10 h-10"
                placeholder="Search by name or ID..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <Select
              className="h-10 min-w-[160px]"
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
            >
              <option value="all">All Statuses</option>
              <option value="available">Available</option>
              <option value="attached">Attached</option>
              <option value="error">Error</option>
              <option value="creating">Creating</option>
              <option value="deleting">Deleting</option>
            </Select>
          </div>
        </CardContent>
      </Card>
      </FadeIn>

      {/* Volumes Table */}
      {loading ? (
        <div className="space-y-3">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-[72px] rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <Card className="border-border/60 overflow-hidden">
          <div className="relative py-16 px-8 text-center">
            <div className="absolute inset-0 bg-gradient-to-b from-accent-violet/5 to-transparent" />
            <div className="relative">
              <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-2xl bg-surface-hover ring-1 ring-border">
                <HardDrive className="h-10 w-10 text-text-muted" />
              </div>
              <h3 className="text-xl font-semibold mb-2">No volumes yet</h3>
              <p className="text-sm text-text-secondary mb-8 max-w-md mx-auto">
                {team.isTeamMember ? t("dash.volumes.empty_team") : "Create persistent storage to keep your data safe across instance restarts and migrations."}
              </p>
              <Button
                className="h-11 bg-accent-violet hover:bg-accent-violet/90 text-white"
                onClick={() => setShowCreate(true)}
                disabled={!canWrite}
              >
                <Plus className="h-4 w-4" /> Create Your First Volume
              </Button>
            </div>
          </div>
        </Card>
      ) : (
        <FadeIn delay={0.18}>
        <Card className="brand-top-accent border-border/60 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/60 bg-gradient-to-r from-surface-hover/80 to-surface-hover/40">
                  <th
                    className="h-12 px-5 text-left font-semibold text-text-primary cursor-pointer select-none hover:bg-surface-hover/60 transition-colors"
                    onClick={() => toggleSort("name")}
                  >
                    <span className="inline-flex items-center gap-2">
                      <HardDrive className="h-3.5 w-3.5 text-text-muted" />
                      Volume
                      <SortIcon col="name" />
                    </span>
                  </th>
                  <th
                    className="h-12 px-4 text-center font-semibold text-text-primary cursor-pointer select-none hover:bg-surface-hover/60 transition-colors"
                    onClick={() => toggleSort("size_gb")}
                  >
                    <span className="inline-flex items-center gap-2 justify-center">
                      <Database className="h-3.5 w-3.5 text-text-muted" />
                      Size
                      <SortIcon col="size_gb" />
                    </span>
                  </th>
                  <th
                    className="h-12 px-4 text-center font-semibold text-text-primary cursor-pointer select-none hover:bg-surface-hover/60 transition-colors"
                    onClick={() => toggleSort("status")}
                  >
                    <span className="inline-flex items-center gap-2 justify-center">
                      Status
                      <SortIcon col="status" />
                    </span>
                  </th>
                  <th
                    className="h-12 px-4 text-center font-semibold text-text-primary cursor-pointer select-none hover:bg-surface-hover/60 transition-colors"
                    onClick={() => toggleSort("region")}
                  >
                    <span className="inline-flex items-center gap-2 justify-center">
                      <Globe className="h-3.5 w-3.5 text-text-muted" />
                      Region
                      <SortIcon col="region" />
                    </span>
                  </th>
                  <th
                    className="h-12 px-4 text-center font-semibold text-text-primary cursor-pointer select-none hover:bg-surface-hover/60 transition-colors"
                    onClick={() => toggleSort("monthly_cost")}
                  >
                    <span className="inline-flex items-center gap-2 justify-center">
                      <DollarSign className="h-3.5 w-3.5 text-text-muted" />
                      Cost
                      <SortIcon col="monthly_cost" />
                    </span>
                  </th>
                  <th className="h-12 px-5 text-right font-semibold text-text-primary">Actions</th>
                </tr>
              </thead>
              <tbody>
                {pageItems.map((vol, idx) => {
                  const attachedInstance = vol.status === "attached"
                    ? instances.find((inst) =>
                        inst.attached_volumes?.some((av) => av.volume_id === vol.volume_id)
                        || vol.attached_to === inst.job_id
                      )
                    : undefined;

                  return (
                    <tr
                      key={vol.volume_id}
                      className={cn(
                        "group transition-colors hover:bg-accent-violet/5",
                        idx !== pageItems.length - 1 && "border-b border-border/40"
                      )}
                    >
                      {/* Volume name + ID */}
                      <td className="py-4 px-5">
                        <div className="flex items-center gap-3">
                          <div className={cn(
                            "flex h-10 w-10 items-center justify-center rounded-lg ring-1 transition-colors",
                            vol.status === "attached"
                              ? "bg-emerald/10 ring-emerald/20 text-emerald"
                              : "bg-surface-hover ring-border text-text-muted"
                          )}>
                            <HardDrive className="h-5 w-5" />
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              {renamingId === vol.volume_id ? (
                                <form
                                  className="flex items-center gap-1"
                                  onSubmit={(e) => { e.preventDefault(); handleRename(vol.volume_id); }}
                                >
                                  <input
                                    autoFocus
                                    value={renameValue}
                                    onChange={(e) => setRenameValue(e.target.value)}
                                    onKeyDown={(e) => { if (e.key === "Escape") { setRenamingId(null); setRenameValue(""); } }}
                                    onBlur={() => { setRenamingId(null); setRenameValue(""); }}
                                    className="h-7 w-40 rounded border border-border bg-background px-2 text-sm font-medium text-text-primary focus:outline-none focus:ring-1 focus:ring-ice-blue"
                                  />
                                  <button type="submit" onMouseDown={(e) => e.preventDefault()} className="text-emerald hover:text-emerald/80 p-0.5"><Check className="h-3.5 w-3.5" /></button>
                                  <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => { setRenamingId(null); setRenameValue(""); }} className="text-text-muted hover:text-text-primary p-0.5"><X className="h-3.5 w-3.5" /></button>
                                </form>
                              ) : (
                                <>
                                  <p className="font-medium text-text-primary">{vol.name || "Unnamed"}</p>
                                  {canWrite && (
                                    <button
                                      onClick={() => { setRenamingId(vol.volume_id); setRenameValue(vol.name || ""); }}
                                      className="opacity-0 group-hover:opacity-100 text-text-muted hover:text-ice-blue transition-all p-0.5"
                                      title="Rename volume"
                                    >
                                      <Pencil className="h-3 w-3" />
                                    </button>
                                  )}
                                </>
                              )}
                              {vol.encrypted && (
                                <Lock className="h-3 w-3 text-accent-violet" />
                              )}
                            </div>
                            <div className="flex items-center gap-2 mt-0.5">
                              <CopyableId id={vol.volume_id} />
                              {attachedInstance && (
                                <NextLink
                                  href={`/dashboard/instances/${attachedInstance.job_id}`}
                                  className="inline-flex items-center gap-1 text-xs text-ice-blue hover:underline"
                                >
                                  <ExternalLink className="h-3 w-3" />
                                  {attachedInstance.name || attachedInstance.job_id.slice(0, 12)}
                                </NextLink>
                              )}
                            </div>
                          </div>
                        </div>
                      </td>

                      {/* Size */}
                      <td className="py-4 px-4 text-center">
                        <span className="font-mono text-sm text-text-secondary">
                          {vol.size_gb} GB
                        </span>
                      </td>

                      {/* Status */}
                      <td className="py-4 px-4 text-center">
                        <div className="inline-flex flex-col items-center gap-0.5">
                          <Badge variant={statusColor(vol.status)}>{vol.status}</Badge>
                          {vol.status === "error" && (
                            <span className="text-[10px] text-text-muted max-w-[140px] leading-tight" title={t("dash.volumes.error_hint")}>
                              {t("dash.volumes.error_hint")}
                            </span>
                          )}
                        </div>
                      </td>

                      {/* Region */}
                      <td className="py-4 px-4 text-center">
                        <span className="text-sm text-text-secondary">{vol.region || "—"}</span>
                      </td>

                      {/* Cost */}
                      <td className="py-4 px-4 text-center">
                        <span className="font-mono text-sm font-medium text-accent-violet">
                          ${(vol.monthly_cost_cad ?? vol.size_gb * PRICE_PER_GB).toFixed(2)}/mo
                        </span>
                      </td>

                      {/* Actions */}
                      <td className="py-4 px-5 text-right">
                        <div className="flex items-center justify-end gap-2">
                          {!canWrite ? (
                            <span className="text-xs text-text-muted">{t("dash.team.instances_read_only")}</span>
                          ) : vol.status === "attached" ? (
                            <>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 text-text-secondary hover:text-accent-gold hover:bg-accent-gold/10"
                                onClick={() => {
                                  api.detachVolume(vol.volume_id)
                                    .then(() => { toast.success("Volume detached"); load(); })
                                    .catch((e: unknown) => toast.error(volumeActionError(e, t("dash.volumes.viewer_blocked"))));
                                }}
                              >
                                <Unlink className="h-3.5 w-3.5 mr-1" /> Detach
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 text-text-secondary hover:text-accent-violet hover:bg-accent-violet/10"
                                onClick={() => openSnapshots(vol.volume_id)}
                                title={t("dash.volumes.snapshots")}
                              >
                                <Camera className="h-3.5 w-3.5" />
                              </Button>
                            </>
                          ) : vol.status === "creating" || vol.status === "provisioning" || vol.status === "deleting" ? (
                            <span className="inline-flex items-center gap-1.5 text-xs text-text-muted">
                              <Loader2 className="h-3.5 w-3.5 animate-spin" /> {vol.status}…
                            </span>
                          ) : vol.status === "error" ? (
                            <>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 text-accent-gold hover:bg-accent-gold/10"
                                onClick={() => handleRetry(vol.volume_id)}
                              >
                                <RefreshCw className="h-3.5 w-3.5 mr-1" /> Retry
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 text-text-muted hover:text-red-500 hover:bg-red-500/10"
                                onClick={() => setConfirmDeleteId(vol.volume_id)}
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </Button>
                            </>
                          ) : attachingId === vol.volume_id ? (
                            <div className="flex flex-col items-end gap-1">
                              <div className="flex items-center gap-2">
                                <select
                                  value={selectedInstance}
                                  onChange={(e) => setSelectedInstance(e.target.value)}
                                  className="h-8 rounded-md border border-border bg-background px-2 text-xs"
                                >
                                  <option value="">Select instance…</option>
                                  {runningInstances.map((inst) => (
                                    <option key={inst.job_id} value={inst.job_id}>
                                      {inst.name || inst.job_id}{inst.gpu_model ? ` (${inst.gpu_model})` : ""}
                                    </option>
                                  ))}
                                </select>
                                <Button size="sm" className="h-8" onClick={() => handleAttach(vol.volume_id)}>
                                  <Link className="h-3.5 w-3.5" /> Go
                                </Button>
                                <Button variant="ghost" size="sm" className="h-8" onClick={() => { setAttachingId(null); setSelectedInstance(""); }}>
                                  ✕
                                </Button>
                              </div>
                              {selectedInstance && attachRegionMismatch(vol.volume_id, selectedInstance) && (
                                <span className="inline-flex items-center gap-1 text-[10px] text-accent-gold max-w-[220px] text-right leading-tight">
                                  <AlertTriangle className="h-3 w-3 shrink-0" />
                                  {t("dash.volumes.attach_region_mismatch")}
                                </span>
                              )}
                            </div>
                          ) : (
                            <>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 text-text-secondary hover:text-accent-cyan hover:bg-accent-cyan/10"
                                onClick={() => setAttachingId(vol.volume_id)}
                              >
                                <Link className="h-3.5 w-3.5 mr-1" /> Attach
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 text-text-secondary hover:text-accent-violet hover:bg-accent-violet/10"
                                onClick={() => openSnapshots(vol.volume_id)}
                                title={t("dash.volumes.snapshots")}
                              >
                                <Camera className="h-3.5 w-3.5" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 text-ice-blue hover:bg-ice-blue/10"
                                onClick={() => setLaunchVolumeId(vol.volume_id)}
                              >
                                <Rocket className="h-3.5 w-3.5 mr-1" /> Launch
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 text-text-muted hover:text-red-500 hover:bg-red-500/10"
                                onClick={() => setConfirmDeleteId(vol.volume_id)}
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </Button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="border-t border-border/60 bg-surface-hover/30 px-5 py-3 flex items-center justify-between">
              <span className="text-xs text-text-muted">
                Showing {pageItems.length} of {filtered.length} volume{filtered.length !== 1 ? "s" : ""}
              </span>
              <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
            </div>
          )}
        </Card>
        </FadeIn>
      )}

      <ConfirmDialog
        open={confirmDeleteId !== null}
        title="Delete this volume?"
        description="All data on the volume will be permanently destroyed. This cannot be undone."
        confirmLabel="Delete volume"
        variant="danger"
        onConfirm={() => confirmDeleteId && handleDelete(confirmDeleteId)}
        onCancel={() => setConfirmDeleteId(null)}
      />

      <ConfirmDialog
        open={confirmSnapshotId !== null}
        title="Delete this snapshot?"
        description="The snapshot will be permanently removed. The volume itself is not affected."
        confirmLabel="Delete snapshot"
        variant="danger"
        onConfirm={() => confirmSnapshotId && handleDeleteSnapshot(confirmSnapshotId)}
        onCancel={() => setConfirmSnapshotId(null)}
      />

      {/* Launch Instance Modal — pre-selects the chosen volume */}
      <LaunchInstanceModal
        open={launchVolumeId !== null}
        onClose={() => setLaunchVolumeId(null)}
        onLaunched={() => { load(); }}
        preSelectedVolumeIds={launchVolumeId ? [launchVolumeId] : undefined}
      />
    </div>
  );
}
