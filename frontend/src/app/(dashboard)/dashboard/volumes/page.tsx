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
  HardDrive, RefreshCw, Plus, Trash2, Loader2, Link, Unlink, Copy, Check, ChevronDown, Globe, Lock,
  Rocket, ExternalLink, Search, ArrowUpDown, ArrowUp, ArrowDown, DollarSign, Database,
} from "lucide-react";
import { FadeIn, StaggerList, StaggerItem } from "@/components/ui/motion";
import { useAuth } from "@/lib/auth";
import * as api from "@/lib/api";
import type { Volume, Instance, GpuAvailability } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

const PRICE_PER_GB = 0.07;

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

export default function VolumesPage() {
  const { user } = useAuth();
  const [volumes, setVolumes] = useState<Volume[]>([]);
  const [instances, setInstances] = useState<Instance[]>([]);
  const [gpus, setGpus] = useState<GpuAvailability[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newSize, setNewSize] = useState(50);
  const [newRegion, setNewRegion] = useState("ca-east");
  const [attachingId, setAttachingId] = useState<string | null>(null);
  const [selectedInstance, setSelectedInstance] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [page, setPage] = useState(1);

  const load = useCallback(() => {
    setLoading(true);
    Promise.all([
      api.listVolumes().then((res) => setVolumes(res.volumes || [])),
      api.fetchInstances().then((res) => setInstances(res.instances || [])).catch(() => {}),
      api.fetchAvailableGPUs().then((res) => setGpus(res.gpus || [])).catch(() => {}),
    ])
      .catch(() => toast.error("Failed to load volumes"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  // Live updates — re-fetch on volume/instance lifecycle changes
  useEventStream({
    eventTypes: ["volume_created", "volume_deleted", "volume_attached", "volume_detached", "job_status"],
    onEvent: () => { load(); },
  });

  const handleCreate = async () => {
    if (!newName.trim()) return toast.error("Volume name required");
    setCreating(true);
    try {
      await api.createVolume({ name: newName.trim(), size_gb: newSize, region: newRegion });
      toast.success("Volume created");
      setNewName("");
      setShowCreate(false);
      load();
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to create volume");
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (volumeId: string) => {
    if (!confirm("Delete this volume? This action cannot be undone.")) return;
    try {
      await api.deleteVolume(volumeId);
      toast.success("Volume deleted");
      load();
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Cannot delete — may have active attachments");
    }
  };

  const handleAttach = async (volumeId: string) => {
    if (!selectedInstance) return toast.error("Select an instance first");
    try {
      await api.attachVolume(volumeId, selectedInstance);
      toast.success("Volume attached");
      setAttachingId(null);
      setSelectedInstance("");
      load();
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to attach volume");
    }
  };

  const totalStorage = volumes.reduce((sum, v) => sum + (v.size_gb || 0), 0);
  const attachedCount = volumes.filter((v) => v.status === "attached").length;
  const estimatedCost = (newSize * PRICE_PER_GB).toFixed(2);
  const runningInstances = instances.filter((i) => i.status === "running");
  const regions = [...new Set(gpus.map((g) => g.region))];
  if (regions.length === 0) regions.push("ca-east");
  const totalMonthlyCost = volumes.reduce((sum, v) => sum + (v.monthly_cost_cad ?? v.size_gb * PRICE_PER_GB), 0);

  const statusColor = (s: string): "default" | "active" | "warning" => {
    if (s === "available") return "default";
    if (s === "attached") return "active";
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
              Create and manage encrypted block storage volumes. Attach them to running instances for persistent data that survives instance restarts.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button variant="outline" className="h-10" onClick={load}>
              <RefreshCw className="h-4 w-4" /> Refresh
            </Button>
            <Button
              className="h-10 bg-accent-violet hover:bg-accent-violet/90 text-white"
              onClick={() => setShowCreate(true)}
            >
              <Plus className="h-4 w-4" /> Create Volume
            </Button>
          </div>
        </div>
      </div>
      </FadeIn>

      {/* Stats Row */}
      <StaggerList className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StaggerItem><StatCard label="Total Volumes" value={volumes.length} icon={HardDrive} glow="violet" /></StaggerItem>
        <StaggerItem><StatCard label="Total Storage" value={`${totalStorage} GB`} icon={Database} glow="cyan" /></StaggerItem>
        <StaggerItem><StatCard label="Attached" value={attachedCount} icon={Link} glow="emerald" /></StaggerItem>
        <StaggerItem><StatCard label="Monthly Cost" value={`$${totalMonthlyCost.toFixed(2)}`} icon={DollarSign} glow="gold" /></StaggerItem>
      </StaggerList>

      {/* Create Volume Modal */}
      <Dialog
        open={showCreate}
        onClose={() => setShowCreate(false)}
        title="Create Volume"
        description="Provision encrypted persistent storage. Billed in real-time from your credits."
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
              <div className="relative">
                <Input
                  type="number"
                  min={1}
                  max={2000}
                  placeholder="Size"
                  value={newSize}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewSize(Math.max(1, Number(e.target.value) || 1))}
                  className="pr-10"
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-text-muted pointer-events-none">GB</span>
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
          <div className="rounded-lg border border-border/60 bg-surface-hover/40 p-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-text-muted">${PRICE_PER_GB}/GB/month &middot; {newSize} GB</span>
              <span className="font-semibold text-accent-violet">${estimatedCost} CAD/mo</span>
            </div>
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
                Create encrypted persistent storage to keep your data safe across instance restarts and migrations.
              </p>
              <Button
                className="h-11 bg-accent-violet hover:bg-accent-violet/90 text-white"
                onClick={() => setShowCreate(true)}
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
                              <p className="font-medium text-text-primary">{vol.name || "Unnamed"}</p>
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
                        <Badge variant={statusColor(vol.status)}>{vol.status}</Badge>
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
                          {vol.status === "attached" ? (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 text-text-secondary hover:text-accent-gold hover:bg-accent-gold/10"
                              onClick={() => {
                                api.detachVolume(vol.volume_id)
                                  .then(() => { toast.success("Volume detached"); load(); })
                                  .catch((e: unknown) => toast.error(e instanceof Error ? e.message : "Failed to detach volume"));
                              }}
                            >
                              <Unlink className="h-3.5 w-3.5 mr-1" /> Detach
                            </Button>
                          ) : attachingId === vol.volume_id ? (
                            <div className="flex items-center gap-2">
                              <select
                                value={selectedInstance}
                                onChange={(e) => setSelectedInstance(e.target.value)}
                                className="h-8 rounded-md border border-border bg-background px-2 text-xs"
                              >
                                <option value="">Select instance…</option>
                                {runningInstances.map((inst) => (
                                  <option key={inst.job_id} value={inst.job_id}>
                                    {inst.name || inst.job_id} ({inst.gpu_model})
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
                              <NextLink href="/dashboard/instances">
                                <Button variant="ghost" size="sm" className="h-8 text-ice-blue hover:bg-ice-blue/10">
                                  <Rocket className="h-3.5 w-3.5 mr-1" /> Launch
                                </Button>
                              </NextLink>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 text-text-muted hover:text-red-500 hover:bg-red-500/10"
                                onClick={() => handleDelete(vol.volume_id)}
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
    </div>
  );
}
