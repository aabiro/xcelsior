"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  HardDrive, RefreshCw, Plus, Trash2, Loader2, Link, Unlink, Copy, Check, ChevronDown, Globe,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import * as api from "@/lib/api";
import type { Volume, Instance, GpuAvailability } from "@/lib/api";
import { toast } from "sonner";

const PRICE_PER_GB = 0.07;

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

  const handleCreate = async () => {
    if (!newName.trim()) return toast.error("Volume name required");
    setCreating(true);
    try {
      await api.createVolume({ name: newName.trim(), size_gb: newSize, region: newRegion });
      toast.success("Volume created");
      setNewName("");
      load();
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to create volume");
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (volumeId: string) => {
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

  const statusColor = (s: string): "default" | "active" | "warning" => {
    if (s === "available") return "default";
    if (s === "attached") return "active";
    return "warning";
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Persistent Volumes</h1>
          <p className="text-sm text-text-muted">
            NFS-backed encrypted storage. Attach to instances for persistent workspaces.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-3.5 w-3.5" /> Refresh
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard label="Total Volumes" value={volumes.length} icon={HardDrive} />
        <StatCard label="Total Storage" value={`${totalStorage} GB`} icon={HardDrive} />
        <StatCard label="Attached" value={attachedCount} icon={Link} />
      </div>

      {/* Create Volume */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Create Volume</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <div className="flex-1">
              <Input
                placeholder="Volume name"
                value={newName}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewName(e.target.value)}
              />
            </div>
            <div className="flex items-center gap-2">
              <div className="relative">
                <Input
                  type="number"
                  min={1}
                  max={2000}
                  placeholder="Size"
                  value={newSize}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewSize(Number(e.target.value))}
                  className="w-28 pr-10"
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-text-muted pointer-events-none">
                  GB
                </span>
              </div>
              <div className="relative">
                <select
                  value={newRegion}
                  onChange={(e) => setNewRegion(e.target.value)}
                  className="h-9 rounded-md border border-border bg-background px-2 pr-7 text-xs appearance-none"
                >
                  {regions.map((r) => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-1.5 top-1/2 -translate-y-1/2 h-3 w-3 text-text-muted pointer-events-none" />
              </div>
              <Button onClick={handleCreate} disabled={creating}>
                {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                Create
              </Button>
            </div>
          </div>
          <p className="mt-2 text-xs text-text-muted">
            ${PRICE_PER_GB}/GB/month &middot; {newSize} GB = <span className="font-semibold text-text-primary">${estimatedCost} CAD/mo</span> &middot; Billed in real-time from your credits
          </p>
        </CardContent>
      </Card>

      {/* Volume List */}
      <div className="space-y-3">
        {loading ? (
          <div className="flex justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
          </div>
        ) : volumes.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center text-text-muted">
              <HardDrive className="mx-auto h-8 w-8 mb-2 opacity-50" />
              No volumes yet. Create one above.
            </CardContent>
          </Card>
        ) : (
          volumes.map((vol) => (
            <Card key={vol.volume_id}>
              <CardContent className="flex items-center justify-between py-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <HardDrive className="h-4 w-4 text-text-muted" />
                    <span className="font-medium">{vol.name}</span>
                    <Badge variant={statusColor(vol.status)}>{vol.status}</Badge>
                    {vol.encrypted && (
                      <Badge variant="info" className="text-xs">🔒 Encrypted</Badge>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-text-muted">
                    <span>{vol.size_gb} GB</span>
                    <span>{vol.region}</span>
                    <span className="font-medium text-text-primary">
                      ${(vol.monthly_cost_cad ?? vol.size_gb * PRICE_PER_GB).toFixed(2)}/mo
                    </span>
                    <CopyableId id={vol.volume_id} />
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {vol.status === "attached" ? (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => api.detachVolume(vol.volume_id).then(load)}
                    >
                      <Unlink className="h-3.5 w-3.5" /> Detach
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
                      <Button size="sm" onClick={() => handleAttach(vol.volume_id)}>
                        <Link className="h-3.5 w-3.5" /> Attach
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => { setAttachingId(null); setSelectedInstance(""); }}>
                        Cancel
                      </Button>
                    </div>
                  ) : (
                    <>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setAttachingId(vol.volume_id)}
                      >
                        <Link className="h-3.5 w-3.5" /> Attach
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDelete(vol.volume_id)}
                        className="text-red-500 hover:text-red-600"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </>
                  )}
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}
