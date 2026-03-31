"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  HardDrive, RefreshCw, Plus, Trash2, Loader2, Link, Unlink,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import * as api from "@/lib/api";
import type { Volume } from "@/lib/api";
import { toast } from "sonner";

export default function VolumesPage() {
  const { user } = useAuth();
  const [volumes, setVolumes] = useState<Volume[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newSize, setNewSize] = useState(50);

  const load = useCallback(() => {
    setLoading(true);
    api.listVolumes()
      .then((res) => setVolumes(res.volumes || []))
      .catch(() => toast.error("Failed to load volumes"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleCreate = async () => {
    if (!newName.trim()) return toast.error("Volume name required");
    setCreating(true);
    try {
      await api.createVolume({ name: newName.trim(), size_gb: newSize });
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

  const totalStorage = volumes.reduce((sum, v) => sum + (v.size_gb || 0), 0);
  const attachedCount = volumes.filter((v) => v.status === "attached").length;

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
          <div className="flex flex-col gap-3 sm:flex-row">
            <Input
              placeholder="Volume name"
              value={newName}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewName(e.target.value)}
              className="flex-1"
            />
            <Input
              type="number"
              placeholder="Size (GB)"
              value={newSize}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewSize(Number(e.target.value))}
              className="w-32"
            />
            <Button onClick={handleCreate} disabled={creating}>
              {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
              Create
            </Button>
          </div>
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
                  <div className="flex gap-4 text-xs text-text-muted">
                    <span>{vol.size_gb} GB</span>
                    <span>{vol.region}</span>
                    <span>ID: {vol.volume_id.slice(0, 8)}…</span>
                  </div>
                </div>
                <div className="flex gap-2">
                  {vol.status === "attached" ? (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => api.detachVolume(vol.volume_id).then(load)}
                    >
                      <Unlink className="h-3.5 w-3.5" /> Detach
                    </Button>
                  ) : (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDelete(vol.volume_id)}
                      className="text-red-500 hover:text-red-600"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
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
