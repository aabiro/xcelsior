"use client";

import { useEffect, useState } from "react";
import { Eye, FileCode, Loader2, Play, Server } from "lucide-react";
import { toast } from "sonner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input, Label, NumberInput, Select } from "@/components/ui/input";
import * as api from "@/lib/api";
import type { SlurmProfileDetail } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";

interface ProfileCard extends SlurmProfileDetail {
  id: string;
}

const WALLTIME = /^(\d+-)?\d{1,2}:\d{2}:\d{2}$|^\d+$/;

/** Operator submit through the admin-only bridge (`/api/slurm/submit`). */
export function SubmitPanel({ onSubmitted }: { onSubmitted: () => void }) {
  const { t } = useLocale();
  const [profiles, setProfiles] = useState<ProfileCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedProfile, setSelectedProfile] = useState("");
  const [jobName, setJobName] = useState("");
  const [command, setCommand] = useState("nvidia-smi");
  const [nodes, setNodes] = useState("1");
  const [gpus, setGpus] = useState("1");
  const [walltime, setWalltime] = useState("01:00:00");
  const [submitting, setSubmitting] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api.fetchSlurmProfiles()
      .then((res) => {
        if (!live) return;
        const cards = Object.entries(res.details || {}).map(([id, d]) => ({ id, ...d }));
        setProfiles(cards);
        setSelectedProfile((cur) => cur || cards.find((c) => !c.retired)?.id || "");
      })
      .catch((err) => {
        if (live) setLoadError(err instanceof Error ? err.message : "Could not load cluster profiles");
      })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, []);

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
      onSubmitted();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to submit job");
    } finally {
      setSubmitting(false);
    }
  };

  const selected = profiles.find((p) => p.id === selectedProfile);

  if (loading) {
    return (
      <div className="flex justify-center py-12" role="status" aria-label="Loading cluster profiles">
        <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {loadError && (
        <Card className="border-accent-red/30 p-4 text-sm text-accent-red">{loadError}</Card>
      )}

      {profiles.length > 0 && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {profiles.map((p) => (
            <Card
              key={p.id}
              className={cn(
                "p-4 transition-all",
                p.retired ? "opacity-60" : "cursor-pointer",
                selectedProfile === p.id ? "border-ice-blue ring-2 ring-ice-blue" : "hover:border-text-muted",
              )}
              onClick={() => { if (!p.retired) setSelectedProfile(p.id); }}
            >
              <div className="mb-2 flex items-center gap-2">
                <Server className="h-4 w-4 shrink-0 text-ice-blue" />
                <span className="truncate text-sm font-bold">{p.name}</span>
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
                {p.retired && <Badge variant="outline" className="text-[10px]">retired</Badge>}
              </div>
            </Card>
          ))}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileCode className="h-4 w-4" /> {t("dash.hpc.submit_title")}
          </CardTitle>
          <CardDescription>
            {selected ? `Submitting to ${selected.name} as an operator.` : "Select a cluster profile above."}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="hpc-job-name" className="text-xs">Job name</Label>
              <Input id="hpc-job-name" placeholder="my-training-run" value={jobName} onChange={(e) => setJobName(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="hpc-profile" className="text-xs">Cluster profile</Label>
              <Select id="hpc-profile" value={selectedProfile} onChange={(e) => setSelectedProfile(e.target.value)}>
                {profiles.filter((p) => !p.retired).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </Select>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Nodes</Label>
              <NumberInput min={1} value={nodes} onChange={(v) => setNodes(String(v))} />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">GPUs per node (0 = CPU only)</Label>
              <NumberInput min={0} value={gpus} onChange={(v) => setGpus(String(v))} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="hpc-walltime" className="text-xs">Wall time</Label>
              <Input id="hpc-walltime" placeholder="HH:MM:SS" value={walltime} onChange={(e) => setWalltime(e.target.value)} />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="hpc-command" className="text-xs">Command (the adapter writes the batch script around it)</Label>
            <textarea
              id="hpc-command"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              rows={4}
              className="w-full resize-y rounded-lg border border-border bg-background px-3 py-2 font-mono text-xs text-text-primary outline-none focus:border-ice-blue focus:ring-2 focus:ring-ice-blue/50"
              spellCheck={false}
            />
          </div>

          {preview !== null && (
            <pre className="max-h-72 overflow-auto rounded-lg border border-border bg-background p-3 font-mono text-[11px] text-text-secondary">
              {preview}
            </pre>
          )}

          <div className="flex flex-wrap justify-end gap-2">
            <Button variant="outline" onClick={handlePreview} disabled={!selectedProfile || !jobName.trim()}>
              <Eye className="h-4 w-4" /> Preview script
            </Button>
            <Button onClick={handleSubmit} disabled={submitting || !selectedProfile || !jobName.trim()}>
              {submitting
                ? <><Loader2 className="h-4 w-4 animate-spin" /> Submitting…</>
                : <><Play className="h-4 w-4" /> Submit Job</>}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
