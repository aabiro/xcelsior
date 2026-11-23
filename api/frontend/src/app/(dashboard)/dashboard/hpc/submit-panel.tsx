"use client";

import { useEffect, useState } from "react";
import { Check, Eye, FileCode, Loader2, Play, Server } from "lucide-react";
import { toast } from "sonner";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input, Label, NumberInput } from "@/components/ui/input";
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
  const [previewing, setPreviewing] = useState(false);
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
    setPreviewing(true);
    try {
      const res = await api.submitSlurmInstance(req);
      setPreview(res.script || "");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Preview failed");
    } finally {
      setPreviewing(false);
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
  const totalGpus = Math.max(0, Number(gpus)) * Math.max(1, Number(nodes));
  const ready = !!selectedProfile && !!jobName.trim();

  if (loading) {
    return (
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2" role="status" aria-label="Loading cluster profiles">
        <div className="h-80 animate-pulse rounded-xl border border-border bg-surface" />
        <div className="h-80 animate-pulse rounded-xl border border-border bg-surface" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {loadError && (
        <Card className="border-accent-red/30 p-4 text-sm text-accent-red">{loadError}</Card>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card className="space-y-5 p-5">
          <div>
            <h3 className="flex items-center gap-2 text-base font-semibold">
              <FileCode className="h-4 w-4 text-accent-cyan" /> {t("dash.hpc.submit_title")}
            </h3>
            <p className="mt-0.5 text-sm text-text-muted">
              Submitted as an operator through the admin bridge. The adapter writes the batch script around your
              command.
            </p>
          </div>

          <fieldset>
            <legend className="mb-2 text-xs font-medium text-text-secondary">Cluster profile</legend>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2" role="radiogroup" aria-label="Cluster profile">
              {profiles.map((p) => {
                const active = selectedProfile === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    disabled={!!p.retired}
                    onClick={() => setSelectedProfile(p.id)}
                    className={cn(
                      "relative rounded-lg border p-3 text-left transition-colors",
                      "disabled:cursor-not-allowed disabled:opacity-50",
                      active
                        ? "border-accent-cyan/60 bg-accent-cyan/10"
                        : "border-border enabled:hover:border-text-muted",
                    )}
                  >
                    {active && (
                      <span className="absolute right-2 top-2 flex h-4 w-4 items-center justify-center rounded-full bg-accent-cyan text-background">
                        <Check className="h-3 w-3" />
                      </span>
                    )}
                    <span className="flex items-center gap-2 pr-5">
                      <Server className="h-3.5 w-3.5 shrink-0 text-ice-blue" />
                      <span className="truncate text-sm font-semibold">{p.name}</span>
                    </span>
                    <span className="mt-1 block text-[11px] text-text-muted">
                      Partition: {p.partition_gpu ?? "assigned by cluster"}
                      {p.max_walltime ? ` · Max: ${p.max_walltime}` : ""}
                    </span>
                    <span className="mt-1.5 flex flex-wrap gap-1">
                      {p.gpu_type && (
                        <Badge variant="info" className="text-[10px]">
                          {p.gpu_type}{p.gpus_per_node ? ` × ${p.gpus_per_node}/node` : ""}
                        </Badge>
                      )}
                      {p.retired && <Badge variant="outline" className="text-[10px]">retired</Badge>}
                    </span>
                  </button>
                );
              })}
            </div>
          </fieldset>

          <div className="space-y-1.5">
            <Label htmlFor="hpc-job-name" className="text-xs">Job name</Label>
            <Input autoComplete="off" autoCapitalize="none" autoCorrect="off" spellCheck={false} id="hpc-job-name" placeholder="my-training-run" value={jobName} onChange={(e) => setJobName(e.target.value)} />
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Nodes</Label>
              <NumberInput min={1} value={nodes} onChange={(v) => setNodes(String(v))} />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">GPUs per node</Label>
              <NumberInput min={0} value={gpus} onChange={(v) => setGpus(String(v))} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="hpc-walltime" className="text-xs">Wall time</Label>
              <Input autoComplete="off" autoCapitalize="none" autoCorrect="off" spellCheck={false} id="hpc-walltime" placeholder="HH:MM:SS" value={walltime} onChange={(e) => setWalltime(e.target.value)} className="font-mono" />
            </div>
          </div>
          <p className="-mt-2 text-[11px] text-text-muted">
            {totalGpus === 0 ? "CPU only: no GPU is requested." : `${totalGpus} GPU(s) in total.`}
          </p>

          <div className="space-y-1.5">
            <Label htmlFor="hpc-command" className="text-xs">Command</Label>
            <textarea autoComplete="off" autoCapitalize="none" autoCorrect="off"
              id="hpc-command"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              rows={5}
              className="w-full resize-y rounded-lg border border-border bg-background px-3 py-2 font-mono text-xs text-text-primary outline-none focus:border-ice-blue focus:ring-2 focus:ring-ice-blue/50"
              spellCheck={false}
            />
          </div>

          <div className="flex flex-wrap justify-end gap-2 border-t border-border pt-4">
            <Button variant="outline" onClick={() => void handlePreview()} disabled={!ready || previewing}>
              {previewing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Eye className="h-4 w-4" />} Preview script
            </Button>
            <Button onClick={() => void handleSubmit()} disabled={submitting || !ready}>
              {submitting
                ? <><Loader2 className="h-4 w-4 animate-spin" /> Submitting…</>
                : <><Play className="h-4 w-4" /> Submit Job</>}
            </Button>
          </div>
        </Card>

        <Card className="flex min-h-64 flex-col p-5">
          <h3 className="text-base font-semibold">Batch script</h3>
          <p className="mt-0.5 text-sm text-text-muted">
            {selected
              ? `What ${selected.name} would be sent. A preview is a dry run: nothing is submitted.`
              : "Select a cluster profile."}
          </p>
          {preview !== null ? (
            <pre
              className="mt-4 max-h-[32rem] flex-1 overflow-auto rounded-lg border border-border bg-background p-3 font-mono text-[11px] leading-relaxed text-text-secondary"
              data-testid="hpc-script-preview"
              tabIndex={0}
            >
              {preview}
            </pre>
          ) : (
            <div className="mt-4 flex flex-1 flex-col items-center justify-center rounded-lg border border-dashed border-border px-6 py-10 text-center">
              <FileCode className="mb-2 h-6 w-6 text-text-muted" />
              <p className="text-sm text-text-muted">
                Name the job and choose <span className="font-medium text-text-secondary">Preview script</span> to see
                the exact #SBATCH lines before anything runs.
              </p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
