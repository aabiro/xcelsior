"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Archive, HardDrive, Loader2 } from "lucide-react";
import { toast } from "sonner";
import {
  fetchArtifactExpiry,
  listVolumes,
  promoteArtifactsToVolume,
  type ArtifactExpiryEntry,
} from "@/lib/api";

/**
 * P3's frontend clause: on a completed instance, *"this output expires in N
 * days"* with the promote action beside it.
 *
 * The retention clock existed on the API (`GET /api/artifacts/{job_id}/expiry`)
 * and on the tool surface (`get_artifact_expiry`) and nowhere a human could see
 * it. The plan is blunt about the consequence — "it is currently invisible,
 * which is how work gets lost" — because the output of a finished run is deleted
 * on a schedule nobody is shown.
 *
 * Promotion is the lever beside it, and it is the whole point of showing the
 * clock: copying the artifacts onto a volume is what stops them expiring.
 */

/** Under a week is where "I'll get to it" stops being safe. */
const URGENT_DAYS = 7;

function toneFor(days: number): { border: string; text: string; label: string } {
  if (days <= 0) return { border: "border-red-500/40", text: "text-red-400", label: "expired" };
  if (days <= URGENT_DAYS) return { border: "border-amber-500/40", text: "text-amber-400", label: `${days}d left` };
  return { border: "border-slate-700", text: "text-slate-400", label: `${days}d left` };
}

export function ArtifactRetentionCard({ jobId }: { jobId: string }) {
  const [artifacts, setArtifacts] = useState<ArtifactExpiryEntry[] | null>(null);
  const [volumes, setVolumes] = useState<Array<{ volume_id: string; name?: string }>>([]);
  const [promoting, setPromoting] = useState(false);
  const [target, setTarget] = useState("");

  const load = useCallback(async () => {
    try {
      const res = await fetchArtifactExpiry(jobId);
      setArtifacts(res.artifacts ?? []);
    } catch {
      // A job with no stored outputs is the common case, not an error worth
      // shouting about — the card simply does not render.
      setArtifacts([]);
    }
  }, [jobId]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!artifacts?.length) return;
    void listVolumes()
      .then((r) => setVolumes(((r as { volumes?: Array<{ volume_id: string; name?: string }> })?.volumes) ?? []))
      .catch(() => setVolumes([]));
  }, [artifacts]);

  if (!artifacts || artifacts.length === 0) return null;

  const soonest = Math.min(...artifacts.map((a) => a.days_remaining));
  const tone = toneFor(soonest);
  const urgent = soonest <= URGENT_DAYS;

  async function promote() {
    if (!target) {
      toast.error("Choose a volume to copy these outputs onto");
      return;
    }
    setPromoting(true);
    try {
      await promoteArtifactsToVolume(target, jobId);
      toast.success("Promotion started — these outputs will stop expiring once it completes");
      await load();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Could not promote these outputs");
    } finally {
      setPromoting(false);
    }
  }

  return (
    <Card className={`p-5 ${tone.border}`}>
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex items-center gap-2">
          <Archive className="w-4 h-4 text-slate-400" />
          <h3 className="font-semibold text-slate-200">Output retention</h3>
        </div>
        <span className={`text-sm font-medium ${tone.text}`}>
          {urgent && <AlertTriangle className="w-3.5 h-3.5 inline mr-1 -mt-0.5" />}
          {tone.label}
        </span>
      </div>

      <p className="text-sm text-slate-400 mb-4">
        {soonest <= 0
          ? "Some of this run's output has passed its retention date and may already be gone."
          : `This run's output is deleted in ${soonest} day${soonest === 1 ? "" : "s"}. Copy it to a volume to keep it.`}
      </p>

      <ul className="space-y-1.5 mb-4">
        {artifacts.map((a) => {
          const t = toneFor(a.days_remaining);
          return (
            <li key={a.artifact_id} className="flex items-center justify-between text-sm">
              <span className="text-slate-300 truncate mr-3">{a.artifact_type}</span>
              <span className={t.text}>{t.label}</span>
            </li>
          );
        })}
      </ul>

      {volumes.length > 0 ? (
        <div className="flex items-center gap-2">
          <select
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200"
          >
            <option value="">Copy to volume…</option>
            {volumes.map((v) => (
              <option key={v.volume_id} value={v.volume_id}>
                {v.name || v.volume_id}
              </option>
            ))}
          </select>
          <Button onClick={promote} disabled={promoting || !target}>
            {promoting ? <Loader2 className="w-4 h-4 animate-spin" /> : <HardDrive className="w-4 h-4" />}
            <span className="ml-2">Promote</span>
          </Button>
        </div>
      ) : (
        <Link href="/dashboard/volumes" className="text-sm text-cyan-400 hover:underline">
          Create a volume to keep this output →
        </Link>
      )}
    </Card>
  );
}
