"use client";

import { useCallback, useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label, NumberInput } from "@/components/ui/input";
import { Loader2, TrendingDown, Zap } from "lucide-react";
import {
  fetchHostSpotPreview,
  fetchSpotFloorSuggestion,
  updateHostSpotSettings,
} from "@/lib/api";
import type { Host, SpotRatePreview } from "@/lib/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import {
  SpotInterruptWarning,
  SpotRateDisplay,
  SpotSurface,
} from "@/components/spot/spot-surface";

interface HostSpotSettingsProps {
  host: Host;
  onUpdated?: (host: Host) => void;
  compact?: boolean;
}

export function HostSpotSettings({ host, onUpdated, compact }: HostSpotSettingsProps) {
  const gpuCount = Math.max(1, host.gpu_count ?? 1);
  const [spotEnabled, setSpotEnabled] = useState(host.spot_enabled !== false);
  const [spotSlots, setSpotSlots] = useState(host.spot_gpu_slots ?? gpuCount);
  const [spotMinCents, setSpotMinCents] = useState(host.spot_min_cents ?? 0);
  const [suggestedMin, setSuggestedMin] = useState<number | null>(null);
  const [preview, setPreview] = useState<SpotRatePreview | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [saving, setSaving] = useState(false);

  const loadPreview = useCallback(async (floorCents: number) => {
    if (!host.gpu_model) return;
    setLoadingPreview(true);
    try {
      if (host.host_id) {
        const res = await fetchHostSpotPreview(host.host_id, floorCents);
        setPreview(res.spot_preview);
      } else {
        const res = await fetchSpotFloorSuggestion(host.gpu_model, floorCents);
        setPreview(res);
      }
    } catch {
      setPreview(null);
    } finally {
      setLoadingPreview(false);
    }
  }, [host.gpu_model, host.host_id]);

  useEffect(() => {
    if (!host.gpu_model) return;
    fetchSpotFloorSuggestion(host.gpu_model)
      .then((res) => {
        setSuggestedMin(res.suggested_min_cents);
        if (!host.spot_min_cents) {
          setSpotMinCents(res.suggested_min_cents);
        }
      })
      .catch(() => {});
  }, [host.gpu_model, host.spot_min_cents]);

  useEffect(() => {
    if (!spotEnabled || !host.gpu_model) return;
    const timer = setTimeout(() => loadPreview(spotMinCents), 300);
    return () => clearTimeout(timer);
  }, [spotEnabled, spotMinCents, host.gpu_model, loadPreview]);

  async function handleSave() {
    if (!host.host_id) return;
    setSaving(true);
    try {
      const res = await updateHostSpotSettings(host.host_id, {
        spot_enabled: spotEnabled,
        spot_gpu_slots: spotSlots,
        spot_min_cents: spotMinCents,
      });
      setPreview(res.spot_preview);
      onUpdated?.(res.host);
      toast.success("Spot settings saved");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save spot settings");
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card className={cn(compact ? "p-4" : "p-5", "brand-top-accent overflow-hidden")}>
      <div className="flex items-center gap-2 mb-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald/10 ring-1 ring-emerald/20">
          <TrendingDown className="h-4 w-4 text-emerald" />
        </div>
        <div>
          <h2 className="text-sm font-semibold">Spot Instance Settings</h2>
          <p className="text-[10px] text-text-muted">Host interruptible capacity & floor rates</p>
        </div>
      </div>

      <button
        type="button"
        onClick={() => {
          setSpotEnabled((prev) => {
            const next = !prev;
            if (!next) setPreview(null);
            return next;
          });
        }}
        className={cn(
          "flex w-full items-center justify-between rounded-lg border p-3 text-left transition-colors mb-4",
          spotEnabled ? "border-emerald/40 bg-emerald/5" : "border-border hover:border-text-muted",
        )}
      >
        <div>
          <p className="text-sm font-medium">Enable spot instances</p>
          <p className="text-xs text-text-muted mt-0.5">
            Accept interruptible workloads at published spot rates
          </p>
        </div>
        <div
          className={cn(
            "h-5 w-9 rounded-full relative shrink-0 transition-colors",
            spotEnabled ? "bg-emerald" : "bg-border",
          )}
        >
          <div
            className={cn(
              "absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform shadow-sm",
              spotEnabled ? "translate-x-4" : "translate-x-0.5",
            )}
          />
        </div>
      </button>

      {spotEnabled && (
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label className="text-xs">Spot GPU slots</Label>
              <NumberInput
                min={0}
                max={gpuCount}
                value={spotSlots}
                onChange={(v) => setSpotSlots(Math.min(gpuCount, Math.max(0, v)))}
              />
              <p className="text-[10px] text-text-muted">Max {gpuCount} (total GPUs on host)</p>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Minimum spot floor (¢/hr)</Label>
              <NumberInput
                min={0}
                max={100000}
                value={spotMinCents}
                onChange={(v) => setSpotMinCents(Math.max(0, v))}
              />
              {suggestedMin != null && (
                <p className="text-[10px] text-text-muted">
                  Suggested for {host.gpu_model}: ¢{suggestedMin}/hr
                </p>
              )}
            </div>
          </div>

          <SpotInterruptWarning title="Interruptible workloads">
            <p className="text-xs text-text-secondary">
              Spot jobs may be interrupted when capacity is needed. You are paid for actual runtime only.
            </p>
          </SpotInterruptWarning>

          <SpotSurface className="p-3.5">
            <p className="text-xs text-text-muted mb-1.5 font-medium uppercase tracking-wide">Effective spot rate</p>
            <SpotRateDisplay
              rateCad={preview?.effective_spot_cad}
              onDemandCad={preview?.on_demand_cad}
              savingsPct={preview?.savings_pct}
              loading={loadingPreview}
            />
          </SpotSurface>
        </div>
      )}

      {host.host_id && (
        <Button
          className="w-full mt-4"
          size="sm"
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? (
            <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Saving…</>
          ) : (
            <><Zap className="h-3.5 w-3.5" /> Save spot settings</>
          )}
        </Button>
      )}
    </Card>
  );
}