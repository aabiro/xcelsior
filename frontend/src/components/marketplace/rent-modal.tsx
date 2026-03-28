"use client";

import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label, Select } from "@/components/ui/input";
import {
  X, Cpu, MapPin, Zap, Clock, DollarSign, Loader2, CheckCircle,
} from "lucide-react";
import * as api from "@/lib/api";
import type { MarketplaceListing } from "@/lib/api";
import { toast } from "sonner";
import { useRouter } from "next/navigation";

interface RentModalProps {
  listing: MarketplaceListing;
  onClose: () => void;
}

const DURATION_PRESETS = [
  { label: "1 hour", hours: 1 },
  { label: "4 hours", hours: 4 },
  { label: "8 hours", hours: 8 },
  { label: "24 hours", hours: 24 },
  { label: "3 days", hours: 72 },
  { label: "7 days", hours: 168 },
];

const TIERS = [
  { value: "standard", label: "Standard", desc: "Best effort, no SLA" },
  { value: "premium", label: "Premium", desc: "99.9% uptime guarantee" },
  { value: "reserved", label: "Reserved", desc: "Dedicated allocation, priority support" },
];

export function RentModal({ listing, onClose }: RentModalProps) {
  const router = useRouter();
  const [step, setStep] = useState<"configure" | "confirm" | "success">("configure");
  const [hours, setHours] = useState(1);
  const [customHours, setCustomHours] = useState("");
  const [tier, setTier] = useState("standard");
  const [jobName, setJobName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [estimating, setEstimating] = useState(false);
  const [estimate, setEstimate] = useState<{ total_cad: number; breakdown?: any } | null>(null);
  const [jobId, setJobId] = useState("");

  const pricePerHour = listing.price_per_hour_cad || listing.price_per_hour || 0;
  const effectiveHours = customHours ? Number(customHours) : hours;
  const estimatedCost = pricePerHour * effectiveHours;

  const handleEstimate = async () => {
    setEstimating(true);
    try {
      const res = await api.estimatePrice({
        gpu_model: listing.gpu_model,
        duration_hours: effectiveHours,
        is_canadian: listing.country === "CA" || listing.country === "Canada",
      });
      setEstimate({ total_cad: res.with_rebate_cad ?? res.base_cost_cad, breakdown: res });
    } catch {
      // Fallback to local estimate
      setEstimate({ total_cad: estimatedCost });
    } finally {
      setEstimating(false);
    }
  };

  const handleSubmit = async () => {
    if (!jobName.trim()) { toast.error("Enter a job name"); return; }
    setSubmitting(true);
    try {
      const res = await api.submitJob({
        host_id: listing.host_id,
        gpu_model: listing.gpu_model,
        duration_hours: effectiveHours,
        tier,
        name: jobName.trim(),
      });
      setJobId(res.job_id || res.id || "");
      setStep("success");
      toast.success("Job submitted successfully");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to submit job");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="mx-4 w-full max-w-lg" onClick={(e) => e.stopPropagation()}>
        <Card className="border-border/50">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Rent GPU</CardTitle>
              <CardDescription>{listing.gpu_model} · {listing.region || "Canada"}</CardDescription>
            </div>
            <button onClick={onClose} className="text-text-muted hover:text-text-primary">
              <X className="h-5 w-5" />
            </button>
          </CardHeader>
          <CardContent className="space-y-4">
            {step === "configure" && (
              <>
                {/* Listing summary */}
                <div className="flex items-center gap-4 rounded-lg bg-surface p-3">
                  <Cpu className="h-8 w-8 text-ice-blue" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">{listing.gpu_model}</p>
                    <div className="flex gap-3 text-xs text-text-muted">
                      <span className="flex items-center gap-1"><MapPin className="h-3 w-3" /> {listing.region || "Canada"}</span>
                      <span className="flex items-center gap-1"><Zap className="h-3 w-3" /> {listing.vram_gb || "—"}GB VRAM</span>
                    </div>
                  </div>
                  <p className="text-sm font-bold font-mono">${pricePerHour.toFixed(2)}/hr</p>
                </div>

                {/* Job name */}
                <div className="space-y-1.5">
                  <Label className="text-xs">Job Name</Label>
                  <Input
                    placeholder="my-training-job"
                    value={jobName}
                    onChange={(e) => setJobName(e.target.value)}
                  />
                </div>

                {/* Duration */}
                <div className="space-y-1.5">
                  <Label className="text-xs">Duration</Label>
                  <div className="grid grid-cols-3 gap-2">
                    {DURATION_PRESETS.map((p) => (
                      <button
                        key={p.hours}
                        onClick={() => { setHours(p.hours); setCustomHours(""); }}
                        className={`rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${
                          !customHours && hours === p.hours
                            ? "border-ice-blue bg-ice-blue/10 text-ice-blue"
                            : "border-border text-text-muted hover:border-text-muted"
                        }`}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                  <Input
                    type="number"
                    min={1}
                    placeholder="Custom hours..."
                    value={customHours}
                    onChange={(e) => setCustomHours(e.target.value)}
                    className="mt-2"
                  />
                </div>

                {/* Tier */}
                <div className="space-y-1.5">
                  <Label className="text-xs">Service Tier</Label>
                  <div className="space-y-2">
                    {TIERS.map((t) => (
                      <button
                        key={t.value}
                        onClick={() => setTier(t.value)}
                        className={`w-full text-left rounded-lg border p-3 transition-colors ${
                          tier === t.value
                            ? "border-ice-blue bg-ice-blue/5"
                            : "border-border hover:border-text-muted"
                        }`}
                      >
                        <p className="text-sm font-medium">{t.label}</p>
                        <p className="text-xs text-text-muted">{t.desc}</p>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Cost estimate */}
                <div className="rounded-lg border border-border p-3 bg-surface">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-text-secondary">Estimated Cost</span>
                    <span className="text-lg font-bold font-mono">
                      ${(estimate?.total_cad ?? estimatedCost).toFixed(2)}
                      <span className="text-xs text-text-muted ml-1">CAD</span>
                    </span>
                  </div>
                  <p className="text-xs text-text-muted mt-1">
                    {effectiveHours}h × ${pricePerHour.toFixed(2)}/hr
                    {tier !== "standard" && " + tier premium"}
                  </p>
                </div>

                <div className="flex gap-2">
                  <Button variant="outline" className="flex-1" onClick={handleEstimate} disabled={estimating}>
                    {estimating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <DollarSign className="h-3.5 w-3.5" />}
                    Get Estimate
                  </Button>
                  <Button
                    className="flex-1"
                    onClick={() => setStep("confirm")}
                    disabled={!jobName.trim() || effectiveHours < 1}
                  >
                    Continue
                  </Button>
                </div>
              </>
            )}

            {step === "confirm" && (
              <>
                <div className="rounded-lg border border-accent-gold/30 bg-accent-gold/5 p-4">
                  <p className="text-sm font-medium text-accent-gold mb-2">Confirm Rental</p>
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between"><span className="text-text-muted">GPU</span><span>{listing.gpu_model}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Job</span><span>{jobName}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Duration</span><span>{effectiveHours}h</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Tier</span><span className="capitalize">{tier}</span></div>
                    <div className="flex justify-between font-medium text-sm pt-1 border-t border-accent-gold/20">
                      <span>Total</span>
                      <span>${(estimate?.total_cad ?? estimatedCost).toFixed(2)} CAD</span>
                    </div>
                  </div>
                </div>
                <p className="text-xs text-text-muted">
                  Your wallet will be charged. Unused time may be refunded based on your service tier.
                </p>
                <div className="flex gap-2">
                  <Button variant="outline" className="flex-1" onClick={() => setStep("configure")}>
                    Back
                  </Button>
                  <Button className="flex-1" onClick={handleSubmit} disabled={submitting}>
                    {submitting ? <><Loader2 className="h-4 w-4 animate-spin" /> Submitting…</> : "Confirm & Pay"}
                  </Button>
                </div>
              </>
            )}

            {step === "success" && (
              <div className="text-center py-4">
                <CheckCircle className="mx-auto h-12 w-12 text-emerald mb-3" />
                <h3 className="text-lg font-semibold mb-1">Job Submitted!</h3>
                <p className="text-sm text-text-secondary mb-4">
                  Your job is being provisioned. You can track its status in the Jobs dashboard.
                </p>
                <div className="flex gap-2">
                  <Button variant="outline" className="flex-1" onClick={onClose}>
                    Close
                  </Button>
                  <Button className="flex-1" onClick={() => router.push(`/dashboard/jobs/${jobId}`)}>
                    View Job
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
