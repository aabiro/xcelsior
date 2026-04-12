"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label, Select, NumberInput } from "@/components/ui/input";
import { TemplateArtwork } from "@/components/instances/template-artwork";
import {
  launchInstance,
  fetchPricingReference,
  fetchProvinces,
  fetchImageTemplates,
  fetchSpotPrices,
  classifyLaunchError,
} from "@/lib/api";
import type {
  PricingReference,
  ImageTemplate,
  LaunchErrorInfo,
  LaunchInstanceParams,
  MarketplaceListing,
} from "@/lib/api";
import { cn, generateFunName } from "@/lib/utils";
import { markInstanceLaunched } from "@/components/InstallBanner";
import {
  Activity,
  AlertTriangle,
  Box,
  CheckCircle,
  CreditCard,
  DollarSign,
  Loader2,
  MapPin,
  RefreshCw,
  Rocket,
  TrendingDown,
  X,
  Zap,
} from "lucide-react";
import { toast } from "sonner";

/* ───────────────────────── Constants ───────────────────────── */

const PRICING_MODES = [
  {
    value: "on_demand" as const,
    label: "On-Demand",
    icon: Activity,
    desc: "Standard rate, guaranteed availability",
    badge: null,
  },
  {
    value: "spot" as const,
    label: "Spot",
    icon: TrendingDown,
    desc: "Up to 30% cheaper, may be preempted",
    badge: "Save up to 30%",
  },
];

const TIERS = [
  { value: "standard", label: "Standard", desc: "Best effort, no SLA" },
  { value: "premium", label: "Premium", desc: "99.9% uptime guarantee" },
  { value: "reserved", label: "Reserved", desc: "Dedicated allocation, priority support" },
];

const PRIORITIES = [
  { label: "Low", value: 0 },
  { label: "Normal", value: 1 },
  { label: "High", value: 2 },
  { label: "Urgent", value: 3 },
] as const;

const FALLBACK_TEMPLATES: { id: string; label: string; image: string; vram: string; icon: string }[] = [
  { id: "pytorch", label: "PyTorch", image: "nvcr.io/nvidia/pytorch:24.12-py3", vram: "24", icon: "pytorch" },
  { id: "tensorflow", label: "TensorFlow", image: "nvcr.io/nvidia/tensorflow:24.12-tf2-py3", vram: "24", icon: "tensorflow" },
  { id: "vllm", label: "vLLM", image: "vllm/vllm-openai:v0.6.6.post1", vram: "24", icon: "vllm" },
  { id: "comfyui", label: "ComfyUI", image: "runpod/comfyui:1.3.0-cuda12.8", vram: "12", icon: "comfyui" },
  { id: "jupyter", label: "Jupyter Lab", image: "quay.io/jupyter/pytorch-notebook:cuda12-latest", vram: "8", icon: "jupyter" },
  { id: "ubuntu", label: "Ubuntu + CUDA", image: "nvidia/cuda:12.4.1-devel-ubuntu22.04", vram: "8", icon: "ubuntu" },
];

/* ───────────────────────── Props ───────────────────────── */

export interface LaunchInstanceModalProps {
  open: boolean;
  onClose: () => void;
  onLaunched?: () => void;
  /** Pre-fill from a marketplace listing */
  listing?: MarketplaceListing;
}

/* ───────────────────────── Component ───────────────────────── */

export function LaunchInstanceModal({
  open,
  onClose,
  onLaunched,
  listing,
}: LaunchInstanceModalProps) {
  const router = useRouter();
  const [step, setStep] = useState<"configure" | "confirm" | "success">("configure");

  // Form state
  const [instanceName, setInstanceName] = useState(() => generateFunName());
  const [image, setImage] = useState("");
  const [gpuModel, setGpuModel] = useState("");
  const [numGpus, setNumGpus] = useState("1");
  const [pricingMode, setPricingMode] = useState<"on_demand" | "spot">("on_demand");
  const [tier, setTier] = useState("standard");
  const [maxBid, setMaxBid] = useState("");
  const [priority, setPriority] = useState(1);
  const [province, setProvince] = useState("");
  const [durationHrs, setDurationHrs] = useState("1");
  const [nfsMount, setNfsMount] = useState("");

  // Fetched data
  const [pricing, setPricing] = useState<PricingReference[]>([]);
  const [provinces, setProvinces] = useState<Record<string, { tax_rate: number; description: string }>>({});
  const [templates, setTemplates] = useState(FALLBACK_TEMPLATES);
  const [spotPrices, setSpotPrices] = useState<Record<string, number>>({});

  // UI state
  const [submitting, setSubmitting] = useState(false);
  const [launchError, setLaunchError] = useState<LaunchErrorInfo | null>(null);
  const [instanceId, setInstanceId] = useState("");

  const resolvedImage = image.trim();
  const selectedTemplate = templates.find((t) => t.image === image);
  const resolvedGpu = listing?.gpu_model || gpuModel;

  // Pricing calculations
  const selectedPricing = pricing.find((p) => p.gpu_model === resolvedGpu);
  const listingRate = listing ? (listing.price_per_hour_cad || listing.price_per_hour || 0) : 0;
  const onDemandRate = listing ? listingRate : (selectedPricing?.on_demand_cad ?? 0);
  const spotRate = listing
    ? (spotPrices[listing.gpu_model] ?? listingRate * 0.7)
    : (selectedPricing?.spot_cad ?? onDemandRate * 0.6);
  const effectiveRate = pricingMode === "spot" ? spotRate : onDemandRate;
  const estimatedCost = effectiveRate * Number(durationHrs) * Number(numGpus);
  const taxRate = province && provinces[province] ? provinces[province].tax_rate : 0;
  const totalWithTax = estimatedCost * (1 + taxRate);

  // Fetch reference data
  useEffect(() => {
    if (!open) return;
    fetchPricingReference()
      .then((r) => setPricing(r.reference || []))
      .catch(() => {});
    fetchProvinces()
      .then((r) => setProvinces(r.provinces || {}))
      .catch(() => {});
    fetchImageTemplates()
      .then((r) => {
        if (r.templates?.length) {
          setTemplates(
            r.templates.map((t: ImageTemplate) => ({
              id: t.id,
              label: t.label,
              image: t.image,
              vram: String(t.default_vram_gb),
              icon: t.icon,
            })),
          );
        }
      })
      .catch(() => {});
    fetchSpotPrices()
      .then((res) => setSpotPrices(res.spot_prices || res.prices || {}))
      .catch(() => {});
  }, [open]);

  // Pre-fill from listing
  useEffect(() => {
    if (listing) {
      setGpuModel(listing.gpu_model);
    }
  }, [listing]);

  // Reset form when modal re-opens
  useEffect(() => {
    if (open) {
      setStep("configure");
      setInstanceName(generateFunName());
      setImage("");
      setNumGpus("1");
      setPricingMode("on_demand");
      setTier("standard");
      setMaxBid("");
      setPriority(1);
      setDurationHrs("1");
      setNfsMount("");
      setLaunchError(null);
      setInstanceId("");
      if (!listing) setGpuModel("");
    }
  }, [open, listing]);

  const gpuModels = pricing.map((p) => p.gpu_model);

  async function handleSubmit() {
    if (!instanceName.trim()) { toast.error("Enter an instance name"); return; }
    if (!resolvedImage) { toast.error("Select a Docker image"); return; }
    setSubmitting(true);
    try {
      const params: LaunchInstanceParams = {
        name: instanceName.trim(),
        image: resolvedImage,
        num_gpus: Number(numGpus),
        priority,
        tier,
        gpu_model: resolvedGpu || undefined,
        nfs_path: nfsMount || undefined,
      };
      if (listing?.host_id) params.host_id = listing.host_id;
      if (pricingMode === "spot") {
        const bid = maxBid ? Number(maxBid) : spotRate;
        if (bid <= 0) { toast.error("Enter a valid max bid"); setSubmitting(false); return; }
        params.max_bid = bid;
      }
      const res = await launchInstance(params);
      const jobId = res.instance?.job_id || "";
      setInstanceId(jobId);
      setStep("success");
      markInstanceLaunched();
      toast.success("Instance launched successfully");
      onLaunched?.();
    } catch (err) {
      const info = classifyLaunchError(err);
      if (info.action) {
        setLaunchError(info);
      } else {
        toast.error(info.message);
      }
    } finally {
      setSubmitting(false);
    }
  }

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="mx-4 w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl" onClick={(e) => e.stopPropagation()}>
        <Card className="border-border/50">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Launch Instance</CardTitle>
              <CardDescription>
                {listing
                  ? `${listing.gpu_model} · ${listing.region || "Canada"}`
                  : "Configure and launch a new GPU instance"}
              </CardDescription>
            </div>
            <button onClick={onClose} className="text-text-muted hover:text-text-primary transition-colors">
              <X className="h-5 w-5" />
            </button>
          </CardHeader>

          <CardContent className="space-y-5">
            {/* ─── Step: Configure ─── */}
            {step === "configure" && (
              <>
                {/* Listing summary (marketplace only) */}
                {listing && (
                  <div className="flex items-center gap-4 rounded-lg bg-surface p-3">
                    <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-ice-blue/20 bg-ice-blue/5">
                      <Zap className="h-6 w-6 text-ice-blue" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium">{listing.gpu_model}</p>
                      <div className="flex gap-3 text-xs text-text-muted">
                        <span className="flex items-center gap-1"><MapPin className="h-3 w-3" /> {listing.region || "Canada"}</span>
                        <span className="flex items-center gap-1"><Zap className="h-3 w-3" /> {listing.vram_gb || "—"}GB VRAM</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-bold font-mono">${listingRate.toFixed(2)}<span className="text-xs font-normal text-text-muted">/hr</span></p>
                      {spotPrices[listing.gpu_model] != null && (
                        <p className="text-xs text-emerald font-mono">${spotPrices[listing.gpu_model].toFixed(2)}<span className="font-normal text-text-muted">/hr spot</span></p>
                      )}
                    </div>
                  </div>
                )}

                {/* Instance name */}
                <div className="space-y-1.5">
                  <Label className="text-xs">Instance Name</Label>
                  <div className="flex gap-2">
                    <Input
                      placeholder="chunky-narwhal"
                      value={instanceName}
                      onChange={(e) => setInstanceName(e.target.value)}
                      className="flex-1"
                    />
                    <button
                      type="button"
                      onClick={() => setInstanceName(generateFunName())}
                      title="Generate a new name"
                      className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border bg-surface-hover/50 text-text-muted transition-colors hover:border-ice-blue/30 hover:text-ice-blue"
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                {/* Docker image / templates */}
                <div className="space-y-1.5">
                  <Label className="text-xs flex items-center gap-1.5">
                    <Box className="h-3.5 w-3.5" />
                    Docker Image
                  </Label>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                    <button
                      type="button"
                      onClick={() => setImage("")}
                      className={cn(
                        "group flex min-h-[108px] flex-col items-center justify-center gap-2 rounded-xl border p-3 text-center transition-colors",
                        !selectedTemplate
                          ? "border-ice-blue/40 bg-ice-blue/5"
                          : "border-border/80 bg-background/30 text-text-secondary hover:border-text-muted hover:text-text-primary",
                      )}
                    >
                      <div
                        className={cn(
                          "flex h-14 w-14 items-center justify-center rounded-2xl border bg-surface/80 transition-transform group-hover:scale-105",
                          !selectedTemplate
                            ? "border-transparent bg-ice-blue/10 text-ice-blue"
                            : "border-border/60 text-text-secondary",
                        )}
                      >
                        <Box className="h-6 w-6" />
                      </div>
                      <span className="font-medium text-text-primary">Custom</span>
                      <span className="text-[11px] uppercase tracking-[0.16em] text-text-muted">Any image</span>
                    </button>
                    {templates.map((tpl) => (
                      <button
                        key={tpl.id}
                        type="button"
                        onClick={() => setImage(tpl.image)}
                        className={cn(
                          "group flex min-h-[108px] flex-col items-center justify-center gap-2 rounded-xl border p-3 text-center transition-colors",
                          image === tpl.image
                            ? "border-ice-blue/40 bg-ice-blue/5"
                            : "border-border/80 bg-background/30 text-text-secondary hover:border-text-muted hover:text-text-primary",
                        )}
                      >
                        <TemplateArtwork template={tpl.id} size={44} className="transition-transform duration-300 group-hover:scale-[1.03]" />
                        <span className="font-medium text-text-primary">{tpl.label}</span>
                        <span className="text-[11px] uppercase tracking-[0.16em] text-text-muted">{tpl.vram} GB VRAM</span>
                      </button>
                    ))}
                  </div>
                  <Input
                    placeholder="docker.io/my-org/my-image:latest"
                    value={image}
                    onChange={(e) => setImage(e.target.value)}
                    className="mt-1.5"
                  />
                </div>

                {/* GPU model + count (non-listing mode) */}
                {!listing && (
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="space-y-1.5">
                      <Label className="text-xs">GPU Model</Label>
                      <Select
                        value={gpuModel}
                        onChange={(e) => setGpuModel(e.target.value)}
                      >
                        <option value="">Auto-select best available</option>
                        {gpuModels.map((model) => (
                          <option key={model} value={model}>{model}</option>
                        ))}
                      </Select>
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-xs">GPUs</Label>
                      <NumberInput
                        min={1}
                        max={8}
                        value={numGpus}
                        onChange={(v) => setNumGpus(String(v))}
                      />
                    </div>
                  </div>
                )}

                {/* Pricing Mode */}
                <div className="space-y-1.5">
                  <Label className="text-xs">Pricing</Label>
                  <div className="grid grid-cols-2 gap-2">
                    {PRICING_MODES.map((mode) => {
                      const Icon = mode.icon;
                      return (
                        <button
                          key={mode.value}
                          type="button"
                          onClick={() => setPricingMode(mode.value)}
                          className={cn(
                            "rounded-lg border p-3 text-left transition-colors",
                            pricingMode === mode.value
                              ? "border-ice-blue/40 bg-ice-blue/5"
                              : "border-border hover:border-text-muted",
                          )}
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <Icon className="h-4 w-4" />
                            <span className="text-sm font-medium">{mode.label}</span>
                            {mode.badge && (
                              <span className="ml-auto rounded-full bg-emerald/20 px-1.5 py-0.5 text-[10px] font-medium text-emerald">
                                {mode.badge}
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-text-muted">{mode.desc}</p>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Spot max bid */}
                {pricingMode === "spot" && (
                  <div className="space-y-1.5">
                    <Label className="text-xs">Max Bid ($/hr CAD)</Label>
                    <Input
                      type="number"
                      step="0.01"
                      min={0.01}
                      placeholder={`Current spot: $${spotRate.toFixed(2)}/hr`}
                      value={maxBid}
                      onChange={(e) => setMaxBid(e.target.value)}
                    />
                    <p className="text-xs text-text-muted">
                      Leave empty to bid at current spot price. Your instance may be preempted if outbid.
                    </p>
                  </div>
                )}

                {/* Service Tier */}
                <div className="space-y-1.5">
                  <Label className="text-xs">Service Tier</Label>
                  <div className="space-y-2">
                    {TIERS.map((t) => (
                      <button
                        key={t.value}
                        type="button"
                        onClick={() => setTier(t.value)}
                        className={cn(
                          "w-full text-left rounded-lg border p-3 transition-colors",
                          tier === t.value
                            ? "border-ice-blue/40 bg-ice-blue/5"
                            : "border-border hover:border-text-muted",
                        )}
                      >
                        <p className="text-sm font-medium">{t.label}</p>
                        <p className="text-xs text-text-muted">{t.desc}</p>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Priority */}
                <div className="space-y-1.5">
                  <Label className="text-xs">Priority</Label>
                  <div className="grid grid-cols-4 gap-2">
                    {PRIORITIES.map((p) => (
                      <button
                        key={p.value}
                        type="button"
                        onClick={() => setPriority(p.value)}
                        className={cn(
                          "rounded-lg border px-3 py-2 text-sm font-medium transition-colors",
                          priority === p.value
                            ? "border-ice-blue/40 bg-ice-blue/5 text-text-primary"
                            : "border-border text-text-secondary hover:border-text-muted",
                        )}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Duration + Province */}
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-1.5">
                    <Label className="text-xs">Duration (hours)</Label>
                    <NumberInput
                      min={0.5}
                      step={0.5}
                      value={durationHrs}
                      onChange={(v) => setDurationHrs(String(v))}
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label className="text-xs">Province</Label>
                    <Select
                      value={province}
                      onChange={(e) => setProvince(e.target.value)}
                    >
                      <option value="">Auto-detect</option>
                      {Object.entries(provinces).map(([code, info]) => (
                        <option key={code} value={code}>{code} — {info.description}</option>
                      ))}
                    </Select>
                  </div>
                </div>

                {/* NFS Mount */}
                <div className="space-y-1.5">
                  <Label className="text-xs">NFS Mount (optional)</Label>
                  <Input
                    placeholder="host:/path/to/share"
                    value={nfsMount}
                    onChange={(e) => setNfsMount(e.target.value)}
                  />
                  <p className="text-xs text-text-muted">Mount a shared NFS volume into your container at /mnt/nfs.</p>
                </div>

                {/* Hourly Rate summary */}
                <div className="rounded-lg border border-border p-3 bg-surface">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-text-secondary">Hourly Rate</span>
                    <span className="text-lg font-bold font-mono">
                      ${effectiveRate.toFixed(2)}
                      <span className="text-xs text-text-muted ml-1">CAD/hr</span>
                    </span>
                  </div>
                  <p className="text-xs text-text-muted mt-1">
                    {pricingMode === "spot"
                      ? "Spot pricing · Billed per second of actual usage"
                      : "On-demand · Billed per second of actual usage"}
                    {tier !== "standard" && ` · tier ${tier} applies`}
                  </p>
                  <div className="flex items-center gap-1.5 mt-2 text-xs text-ice-blue">
                    <DollarSign className="h-3 w-3" />
                    <span>Real-time metered — you only pay for what you use</span>
                  </div>
                </div>

                <Button
                  className="w-full"
                  onClick={() => setStep("confirm")}
                  disabled={!instanceName.trim() || !resolvedImage}
                >
                  Continue
                </Button>
              </>
            )}

            {/* ─── Step: Confirm ─── */}
            {step === "confirm" && (
              <>
                <div className="rounded-lg border border-accent-gold/30 bg-accent-gold/5 p-4">
                  <p className="text-sm font-medium text-accent-gold mb-2">Confirm Launch</p>
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between"><span className="text-text-muted">GPU</span><span>{resolvedGpu || "Auto"}</span></div>
                    {Number(numGpus) > 1 && (
                      <div className="flex justify-between"><span className="text-text-muted">GPUs</span><span>×{numGpus}</span></div>
                    )}
                    <div className="flex justify-between"><span className="text-text-muted">Instance</span><span>{instanceName}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Image</span><span className="text-right max-w-[220px] truncate" title={resolvedImage}>{resolvedImage}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Pricing</span><span className="capitalize">{pricingMode === "on_demand" ? "On-Demand" : "Spot"}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Tier</span><span className="capitalize">{tier}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Priority</span><span>{PRIORITIES.find((p) => p.value === priority)?.label ?? "Normal"}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Duration</span><span>{durationHrs}h</span></div>
                    {nfsMount && (
                      <div className="flex justify-between"><span className="text-text-muted">NFS Mount</span><span className="text-right max-w-[220px] truncate" title={nfsMount}>{nfsMount}</span></div>
                    )}
                    <div className="flex justify-between font-medium text-sm pt-1 border-t border-accent-gold/20">
                      <span>Rate</span>
                      <span>${effectiveRate.toFixed(2)}/hr CAD</span>
                    </div>
                    {resolvedGpu && (
                      <>
                        <div className="flex justify-between text-sm">
                          <span>Estimated</span>
                          <span>${estimatedCost.toFixed(2)} CAD</span>
                        </div>
                        {taxRate > 0 && (
                          <div className="flex justify-between text-sm">
                            <span>Tax ({(taxRate * 100).toFixed(1)}%)</span>
                            <span>${(estimatedCost * taxRate).toFixed(2)} CAD</span>
                          </div>
                        )}
                        {taxRate > 0 && (
                          <div className="flex justify-between font-medium text-sm pt-1 border-t border-accent-gold/20">
                            <span>Total</span>
                            <span>${totalWithTax.toFixed(2)} CAD</span>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
                <p className="text-xs text-text-muted">
                  Your wallet will be charged based on actual usage. You can stop the instance at any time.
                </p>

                {launchError && (
                  <div className="rounded-lg border border-accent-red/30 bg-accent-red/10 p-3 flex items-start gap-3">
                    <AlertTriangle className="h-4 w-4 text-accent-red mt-0.5 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-accent-red font-medium">{launchError.message}</p>
                      {launchError.action && (
                        <Button
                          size="sm"
                          className="mt-2"
                          onClick={() => router.push(launchError.action!.href)}
                        >
                          <CreditCard className="h-3.5 w-3.5" />
                          {launchError.action.label}
                        </Button>
                      )}
                    </div>
                    <button onClick={() => setLaunchError(null)} className="text-text-muted hover:text-text-primary">
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )}

                <div className="flex gap-2">
                  <Button variant="outline" className="flex-1" type="button" onClick={() => { setLaunchError(null); setStep("configure"); }}>
                    Back
                  </Button>
                  <Button className="flex-1" type="button" onClick={handleSubmit} disabled={submitting}>
                    {submitting ? <><Loader2 className="h-4 w-4 animate-spin" /> Launching…</> : <>
                      <Rocket className="h-4 w-4" /> Launch Instance
                    </>}
                  </Button>
                </div>
              </>
            )}

            {/* ─── Step: Success ─── */}
            {step === "success" && (
              <div className="text-center py-4">
                <CheckCircle className="mx-auto h-12 w-12 text-emerald mb-3" />
                <h3 className="text-lg font-semibold mb-1">Instance Launched!</h3>
                <p className="text-sm text-text-secondary mb-4">
                  Your instance is being provisioned. Billing begins when it starts running.
                </p>
                <div className="flex gap-2">
                  <Button variant="outline" className="flex-1" type="button" onClick={onClose}>
                    Close
                  </Button>
                  <Button className="flex-1" type="button" onClick={() => router.push(`/dashboard/instances/${instanceId}`)}>
                    View Instance
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
