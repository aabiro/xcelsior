"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label, Select, NumberInput } from "@/components/ui/input";
import { TemplateArtwork } from "@/components/instances/template-artwork";
import {
  launchInstance,
  fetchAvailableGPUs,
  fetchPricingReference,
  fetchProvinces,
  fetchImageTemplates,
  fetchSpotPrices,
  classifyLaunchError,
  listAvailableVolumes,
  detectProvince,
  fetchPricingRates,
} from "@/lib/api";
import type {
  PricingReference,
  GpuAvailability,
  ImageTemplate,
  LaunchErrorInfo,
  LaunchInstanceParams,
  MarketplaceListing,
  Volume,
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
  HardDrive,
  Loader2,
  Lock,
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
  { value: "sovereign", label: "Sovereign", desc: "Canada-only, Canadian-jurisdiction operator" },
];

const PRIORITIES = [
  { label: "Low", value: "low", numeric: 1 },
  { label: "Normal", value: "normal", numeric: 3 },
  { label: "High", value: "high", numeric: 6 },
  { label: "Critical", value: "critical", numeric: 9 },
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
  onLaunched?: (instanceId: string, instance?: import("@/lib/api").Instance) => void;
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
  const [vramGb, setVramGb] = useState<number | "">(""); 
  const [numGpus, setNumGpus] = useState("1");
  const [pricingMode, setPricingMode] = useState<"on_demand" | "spot">("on_demand");
  const [tier, setTier] = useState("standard");
  const [maxBid, setMaxBid] = useState("");
  const [priority, setPriority] = useState<string>("normal");
  const [province, setProvince] = useState("");

  // Volume picker state
  const [availableVolumes, setAvailableVolumes] = useState<Volume[]>([]);
  const [selectedVolumeIds, setSelectedVolumeIds] = useState<string[]>([]);

  // Encrypted workspace
  const [encryptedWorkspace, setEncryptedWorkspace] = useState(false);

  // Fetched data
  const [pricing, setPricing] = useState<PricingReference[]>([]);
  const [availableGpus, setAvailableGpus] = useState<GpuAvailability[]>([]);
  const [provinces, setProvinces] = useState<Record<string, { name: string; tax_rate: number; tax_description: string }>>({});
  const [templates, setTemplates] = useState(FALLBACK_TEMPLATES);
  const [spotPrices, setSpotPrices] = useState<Record<string, number>>({});

  // Dynamic pricing from backend
  const [dynamicRate, setDynamicRate] = useState<{
    effective_rate_per_gpu: number;
    total_per_hour: number;
    tax_rate: number;
    tax_description: string;
    tax_amount: number;
    total_with_tax: number;
    base_rate_cad: number;
    priority_multiplier: number;
    sovereignty_premium: number;
    multi_gpu_discount: number;
  } | null>(null);

  // UI state
  const [submitting, setSubmitting] = useState(false);
  const [launchError, setLaunchError] = useState<LaunchErrorInfo | null>(null);
  const [instanceId, setInstanceId] = useState("");

  const resolvedImage = image.trim();
  const selectedTemplate = templates.find((t) => t.image === image);
  const resolvedGpu = listing?.gpu_model || gpuModel;
  const templateVramGb = selectedTemplate ? Number(selectedTemplate.vram) : undefined;
  const isAutoGpuSelection = !listing && !resolvedGpu;
  const eligibleInventory = availableGpus.filter((gpu) => templateVramGb == null || gpu.vram_gb >= templateVramGb);
  const liveInventory = eligibleInventory.filter((gpu) => gpu.count_available > 0);
  const hasLiveInventory = liveInventory.length > 0;
  const hasHostedInventory = eligibleInventory.length > 0;
  const inventorySource = liveInventory.length > 0 ? liveInventory : eligibleInventory;
  const gpuInventoryOptions = Array.from(
    inventorySource.reduce((acc, gpu) => {
      const existing = acc.get(gpu.gpu_model);
      const price = gpu.price_per_hour_cad > 0 ? gpu.price_per_hour_cad : undefined;

      if (!existing) {
        acc.set(gpu.gpu_model, {
          gpu_model: gpu.gpu_model,
          maxVramGb: gpu.vram_gb,
          countAvailable: gpu.count_available,
          minPricePerHourCad: price,
        });
        return acc;
      }

      existing.maxVramGb = Math.max(existing.maxVramGb, gpu.vram_gb);
      existing.countAvailable += gpu.count_available;
      if (price != null) {
        existing.minPricePerHourCad = existing.minPricePerHourCad == null
          ? price
          : Math.min(existing.minPricePerHourCad, price);
      }
      return acc;
    }, new Map<string, {
      gpu_model: string;
      maxVramGb: number;
      countAvailable: number;
      minPricePerHourCad?: number;
    }>()).values(),
  ).sort((a, b) => a.gpu_model.localeCompare(b.gpu_model));
  const fallbackGpuOptions = pricing.map((entry) => ({
    gpu_model: entry.gpu_model,
    maxVramGb: 0,
    countAvailable: 0,
    minPricePerHourCad: entry.on_demand_cad,
  }));
  const gpuModelOptions = gpuInventoryOptions.length > 0 ? gpuInventoryOptions : fallbackGpuOptions;
  const selectedGpuStillAvailable = !gpuModel || gpuModelOptions.some((option) => option.gpu_model === gpuModel);

  // VRAM options per selected GPU model — derived from inventory
  const FALLBACK_VRAM: Record<string, number[]> = {
    "RTX 2060": [6], "RTX 3090": [24], "RTX 4080": [16], "RTX 4090": [24], "RTX 5090": [32],
    "A100 40GB": [40], "A100 80GB": [80], "A100": [40, 80],
    "L40": [48], "L40S": [48], "H100": [80], "H200": [141],
  };
  const vramOptionsForGpu: number[] = (() => {
    if (!resolvedGpu) return [];
    // Collect unique VRAM values from live inventory for this model
    const fromInventory = [...new Set(
      availableGpus
        .filter((g) => g.gpu_model === resolvedGpu && g.vram_gb > 0)
        .map((g) => g.vram_gb),
    )].sort((a, b) => a - b);
    if (fromInventory.length > 0) return fromInventory;
    return FALLBACK_VRAM[resolvedGpu] ?? [];
  })();
  const resolvedVramGb = vramGb || (vramOptionsForGpu.length === 1 ? vramOptionsForGpu[0] : undefined);

  // Pricing — use dynamic rate from backend when available, else fall back to reference
  const selectedPricing = pricing.find((p) => p.gpu_model === resolvedGpu);
  const selectedInventory = gpuModelOptions.find((gpu) => gpu.gpu_model === resolvedGpu);
  const listingRate = listing ? (listing.price_per_hour_cad || listing.price_per_hour || 0) : 0;
  // Cheapest available rate for "auto" GPU selection
  const cheapestAvailableRate = (() => {
    const candidates = gpuModelOptions
      .map((g) => g.minPricePerHourCad)
      .filter((p): p is number => p != null && p > 0);
    if (candidates.length === 0) {
      const refCandidates = pricing
        .map((p) => pricingMode === "spot" ? (p.spot_cad ?? p.on_demand_cad) : p.on_demand_cad)
        .filter((p) => p > 0);
      return refCandidates.length > 0 ? Math.min(...refCandidates) : null;
    }
    return Math.min(...candidates);
  })();
  const isEstimatedRate = !dynamicRate && !listing && !resolvedGpu;
  const effectiveRate = dynamicRate
    ? dynamicRate.effective_rate_per_gpu
    : listing
      ? listingRate
      : pricingMode === "spot"
        ? (spotPrices[resolvedGpu] ?? selectedPricing?.spot_cad ?? cheapestAvailableRate)
        : (selectedPricing?.on_demand_cad ?? selectedInventory?.minPricePerHourCad ?? cheapestAvailableRate);
  const spotRate = listing
    ? (spotPrices[listing.gpu_model] ?? listingRate * 0.7)
    : spotPrices[resolvedGpu] ?? selectedPricing?.spot_cad;
  const totalPerHour = dynamicRate
    ? dynamicRate.total_per_hour
    : effectiveRate != null ? effectiveRate * Number(numGpus) : null;
  const taxRate = dynamicRate?.tax_rate ?? (province && provinces[province] ? provinces[province].tax_rate : 0);
  const totalWithTax = dynamicRate
    ? dynamicRate.total_with_tax
    : totalPerHour != null ? totalPerHour * (1 + taxRate) : null;

  // Fetch reference data
  useEffect(() => {
    if (!open) return;
    fetchPricingReference()
      .then((r) => setPricing(r.reference || []))
      .catch(() => {});
    fetchAvailableGPUs()
      .then((r) => setAvailableGpus(r.gpus || []))
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
    listAvailableVolumes()
      .then((r) => setAvailableVolumes(r.volumes || []))
      .catch(() => {});
    // Auto-detect province
    if (!province) {
      detectProvince()
        .then((r) => setProvince(r.province || "ON"))
        .catch(() => setProvince("ON"));
    }
  }, [open]);

  // Dynamic pricing — recompute when any pricing variable changes
  useEffect(() => {
    if (!open || !resolvedGpu) { setDynamicRate(null); return; }
    const controller = new AbortController();
    fetchPricingRates({
      gpu_model: resolvedGpu,
      tier,
      mode: pricingMode,
      priority,
      num_gpus: Number(numGpus),
      province: province || "ON",
    })
      .then((r) => { if (!controller.signal.aborted) setDynamicRate(r); })
      .catch(() => { if (!controller.signal.aborted) setDynamicRate(null); });
    return () => controller.abort();
  }, [open, resolvedGpu, tier, pricingMode, priority, numGpus, province]);

  useEffect(() => {
    if (listing || !gpuModel || selectedGpuStillAvailable) return;
    if (!selectedGpuStillAvailable) {
      setGpuModel("");
    }
  }, [gpuModel, listing, selectedGpuStillAvailable]);

  // Auto-fill VRAM when GPU model changes
  useEffect(() => {
    if (vramOptionsForGpu.length === 1) {
      setVramGb(vramOptionsForGpu[0]);
    } else {
      setVramGb("");
    }
  }, [resolvedGpu]); // eslint-disable-line react-hooks/exhaustive-deps

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
      setVramGb("");
      setPricingMode("on_demand");
      setTier("standard");
      setMaxBid("");
      setPriority("normal");
      setSelectedVolumeIds([]);
      setLaunchError(null);
      setInstanceId("");
      setDynamicRate(null);
      if (!listing) setGpuModel("");
    }
  }, [open, listing]);

  async function handleSubmit() {
    if (!instanceName.trim()) { toast.error("Enter an instance name"); return; }
    if (!resolvedImage) { toast.error("Select a Docker image"); return; }
    setSubmitting(true);
    try {
      const numericPriority = PRIORITIES.find((p) => p.value === priority)?.numeric ?? 3;
      const params: LaunchInstanceParams = {
        name: instanceName.trim(),
        image: resolvedImage,
        vram_needed_gb: resolvedVramGb,
        num_gpus: Number(numGpus),
        priority: numericPriority,
        tier,
        gpu_model: resolvedGpu || undefined,
        volume_ids: selectedVolumeIds.length > 0 ? selectedVolumeIds : undefined,
        encrypted_workspace: encryptedWorkspace || undefined,
      };
      if (listing?.host_id) params.host_id = listing.host_id;
      if (pricingMode === "spot") {
        const bid = maxBid ? Number(maxBid) : spotRate;
        if (!(bid && bid > 0)) {
          toast.error(isAutoGpuSelection ? "Enter a max bid when using Auto GPU with spot pricing" : "Enter a valid max bid");
          setSubmitting(false);
          return;
        }
        params.max_bid = bid;
      }
      const res = await launchInstance(params);
      const jobId = res.instance?.job_id || "";
      setInstanceId(jobId);
      setStep("success");
      markInstanceLaunched();
      toast.success("Instance launched successfully");
      onLaunched?.(jobId, res.instance);
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
                          "flex h-14 w-14 items-center justify-center rounded-2xl transition-transform group-hover:scale-105",
                          !selectedTemplate
                            ? "bg-ice-blue/10 text-ice-blue"
                            : "bg-surface/60 text-text-secondary",
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

                {/* GPU model + VRAM + count (non-listing mode) */}
                {!listing && (
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="space-y-1.5">
                      <Label className="text-xs">GPU Model</Label>
                      <Select
                        value={gpuModel}
                        onChange={(e) => setGpuModel(e.target.value)}
                      >
                        <option value="">Auto-select best available</option>
                        {gpuModelOptions.map((option) => (
                          <option key={option.gpu_model} value={option.gpu_model}>
                            {option.gpu_model}
                            {option.maxVramGb > 0 ? ` · ${option.maxVramGb} GB` : ""}
                            {option.countAvailable > 0 ? ` · ${option.countAvailable} available` : ""}
                          </option>
                        ))}
                      </Select>
                      <p className="text-[11px] text-text-muted">
                        {hasLiveInventory
                          ? "Live GPU inventory is aggregated across active hosts."
                          : hasHostedInventory
                            ? "Hosted GPU inventory is aggregated from registered hosts; none are active right now."
                          : "Live inventory is unavailable, so the picker is using reference models as a fallback."}
                      </p>
                    </div>
                    <div className="grid gap-4 grid-rows-[auto_auto]">
                      {/* VRAM selector — auto-fill or dropdown */}
                      <div className="space-y-1.5">
                        <Label className="text-xs">VRAM</Label>
                        {!resolvedGpu ? (
                          <Select disabled>
                            <option>Select a GPU first</option>
                          </Select>
                        ) : vramOptionsForGpu.length <= 1 ? (
                          <Select disabled value={String(resolvedVramGb ?? "")}>
                            <option>{resolvedVramGb ? `${resolvedVramGb} GB` : "—"}</option>
                          </Select>
                        ) : (
                          <Select
                            value={String(vramGb)}
                            onChange={(e) => setVramGb(e.target.value ? Number(e.target.value) : "")}
                          >
                            <option value="">Select VRAM</option>
                            {vramOptionsForGpu.map((v) => (
                              <option key={v} value={v}>{v} GB</option>
                            ))}
                          </Select>
                        )}
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
                      placeholder={spotRate != null ? `Current spot: $${spotRate.toFixed(2)}/hr` : "Enter your max bid"}
                      value={maxBid}
                      onChange={(e) => setMaxBid(e.target.value)}
                    />
                    <p className="text-xs text-text-muted">
                      {spotRate != null
                        ? "Leave empty to bid at current spot price. Your instance may be preempted if outbid."
                        : "Auto GPU selection needs an explicit max bid because the final spot market depends on the assigned host."}
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

                {/* Province / Region */}
                <div className="space-y-1.5">
                  <Label className="text-xs flex items-center gap-1.5">
                    <MapPin className="h-3.5 w-3.5" />
                    Province / Region
                  </Label>
                  <Select
                    value={province}
                    onChange={(e) => setProvince(e.target.value)}
                  >
                    <option value="">Auto-detect</option>
                    {Object.entries(provinces).map(([code, info]) => (
                      <option key={code} value={code}>
                        {code} — {info.name} ({info.tax_description})
                      </option>
                    ))}
                  </Select>
                </div>

                {/* Attached Volumes */}
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-1.5">
                    <HardDrive className="h-3.5 w-3.5 text-text-muted" />
                    Volumes
                    {selectedVolumeIds.length > 0 && (
                      <span className="ml-auto text-[10px] font-mono text-ice-blue">
                        {selectedVolumeIds.length} selected
                      </span>
                    )}
                  </Label>
                  {availableVolumes.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-border/60 p-4 text-center">
                      <HardDrive className="h-5 w-5 text-text-muted mx-auto mb-1.5 opacity-40" />
                      <p className="text-xs text-text-muted">No volumes available</p>
                      <p className="text-[10px] text-text-muted/60 mt-0.5">Create a volume from the Volumes page first</p>
                    </div>
                  ) : (
                    <div className="space-y-1.5 max-h-[140px] overflow-y-auto pr-1 scrollbar-thin">
                      {availableVolumes.map((vol) => {
                        const isSelected = selectedVolumeIds.includes(vol.volume_id);
                        return (
                          <button
                            key={vol.volume_id}
                            type="button"
                            onClick={() =>
                              setSelectedVolumeIds((prev) =>
                                isSelected
                                  ? prev.filter((id) => id !== vol.volume_id)
                                  : [...prev, vol.volume_id]
                              )
                            }
                            className={cn(
                              "w-full flex items-center gap-3 rounded-lg border p-2.5 text-left transition-all duration-150",
                              isSelected
                                ? "border-ice-blue/40 bg-ice-blue/5 shadow-[0_0_12px_rgba(0,212,255,0.06)]"
                                : "border-border hover:border-text-muted/40 hover:bg-surface/50"
                            )}
                          >
                            <div className={cn(
                              "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-colors",
                              isSelected ? "bg-ice-blue/15 text-ice-blue" : "bg-surface text-text-muted"
                            )}>
                              <HardDrive className="h-4 w-4" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-1.5">
                                <span className={cn("text-xs font-medium truncate", isSelected && "text-ice-blue")}>
                                  {vol.name}
                                </span>
                                {vol.encrypted && (
                                  <Lock className="h-2.5 w-2.5 text-accent-gold shrink-0" />
                                )}
                              </div>
                              <div className="flex items-center gap-2 mt-0.5">
                                <span className="text-[10px] text-text-muted font-mono">{vol.size_gb} GB</span>
                                {vol.region && (
                                  <span className="text-[10px] text-text-muted/60">{vol.region}</span>
                                )}
                              </div>
                            </div>
                            <div className={cn(
                              "h-4 w-4 rounded-full border-2 shrink-0 transition-all duration-150 flex items-center justify-center",
                              isSelected
                                ? "border-ice-blue bg-ice-blue"
                                : "border-text-muted/30"
                            )}>
                              {isSelected && (
                                <svg viewBox="0 0 12 12" className="h-2.5 w-2.5 text-navy fill-current">
                                  <path d="M10 3L4.5 8.5L2 6" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                              )}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  )}
                  <p className="text-[10px] text-text-muted/60">Volumes mount at <code className="text-ice-blue/70">/workspace</code> inside your container.</p>
                </div>

                {/* Encrypted Workspace toggle */}
                <button
                  type="button"
                  onClick={() => setEncryptedWorkspace(!encryptedWorkspace)}
                  className={cn(
                    "flex items-center gap-3 w-full rounded-lg border p-3 text-sm transition-colors text-left",
                    encryptedWorkspace
                      ? "border-accent-violet/50 bg-accent-violet/5"
                      : "border-border/60 bg-surface-hover/40 hover:border-border",
                  )}
                >
                  <Lock className={cn("h-4 w-4 shrink-0", encryptedWorkspace ? "text-accent-violet" : "text-text-muted")} />
                  <div className="flex-1 min-w-0">
                    <span className={cn("font-medium", encryptedWorkspace ? "text-text-primary" : "text-text-secondary")}>
                      Encrypted Workspace
                    </span>
                    <p className="text-xs text-text-muted mt-0.5">LUKS2 AES-256 encrypted ephemeral storage · destroyed on termination</p>
                  </div>
                  <div className={cn(
                    "h-5 w-9 rounded-full relative transition-colors shrink-0",
                    encryptedWorkspace ? "bg-accent-violet" : "bg-border",
                  )}>
                    <div className={cn(
                      "absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform shadow-sm",
                      encryptedWorkspace ? "translate-x-4" : "translate-x-0.5",
                    )} />
                  </div>
                </button>

                {/* Hourly Rate summary */}
                <div className="rounded-lg border border-border p-3 bg-surface">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-text-secondary">Hourly Rate</span>
                    {effectiveRate != null ? (
                      <span className="text-lg font-bold font-mono">
                        {isEstimatedRate && "~"}${effectiveRate.toFixed(2)}
                        <span className="text-xs text-text-muted ml-1">CAD/hr{isEstimatedRate && " est."}</span>
                      </span>
                    ) : (
                      <span className="text-sm text-text-muted italic">No pricing data</span>
                    )}
                  </div>
                  <p className="text-xs text-text-muted mt-1">
                    {isEstimatedRate
                      ? `Best available GPU · exact rate depends on assigned host`
                      : pricingMode === "spot"
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
                    <div className="flex justify-between"><span className="text-text-muted">GPU</span><span>{resolvedGpu || (gpuModelOptions[0]?.gpu_model ? `Best Available (${gpuModelOptions[0].gpu_model})` : "Best Available")}</span></div>
                    {Number(numGpus) > 1 && (
                      <div className="flex justify-between"><span className="text-text-muted">GPUs</span><span>×{numGpus}</span></div>
                    )}
                    <div className="flex justify-between"><span className="text-text-muted">Instance</span><span>{instanceName}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Image</span><span className="text-right max-w-[220px] truncate" title={resolvedImage}>{resolvedImage}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Pricing</span><span className="capitalize">{pricingMode === "on_demand" ? "On-Demand" : "Spot"}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Tier</span><span className="capitalize">{tier}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Priority</span><span>{PRIORITIES.find((p) => p.value === priority)?.label ?? "Normal"}</span></div>
                    {selectedVolumeIds.length > 0 && (
                      <div className="flex justify-between">
                        <span className="text-text-muted">Volumes</span>
                        <span className="text-right max-w-[220px] truncate" title={selectedVolumeIds.map((id) => availableVolumes.find((v) => v.volume_id === id)?.name || id.slice(0, 8)).join(", ")}>
                          {selectedVolumeIds.map((id) => availableVolumes.find((v) => v.volume_id === id)?.name || id.slice(0, 8)).join(", ")}
                        </span>
                      </div>
                    )}
                    {encryptedWorkspace && (
                      <div className="flex justify-between">
                        <span className="text-text-muted">Workspace</span>
                        <span className="flex items-center gap-1"><Lock className="h-3 w-3 text-accent-violet" /> Encrypted (LUKS2)</span>
                      </div>
                    )}
                    <div className="flex justify-between font-medium text-sm pt-1 border-t border-accent-gold/20">
                      <span>Rate</span>
                      <span>{effectiveRate != null ? `${isEstimatedRate ? "~" : ""}$${effectiveRate.toFixed(2)}/hr CAD${isEstimatedRate ? " est." : ""}` : "—"}</span>
                    </div>
                    {totalPerHour != null && (
                      <>
                        <div className="flex justify-between text-sm">
                          <span>Per hour ({Number(numGpus) > 1 ? `${numGpus} GPUs` : "1 GPU"})</span>
                          <span>{isEstimatedRate && "~"}${totalPerHour.toFixed(2)} CAD/hr</span>
                        </div>
                        {taxRate > 0 && (
                          <div className="flex justify-between text-sm">
                            <span>Tax ({(taxRate * 100).toFixed(1)}%)</span>
                            <span>+${(totalPerHour * taxRate).toFixed(2)} CAD/hr</span>
                          </div>
                        )}
                        {taxRate > 0 && totalWithTax != null && (
                          <div className="flex justify-between font-medium text-sm pt-1 border-t border-accent-gold/20">
                            <span>Total</span>
                            <span>{isEstimatedRate && "~"}${totalWithTax.toFixed(2)} CAD/hr</span>
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
