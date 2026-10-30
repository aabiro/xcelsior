"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "next/navigation";
import posthog from "posthog-js";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label, Select, NumberInput } from "@/components/ui/input";
import { TemplateArtwork } from "@/components/instances/template-artwork";
import {
  launchInstance,
  fetchAvailableGPUs,
  fetchPricingReference,
  fetchImageTemplates,
  fetchSpotPrices,
  classifyLaunchError,
  listAvailableVolumes,
  fetchPricingRates,
  fetchServiceTierCatalog,
  fetchSpotFeatureStatus,
} from "@/lib/api";
import type {
  PricingReference,
  GpuAvailability,
  ImageTemplate,
  LaunchErrorInfo,
  LaunchInstanceParams,
  MarketplaceListing,
  Volume,
  ServiceTierCatalog,
} from "@/lib/api";
import { PlacementPreferenceControl } from "@/components/instances/placement-preference-control";
import { cn, generateFunName } from "@/lib/utils";
import { getVramOptions } from "@/lib/gpu-models";
import { markInstanceLaunched } from "@/components/InstallBanner";
import {
  Activity,
  AlertTriangle,
  Box,
  CheckCircle,
  CreditCard,
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
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import { getTeamContext } from "@/lib/team-context";
import {
  SpotInterruptWarning,
  SpotKillSwitchBanner,
  SpotRateDisplay,
  SpotSavingsPill,
} from "@/components/spot/spot-surface";

/* ───────────────────────── Constants ───────────────────────── */

const PRICING_MODES = [
  {
    value: "on_demand" as const,
    label: "On-Demand",
    icon: Activity,
    desc: "Keep it running. Your allocated GPU cannot be reclaimed by another customer.",
  },
  {
    value: "spot" as const,
    label: "Spot",
    icon: TrendingDown,
    desc: "Lower cost. Your instance can be interrupted; save checkpoints so work can resume.",
  },
];

// Same conversion used by persistent-volume metering in billing/billing.py.
const HOURS_PER_MONTH = 730;

function hourlyPrice(value: number) {
  const exact = (Math.round(value * 1_000_000) / 1_000_000).toFixed(6).replace(/0+$/, "");
  const [whole, fraction = ""] = exact.split(".");
  return `$${whole}.${fraction.padEnd(2, "0")}/hr`;
}

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
  /** Pre-select volumes to attach (e.g. launched from volumes page) */
  preSelectedVolumeIds?: string[];
  /** Pre-select a user template image */
  templateId?: string;
  /** Pre-select GPU model (e.g. from spot pricing page) */
  initialGpuModel?: string;
  /** Pre-select pricing mode */
  initialPricingMode?: "on_demand" | "spot";
}

/* ───────────────────────── Component ───────────────────────── */

export function LaunchInstanceModal({
  open,
  onClose,
  onLaunched,
  listing,
  preSelectedVolumeIds,
  templateId,
  initialGpuModel,
  initialPricingMode,
}: LaunchInstanceModalProps) {
  const { user } = useAuth();
  const { t } = useLocale();
  const team = getTeamContext(user);
  const launchBlocked = !team.canWriteInstances;
  const router = useRouter();
  const [step, setStep] = useState<"configure" | "confirm" | "success">("configure");

  // Form state
  const [instanceName, setInstanceName] = useState(() => generateFunName());
  // Follow the default template until the user chooses an image.
  const [image, setImage] = useState<string | null>(null);
  const [gpuModel, setGpuModel] = useState("");
  const [vramGb, setVramGb] = useState<number | "">("");
  const [numGpus, setNumGpus] = useState("1");
  const [pricingMode, setPricingMode] = useState<"on_demand" | "spot">("on_demand");
  const [tier, setTier] = useState<"standard" | "premium" | "dedicated">("standard");
  const [gpuExclusive, setGpuExclusive] = useState(false);
  const [queueBid, setQueueBid] = useState("0.00");
  const [tierCatalog, setTierCatalog] = useState<ServiceTierCatalog | null>(null);
  const [tierPricingError, setTierPricingError] = useState<string | null>(null);
  // No province state, deliberately. Stripe Tax is the merchant-of-record
  // calculation and reads the *account* country/province — `api_create_payment_intent`
  // says so in as many words. `/api/pricing/rates` now follows the same source,
  // so the estimate shown here and the amount charged agree. Sending a province
  // from the client is what made them disagree: the old `detectProvince()` call
  // hit a route that does not exist and its `.catch` hard-coded "ON", quoting
  // 13% HST to every customer outside Ontario.

  // Volume picker state
  const [availableVolumes, setAvailableVolumes] = useState<Volume[]>([]);
  const [selectedVolumeIds, setSelectedVolumeIds] = useState<string[]>([]);

  // Template image
  const [templateImageId, setTemplateImageId] = useState<string>(templateId ?? "");

  // Encrypted workspace
  const [encryptedWorkspace, setEncryptedWorkspace] = useState(false);

  // Fetched data
  const [pricing, setPricing] = useState<PricingReference[]>([]);
  const [availableGpus, setAvailableGpus] = useState<GpuAvailability[]>([]);
  const [templates, setTemplates] = useState(FALLBACK_TEMPLATES);
  const [spotPrices, setSpotPrices] = useState<Record<string, number>>({});

  // Dynamic pricing from backend
  const [dynamicRate, setDynamicRate] = useState<{
    gpu_model: string;
    pricing_mode: string;
    effective_rate_per_gpu: number;
    total_per_hour: number;
    tax_rate: number;
    tax_description: string;
    tax_amount: number;
    total_with_tax: number;
    base_rate_cad: number;
    priority_multiplier: number;
    multi_gpu_discount: number;
  } | null>(null);

  // UI state
  const [submitting, setSubmitting] = useState(false);
  const [launchError, setLaunchError] = useState<LaunchErrorInfo | null>(null);
  const [instanceId, setInstanceId] = useState("");
  const [gpuLoading, setGpuLoading] = useState(true);
  const [spotQuoteLoading, setSpotQuoteLoading] = useState(false);
  const [spotQuoteError, setSpotQuoteError] = useState<string | null>(null);
  const [quoteRefresh, setQuoteRefresh] = useState(0);
  const [spotFeature, setSpotFeature] = useState<{
    enabled: boolean;
    message?: string | null;
  }>({ enabled: true });
  const effectivePricingMode =
    !spotFeature.enabled && pricingMode === "spot" ? "on_demand" : pricingMode;
  const priorityLabel = tier === "dedicated" ? "Highest" : tier === "premium" ? "Premium" : "Standard";

  const defaultImage = templates.find((t) => t.id === "pytorch")?.image ?? FALLBACK_TEMPLATES[0].image;
  const resolvedImage = (image ?? defaultImage).trim();
  const selectedTemplate = templates.find((t) => t.image === resolvedImage);
  const resolvedGpu = listing?.gpu_model || gpuModel;
  const templateVramGb = selectedTemplate ? Number(selectedTemplate.vram) : undefined;
  const isAutoGpuSelection = !listing && !resolvedGpu;
  // Show ALL GPUs in the dropdown, don't hide GPUs that have less VRAM than the template recommends.
  // Users can still launch on a smaller GPU (they'll see a warning).
  const eligibleInventory = availableGpus;
  const liveInventory = eligibleInventory.filter((gpu) => gpu.count_available > 0);
  const hasLiveInventory = liveInventory.length > 0;
  const hasHostedInventory = eligibleInventory.length > 0;
  const inventorySource = liveInventory.length > 0 ? liveInventory : eligibleInventory;
  const gpuInventoryOptions = Array.from(
    inventorySource.reduce((acc, gpu) => {
      const existing = acc.get(gpu.gpu_model);
      const price = gpu.price_per_hour_cad != null && gpu.price_per_hour_cad > 0 ? gpu.price_per_hour_cad : undefined;

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
  const gpuModelOptions = gpuInventoryOptions;
  const selectedGpuStillAvailable = !gpuModel || gpuModelOptions.some((option) => option.gpu_model === gpuModel);

  // VRAM options per selected GPU model, derived from live inventory
  // when available, otherwise from the canonical catalog in
  // frontend/src/lib/gpu-models.ts.
  const vramOptionsForGpu: number[] = (() => {
    if (!resolvedGpu) return [];
    // Collect unique VRAM values from live inventory for this model
    const fromInventory = [...new Set(
      availableGpus
        .filter((g) => g.gpu_model === resolvedGpu && g.vram_gb > 0)
        .map((g) => g.vram_gb),
    )].sort((a, b) => a - b);
    if (fromInventory.length > 0) return fromInventory;
    return getVramOptions(resolvedGpu);
  })();
  const resolvedVramGb = vramGb || (vramOptionsForGpu.length === 1 ? vramOptionsForGpu[0] : undefined);

  // Fleet rates are estimates until placement selects a host.
  const listingRate = listing ? (listing.price_per_hour_cad || listing.price_per_hour || 0) : 0;
  const estimatedGpu = resolvedGpu || gpuModelOptions
    .filter((entry) => entry.minPricePerHourCad != null)
    .sort((a, b) => a.minPricePerHourCad! - b.minPricePerHourCad!)[0]?.gpu_model
    || pricing[0]?.gpu_model;
  const estimatedInventory = gpuModelOptions.find((entry) => entry.gpu_model === estimatedGpu);
  const estimatedPricing = pricing.find((entry) => entry.gpu_model === estimatedGpu);
  const currentRate = dynamicRate && dynamicRate.gpu_model === estimatedGpu && dynamicRate.pricing_mode === effectivePricingMode
    ? dynamicRate
    : null;
  const currentCatalog = tierCatalog?.gpu_model === estimatedGpu && tierCatalog?.mode === effectivePricingMode && (!resolvedVramGb || tierCatalog.vram_gb === resolvedVramGb) ? tierCatalog : null;
  const selectedTierPrice = currentCatalog?.tiers.find((entry) => entry.tier === tier);
  const exclusiveAddon = gpuExclusive && tier !== "dedicated" ? currentCatalog?.gpu_exclusive_addon : null;
  const addonReady = !gpuExclusive || tier === "dedicated" || !!exclusiveAddon;
  const bidUnit = currentCatalog?.queue_bid_unit;
  const bidAmount = queueBid.trim() === "" ? 0 : Number(queueBid);
  const bidMicros = Math.round(bidAmount * 1_000_000);
  const bidUnits = bidUnit ? Math.round(bidMicros / bidUnit.unit_amount_micros) : 0;
  const bidReady = Number.isFinite(bidAmount) && bidAmount >= 0 && Math.abs(bidMicros / 1_000_000 - bidAmount) < 1e-12 && bidUnits <= 1_000_000 && (bidAmount === 0 || bidMicros > 0 && !!bidUnit && bidUnits * bidUnit.unit_amount_micros === bidMicros);
  const effectiveRate = selectedTierPrice && addonReady && bidReady ? (selectedTierPrice.unit_amount_micros + (exclusiveAddon?.unit_amount_micros ?? 0) + bidMicros) / 1_000_000 : null;
  const nextQueueTier = currentCatalog?.tiers.find((entry) => entry.priority === (selectedTierPrice?.priority ?? 0) + 1);
  const promotionMinutes = bidReady && bidMicros > 0 && nextQueueTier && selectedTierPrice
    ? Math.ceil(Math.max(0, nextQueueTier.unit_amount_micros - selectedTierPrice.unit_amount_micros - bidMicros) * 10 / bidMicros) * 15
    : null;
  const isEstimatedRate = false;
  const physicalHosts = currentCatalog?.physical_hosts.filter((host) => !listing?.host_id || host.host_id === listing.host_id) ?? [];
  const wholeHostSizes = [...new Set(physicalHosts.map((host) => host.num_gpus))].sort((a, b) => a-b);
  const wholeHostReady = tier !== "dedicated" || wholeHostSizes.includes(Number(numGpus));
  const computeBasePerHour = currentCatalog ? currentCatalog.base_rate_micros * Number(numGpus) / 1_000_000 : null;
  const tierAdditionPerHour = selectedTierPrice ? selectedTierPrice.tier_addition_micros * Number(numGpus) / 1_000_000 : null;
  const exclusivityPerHour = exclusiveAddon ? exclusiveAddon.unit_amount_micros * Number(numGpus) / 1_000_000 : 0;
  const onDemandRate = listing
    ? listingRate
    : estimatedInventory?.minPricePerHourCad ?? estimatedPricing?.on_demand_cad ?? null;
  const spotRate = effectivePricingMode === "spot"
    ? effectiveRate
    : spotPrices[estimatedGpu ?? ""] ?? estimatedPricing?.spot_cad ?? null;
  const savingsPct =
    onDemandRate != null && spotRate != null && onDemandRate > spotRate
      ? Math.round((1 - spotRate / onDemandRate) * 100)
      : null;
  const computePerHour = effectiveRate != null ? effectiveRate * Number(numGpus) : null;
  const selectedVolumes = availableVolumes.filter((v) => selectedVolumeIds.includes(v.volume_id));
  const volumePrices = selectedVolumes.map((volume) =>
    volume.monthly_cost_cad ?? (volume.price_per_gb_month_cad != null
      ? volume.size_gb * volume.price_per_gb_month_cad
      : null),
  );
  const volumePricingKnown = selectedVolumes.length === selectedVolumeIds.length && volumePrices.every((cost) => cost != null);
  const volumeMonthlyCost = volumePrices.reduce<number>((total, cost) => total + (cost ?? 0), 0);
  const volumePerHour = volumePricingKnown ? volumeMonthlyCost / HOURS_PER_MONTH : null;
  const totalPerHour = computePerHour != null && volumePerHour != null ? computePerHour + volumePerHour : null;
  // `/api/pricing/rates` is the only source of a tax rate now; the removed
  // fallback read a map that the 404ing fetch always left empty, so it only
  // ever yielded 0.
  const taxRate = currentRate?.tax_rate;
  const totalWithTax = totalPerHour != null && taxRate != null ? totalPerHour * (1 + taxRate) : null;

  const priceSummary = (
    <div className="rounded-lg border border-border p-3 bg-surface space-y-2" aria-live="polite">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm text-text-secondary">Estimated total (CAD)</span>
        <span className="text-lg font-bold font-mono" data-testid="launch-total">
          {totalPerHour != null ? hourlyPrice(totalWithTax ?? totalPerHour) : "Unavailable"}
        </span>
      </div>
      <div className="flex justify-between gap-3 text-xs">
        <span className="text-text-muted">Compute ({numGpus} {Number(numGpus) === 1 ? "GPU" : "GPUs"})</span>
        <span>{computeBasePerHour != null ? hourlyPrice(computeBasePerHour) : "Loading prices…"}</span>
      </div>
      <div className="flex justify-between gap-3 text-xs">
        <span className="text-text-muted">{selectedTierPrice?.label ?? tier} service tier</span>
        <span>{tierAdditionPerHour != null ? `+${hourlyPrice(tierAdditionPerHour)}` : "Loading prices…"}</span>
      </div>
      {exclusiveAddon && <div className="flex justify-between gap-3 text-xs"><span className="text-text-muted">Exclusive allocated GPUs</span><span>+{hourlyPrice(exclusivityPerHour)}</span></div>}
      {bidMicros > 0 && bidReady && <div className="flex justify-between gap-3 text-xs"><span className="text-text-muted">Queue bid ({numGpus} {Number(numGpus) === 1 ? "GPU" : "GPUs"})</span><span>+{hourlyPrice(bidMicros * Number(numGpus) / 1_000_000)}</span></div>}
      {selectedVolumeIds.length > 0 && (
        <>
          <div className="flex justify-between gap-3 text-xs">
            <span className="text-text-muted">Volume storage ({selectedVolumes.reduce((total, volume) => total + volume.size_gb, 0)} GB)</span>
            <span>{volumePerHour != null ? hourlyPrice(volumePerHour) : "Price unavailable"}</span>
          </div>
          {volumePricingKnown && <p className="text-[11px] text-text-muted">${volumeMonthlyCost.toFixed(2)}/month of existing storage. Already billed separately; attaching adds no new fee. Storage billing continues when compute stops.</p>}
        </>
      )}
      <div className="flex justify-between gap-3 text-xs">
        <span className="text-text-muted">Estimated tax{taxRate != null ? ` (${(taxRate * 100).toFixed(1)}%)` : ""}</span>
        <span>{taxRate != null && totalPerHour != null ? hourlyPrice(totalPerHour * taxRate) : "Calculated when available"}</span>
      </div>
      <p className="text-[11px] text-text-muted">
        Prices are locked when your launch is accepted. Usage is billed per second.{" "}
        {taxRate == null ? "Total excludes tax until the estimate is available. " : ""}
        {effectivePricingMode === "spot" ? "Spot uses spare capacity and can be interrupted." : "On-demand protects your allocation from reclamation by another customer."}
      </p>
    </div>
  );

  // Fetch reference data
  useEffect(() => {
    if (!open) return;
    fetchPricingReference()
      .then((r) => setPricing(r.reference || []))
      .catch(() => {});
    setGpuLoading(true);
    fetchAvailableGPUs()
      .then((r) => {
        console.debug("[gpu] fetchAvailableGPUs →", r.gpus?.length ?? 0, "models");
        setAvailableGpus(r.gpus || []);
      })
      .catch((err) => { console.error("[gpu] fetchAvailableGPUs failed:", err); })
      .finally(() => setGpuLoading(false));
    fetchImageTemplates()
      .then((r) => {
        if (r.templates?.length) {
          setTemplates(
            r.templates
              .map((t: ImageTemplate) => ({
                id: t.id,
                label: t.label,
                image: t.image,
                vram: String(t.default_vram_gb),
                icon: t.icon,
              }))
              .sort((a, b) => Number(b.id === "pytorch") - Number(a.id === "pytorch")),
          );
        }
      })
      .catch(() => {});
    fetchSpotPrices()
      // `prices` is the lookup this component wants — `{"A10": 0.42}`.
      // `spot_prices` is a **list** of rows. This read `res.spot_prices ||
      // res.prices`, and a non-empty array is truthy, so the correctly-shaped
      // fallback was unreachable and every `spotPrices[gpuModel]` was
      // `undefined` — the per-listing spot price simply never rendered.
      //
      // The list is still the fallback rather than being dropped: it carries
      // the same rates, so deriving the lookup from it keeps this working if
      // `prices` is ever retired.
      .then((res) =>
        setSpotPrices(
          res.prices ??
            Object.fromEntries((res.spot_prices ?? []).map((row) => [row.gpu_model, row.rate_cad])),
        ),
      )
      .catch(() => {});
    fetchSpotFeatureStatus()
      .then((res) => setSpotFeature({ enabled: res.enabled, message: res.message }))
      .catch(() => setSpotFeature({ enabled: true }));
    listAvailableVolumes()
      .then((r) => setAvailableVolumes(r.volumes || []))
      .catch(() => {});
    // Sync templateId prop into state on open
    setTemplateImageId(templateId ?? "");
  }, [open, templateId]);

  useEffect(() => {
    if (!open) return;
    const reloadVolumes = () => {
      listAvailableVolumes()
        .then((r) => setAvailableVolumes(r.volumes || []))
        .catch(() => {});
    };
    window.addEventListener("xcelsior-team-changed", reloadVolumes);
    return () => window.removeEventListener("xcelsior-team-changed", reloadVolumes);
  }, [open]);

  const launchRegion = listing?.region?.trim().toLowerCase() || "";
  const volumeRegionMismatch =
    launchRegion.length > 0 &&
    selectedVolumes.some((v) => v.region && v.region.toLowerCase() !== launchRegion);

  // Dynamic pricing, recompute when any pricing variable changes
  useEffect(() => {
    if (!open || !estimatedGpu) {
      setDynamicRate(null);
      setSpotQuoteLoading(false);
      return;
    }
    const controller = new AbortController();
    setDynamicRate(null);
    setTierCatalog(null);
    setTierPricingError(null);
    fetchServiceTierCatalog(estimatedGpu, effectivePricingMode, resolvedVramGb)
      .then((catalog) => { if (!controller.signal.aborted) setTierCatalog(catalog); })
      .catch(() => { if (!controller.signal.aborted) setTierPricingError("Unable to load service-tier prices. Retry before launching."); });
    if (effectivePricingMode === "spot") {
      setSpotQuoteLoading(true);
      setSpotQuoteError(null);
    }
    fetchPricingRates({
      gpu_model: estimatedGpu,
      vram_gb: resolvedVramGb,
      tier: "standard",
      mode: effectivePricingMode,
      priority: "normal",
      num_gpus: 1,
    })
      .then((r) => {
        if (controller.signal.aborted) return;
        setDynamicRate({ ...r, gpu_model: estimatedGpu, pricing_mode: effectivePricingMode });
        if (effectivePricingMode === "spot") setSpotQuoteError(null);
      })
      .catch(() => {
        if (controller.signal.aborted) return;
        setDynamicRate(null);
        if (effectivePricingMode === "spot") {
          setSpotQuoteError("Unable to load spot rate. Check your connection and retry.");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted && effectivePricingMode === "spot") {
          setSpotQuoteLoading(false);
        }
      });
    return () => controller.abort();
  }, [open, estimatedGpu, effectivePricingMode, resolvedVramGb, quoteRefresh]);

  function reloadSpotQuote() {
    setQuoteRefresh((value) => value + 1);
  }

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
      setImage(null);
      setNumGpus("1");
      setVramGb("");
      setPricingMode(initialPricingMode ?? "on_demand");
      setTier("standard");
      setGpuExclusive(false);
      setQueueBid("0.00");
      setSelectedVolumeIds(preSelectedVolumeIds ?? []);
      setLaunchError(null);
      setInstanceId("");
      setDynamicRate(null);
      setSpotQuoteError(null);
      setSpotQuoteLoading(false);
      if (!listing) setGpuModel(initialGpuModel ?? "");
    }
  }, [open, listing, initialGpuModel, initialPricingMode, preSelectedVolumeIds]);

  async function handleSubmit() {
    if (launchBlocked) {
      toast.error(t("dash.instances.viewer_launch_blocked"));
      return;
    }
    if (!instanceName.trim()) { toast.error("Enter an instance name"); return; }
    if (!resolvedImage) { toast.error("Select a Docker image"); return; }
    if (!selectedTierPrice || !wholeHostReady || !addonReady || !volumePricingKnown || !bidReady) { toast.error("Refresh resource prices and check your queue bid and host configuration"); return; }
    setSubmitting(true);
    try {
      const params: LaunchInstanceParams = {
        name: instanceName.trim(),
        image: resolvedImage,
        vram_needed_gb: resolvedVramGb,
        num_gpus: Number(numGpus),
        tier,
        service_tier_price_id: selectedTierPrice.stripe_price_id,
        gpu_exclusive_price_id: exclusiveAddon?.stripe_price_id,
        queue_bid_units: bidUnits,
        queue_bid_price_id: bidUnits > 0 ? bidUnit?.stripe_price_id : undefined,
        exclusive_gpu: gpuExclusive && tier !== "dedicated",
        gpu_model: estimatedGpu,
        volume_ids: selectedVolumeIds.length > 0 ? selectedVolumeIds : undefined,
        encrypted_workspace: encryptedWorkspace || undefined,
        template_image_id: templateImageId || undefined,
      };
      if (listing?.host_id) params.host_id = listing.host_id;
      if (effectivePricingMode === "spot") {
        params.pricing_mode = "spot";
      }
      const res = await launchInstance(params);
      const jobId = res.instance?.job_id || "";
      setInstanceId(jobId);
      markInstanceLaunched();
      toast.success("Instance launched, provisioning now.");
      posthog.capture("gpu_instance_launched", {
        gpu_model: resolvedGpu || null,
        pricing_mode: effectivePricingMode,
        tier: params.tier,
        num_gpus: params.num_gpus,
        has_volume: (params.volume_ids?.length ?? 0) > 0,
        from_marketplace: Boolean(listing?.host_id),
        encrypted_workspace: Boolean(params.encrypted_workspace),
      });
      onLaunched?.(jobId, res.instance);
      onClose();
      if (jobId) {
        router.push(`/dashboard/instances/${encodeURIComponent(jobId)}`);
      }
    } catch (err) {
      const info = classifyLaunchError(err);
      // Report only genuinely unexpected failures. Recognized user-state
      // rejections (empty or suspended wallet, concurrency limit, invalid
      // config) already reach the user as a friendly, actionable message, so
      // capturing them just opens a fresh error issue for a normal state.
      // Server errors still get captured — a 503 with no capacity is worth
      // knowing about even though the user is shown a tidy message.
      const isServerError = info.status !== undefined && info.status >= 500;
      if (!info.handled || isServerError) {
        // A status-stable message so captures group under one fingerprint
        // rather than splitting on the amounts interpolated into the detail.
        const captured =
          info.status !== undefined
            ? new Error(`Instance launch failed (HTTP ${info.status})`)
            : err instanceof Error
              ? err
              : new Error(String(err));
        posthog.captureException(captured);
      }
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

  return createPortal(
    <div className="dashboard-site-modal-overlay fixed inset-0 z-[300] flex items-center justify-center" onClick={onClose}>
      <div className="dashboard-site-modal-panel mx-4 w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl border border-border/50 bg-[var(--popover-solid)]" onClick={(e) => e.stopPropagation()}>
        <Card className="border-0 bg-[var(--popover-solid)] shadow-none">
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
            {launchBlocked && (
              <div className="rounded-lg border border-accent-gold/30 bg-accent-gold/10 px-3 py-2 text-sm text-accent-gold">
                {t("dash.instances.viewer_launch_blocked")}
              </div>
            )}
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
                        <span className="flex items-center gap-1"><Zap className="h-3 w-3" /> {listing.vram_gb || "-"}GB VRAM</span>
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
                    {templates.map((tpl) => (
                      <button
                        key={tpl.id}
                        type="button"
                        onClick={() => setImage(tpl.image)}
                        className={cn(
                          "group flex min-h-[108px] flex-col items-center justify-center gap-2 rounded-xl border p-3 text-center transition-colors",
                          resolvedImage === tpl.image
                            ? "border-ice-blue/40 bg-ice-blue/5"
                            : "border-border/80 bg-background/30 text-text-secondary hover:border-text-muted hover:text-text-primary",
                        )}
                      >
                        <div
                          className={cn(
                            "flex h-14 w-14 items-center justify-center rounded-2xl transition-transform group-hover:scale-105",
                            resolvedImage === tpl.image
                              ? "bg-ice-blue/10"
                              : "bg-surface/60",
                          )}
                        >
                          <TemplateArtwork template={tpl.id} size={32} />
                        </div>
                        <span className="font-medium text-text-primary">{tpl.label}</span>
                        <span className="text-[11px] uppercase tracking-[0.16em] text-text-muted">{tpl.vram} GB VRAM</span>
                      </button>
                    ))}
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
                  </div>
                  <Input
                    placeholder="docker.io/my-org/my-image:latest"
                    value={image ?? defaultImage}
                    onChange={(e) => setImage(e.target.value)}
                    className="mt-1.5"
                  />
                </div>

                {/* GPU model + VRAM + count (non-listing mode) */}
                {!listing && (
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="space-y-1.5">
                      <Label htmlFor="launch-gpu-model" className="text-xs">GPU Model</Label>
                      <Select
                        id="launch-gpu-model"
                        value={gpuModel}
                        onChange={(e) => setGpuModel(e.target.value)}
                        disabled={gpuLoading}
                      >
                        <option value="">{gpuLoading ? "Loading GPU inventory…" : "Auto-select best available"}</option>
                        {gpuModelOptions.map((option) => (
                          <option key={option.gpu_model} value={option.gpu_model}>
                            {option.gpu_model}
                            {option.maxVramGb > 0 ? ` · ${option.maxVramGb} GB` : ""}
                            {option.countAvailable > 0 ? ` · ${option.countAvailable} available` : ""}
                          </option>
                        ))}
                      </Select>
                      <p className="text-[11px] text-text-muted">
                        {gpuLoading
                          ? "Loading live GPU inventory…"
                          : hasLiveInventory
                          ? "Live GPU inventory is aggregated across active hosts."
                          : hasHostedInventory
                            ? "Hosted GPU inventory is aggregated from registered hosts; none are active right now."
                          : "Live inventory is unavailable, so the picker is using reference models as a fallback."}
                      </p>
                    </div>
                    <div className="grid gap-4 grid-rows-[auto_auto]">
                      {/* VRAM selector, auto-fill or dropdown */}
                      <div className="space-y-1.5">
                        <Label className="text-xs">VRAM</Label>
                        {!resolvedGpu ? (
                          <Select disabled>
                            <option>Select a GPU first</option>
                          </Select>
                        ) : vramOptionsForGpu.length <= 1 ? (
                          <Select disabled value={String(resolvedVramGb ?? "")}>
                            <option>{resolvedVramGb ? `${resolvedVramGb} GB` : "-"}</option>
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
                        <Label htmlFor="launch-gpu-count" className="text-xs">GPUs</Label>
                        <NumberInput
                          id="launch-gpu-count"
                          disabled={tier === "dedicated"}
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
                  <Label className="text-xs">Can your workload handle interruptions?</Label>
                  {!spotFeature.enabled && (
                    <SpotKillSwitchBanner message={spotFeature.message} />
                  )}
                  <div className="grid grid-cols-2 gap-2">
                    {PRICING_MODES.map((mode) => {
                      const Icon = mode.icon;
                      const spotDisabled = mode.value === "spot" && !spotFeature.enabled;
                      return (
                        <button
                          key={mode.value}
                          type="button"
                          disabled={spotDisabled}
                          onClick={() => {
                            if (spotDisabled) return;
                            setPricingMode(mode.value);
                          }}
                          className={cn(
                            spotDisabled && "opacity-50 cursor-not-allowed",
                            "rounded-lg border p-3 text-left transition-colors",
                            effectivePricingMode === mode.value
                              ? "border-ice-blue/40 bg-ice-blue/5"
                              : "border-border hover:border-text-muted",
                          )}
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <Icon className="h-4 w-4" />
                            <span className="text-sm font-medium">{mode.label}</span>
                            {mode.value === "spot" && savingsPct != null && savingsPct > 0 && (
                              <span className="ml-auto rounded-full bg-emerald/20 px-1.5 py-0.5 text-[10px] font-medium text-emerald">
                                Save {savingsPct}%
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-text-muted">{mode.desc}</p>
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Service Tier</Label>
                  <div className="grid grid-cols-3 gap-2">
                    {(["standard", "premium", "dedicated"] as const).map((value) => {
                      const price = currentCatalog?.tiers.find((entry) => entry.tier === value);
                      const count = value === "dedicated" ? wholeHostSizes[0] ?? Number(numGpus) : Number(numGpus);
                      return <button
                        key={value} type="button" disabled={!price}
                        aria-pressed={tier === value}
                        onClick={() => {
                          setTier(value);
                          if (value === "dedicated") {
                            setGpuExclusive(false);
                            if (wholeHostSizes.length) setNumGpus(String(wholeHostSizes[0]));
                          }
                        }}
                        className={cn("rounded-lg border p-3 text-left transition-colors disabled:opacity-50", tier === value ? "border-ice-blue/40 bg-ice-blue/5" : "border-border hover:border-text-muted")}
                      >
                        <p className="text-sm font-medium capitalize">{value}</p>
                        <p className="mt-1 text-xs font-mono">{price ? `+${hourlyPrice(price.tier_addition_micros * count / 1_000_000)}` : "Loading prices…"}</p>
                        <p className="mt-1 text-[11px] text-text-muted">{value === "standard" ? "Standard scheduling" : value === "premium" ? "Priority scheduling" : "Entire physical host · highest priority"}</p>
                      </button>;
                    })}
                  </div>
                  {tierPricingError && <div className="flex items-center justify-between gap-2 text-xs text-accent-red"><span>{tierPricingError}</span><Button variant="outline" size="sm" onClick={reloadSpotQuote}>Retry pricing</Button></div>}
                </div>

                <div className="space-y-1.5 rounded-lg border border-border p-3">
                  <Label htmlFor="launch-queue-bid" className="text-xs">Additional queue bid (CAD / GPU-hour)</Label>
                  <Input id="launch-queue-bid" type="number" min="0" step={bidUnit ? bidUnit.unit_amount_micros / 1_000_000 : 0.01} value={queueBid} onChange={(event) => setQueueBid(event.target.value)} disabled={!bidUnit} />
                  <p className="text-[11px] text-text-muted">Optional. Moves your job ahead within its service tier. Added to the running compute rate; no charge while waiting.</p>
                  <details className="text-[11px] text-text-muted"><summary className="cursor-pointer">How queue aging works</summary><p className="mt-1">Every 15 minutes adds free scheduling credit equal to 10% of your original bid. Your bid plus that credit can promote queue position by one tier. Promoted jobs join a FIFO section behind customers who purchased that tier. Your price and hardware isolation stay unchanged.</p></details>
                  {promotionMinutes != null && <p className="text-[11px] text-text-secondary">{promotionMinutes === 0 ? `Qualifies for the back of the ${nextQueueTier?.label} queue.` : `Can join the back of the ${nextQueueTier?.label} queue after ${promotionMinutes} minutes waiting.`} Starting still depends on available capacity.</p>}
                  {!bidReady && <p className="text-xs text-accent-red">Enter a nonnegative amount in {bidUnit ? hourlyPrice(bidUnit.unit_amount_micros / 1_000_000) : "$0.01/hr"} increments.</p>}
                </div>

                {tier === "dedicated" ? (
                  <div className="space-y-1.5">
                    <Label htmlFor="launch-whole-host" className="text-xs">Physical host size</Label>
                    <Select id="launch-whole-host" value={numGpus} onChange={(event) => setNumGpus(event.target.value)} disabled={!wholeHostSizes.length}>
                      {!wholeHostSizes.length && <option>No qualified physical hosts available</option>}
                      {wholeHostSizes.map((count) => <option key={count} value={count}>Entire host · {count} {count === 1 ? "GPU" : "GPUs"}</option>)}
                    </Select>
                    <p className="text-[11px] text-text-muted">All GPUs on one physical host are reserved for you. The total includes every GPU; another customer&apos;s workload cannot share that host.</p>
                  </div>
                ) : (
                  <label className="flex items-start gap-3 rounded-lg border border-border p-3 text-xs">
                    <input type="checkbox" checked={gpuExclusive} disabled={!currentCatalog?.gpu_exclusive_addon} onChange={(event) => setGpuExclusive(event.target.checked)} />
                    <span><span className="font-medium">Exclusive allocated GPUs</span><span className="ml-2 font-mono">{currentCatalog?.gpu_exclusive_addon ? `+${hourlyPrice(currentCatalog.gpu_exclusive_addon.rate_per_gpu_cad * Number(numGpus))}` : "Loading price…"}</span><span className="mt-1 block text-text-muted">Keep the allocated GPUs free of other GPU processes while allowing other GPUs on the host to serve other customers.</span></span>
                  </label>
                )}

                {effectivePricingMode === "spot" && (
                  <SpotInterruptWarning>
                    {spotQuoteError ? (
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-xs text-accent-red">{spotQuoteError}</p>
                        <Button type="button" variant="outline" size="sm" onClick={reloadSpotQuote}>
                          <RefreshCw className="h-3 w-3" /> Retry
                        </Button>
                      </div>
                    ) : spotRate != null || spotQuoteLoading ? (
                      <div className="flex flex-wrap items-center gap-2">
                        <SpotRateDisplay
                          rateCad={spotRate}
                          onDemandCad={onDemandRate}
                          savingsPct={savingsPct}
                          loading={spotQuoteLoading && spotRate == null}
                          size="lg"
                        />
                        {savingsPct != null && savingsPct > 0 && (
                          <SpotSavingsPill pct={savingsPct} className="text-xs" />
                        )}
                      </div>
                    ) : (
                      <p className="text-xs text-text-muted">Select a GPU to see the spot rate.</p>
                    )}
                    <p className="text-xs text-text-muted">
                      Published spot pricing with no bidding. Your instance may be reclaimed when
                      capacity is needed and will automatically requeue when GPUs are free.
                    </p>
                  </SpotInterruptWarning>
                )}

                {/* Attached Volumes */}
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-1.5">
                    <HardDrive className="h-3.5 w-3.5 text-text-muted" />
                    {t("dash.launch.volumes_label")}
                    {selectedVolumeIds.length > 0 && (
                      <span className="ml-auto text-[10px] font-mono text-ice-blue">
                        {selectedVolumeIds.length} selected
                      </span>
                    )}
                  </Label>
                  {volumeRegionMismatch && (
                    <div className="flex items-start gap-2 rounded-lg border border-accent-gold/30 bg-accent-gold/5 px-3 py-2">
                      <AlertTriangle className="h-3.5 w-3.5 text-accent-gold shrink-0 mt-0.5" />
                      <p className="text-[11px] text-text-secondary leading-snug">
                        {t("dash.launch.volumes_region_mismatch")}
                      </p>
                    </div>
                  )}
                  {availableVolumes.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-border/60 p-4 text-center">
                      <HardDrive className="h-5 w-5 text-text-muted mx-auto mb-1.5 opacity-40" />
                      <p className="text-xs text-text-muted">{t("dash.launch.volumes_empty")}</p>
                      <p className="text-[10px] text-text-muted/60 mt-0.5">{t("dash.launch.volumes_empty_hint")}</p>
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
                                {vol.monthly_cost_cad != null && (
                                  <span className="text-[10px] text-text-muted font-mono">${vol.monthly_cost_cad.toFixed(2)}/month</span>
                                )}
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

                {/* Template image */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-text-secondary">Template Image <span className="text-text-muted">(optional)</span></label>
                  <input
                    type="text"
                    className="w-full rounded-md border border-border/60 bg-surface-hover/40 px-3 py-2 text-sm placeholder:text-text-muted focus:border-ice-blue/50 focus:outline-none focus:ring-1 focus:ring-ice-blue/30"
                    placeholder="Image ID or leave blank to use standard image"
                    value={templateImageId}
                    onChange={(e) => setTemplateImageId(e.target.value)}
                  />
                  <p className="text-[10px] text-text-muted/60">Paste a template image ID from your Templates page to launch with a saved environment.</p>
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

                {priceSummary}

                <Button
                  className="w-full"
                  onClick={() => setStep("confirm")}
                  disabled={launchBlocked || !instanceName.trim() || !resolvedImage || gpuLoading || !selectedTierPrice || !wholeHostReady || !addonReady || !volumePricingKnown || !bidReady}
                >
                  {gpuLoading ? <>Loading GPU inventory…</> : "Continue"}
                </Button>
              </>
            )}

            {/* ─── Step: Confirm ─── */}
            {step === "confirm" && (
              <>
                {/* P5: the price/reliability trade-off, stated *before* launch.
                    A refusal renders as a refusal here — the gate is explicit
                    that an unsatisfiable preference must not quietly fall back
                    to the cheapest host.

                    **Currently a preview, and it says so.** No route accepts a
                    placement preference at launch, no column stores one, and
                    `scheduler.py` ranks with `host_efficiency_score` without
                    ever consulting one. Shipping this as a silent no-op would be
                    the exact trust failure the gate names — a user states
                    "verified only", sees "satisfiable", and lands on the
                    cheapest host. `notApplied` makes that visible until the
                    preference is carried end to end. */}
                <PlacementPreferenceControl
                  notApplied
                  spec={{
                    gpu_model: resolvedGpu || undefined,
                    num_gpus: Number(numGpus) || 1,
                    vram_gb: typeof vramGb === "number" ? vramGb : undefined,
                  }}
                />
                <div className="rounded-lg border border-accent-gold/30 bg-accent-gold/5 p-4">
                  <p className="text-sm font-medium text-accent-gold mb-2">Confirm Launch</p>
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between"><span className="text-text-muted">GPU</span><span>{resolvedGpu || (gpuModelOptions[0]?.gpu_model ? `Best Available (${gpuModelOptions[0].gpu_model})` : "Best Available")}</span></div>
                    {Number(numGpus) > 1 && (
                      <div className="flex justify-between"><span className="text-text-muted">GPUs</span><span>×{numGpus}</span></div>
                    )}
                    <div className="flex justify-between"><span className="text-text-muted">Instance</span><span>{instanceName}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Image</span><span className="text-right" title={resolvedImage}>{resolvedImage}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Pricing</span><span className="capitalize">{effectivePricingMode === "on_demand" ? "On-Demand" : "Spot"}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Service tier</span><span>{selectedTierPrice?.label}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Queue priority</span><span>{priorityLabel}</span></div>
                    {selectedVolumeIds.length > 0 && (
                      <div className="flex justify-between">
                        <span className="text-text-muted">Volumes</span>
                        <span className="text-right" title={selectedVolumeIds.map((id) => availableVolumes.find((v) => v.volume_id === id)?.name || id.slice(0, 8)).join(", ")}>
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
                      <span>{effectiveRate != null ? `${isEstimatedRate ? "~" : ""}$${effectiveRate.toFixed(2)}/hr${isEstimatedRate ? " est." : ""}` : "-"}</span>
                    </div>
                  </div>
                </div>
                {priceSummary}
                <p className="text-xs text-text-muted">
                  Your wallet will be charged based on actual usage. You can stop the instance at any time.
                </p>
                {effectivePricingMode === "spot" && effectiveRate != null && (
                  <p className="text-xs text-amber-200/90 rounded-md border border-amber-500/20 bg-amber-500/5 px-3 py-2">
                    Spot rate of ${effectiveRate.toFixed(2)}/hr is locked at launch for billing.
                    Future launches may reflect updated spot prices.
                  </p>
                )}

                {launchError && (
                  <div className="rounded-lg border border-accent-red/30 bg-accent-red/10 p-3 flex items-start gap-3">
                    <AlertTriangle className="h-4 w-4 text-accent-red mt-0.5 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-accent-red font-medium">{launchError.message}</p>
                      {launchError.action && (
                        <Button
                          size="sm"
                          className="mt-2"
                          onClick={() => {
                            const href = launchError.action!.href;
                            onClose();
                            router.push(href);
                          }}
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
                  <Button className="flex-1" type="button" onClick={handleSubmit} disabled={submitting || launchBlocked}>
                    {submitting ? <><Loader2 className="h-4 w-4 animate-spin" /> Launching…</> : <>
                      <Rocket className="h-4 w-4" /> Launch Instance
                    </>}
                  </Button>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>,
    document.body,
  );
}
