"use client";

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import {
  X, Cpu, MapPin, Zap, DollarSign, Loader2, CheckCircle, TrendingDown, Activity,
  AlertTriangle, CreditCard, Box, RefreshCw,
} from "lucide-react";
import * as api from "@/lib/api";
import type { MarketplaceListing, LaunchErrorInfo, ImageTemplate } from "@/lib/api";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { cn, generateFunName } from "@/lib/utils";

interface RentModalProps {
  listing: MarketplaceListing;
  onClose: () => void;
}

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

const FALLBACK_TEMPLATES: { id: string; label: string; image: string; icon: string }[] = [
  { id: "pytorch", label: "PyTorch", image: "nvcr.io/nvidia/pytorch:24.12-py3", icon: "🔥" },
  { id: "tensorflow", label: "TensorFlow", image: "nvcr.io/nvidia/tensorflow:24.12-tf2-py3", icon: "🧠" },
  { id: "vllm", label: "vLLM", image: "vllm/vllm-openai:v0.6.6.post1", icon: "⚡" },
  { id: "comfyui", label: "ComfyUI", image: "runpod/comfyui:1.3.0-cuda12.8", icon: "🎨" },
  { id: "jupyter", label: "Jupyter Lab", image: "quay.io/jupyter/pytorch-notebook:cuda12-latest", icon: "📓" },
  { id: "ubuntu", label: "Ubuntu + CUDA", image: "nvidia/cuda:12.4.1-devel-ubuntu22.04", icon: "🐧" },
];

export function RentModal({ listing, onClose }: RentModalProps) {
  const router = useRouter();
  const [step, setStep] = useState<"configure" | "confirm" | "success">("configure");
  const [pricingMode, setPricingMode] = useState<"on_demand" | "spot">("on_demand");
  const [maxBid, setMaxBid] = useState("");
  const [tier, setTier] = useState("standard");
  const [instanceName, setInstanceName] = useState(() => generateFunName());
  const [submitting, setSubmitting] = useState(false);
  const [instanceId, setInstanceId] = useState("");
  const [spotPrice, setSpotPrice] = useState<number | null>(null);
  const [launchError, setLaunchError] = useState<LaunchErrorInfo | null>(null);
  const [templates, setTemplates] = useState(FALLBACK_TEMPLATES);
  const [selectedImage, setSelectedImage] = useState("");
  const [customImage, setCustomImage] = useState("");
  const [isCustom, setIsCustom] = useState(false);

  const resolvedImage = isCustom ? customImage.trim() : selectedImage;

  const pricePerHour = listing.price_per_hour_cad || listing.price_per_hour || 0;
  const spotRate = spotPrice ?? pricePerHour * 0.7;
  const effectiveRate = pricingMode === "spot" ? spotRate : pricePerHour;

  // Fetch current spot price for this GPU model
  useEffect(() => {
    api.fetchSpotPrices()
      .then((res) => {
        const prices = res.spot_prices || res.prices || {};
        const gpu = listing.gpu_model;
        if (prices[gpu] != null) setSpotPrice(prices[gpu]);
      })
      .catch((e) => console.error("Failed to fetch spot prices", e));
    api.fetchImageTemplates()
      .then((r) => {
        if (r.templates?.length) {
          setTemplates(
            r.templates.map((t: ImageTemplate) => ({
              id: t.id,
              label: t.label,
              image: t.image,
              icon: t.icon,
            })),
          );
        }
      })
      .catch((e) => console.error("Failed to load templates", e));
  }, [listing.gpu_model]);

  const handleSubmit = async () => {
    if (!instanceName.trim()) { toast.error("Enter an instance name"); return; }
    if (!resolvedImage) { toast.error("Select a Docker image"); return; }
    setSubmitting(true);
    try {
      const params: api.LaunchInstanceParams = {
        name: instanceName.trim(),
        image: resolvedImage,

        tier,
        host_id: listing.host_id,
        gpu_model: listing.gpu_model,
      };
      if (pricingMode === "spot") {
        const bid = maxBid ? Number(maxBid) : spotRate;
        if (bid <= 0) { toast.error("Enter a valid max bid"); setSubmitting(false); return; }
        params.max_bid = bid;
      }
      const res = await api.launchInstance(params);
      setInstanceId(res.instance?.job_id || "");
      setStep("success");
      toast.success("Instance launched successfully");
    } catch (err) {
      const info = api.classifyLaunchError(err);
      if (info.action) {
        setLaunchError(info);
      } else {
        toast.error(info.message);
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="mx-4 w-full max-w-lg max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <Card className="border-border/50">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Launch Instance</CardTitle>
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
                  <div className="text-right">
                    <p className="text-sm font-bold font-mono">${pricePerHour.toFixed(2)}<span className="text-xs font-normal text-text-muted">/hr</span></p>
                    {spotPrice != null && (
                      <p className="text-xs text-emerald font-mono">${spotPrice.toFixed(2)}<span className="font-normal text-text-muted">/hr spot</span></p>
                    )}
                  </div>
                </div>

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
                      className="flex items-center justify-center h-10 w-10 rounded-lg border border-border bg-surface-hover/50 text-text-muted hover:text-ice-blue hover:border-ice-blue/30 transition-colors shrink-0"
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                {/* Docker image picker */}
                <div className="space-y-1.5">
                  <Label className="text-xs flex items-center gap-1.5">
                    <Box className="h-3.5 w-3.5" />
                    Docker Image
                  </Label>
                  <div className="grid grid-cols-3 gap-2">
                    {templates.map((tpl) => (
                      <button
                        key={tpl.id}
                        type="button"
                        onClick={() => { setSelectedImage(tpl.image); setIsCustom(false); }}
                        className={cn(
                          "flex flex-col items-center gap-1 rounded-lg border p-2 text-xs transition-colors",
                          !isCustom && selectedImage === tpl.image
                            ? "border-ice-blue bg-ice-blue/5 text-ice-blue"
                            : "border-border text-text-secondary hover:border-text-muted hover:text-text-primary"
                        )}
                      >
                        <span className="text-base">{tpl.icon}</span>
                        <span className="font-medium">{tpl.label}</span>
                      </button>
                    ))}
                    <button
                      type="button"
                      onClick={() => { setIsCustom(true); setSelectedImage(""); }}
                      className={cn(
                        "flex flex-col items-center gap-1 rounded-lg border p-2 text-xs transition-colors",
                        isCustom
                          ? "border-ice-blue bg-ice-blue/5 text-ice-blue"
                          : "border-border text-text-secondary hover:border-text-muted hover:text-text-primary"
                      )}
                    >
                      <span className="text-base">⚙️</span>
                      <span className="font-medium">Custom</span>
                    </button>
                  </div>
                  {isCustom && (
                    <Input
                      placeholder="docker.io/my-org/my-image:latest"
                      value={customImage}
                      onChange={(e) => setCustomImage(e.target.value)}
                      className="mt-1.5"
                    />
                  )}
                </div>

                {/* Pricing Mode */}
                <div className="space-y-1.5">
                  <Label className="text-xs">Pricing</Label>
                  <div className="grid grid-cols-2 gap-2">
                    {PRICING_MODES.map((mode) => {
                      const Icon = mode.icon;
                      return (
                        <button
                          key={mode.value}
                          onClick={() => setPricingMode(mode.value)}
                          className={`rounded-lg border p-3 text-left transition-colors ${
                            pricingMode === mode.value
                              ? "border-ice-blue bg-ice-blue/5"
                              : "border-border hover:border-text-muted"
                          }`}
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

                {/* Rate summary */}
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
                    {tier !== "standard" && " · tier premium applies"}
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

            {step === "confirm" && (
              <>
                <div className="rounded-lg border border-accent-gold/30 bg-accent-gold/5 p-4">
                  <p className="text-sm font-medium text-accent-gold mb-2">Confirm Launch</p>
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between"><span className="text-text-muted">GPU</span><span>{listing.gpu_model}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Instance</span><span>{instanceName}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Image</span><span className="text-right max-w-[220px] truncate" title={resolvedImage}>{resolvedImage}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Pricing</span><span className="capitalize">{pricingMode === "on_demand" ? "On-Demand" : "Spot"}</span></div>
                    <div className="flex justify-between"><span className="text-text-muted">Tier</span><span className="capitalize">{tier}</span></div>
                    <div className="flex justify-between font-medium text-sm pt-1 border-t border-accent-gold/20">
                      <span>Rate</span>
                      <span>${effectiveRate.toFixed(2)}/hr CAD</span>
                    </div>
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
                  <Button variant="outline" className="flex-1" onClick={() => { setLaunchError(null); setStep("configure"); }}>
                    Back
                  </Button>
                  <Button className="flex-1" onClick={handleSubmit} disabled={submitting}>
                    {submitting ? <><Loader2 className="h-4 w-4 animate-spin" /> Launching…</> : "Launch Instance"}
                  </Button>
                </div>
              </>
            )}

            {step === "success" && (
              <div className="text-center py-4">
                <CheckCircle className="mx-auto h-12 w-12 text-emerald mb-3" />
                <h3 className="text-lg font-semibold mb-1">Instance Launched!</h3>
                <p className="text-sm text-text-secondary mb-4">
                  Your instance is being provisioned. Billing begins when it starts running.
                </p>
                <div className="flex gap-2">
                  <Button variant="outline" className="flex-1" onClick={onClose}>
                    Close
                  </Button>
                  <Button className="flex-1" onClick={() => router.push(`/dashboard/instances/${instanceId}`)}>
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
