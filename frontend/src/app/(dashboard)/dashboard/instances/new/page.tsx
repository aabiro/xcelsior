"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label, Select } from "@/components/ui/input";
import { ArrowLeft, Rocket, DollarSign, Box } from "lucide-react";
import { submitInstance, fetchPricingReference, fetchProvinces } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import type { PricingReference } from "@/lib/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

/* ── Past-values history (for native datalist suggestions) ───── */
const LS_KEY = "xcelsior:instance-history";
const MAX_HISTORY = 15;

function getHistory(): Record<string, string[]> {
  if (typeof window === "undefined") return {};
  try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); }
  catch { return {}; }
}

function saveHistory(vals: Record<string, string>) {
  try {
    const hist = getHistory();
    for (const [k, v] of Object.entries(vals)) {
      if (!v.trim()) continue;
      const arr = hist[k] || [];
      const idx = arr.indexOf(v);
      if (idx !== -1) arr.splice(idx, 1);
      arr.unshift(v);
      if (arr.length > MAX_HISTORY) arr.length = MAX_HISTORY;
      hist[k] = arr;
    }
    localStorage.setItem(LS_KEY, JSON.stringify(hist));
  } catch { /* ignore quota errors */ }
}

const TIERS = ["on-demand", "spot", "reserved"] as const;
const PRIORITIES = [
  { label: "Low", value: 0 },
  { label: "Normal", value: 1 },
  { label: "High", value: 2 },
  { label: "Urgent", value: 3 },
] as const;

const TEMPLATES = [
  { id: "pytorch", label: "PyTorch", image: "nvcr.io/nvidia/pytorch:24.01-py3", vram: "24", icon: "🔥" },
  { id: "tensorflow", label: "TensorFlow", image: "nvcr.io/nvidia/tensorflow:24.01-tf2-py3", vram: "24", icon: "🧠" },
  { id: "vllm", label: "vLLM", image: "vllm/vllm-openai:latest", vram: "24", icon: "⚡" },
  { id: "comfyui", label: "ComfyUI", image: "comfyanonymous/comfyui:latest", vram: "12", icon: "🎨" },
  { id: "jupyter", label: "Jupyter Lab", image: "quay.io/jupyter/pytorch-notebook:cuda12-latest", vram: "8", icon: "📓" },
  { id: "ubuntu", label: "Ubuntu + CUDA", image: "nvidia/cuda:12.4.1-devel-ubuntu22.04", vram: "8", icon: "🐧" },
] as const;

export default function NewInstancePage() {
  const router = useRouter();
  const { t } = useLocale();
  const [submitting, setSubmitting] = useState(false);

  // Form state
  const [name, setName] = useState("");
  const [image, setImage] = useState("");
  const [gpuModel, setGpuModel] = useState("");
  const [vramNeeded, setVramNeeded] = useState("24");
  const [numGpus, setNumGpus] = useState("1");
  const [tier, setTier] = useState<(typeof TIERS)[number]>("on-demand");
  const [priority, setPriority] = useState(1);
  const [durationHrs, setDurationHrs] = useState("1");
  const [province, setProvince] = useState("");
  const [nfsMount, setNfsMount] = useState("");

  // Past values for datalist dropdowns
  const [pastValues, setPastValues] = useState<Record<string, string[]>>({});

  // Reference data
  const [pricing, setPricing] = useState<PricingReference[]>([]);
  const [provinces, setProvinces] = useState<Record<string, { tax_rate: number; description: string }>>({});

  useEffect(() => {
    setPastValues(getHistory());
    fetchPricingReference()
      .then((r) => setPricing(r.reference || []))
      .catch((e) => console.error("Failed to load pricing", e));
    fetchProvinces()
      .then((r) => setProvinces(r.provinces || {}))
      .catch((e) => console.error("Failed to load provinces", e));
  }, []);

  // GPU model options from pricing reference
  const gpuModels = pricing.map((p) => p.gpu_model);

  // Cost estimator
  const selectedPricing = pricing.find((p) => p.gpu_model === gpuModel);
  const hourlyRate = selectedPricing
    ? tier === "spot"
      ? (selectedPricing.spot_cad ?? selectedPricing.on_demand_cad * 0.6)
      : tier === "reserved"
        ? (selectedPricing.reserved_1mo_cad ?? selectedPricing.on_demand_cad * 0.8)
        : selectedPricing.on_demand_cad
    : 0;
  const estimatedCost = hourlyRate * Number(durationHrs) * Number(numGpus);
  const taxRate = province && provinces[province] ? provinces[province].tax_rate : 0;
  const totalWithTax = estimatedCost * (1 + taxRate);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const instanceName = name.trim() || `instance-${Date.now().toString(36)}`;
    setSubmitting(true);
    try {
      const res = await submitInstance({
        name: instanceName,
        vram_needed_gb: Number(vramNeeded) || 24,
        num_gpus: Number(numGpus),
        priority,
        tier,
        image: image || undefined,
        nfs_path: nfsMount || undefined,
      });
      // Save form values for future datalist suggestions
      saveHistory({
        name: instanceName,
        image,
        gpuModel,
        vramNeeded,
        numGpus,
        durationHrs,
        nfsMount,
      });
      toast.success("Instance submitted");
      const jobId = (res as { instance?: { job_id?: string } })?.instance?.job_id;
      router.push(jobId ? `/dashboard/instances/${jobId}` : "/dashboard/instances");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to submit instance");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Link href="/dashboard/instances">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div>
          <h1 className="text-2xl font-bold">{t("dash.newinstance.title")}</h1>
          <p className="text-sm text-text-secondary">
            {t("dash.newinstance.subtitle")}
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} autoComplete="on" className="space-y-6">
        {/* Template Selector */}
        <Card className="space-y-4">
          <div className="flex items-center gap-2">
            <Box className="h-4 w-4 text-ice-blue" />
            <h2 className="text-lg font-semibold">Templates</h2>
          </div>
          <p className="text-sm text-text-secondary">Quick-start with a pre-configured environment, or choose custom.</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <button
              type="button"
              onClick={() => { setImage(""); setVramNeeded("24"); }}
              className={cn(
                "flex flex-col items-center gap-1.5 rounded-lg border p-3 text-sm transition-colors",
                !TEMPLATES.some((t) => t.image === image)
                  ? "border-accent-red bg-accent-red/10 text-accent-red"
                  : "border-border text-text-secondary hover:border-text-muted hover:text-text-primary"
              )}
            >
              <span className="text-lg">⚙️</span>
              <span className="font-medium">Custom</span>
            </button>
            {TEMPLATES.map((tpl) => (
              <button
                key={tpl.id}
                type="button"
                onClick={() => { setImage(tpl.image); setVramNeeded(tpl.vram); }}
                className={cn(
                  "flex flex-col items-center gap-1.5 rounded-lg border p-3 text-sm transition-colors",
                  image === tpl.image
                    ? "border-accent-red bg-accent-red/10 text-accent-red"
                    : "border-border text-text-secondary hover:border-text-muted hover:text-text-primary"
                )}
              >
                <span className="text-lg">{tpl.icon}</span>
                <span className="font-medium">{tpl.label}</span>
              </button>
            ))}
          </div>
        </Card>

        {/* Basic Info */}
        <Card className="space-y-4">
          <h2 className="text-lg font-semibold">{t("dash.newinstance.config")}</h2>

          <div className="space-y-2">
            <Label htmlFor="name">{t("dash.newinstance.name")}</Label>
            <Input
              id="name"
              name="instance-name"
              autoComplete="on"
              list="past-names"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={t("dash.newinstance.name_placeholder")}
            />
            <datalist id="past-names">
              {(pastValues.name || []).map((v) => <option key={v} value={v} />)}
            </datalist>
          </div>

          <div className="space-y-2">
            <Label htmlFor="image">{t("dash.newinstance.docker")}</Label>
            <Input
              id="image"
              name="docker-image"
              autoComplete="on"
              list="past-images"
              value={image}
              onChange={(e) => setImage(e.target.value)}
              placeholder={t("dash.newinstance.docker_placeholder")}
            />
            <datalist id="past-images">
              {(pastValues.image || []).map((v) => <option key={v} value={v} />)}
            </datalist>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="gpu">{t("dash.newinstance.gpu_model")}</Label>
              <Select id="gpu" name="gpu-model" autoComplete="on" value={gpuModel} onChange={(e) => setGpuModel(e.target.value)}>
                <option value="">{t("dash.newinstance.gpu_auto")}</option>
                {gpuModels.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="vram">{t("dash.newinstance.vram")}</Label>
              <Input
                id="vram"
                name="vram-gb"
                autoComplete="on"
                list="past-vram"
                type="number"
                min="1"
                value={vramNeeded}
                onChange={(e) => setVramNeeded(e.target.value)}
              />
              <datalist id="past-vram">
                {(pastValues.vramNeeded || []).map((v) => <option key={v} value={v} />)}
              </datalist>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="numGpus">{t("dash.newinstance.gpu_count")}</Label>
              <Input
                id="numGpus"
                name="num-gpus"
                autoComplete="on"
                list="past-numgpus"
                type="number"
                min="1"
                max="8"
                value={numGpus}
                onChange={(e) => setNumGpus(e.target.value)}
              />
              <datalist id="past-numgpus">
                {(pastValues.numGpus || []).map((v) => <option key={v} value={v} />)}
              </datalist>
            </div>

            <div className="space-y-2">
              <Label htmlFor="tier">{t("dash.newinstance.pricing_tier")}</Label>
              <Select id="tier" name="pricing-tier" autoComplete="on" value={tier} onChange={(e) => setTier(e.target.value as typeof tier)}>
                {TIERS.map((t) => (
                  <option key={t} value={t}>{t.replace("-", " ").replace(/\b\w/g, (c) => c.toUpperCase())}</option>
                ))}
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="priority">{t("dash.newinstance.priority")}</Label>
              <Select id="priority" name="job-priority" autoComplete="on" value={priority} onChange={(e) => setPriority(Number(e.target.value))}>
                {PRIORITIES.map((p) => (
                  <option key={p.value} value={p.value}>{p.label}</option>
                ))}
              </Select>
            </div>
          </div>
        </Card>

        {/* Jurisdiction */}
        <Card className="space-y-4">
          <h2 className="text-lg font-semibold">{t("dash.newinstance.residency")}</h2>
          <p className="text-sm text-text-secondary">
            {t("dash.newinstance.residency_desc")}
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="province">{t("dash.newinstance.province")}</Label>
              <Select id="province" name="province" autoComplete="on" value={province} onChange={(e) => setProvince(e.target.value)}>
                <option value="">{t("dash.newinstance.province_auto")}</option>
                {Object.entries(provinces).map(([code, info]) => (
                  <option key={code} value={code}>{code} — {info.description}</option>
                ))}
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="duration">{t("dash.newinstance.duration")}</Label>
              <Input
                id="duration"
                name="duration-hours"
                autoComplete="on"
                list="past-duration"
                type="number"
                min="0.5"
                step="0.5"
                value={durationHrs}
                onChange={(e) => setDurationHrs(e.target.value)}
              />
              <datalist id="past-duration">
                {(pastValues.durationHrs || []).map((v) => <option key={v} value={v} />)}
              </datalist>
            </div>
          </div>
        </Card>

        {/* Optional */}
        <Card className="space-y-4">
          <h2 className="text-lg font-semibold">{t("dash.newinstance.advanced")}</h2>

          <div className="space-y-2">
            <Label htmlFor="nfs">{t("dash.newinstance.nfs")}</Label>
            <Input
              id="nfs"
              name="nfs-mount"
              autoComplete="on"
              list="past-nfs"
              value={nfsMount}
              onChange={(e) => setNfsMount(e.target.value)}
              placeholder={t("dash.newinstance.nfs_placeholder")}
            />
            <datalist id="past-nfs">
              {(pastValues.nfsMount || []).map((v) => <option key={v} value={v} />)}
            </datalist>
            <p className="text-xs text-text-muted">
              {t("dash.newinstance.nfs_desc")}
            </p>
          </div>
        </Card>

        {/* Cost Estimator */}
        {gpuModel && (
          <Card className="border-ice-blue/30 bg-ice-blue/5">
            <div className="flex items-start gap-3">
              <DollarSign className="mt-0.5 h-5 w-5 text-ice-blue shrink-0" />
              <div className="space-y-1">
                <h3 className="font-semibold text-ice-blue">{t("dash.newinstance.estimated_cost")}</h3>
                <div className="text-sm text-text-secondary space-y-0.5">
                  <p>
                    {gpuModel} × {numGpus} GPU{Number(numGpus) > 1 ? "s" : ""} × {durationHrs}h @ ${hourlyRate.toFixed(2)}/hr
                  </p>
                  <p className="font-mono">
                    Subtotal: <span className="text-text-primary">${estimatedCost.toFixed(2)} CAD</span>
                  </p>
                  {taxRate > 0 && (
                    <p className="font-mono">
                      Tax ({(taxRate * 100).toFixed(1)}%): <span className="text-text-primary">${(estimatedCost * taxRate).toFixed(2)} CAD</span>
                    </p>
                  )}
                  <p className="font-mono text-base font-semibold text-text-primary">
                    Total: ${totalWithTax.toFixed(2)} CAD
                  </p>
                  {tier === "spot" && (
                    <p className="text-xs text-amber-400 mt-1">⚡ Spot instances are interruptible — your job may be preempted if demand increases.</p>
                  )}
                  {tier === "reserved" && (
                    <p className="text-xs text-emerald-400 mt-1">✓ Reserved pricing — 20% discount with commitment.</p>
                  )}
                </div>
              </div>
            </div>
          </Card>
        )}

        {/* Submit */}
        <div className="flex justify-end gap-3">
          <Link href="/dashboard/instances">
            <Button variant="outline" type="button">{t("dash.newinstance.cancel")}</Button>
          </Link>
          <Button type="submit" disabled={submitting}>
            <Rocket className="h-4 w-4" />
            {submitting ? "Submitting..." : t("dash.newinstance.submit")}
          </Button>
        </div>
      </form>
    </div>
  );
}
