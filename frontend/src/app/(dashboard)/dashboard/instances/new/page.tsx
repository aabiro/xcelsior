"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label, Select } from "@/components/ui/input";
import { ArrowLeft, Rocket, DollarSign } from "lucide-react";
import { submitInstance, fetchPricingReference, fetchProvinces } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import type { PricingReference } from "@/lib/api";
import { toast } from "sonner";

const TIERS = ["on-demand", "spot", "reserved"] as const;
const PRIORITIES = ["low", "normal", "high"] as const;

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
  const [priority, setPriority] = useState<(typeof PRIORITIES)[number]>("normal");
  const [durationHrs, setDurationHrs] = useState("1");
  const [province, setProvince] = useState("");
  const [nfsMount, setNfsMount] = useState("");

  // Reference data
  const [pricing, setPricing] = useState<PricingReference[]>([]);
  const [provinces, setProvinces] = useState<Record<string, { tax_rate: number; description: string }>>({});

  useEffect(() => {
    fetchPricingReference()
      .then((r) => setPricing(r.reference || []))
      .catch(() => {});
    fetchProvinces()
      .then((r) => setProvinces(r.provinces || {}))
      .catch(() => {});
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
    setSubmitting(true);
    try {
      const res = await submitInstance({
        name: name || undefined,
        vram_needed_gb: Number(vramNeeded),
        num_gpus: Number(numGpus),
        priority,
        tier,
        image: image || undefined,
        nfs_path: nfsMount || undefined,
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

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Basic Info */}
        <Card className="space-y-4">
          <h2 className="text-lg font-semibold">{t("dash.newinstance.config")}</h2>

          <div className="space-y-2">
            <Label htmlFor="name">{t("dash.newinstance.name")}</Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={t("dash.newinstance.name_placeholder")}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="image">{t("dash.newinstance.docker")}</Label>
            <Input
              id="image"
              value={image}
              onChange={(e) => setImage(e.target.value)}
              placeholder={t("dash.newinstance.docker_placeholder")}
            />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="gpu">{t("dash.newinstance.gpu_model")}</Label>
              <Select id="gpu" value={gpuModel} onChange={(e) => setGpuModel(e.target.value)}>
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
                type="number"
                min="1"
                value={vramNeeded}
                onChange={(e) => setVramNeeded(e.target.value)}
              />
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="numGpus">{t("dash.newinstance.gpu_count")}</Label>
              <Input
                id="numGpus"
                type="number"
                min="1"
                max="8"
                value={numGpus}
                onChange={(e) => setNumGpus(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="tier">{t("dash.newinstance.pricing_tier")}</Label>
              <Select id="tier" value={tier} onChange={(e) => setTier(e.target.value as typeof tier)}>
                {TIERS.map((t) => (
                  <option key={t} value={t}>{t.replace("-", " ").replace(/\b\w/g, (c) => c.toUpperCase())}</option>
                ))}
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="priority">{t("dash.newinstance.priority")}</Label>
              <Select id="priority" value={priority} onChange={(e) => setPriority(e.target.value as typeof priority)}>
                {PRIORITIES.map((p) => (
                  <option key={p} value={p}>{p.charAt(0).toUpperCase() + p.slice(1)}</option>
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
              <Select id="province" value={province} onChange={(e) => setProvince(e.target.value)}>
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
                type="number"
                min="0.5"
                step="0.5"
                value={durationHrs}
                onChange={(e) => setDurationHrs(e.target.value)}
              />
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
              value={nfsMount}
              onChange={(e) => setNfsMount(e.target.value)}
              placeholder={t("dash.newinstance.nfs_placeholder")}
            />
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
