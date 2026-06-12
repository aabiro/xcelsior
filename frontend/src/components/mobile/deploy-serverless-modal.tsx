"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useModalBodyLock } from "@/hooks/useModalBodyLock";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { Loader2, Rocket, X, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input, NumberInput } from "@/components/ui/input";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { GpuAvailability } from "@/lib/api";
import { GPU_MODELS } from "@/lib/gpu-models";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { DEFAULT_FORM, PRESET_MODELS } from "@/features/serverless/constants";
import { deployServerlessEndpoint } from "@/features/serverless/deploy-actions";
import { findGpuInRegion, regionOptionsForGpus } from "@/features/serverless/region-options";
import type { DeployStudioForm } from "@/features/serverless/types";

interface DeployServerlessModalProps {
  open: boolean;
  onClose: () => void;
  canWrite: boolean;
}

export function DeployServerlessModal({ open, onClose, canWrite }: DeployServerlessModalProps) {
  const { t } = useLocale();
  const router = useRouter();
  const [form, setForm] = useState<DeployStudioForm>(DEFAULT_FORM);
  const [gpus, setGpus] = useState<GpuAvailability[]>([]);
  const [loadingGpus, setLoadingGpus] = useState(false);
  const [deploying, setDeploying] = useState(false);
  const deployingRef = useRef(false);
  const loadGenRef = useRef(0);

  useModalBodyLock(open);

  const gpuTypes = useMemo(() => {
    const fromMarket = [...new Set(gpus.map((g) => g.gpu_model))];
    if (fromMarket.length > 0) return fromMarket;
    return GPU_MODELS.slice(0, 8).map((g) => g.value);
  }, [gpus]);

  const regions = useMemo(() => regionOptionsForGpus(gpus, form.gpuTier), [gpus, form.gpuTier]);

  const selectedGpu = findGpuInRegion(gpus, form.gpuTier, form.region);

  const loadGpus = useCallback(async (generation: number) => {
    setLoadingGpus(true);
    try {
      const res = await api.fetchAvailableGPUs();
      if (generation !== loadGenRef.current) return;
      setGpus(res.gpus || []);
      const first = res.gpus?.[0];
      if (first) {
        setForm((prev) => {
          const gpuTier = prev.gpuTier || first.gpu_model;
          const nextRegions = regionOptionsForGpus(res.gpus || [], gpuTier);
          const region = nextRegions.includes(prev.region) ? prev.region : nextRegions[0];
          return { ...prev, gpuTier, region };
        });
      }
    } catch {
      if (generation !== loadGenRef.current) return;
      setForm((prev) => ({
        ...prev,
        gpuTier: prev.gpuTier || GPU_MODELS[0]?.value || "",
        region: prev.region || "ca-east",
      }));
    } finally {
      if (generation === loadGenRef.current) setLoadingGpus(false);
    }
  }, []);

  useEffect(() => {
    if (!open) {
      loadGenRef.current += 1;
      return;
    }
    const generation = ++loadGenRef.current;
    setForm(DEFAULT_FORM);
    void loadGpus(generation);
  }, [open, loadGpus]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const update = <K extends keyof DeployStudioForm>(key: K, value: DeployStudioForm[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const updateGpuTier = (gpuTier: string) => {
    setForm((prev) => {
      const nextRegions = regionOptionsForGpus(gpus, gpuTier);
      const region = nextRegions.includes(prev.region) ? prev.region : nextRegions[0];
      return { ...prev, gpuTier, region };
    });
  };

  const handleDeploy = async () => {
    if (deployingRef.current) return;
    if (!canWrite) return toast.error(t("dash.serverless.viewer_blocked"));
    if (!form.modelRef.trim()) return toast.error(t("dash.serverless.err_model"));
    if (!form.gpuTier) return toast.error(t("dash.serverless.err_gpu"));

    deployingRef.current = true;
    setDeploying(true);
    try {
      const res = await deployServerlessEndpoint({
        ...form,
        method: "preset",
        managedEngine: "vllm",
      });
      toast.success(t("dash.serverless.deploy_success"));
      onClose();
      router.push(`/dashboard/inference/${res.endpoint.endpoint_id}`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : t("dash.serverless.deploy_failed");
      toast.error(/team viewers cannot/i.test(msg) ? t("dash.serverless.viewer_blocked") : msg);
    } finally {
      deployingRef.current = false;
      setDeploying(false);
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[60] flex items-end justify-center bg-black/65 backdrop-blur-sm sm:items-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby="deploy-serverless-title"
            className="mx-0 w-full max-h-[92vh] overflow-y-auto rounded-t-3xl border border-border/80 bg-surface shadow-2xl sm:mx-4 sm:max-w-lg sm:rounded-2xl"
            initial={{ y: 48, opacity: 0, scale: 0.98 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: 32, opacity: 0, scale: 0.98 }}
            transition={{ type: "spring", damping: 28, stiffness: 320 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 z-10 flex items-start justify-between gap-3 border-b border-border/60 bg-surface/95 px-5 py-4 backdrop-blur-md">
              <div>
                <div className="mb-1 flex items-center gap-2 text-accent-violet">
                  <Zap className="h-4 w-4" />
                  <span className="text-xs font-semibold uppercase tracking-wider">
                    {t("dash.mobile.deploy_badge")}
                  </span>
                </div>
                <h2 id="deploy-serverless-title" className="text-lg font-bold">
                  {t("dash.mobile.deploy_modal_title")}
                </h2>
                <p className="mt-0.5 text-sm text-text-muted">{t("dash.mobile.deploy_modal_desc")}</p>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="rounded-full p-2 text-text-muted hover:bg-surface-hover hover:text-text-primary"
                aria-label={t("dash.mobile.deploy_close")}
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-5 px-5 py-5">
              <div>
                <label className="mb-1.5 block text-sm font-medium">{t("dash.serverless.endpoint_name")}</label>
                <Input
                  value={form.name}
                  onChange={(e) => update("name", e.target.value)}
                  placeholder={t("dash.serverless.endpoint_name_ph")}
                  className="h-12 text-base"
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium">{t("dash.serverless.model_library")}</label>
                <div className="grid grid-cols-1 gap-2">
                  {PRESET_MODELS.map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => update("modelRef", m.id)}
                      className={cn(
                        "rounded-xl border px-3 py-3 text-left transition-all",
                        form.modelRef === m.id
                          ? "border-accent-violet/50 bg-accent-violet/10 shadow-[0_0_20px_rgba(139,92,246,0.15)]"
                          : "border-border hover:border-accent-violet/30 hover:bg-surface-hover",
                      )}
                    >
                      <span className="font-medium text-sm">{m.label}</span>
                      <span className="mt-0.5 block text-xs text-text-muted">{m.id}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-1.5 block text-sm font-medium">{t("dash.serverless.gpu_type")}</label>
                  <select
                    value={form.gpuTier}
                    onChange={(e) => updateGpuTier(e.target.value)}
                    disabled={loadingGpus}
                    className="h-12 w-full rounded-lg border border-border bg-background px-3 text-base"
                  >
                    <option value="">{t("dash.serverless.select_gpu")}</option>
                    {gpuTypes.map((g) => (
                      <option key={g} value={g}>{g}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="mb-1.5 block text-sm font-medium">{t("dash.serverless.region")}</label>
                  <select
                    value={form.region}
                    onChange={(e) => update("region", e.target.value)}
                    className="h-12 w-full rounded-lg border border-border bg-background px-3 text-base"
                  >
                    {regions.map((r) => (
                      <option key={r} value={r}>{r}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <label className="mb-1.5 block text-sm font-medium">{t("dash.serverless.max_workers")}</label>
                <NumberInput
                  min={1}
                  max={16}
                  value={form.maxWorkers}
                  onChange={(v) => update("maxWorkers", v)}
                  className="h-12 text-base"
                />
              </div>

              {selectedGpu && (
                <p className="rounded-lg border border-border/60 bg-surface-hover/60 px-3 py-2 text-xs text-text-muted">
                  {t("dash.serverless.est_cost")}{" "}
                  <span className="font-mono text-text-primary">
                    ${selectedGpu.price_per_hour_cad.toFixed(2)} CAD/hr
                  </span>
                  {t("dash.serverless.per_worker")}
                </p>
              )}

              <div className="flex flex-col gap-2 pt-1">
                <Button
                  size="lg"
                  className="h-14 w-full gap-2 bg-accent-violet text-white hover:bg-accent-violet/90 shadow-[0_0_28px_rgba(139,92,246,0.35)]"
                  onClick={() => void handleDeploy()}
                  disabled={deploying || !canWrite}
                >
                  {deploying ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Rocket className="h-5 w-5" />
                  )}
                  {deploying ? t("dash.serverless.deploying") : t("dash.serverless.deploy")}
                </Button>
                <Link
                  href="/dashboard/inference/new"
                  onClick={onClose}
                  className="text-center text-sm text-accent-cyan hover:underline"
                >
                  {t("dash.mobile.deploy_full_studio")}
                </Link>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
