"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useModalBodyLock } from "@/hooks/useModalBodyLock";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Loader2, Rocket, X, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input, NumberInput, Select } from "@/components/ui/input";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { DEFAULT_FORM, PRESET_MODELS } from "@/features/serverless/constants";
import { deployServerlessEndpoint } from "@/features/serverless/deploy-actions";
import {
  EMPTY_DEPLOY_GPU_DATA,
  defaultDeployGpu,
  deployGpuOptions,
  deployRegionOptions,
  freeGpuRow,
  loadDeployGpuData,
  regionFor,
  type DeployGpuData,
} from "@/features/serverless/gpu-options";
import { DeployGpuEmpty, deployGpuLabel } from "@/features/serverless/gpu-picker";
import type { DeployStudioForm } from "@/features/serverless/types";
import { endpointHref } from "@/features/serverless/endpoint-href";
import { RegionPicker } from "@/features/serverless/region-picker";
import { PickerSelect } from "@/components/ui/picker-select";

interface DeployServerlessModalProps {
  open: boolean;
  onClose: () => void;
  canWrite: boolean;
}

/** Same field look as the rest of the dashboard, at 16px so iOS does not zoom on focus. */
const FIELD = "h-12 text-base";

export function DeployServerlessModal({ open, onClose, canWrite }: DeployServerlessModalProps) {
  const { t } = useLocale();
  const router = useRouter();
  const [form, setForm] = useState<DeployStudioForm>(DEFAULT_FORM);
  const [gpuData, setGpuData] = useState<DeployGpuData>(EMPTY_DEPLOY_GPU_DATA);
  const [loadingGpus, setLoadingGpus] = useState(false);
  const [deploying, setDeploying] = useState(false);
  const deployingRef = useRef(false);
  const loadGenRef = useRef(0);
  const panelRef = useRef<HTMLDivElement>(null);

  useModalBodyLock(open);

  // Only GPUs the fleet reports. No catalog stands in for an empty or failed list.
  const gpuOptions = useMemo(() => deployGpuOptions(gpuData), [gpuData]);
  const regions = useMemo(() => deployRegionOptions(gpuOptions, form.gpuTier), [gpuOptions, form.gpuTier]);
  const selectedOption = gpuOptions.find((option) => option.model === form.gpuTier);
  const selectedGpu = freeGpuRow(gpuData.gpus, form.gpuTier, form.region);

  const loadGpus = useCallback(async (generation: number) => {
    setLoadingGpus(true);
    const data = await loadDeployGpuData(api.fetchAvailableGPUs);
    if (generation !== loadGenRef.current) return;
    setGpuData(data);
    const options = deployGpuOptions(data);
    setForm((prev) => {
      const known = options.some((option) => option.model === prev.gpuTier);
      const gpuTier = known ? prev.gpuTier : defaultDeployGpu(options);
      return { ...prev, gpuTier, region: regionFor(options, gpuTier, prev.region) };
    });
    setLoadingGpus(false);
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
    const previous = document.activeElement as HTMLElement | null;
    panelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "Tab") {
        const elements = Array.from(panelRef.current?.querySelectorAll<HTMLElement>(
          'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), [tabindex="0"]',
        ) ?? []).filter(element => element.getClientRects().length > 0);
        const first = elements[0];
        const last = elements[elements.length - 1];
        if (e.shiftKey && (document.activeElement === first || document.activeElement === panelRef.current)) {
          e.preventDefault(); last?.focus();
        } else if (!e.shiftKey && (document.activeElement === last || document.activeElement === panelRef.current)) {
          e.preventDefault(); first?.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      if (previous?.isConnected) previous.focus();
    };
  }, [open, onClose]);

  const update = <K extends keyof DeployStudioForm>(key: K, value: DeployStudioForm[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const updateGpuTier = (gpuTier: string) => {
    setForm((prev) => ({ ...prev, gpuTier, region: regionFor(gpuOptions, gpuTier, prev.region) }));
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
      router.push(endpointHref(res.endpoint.endpoint_id));
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : t("dash.serverless.deploy_failed");
      toast.error(/team viewers cannot/i.test(msg) ? t("dash.serverless.viewer_blocked") : msg);
    } finally {
      deployingRef.current = false;
      setDeploying(false);
    }
  };

  const noGpus = !loadingGpus && gpuOptions.length === 0;

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[60] flex items-end justify-center bg-black/70 backdrop-blur-sm sm:items-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          {/* Opaque panel. Inside the dashboard shell `bg-surface` is a 2% tint, so
              the panel and its sticky header were see-through: the model list
              scrolled visibly under the title. `--popover-solid` is the solid
              colour every dialog uses. */}
          <motion.div
            ref={panelRef}
            role="dialog"
            tabIndex={-1}
            aria-modal="true"
            aria-labelledby="deploy-serverless-title"
            className="mx-0 flex w-full max-h-[92dvh] flex-col overflow-hidden rounded-t-3xl border border-border/80 bg-[var(--popover-solid)] shadow-2xl outline-none sm:mx-4 sm:max-w-lg sm:rounded-2xl"
            initial={{ y: 48, opacity: 0, scale: 0.98 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: 32, opacity: 0, scale: 0.98 }}
            transition={{ type: "spring", damping: 28, stiffness: 320 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex shrink-0 items-start justify-between gap-3 border-b border-border/60 bg-[var(--popover-solid)] px-5 py-4">
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

            {/* The body scrolls inside the panel, under an opaque header, and
                stops at the panel's edge (no scroll chaining to the page). */}
            <div className="min-h-0 flex-1 space-y-5 overflow-y-auto overscroll-contain px-5 pt-5 pb-[max(1.25rem,env(safe-area-inset-bottom))]">
              <div>
                <label htmlFor="deploy-serverless-name" className="mb-1.5 block text-sm font-medium">
                  {t("dash.serverless.endpoint_name")}
                </label>
                <Input
                  id="deploy-serverless-name"
                  value={form.name}
                  onChange={(e) => update("name", e.target.value)}
                  placeholder={t("dash.serverless.endpoint_name_ph")}
                  autoComplete="off"
                  className={FIELD}
                />
              </div>

              <div>
                <p className="mb-2 block text-sm font-medium">{t("dash.serverless.model_library")}</p>
                <div className="grid grid-cols-1 gap-2" role="radiogroup" aria-label={t("dash.serverless.model_library")}>
                  {PRESET_MODELS.map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      role="radio"
                      aria-checked={form.modelRef === m.id}
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

              {noGpus && (
                <DeployGpuEmpty failed={gpuData.failed} onRetry={() => void loadGpus(++loadGenRef.current)} />
              )}

              <div>
                <label id="deploy-serverless-gpu-label" htmlFor="deploy-serverless-gpu" className="mb-1.5 block text-sm font-medium">
                  {t("dash.serverless.gpu_type")}
                </label>
                <PickerSelect
                  id="deploy-serverless-gpu"
                  aria-labelledby="deploy-serverless-gpu-label"
                  value={form.gpuTier}
                  options={loadingGpus ? [] : gpuOptions.map((option) => ({ value: option.model, label: deployGpuLabel(option, t) }))}
                  onChange={updateGpuTier}
                  placeholder={loadingGpus
                    ? t("dash.serverless.gpus_loading")
                    : gpuOptions.length === 0
                      ? t("dash.serverless.no_gpus_short")
                      : t("dash.serverless.select_gpu")}
                  disabled={loadingGpus}
                  className={FIELD}
                  renderNative={(select) => <Select {...select} className={FIELD} />}
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <RegionPicker
                  regions={regions}
                  value={form.region}
                  onChange={(region) => update("region", region)}
                  labelClassName="mb-1.5 block text-sm font-medium"
                  fieldClassName={FIELD}
                  renderNative={(select) => <Select {...select} className={FIELD} />}
                  field={(label, control, key) => <div key={key} className="min-w-0">{label}{control}</div>}
                />
              </div>

              {selectedOption && selectedOption.available === 0 && (
                <p className="text-sm text-text-muted">
                  {t("dash.serverless.gpu_all_busy", { model: selectedOption.model })}
                </p>
              )}

              <div>
                <label htmlFor="deploy-serverless-max-workers" className="mb-1.5 block text-sm font-medium">
                  {t("dash.serverless.max_workers")}
                </label>
                <NumberInput
                  id="deploy-serverless-max-workers"
                  min={1}
                  max={16}
                  value={form.maxWorkers}
                  onChange={(v) => update("maxWorkers", v)}
                  className={FIELD}
                />
              </div>

              {selectedGpu && (
                <p className="rounded-lg border border-border/60 bg-surface-hover/60 px-3 py-2 text-xs text-text-muted">
                  {t("dash.serverless.est_cost")}{" "}
                  <span className="font-mono text-text-primary">
                    ${selectedGpu.price_per_hour_cad.toFixed(2)}/hr
                  </span>
                  {t("dash.serverless.per_worker")}
                </p>
              )}

              <div className="flex flex-col gap-2 pt-1">
                <Button
                  size="lg"
                  className="h-14 w-full gap-2 bg-accent-violet text-white hover:bg-accent-violet/90 shadow-[0_0_28px_rgba(139,92,246,0.35)]"
                  onClick={() => void handleDeploy()}
                  disabled={deploying || !canWrite || !form.gpuTier}
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
                  className="inline-flex min-h-11 items-center justify-center gap-1.5 py-2 text-center text-sm text-accent-cyan hover:underline"
                >
                  {t("dash.mobile.deploy_full_studio")}
                  <ArrowRight className="h-4 w-4" aria-hidden />
                </Link>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
