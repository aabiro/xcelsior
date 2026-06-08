"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Sparkles, Box, Cpu, Layers, Settings2, Rocket, ChevronLeft, ChevronRight,
  Loader2, Zap, Globe, Timer, Gauge,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input, NumberInput } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { GpuAvailability } from "@/lib/api";
import { GPU_MODELS } from "@/lib/gpu-models";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import {
  DEFAULT_FORM, DEPLOY_STUDIO_STEPS, IDLE_TIMEOUT_OPTIONS, PRESET_IMAGE, PRESET_MODELS,
} from "./constants";
import type { DeployStudioForm } from "./types";

const STEP_ICONS = [Sparkles, Box, Cpu, Layers, Settings2, Rocket];

interface DeployStudioProps {
  gpus: GpuAvailability[];
  canWrite: boolean;
}

function envToRecord(rows: DeployStudioForm["envRows"]): Record<string, string> | undefined {
  const out: Record<string, string> = {};
  for (const row of rows) {
    const k = row.key.trim();
    if (k) out[k] = row.value;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

export function DeployStudio({ gpus, canWrite }: DeployStudioProps) {
  const { t } = useLocale();
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [form, setForm] = useState<DeployStudioForm>(DEFAULT_FORM);
  const [deploying, setDeploying] = useState(false);

  const gpuTypes = useMemo(() => {
    const fromMarket = [...new Set(gpus.map((g) => g.gpu_model))];
    if (fromMarket.length > 0) return fromMarket;
    return GPU_MODELS.slice(0, 12).map((g) => g.value);
  }, [gpus]);

  const regions = useMemo(() => {
    const r = [...new Set(gpus.map((g) => g.region))];
    return r.length > 0 ? r : ["ca-east"];
  }, [gpus]);

  const selectedGpu = gpus.find((g) => g.gpu_model === form.gpuTier && g.region === form.region);
  const costPerHour = selectedGpu?.price_per_hour_cad ?? (form.gpuTier ? 2.5 : 0);

  const update = <K extends keyof DeployStudioForm>(key: K, value: DeployStudioForm[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const validateStep = (idx: number): string | null => {
    if (idx === 0) return null;
    if (idx === 1) {
      if (form.method === "preset" && !form.modelRef.trim()) return t("dash.serverless.err_model");
      if (form.method === "custom" && !form.imageRef.trim()) return t("dash.serverless.err_image");
      return null;
    }
    if (idx === 2) {
      if (!form.gpuTier) return t("dash.serverless.err_gpu");
      return null;
    }
    if (idx === 3) {
      if (form.maxWorkers < form.minWorkers) return t("dash.serverless.err_workers");
      return null;
    }
    return null;
  };

  const next = () => {
    const err = validateStep(step);
    if (err) return toast.error(err);
    setStep((s) => Math.min(s + 1, DEPLOY_STUDIO_STEPS.length - 1));
  };

  const back = () => setStep((s) => Math.max(s - 1, 0));

  const handleDeploy = async () => {
    if (!canWrite) return toast.error(t("dash.serverless.viewer_blocked"));
    for (let i = 1; i < DEPLOY_STUDIO_STEPS.length; i++) {
      const err = validateStep(i);
      if (err) return toast.error(err);
    }
    setDeploying(true);
    try {
      const image = form.method === "preset" ? PRESET_IMAGE : form.imageRef.trim();
      const res = await api.createServerlessEndpoint({
        name: form.name.trim() || form.modelRef || form.imageRef,
        mode: form.method === "preset" ? "preset" : "custom",
        model_name: form.method === "preset" ? form.modelRef.trim() : undefined,
        model_ref: form.method === "preset" ? form.modelRef.trim() : undefined,
        gpu_type: form.gpuTier,
        gpu_tier: form.gpuTier,
        gpu_count: form.gpuCount,
        region: form.region,
        docker_image: image,
        image_ref: image,
        min_workers: form.minWorkers,
        max_workers: form.maxWorkers,
        max_concurrency: form.maxConcurrency,
        idle_timeout_sec: form.idleTimeoutSec,
        scaling_policy_type: form.scalingPolicyType,
        scaling_policy_value: form.scalingPolicyValue,
        startup_command: form.method === "custom" ? form.startupCommand : undefined,
        http_port: form.method === "custom" ? form.httpPort : undefined,
        health_check_path: form.healthCheckPath,
        cuda_version: form.cudaVersion,
        env: envToRecord(form.envRows),
      });
      toast.success(t("dash.serverless.deploy_success"));
      router.push(`/dashboard/inference/${res.endpoint.endpoint_id}`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : t("dash.serverless.deploy_failed");
      toast.error(/team viewers cannot/i.test(msg) ? t("dash.serverless.viewer_blocked") : msg);
    } finally {
      setDeploying(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Hero header */}
      <div className="relative overflow-hidden rounded-2xl border border-accent-violet/20 bg-gradient-to-br from-accent-violet/10 via-surface to-accent-cyan/5 p-6 sm:p-8">
        <div className="absolute -right-16 -top-16 h-48 w-48 rounded-full bg-accent-violet/10 blur-3xl" />
        <div className="absolute -bottom-12 -left-12 h-40 w-40 rounded-full bg-accent-cyan/10 blur-3xl" />
        <div className="relative">
          <div className="flex items-center gap-2 mb-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent-violet/20">
              <Rocket className="h-5 w-5 text-accent-violet" />
            </div>
            <Badge variant="info" className="text-[10px] uppercase tracking-widest">
              {t("dash.serverless.studio_badge")}
            </Badge>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
            {t("dash.serverless.studio_title")}
          </h1>
          <p className="mt-2 text-sm text-text-muted max-w-2xl">
            {t("dash.serverless.studio_desc")}
          </p>
        </div>
      </div>

      {/* Step rail */}
      <div className="flex gap-1 overflow-x-auto pb-1 scrollbar-none">
        {DEPLOY_STUDIO_STEPS.map((s, i) => {
          const Icon = STEP_ICONS[i];
          const active = i === step;
          const done = i < step;
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => i < step && setStep(i)}
              className={cn(
                "flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium whitespace-nowrap transition-all shrink-0",
                active && "bg-accent-violet/15 text-accent-violet border border-accent-violet/30 shadow-[0_0_20px_rgba(139,92,246,0.15)]",
                done && !active && "text-text-secondary hover:bg-surface-hover cursor-pointer",
                !active && !done && "text-text-muted",
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">{t(s.labelKey)}</span>
              <span className="sm:hidden">{i + 1}</span>
            </button>
          );
        })}
      </div>

      {/* Step content */}
      <div className="glow-card brand-top-accent stat-glow-violet rounded-2xl border border-border bg-surface p-6 sm:p-8 min-h-[360px]">
        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, x: 16 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -16 }}
            transition={{ duration: 0.25 }}
            className="space-y-5"
          >
            {step === 0 && (
              <>
                <h2 className="text-lg font-semibold">{t("dash.serverless.method_title")}</h2>
                <p className="text-sm text-text-muted">{t("dash.serverless.method_desc")}</p>
                <div className="grid gap-4 sm:grid-cols-2">
                  {(["preset", "custom"] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => update("method", m)}
                      className={cn(
                        "group relative rounded-xl border p-5 text-left transition-all",
                        form.method === m
                          ? "border-accent-violet/50 bg-accent-violet/10 shadow-[0_0_24px_rgba(139,92,246,0.12)]"
                          : "border-border hover:border-accent-violet/30 hover:bg-surface-hover",
                      )}
                    >
                      <div className={cn(
                        "mb-3 flex h-10 w-10 items-center justify-center rounded-lg",
                        form.method === m ? "bg-accent-violet/20" : "bg-surface-hover",
                      )}>
                        {m === "preset" ? (
                          <Sparkles className={cn("h-5 w-5", form.method === m ? "text-accent-violet" : "text-text-muted")} />
                        ) : (
                          <Box className={cn("h-5 w-5", form.method === m ? "text-accent-violet" : "text-text-muted")} />
                        )}
                      </div>
                      <p className="font-semibold">
                        {m === "preset" ? t("dash.serverless.method_preset") : t("dash.serverless.method_custom")}
                      </p>
                      <p className="mt-1 text-xs text-text-muted">
                        {m === "preset" ? t("dash.serverless.method_preset_desc") : t("dash.serverless.method_custom_desc")}
                      </p>
                    </button>
                  ))}
                </div>
              </>
            )}

            {step === 1 && (
              <>
                <h2 className="text-lg font-semibold">{t("dash.serverless.source_title")}</h2>
                <div>
                  <label className="block text-sm font-medium mb-1">{t("dash.serverless.endpoint_name")}</label>
                  <Input
                    placeholder={t("dash.serverless.endpoint_name_ph")}
                    value={form.name}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => update("name", e.target.value)}
                  />
                </div>
                {form.method === "preset" ? (
                  <div className="space-y-3">
                    <label className="block text-sm font-medium">{t("dash.serverless.model_library")}</label>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {PRESET_MODELS.map((m) => (
                        <button
                          key={m.id}
                          type="button"
                          onClick={() => update("modelRef", m.id)}
                          className={cn(
                            "rounded-lg border px-3 py-2.5 text-left text-sm transition-all",
                            form.modelRef === m.id
                              ? "border-accent-cyan/50 bg-accent-cyan/10"
                              : "border-border hover:bg-surface-hover",
                          )}
                        >
                          <span className="font-medium">{m.label}</span>
                          <span className="block text-xs text-text-muted mt-0.5">{m.id}</span>
                        </button>
                      ))}
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">{t("dash.serverless.model_custom_id")}</label>
                      <Input
                        value={form.modelRef}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => update("modelRef", e.target.value)}
                        placeholder="org/model-name"
                      />
                    </div>
                  </div>
                ) : (
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="sm:col-span-2">
                      <label className="block text-sm font-medium mb-1">{t("dash.serverless.container_image")}</label>
                      <Input
                        value={form.imageRef}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => update("imageRef", e.target.value)}
                        placeholder="registry.io/your-image:tag"
                      />
                    </div>
                    <div className="sm:col-span-2">
                      <label className="block text-sm font-medium mb-1">{t("dash.serverless.start_command")}</label>
                      <Input
                        value={form.startupCommand}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => update("startupCommand", e.target.value)}
                        placeholder="python handler.py"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">{t("dash.serverless.http_port")}</label>
                      <NumberInput min={1} max={65535} value={form.httpPort} onChange={(v) => update("httpPort", v)} />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">{t("dash.serverless.health_path")}</label>
                      <Input
                        value={form.healthCheckPath}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => update("healthCheckPath", e.target.value)}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">{t("dash.serverless.cuda_version")}</label>
                      <Input
                        value={form.cudaVersion}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => update("cudaVersion", e.target.value)}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">{t("dash.serverless.registry_auth")}</label>
                      <Input
                        value={form.registryAuth}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => update("registryAuth", e.target.value)}
                        placeholder={t("dash.serverless.registry_auth_ph")}
                      />
                    </div>
                  </div>
                )}
              </>
            )}

            {step === 2 && (
              <>
                <h2 className="text-lg font-semibold">{t("dash.serverless.hardware_title")}</h2>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div>
                    <label className="block text-sm font-medium mb-1">{t("dash.serverless.gpu_type")}</label>
                    <select
                      value={form.gpuTier}
                      onChange={(e) => update("gpuTier", e.target.value)}
                      className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                    >
                      <option value="">{t("dash.serverless.select_gpu")}</option>
                      {gpuTypes.map((g) => {
                        const info = gpus.find((x) => x.gpu_model === g);
                        return (
                          <option key={g} value={g}>
                            {g}
                            {info ? ` — ${info.vram_gb}GB · $${info.price_per_hour_cad.toFixed(2)}/hr` : ""}
                          </option>
                        );
                      })}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">{t("dash.serverless.region")}</label>
                    <select
                      value={form.region}
                      onChange={(e) => update("region", e.target.value)}
                      className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                    >
                      {regions.map((r) => (
                        <option key={r} value={r}>{r}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">{t("dash.serverless.gpu_count")}</label>
                    <NumberInput min={1} max={8} value={form.gpuCount} onChange={(v) => update("gpuCount", v)} />
                  </div>
                </div>
                {form.gpuTier && (
                  <div className="rounded-xl border border-accent-cyan/20 bg-accent-cyan/5 p-4 flex items-center gap-3">
                    <Gauge className="h-5 w-5 text-accent-cyan shrink-0" />
                    <div className="text-sm">
                      <span className="text-text-muted">{t("dash.serverless.est_cost")}</span>{" "}
                      <span className="font-semibold font-mono">
                        ${(costPerHour * form.gpuCount).toFixed(2)} CAD/hr
                      </span>
                      <span className="text-text-muted"> {t("dash.serverless.per_worker")}</span>
                    </div>
                  </div>
                )}
              </>
            )}

            {step === 3 && (
              <>
                <h2 className="text-lg font-semibold">{t("dash.serverless.scaling_title")}</h2>
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">{t("dash.serverless.min_workers")}</label>
                    <NumberInput min={0} max={32} value={form.minWorkers} onChange={(v) => update("minWorkers", v)} />
                    <p className="text-xs text-text-muted mt-0.5">{t("dash.serverless.scale_zero_hint")}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">{t("dash.serverless.max_workers")}</label>
                    <NumberInput min={1} max={32} value={form.maxWorkers} onChange={(v) => update("maxWorkers", v)} />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">{t("dash.serverless.max_concurrency")}</label>
                    <NumberInput min={1} max={256} value={form.maxConcurrency} onChange={(v) => update("maxConcurrency", v)} />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">{t("dash.serverless.idle_timeout")}</label>
                    <select
                      value={form.idleTimeoutSec}
                      onChange={(e) => update("idleTimeoutSec", Number(e.target.value))}
                      className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                    >
                      {IDLE_TIMEOUT_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{t(o.labelKey)}</option>
                      ))}
                    </select>
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">{t("dash.serverless.scaling_policy")}</label>
                  <div className="flex flex-wrap gap-2 mb-3">
                    {(["queue_request_count", "queue_delay"] as const).map((p) => (
                      <button
                        key={p}
                        type="button"
                        onClick={() => update("scalingPolicyType", p)}
                        className={cn(
                          "rounded-lg px-3 py-1.5 text-xs font-medium border transition-colors",
                          form.scalingPolicyType === p
                            ? "border-accent-violet/50 bg-accent-violet/10 text-accent-violet"
                            : "border-border text-text-muted hover:text-text-primary",
                        )}
                      >
                        {p === "queue_request_count"
                          ? t("dash.serverless.policy_queue_count")
                          : t("dash.serverless.policy_queue_delay")}
                      </button>
                    ))}
                  </div>
                  <div className="max-w-xs">
                    <label className="block text-sm font-medium mb-1">{t("dash.serverless.policy_threshold")}</label>
                    <NumberInput
                      min={1}
                      max={1000}
                      value={form.scalingPolicyValue}
                      onChange={(v) => update("scalingPolicyValue", v)}
                    />
                  </div>
                </div>
              </>
            )}

            {step === 4 && (
              <>
                <h2 className="text-lg font-semibold">{t("dash.serverless.env_title")}</h2>
                <p className="text-sm text-text-muted">{t("dash.serverless.env_desc")}</p>
                <div className="space-y-2">
                  {form.envRows.map((row, i) => (
                    <div key={i} className="flex gap-2">
                      <Input
                        className="flex-1"
                        placeholder="KEY"
                        value={row.key}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                          const rows = [...form.envRows];
                          rows[i] = { ...rows[i], key: e.target.value };
                          update("envRows", rows);
                        }}
                      />
                      <Input
                        className="flex-[2]"
                        placeholder="value"
                        value={row.value}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                          const rows = [...form.envRows];
                          rows[i] = { ...rows[i], value: e.target.value };
                          update("envRows", rows);
                        }}
                      />
                    </div>
                  ))}
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => update("envRows", [...form.envRows, { key: "", value: "" }])}
                  >
                    {t("dash.serverless.env_add")}
                  </Button>
                </div>
              </>
            )}

            {step === 5 && (
              <>
                <h2 className="text-lg font-semibold">{t("dash.serverless.review_title")}</h2>
                <div className="grid gap-3 sm:grid-cols-2">
                  {[
                    { icon: Sparkles, label: t("dash.serverless.review_method"), value: form.method === "preset" ? t("dash.serverless.method_preset") : t("dash.serverless.method_custom") },
                    { icon: Box, label: t("dash.serverless.review_source"), value: form.method === "preset" ? form.modelRef : form.imageRef },
                    { icon: Cpu, label: t("dash.serverless.review_gpu"), value: `${form.gpuCount}× ${form.gpuTier || "—"} · ${form.region}` },
                    { icon: Layers, label: t("dash.serverless.review_scaling"), value: `${form.minWorkers}–${form.maxWorkers} workers · ${form.maxConcurrency} concurrent` },
                    { icon: Timer, label: t("dash.serverless.idle_timeout"), value: `${form.idleTimeoutSec}s` },
                    { icon: Globe, label: t("dash.serverless.scaling_policy"), value: form.scalingPolicyType },
                  ].map((item) => (
                    <div key={item.label} className="flex items-start gap-3 rounded-lg border border-border bg-surface-hover/50 p-3">
                      <item.icon className="h-4 w-4 text-accent-violet mt-0.5 shrink-0" />
                      <div className="min-w-0">
                        <p className="text-xs text-text-muted">{item.label}</p>
                        <p className="text-sm font-medium truncate">{item.value}</p>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="rounded-xl border border-emerald/20 bg-emerald/5 p-4">
                  <div className="flex items-center gap-2 text-sm font-medium text-emerald">
                    <Zap className="h-4 w-4" />
                    {t("dash.serverless.cold_start_note")}
                  </div>
                  <p className="text-xs text-text-muted mt-1">{t("dash.serverless.cold_start_desc")}</p>
                  {form.gpuTier && (
                    <p className="text-sm mt-2 font-mono">
                      ~${(costPerHour * form.gpuCount).toFixed(2)} CAD/hr {t("dash.serverless.per_worker")}
                      {form.minWorkers > 0 && (
                        <span className="text-text-muted">
                          {" "}· ~${(costPerHour * form.gpuCount * form.minWorkers * 24).toFixed(2)}/day min
                        </span>
                      )}
                    </p>
                  )}
                </div>
              </>
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Nav */}
      <div className="flex items-center justify-between">
        <Button variant="outline" onClick={back} disabled={step === 0}>
          <ChevronLeft className="h-4 w-4" /> {t("dash.serverless.back")}
        </Button>
        {step < DEPLOY_STUDIO_STEPS.length - 1 ? (
          <Button onClick={next}>
            {t("dash.serverless.continue")} <ChevronRight className="h-4 w-4" />
          </Button>
        ) : (
          <Button onClick={handleDeploy} disabled={deploying || !canWrite}>
            {deploying ? <Loader2 className="h-4 w-4 animate-spin" /> : <Rocket className="h-4 w-4" />}
            {t("dash.serverless.deploy")}
          </Button>
        )}
      </div>
    </div>
  );
}