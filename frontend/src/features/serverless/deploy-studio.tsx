"use client";

import { useEffect, useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import posthog from "posthog-js";
import { motion, AnimatePresence } from "framer-motion";
import {
  Sparkles, Box, Cpu, Layers, Rocket, ChevronLeft, ChevronRight,
  Loader2, Zap, Globe, Timer, Gauge, Save,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input, NumberInput } from "@/components/ui/input";

import { useLocale } from "@/lib/locale";
import type { GpuAvailability } from "@/lib/api";
import { GPU_MODELS } from "@/lib/gpu-models";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import {
  DEFAULT_FORM, DEPLOY_STUDIO_STEPS, formatTokenRateFromPricing, IDLE_TIMEOUT_OPTIONS, MANAGED_ENGINES, PRESET_MODELS,
  type TokenPricingQuote,
} from "./constants";
import type { DeployStudioForm } from "./types";
import { deployServerlessEndpoint } from "./deploy-actions";
import { TokenPricingTable } from "./token-pricing-table";
import { findGpuInRegion, regionOptionsForGpus } from "./region-options";
import { ServerlessHero, ServerlessSelect, StepRail } from "./serverless-ui";

interface DeployStudioProps {
  gpus: GpuAvailability[];
  canWrite: boolean;
}

const DRAFT_KEY = "xcelsior:serverless-deploy-draft";

interface DeployDraft {
  form: DeployStudioForm;
  step: number;
  savedAt: number;
}

function readDraft(): DeployDraft | null {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as DeployDraft;
    if (!parsed || typeof parsed !== "object" || typeof parsed.form !== "object") return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeDraft(form: DeployStudioForm, step: number) {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify({ form, step, savedAt: Date.now() }));
  } catch {
    // storage full or unavailable, draft saving is best-effort
  }
}

function clearDraft() {
  try {
    localStorage.removeItem(DRAFT_KEY);
  } catch {
    // ignore
  }
}

export function DeployStudio({ gpus, canWrite }: DeployStudioProps) {
  const { t } = useLocale();
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [form, setForm] = useState<DeployStudioForm>(DEFAULT_FORM);
  const [deploying, setDeploying] = useState(false);
  const [draftReady, setDraftReady] = useState(false);
  const [tokenQuotes, setTokenQuotes] = useState<Record<string, TokenPricingQuote>>({});

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/v2/serverless/preset-token-pricing", { credentials: "include" });
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled && data?.quotes) setTokenQuotes(data.quotes);
      } catch {
        // Token rates render once /preset-token-pricing succeeds.
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Restore an in-progress draft so a failed deploy (e.g. insufficient wallet
  // funds) or a detour to billing never loses the user's input.
  useEffect(() => {
    const draft = readDraft();
    if (draft) {
      setForm({ ...DEFAULT_FORM, ...draft.form });
      setStep(Math.min(Math.max(draft.step ?? 0, 0), DEPLOY_STUDIO_STEPS.length - 1));
      toast.info(t("dash.serverless.draft_restored"), {
        action: {
          label: t("dash.serverless.draft_discard"),
          onClick: () => {
            clearDraft();
            setForm(DEFAULT_FORM);
            setStep(0);
          },
        },
      });
    }
    setDraftReady(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Autosave (debounced) once the draft has been hydrated.
  useEffect(() => {
    if (!draftReady) return;
    if (step === 0 && JSON.stringify(form) === JSON.stringify(DEFAULT_FORM)) return;
    const id = window.setTimeout(() => writeDraft(form, step), 400);
    return () => window.clearTimeout(id);
  }, [form, step, draftReady]);

  const gpuTypes = useMemo(() => {
    const fromMarket = [...new Set(gpus.map((g) => g.gpu_model))];
    if (fromMarket.length > 0) return fromMarket;
    return GPU_MODELS.slice(0, 12).map((g) => g.value);
  }, [gpus]);

  useEffect(() => {
    setForm((prev) => {
      const nextGpuTier = prev.gpuTier || (gpus.length > 0 ? gpuTypes[0] || "" : "");
      const nextRegions = regionOptionsForGpus(gpus, nextGpuTier);
      const nextRegion = nextRegions.includes(prev.region) ? prev.region : nextRegions[0];
      if (prev.gpuTier === nextGpuTier && prev.region === nextRegion) return prev;
      return { ...prev, gpuTier: nextGpuTier, region: nextRegion };
    });
  }, [gpus, gpuTypes]);

  const regions = useMemo(() => regionOptionsForGpus(gpus, form.gpuTier), [gpus, form.gpuTier]);
  const selectedGpu = findGpuInRegion(gpus, form.gpuTier, form.region);
  const costPerHour = selectedGpu?.price_per_hour_cad ?? (form.gpuTier ? 2.5 : 0);

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

  const validateStep = (idx: number): string | null => {
    if (idx === 0) return null;
    if (idx === 1) {
      if (form.method === "preset" && !form.modelRef.trim()) return t("dash.serverless.err_model");
      if (form.method === "custom" && form.customSource === "docker" && !form.imageRef.trim()) {
        return t("dash.serverless.err_image");
      }
      if (form.method === "custom" && form.customSource === "github" && !form.githubRepo.trim()) {
        return t("dash.serverless.err_github");
      }
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

  const saveAndExit = () => {
    writeDraft(form, step);
    toast.success(t("dash.serverless.draft_saved"));
    router.push("/dashboard/inference");
  };

  const handleDeploy = async () => {
    if (!canWrite) return toast.error(t("dash.serverless.viewer_blocked"));
    for (let i = 1; i < DEPLOY_STUDIO_STEPS.length; i++) {
      const err = validateStep(i);
      if (err) return toast.error(err);
    }
    setDeploying(true);
    try {
      const res = await deployServerlessEndpoint(form);
      clearDraft();
      toast.success(t("dash.serverless.deploy_success"));
      posthog.capture("serverless_endpoint_deployed", {
        method: form.method,
        gpu_tier: form.gpuTier,
        managed_engine: form.method === "preset" ? form.managedEngine : null,
        model_ref: form.method === "preset" ? form.modelRef : null,
        custom_source: form.method === "custom" ? form.customSource : null,
      });
      router.push(`/dashboard/inference/${res.endpoint.endpoint_id}`);
    } catch (e: unknown) {
      posthog.captureException(e instanceof Error ? e : new Error(String(e)));
      const msg = e instanceof Error ? e.message : t("dash.serverless.deploy_failed");
      if (/wallet|funds|suspended|balance|credit/i.test(msg)) {
        // Draft stays saved, so a detour to billing loses nothing.
        toast.error(`${msg}, ${t("dash.serverless.draft_saved")}`, {
          action: {
            label: t("dash.billing.title"),
            onClick: () => router.push("/dashboard/billing"),
          },
        });
      } else {
        toast.error(/team viewers cannot/i.test(msg) ? t("dash.serverless.viewer_blocked") : msg);
      }
    } finally {
      setDeploying(false);
    }
  };

  const reviewSource = form.method === "preset"
    ? `${MANAGED_ENGINES.find((e) => e.id === form.managedEngine)?.label ?? form.managedEngine} · ${form.modelRef}`
    : form.customSource === "github"
      ? form.githubRepo || form.imageRef
      : form.imageRef;

  return (
    <div className="space-y-6">
      <ServerlessHero
        icon={Rocket}
        badge={t("dash.serverless.studio_badge")}
        title={t("dash.serverless.studio_title")}
        description={t("dash.serverless.studio_desc")}
        accent="violet"
      />

      <StepRail
        steps={DEPLOY_STUDIO_STEPS}
        current={step}
        onStepClick={setStep}
        label={t}
      />

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
                    <div>
                      <label className="block text-sm font-medium mb-2">{t("dash.serverless.managed_engine")}</label>
                      <div className="grid gap-2 sm:grid-cols-3">
                        {MANAGED_ENGINES.map((engine) => (
                          <button
                            key={engine.id}
                            type="button"
                            onClick={() => update("managedEngine", engine.id)}
                            className={cn(
                              "rounded-lg border px-3 py-2.5 text-left text-sm transition-all",
                              form.managedEngine === engine.id
                                ? "border-accent-violet/50 bg-accent-violet/10"
                                : "border-border hover:bg-surface-hover",
                            )}
                          >
                            <span className="font-medium">{engine.label}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                    <label className="block text-sm font-medium">{t("dash.serverless.model_library")}</label>
                    {Object.keys(tokenQuotes).length > 0 && (
                      <TokenPricingTable quotes={tokenQuotes} selectedModel={form.modelRef} />
                    )}
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
                          {m.task !== "rerank" && (
                            <span className="block text-xs font-mono text-accent-cyan/80 mt-1">
                              {formatTokenRateFromPricing(m.id, tokenQuotes[m.id])}
                            </span>
                          )}
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
                    <div className="sm:col-span-2 flex gap-2">
                      {(["docker", "github"] as const).map((src) => (
                        <button
                          key={src}
                          type="button"
                          onClick={() => update("customSource", src)}
                          className={cn(
                            "rounded-lg border px-3 py-2 text-sm font-medium transition-all",
                            form.customSource === src
                              ? "border-accent-violet/50 bg-accent-violet/10"
                              : "border-border hover:bg-surface-hover",
                          )}
                        >
                          {src === "docker" ? t("dash.serverless.source_docker") : t("dash.serverless.source_github")}
                        </button>
                      ))}
                    </div>
                    {form.customSource === "github" ? (
                      <>
                        <div className="sm:col-span-2">
                          <label className="block text-sm font-medium mb-1">{t("dash.serverless.github_repo")}</label>
                          <Input
                            value={form.githubRepo}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => update("githubRepo", e.target.value)}
                            placeholder="https://github.com/org/repo"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium mb-1">{t("dash.serverless.github_branch")}</label>
                          <Input
                            value={form.githubBranch}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => update("githubBranch", e.target.value)}
                            placeholder="main"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium mb-1">{t("dash.serverless.container_image")}</label>
                          <Input
                            value={form.imageRef}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => update("imageRef", e.target.value)}
                            placeholder={t("dash.serverless.github_image_ph")}
                          />
                        </div>
                      </>
                    ) : (
                    <div className="sm:col-span-2">
                      <label className="block text-sm font-medium mb-1">{t("dash.serverless.container_image")}</label>
                      <Input
                        value={form.imageRef}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => update("imageRef", e.target.value)}
                        placeholder="registry.io/your-image:tag"
                      />
                    </div>
                    )}
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
                    <ServerlessSelect
                      value={form.gpuTier}
                      onChange={(e) => updateGpuTier(e.target.value)}
                    >
                      <option value="">{t("dash.serverless.select_gpu")}</option>
                      {gpuTypes.map((g) => {
                        const info = gpus.find((x) => x.gpu_model === g);
                        return (
                          <option key={g} value={g}>
                            {g}
                            {info ? `, ${info.vram_gb}GB · $${info.price_per_hour_cad.toFixed(2)}/hr` : ""}
                          </option>
                        );
                      })}
                    </ServerlessSelect>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">{t("dash.serverless.region")}</label>
                    <ServerlessSelect
                      value={form.region}
                      onChange={(e) => update("region", e.target.value)}
                    >
                      {regions.map((r) => (
                        <option key={r} value={r}>{r}</option>
                      ))}
                    </ServerlessSelect>
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
                    <ServerlessSelect
                      value={form.idleTimeoutSec}
                      onChange={(e) => update("idleTimeoutSec", Number(e.target.value))}
                    >
                      {IDLE_TIMEOUT_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{t(o.labelKey)}</option>
                      ))}
                    </ServerlessSelect>
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
                    { icon: Box, label: t("dash.serverless.review_source"), value: reviewSource },
                    { icon: Cpu, label: t("dash.serverless.review_gpu"), value: `${form.gpuCount}× ${form.gpuTier || "-"} · ${form.region}` },
                    { icon: Layers, label: t("dash.serverless.review_scaling"), value: `${form.minWorkers}-${form.maxWorkers} workers · ${form.maxConcurrency} concurrent` },
                    { icon: Timer, label: t("dash.serverless.idle_timeout"), value: `${form.idleTimeoutSec}s` },
                    {
                      icon: Globe,
                      label: t("dash.serverless.scaling_policy"),
                      value: `${form.scalingPolicyType === "queue_request_count"
                        ? t("dash.serverless.review_policy_queue_count")
                        : t("dash.serverless.review_policy_queue_delay")} · ${form.scalingPolicyValue}`,
                    },
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
                      {form.method === "preset" && formatTokenRateFromPricing(form.modelRef, tokenQuotes[form.modelRef]) && (
                        <span className="text-text-muted">
                          {" "}· {formatTokenRateFromPricing(form.modelRef, tokenQuotes[form.modelRef])}
                        </span>
                      )}
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
      <div className="flex items-center justify-between gap-3">
        <Button variant="outline" onClick={back} disabled={step === 0}>
          <ChevronLeft className="h-4 w-4" /> {t("dash.serverless.back")}
        </Button>
        <div className="flex items-center gap-2">
          <Button variant="ghost" onClick={saveAndExit}>
            <Save className="h-4 w-4" /> {t("dash.serverless.save_exit")}
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
    </div>
  );
}
