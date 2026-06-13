"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import {
  Cpu, Activity, MapPin, Clock, RefreshCw, AlertCircle,
  Leaf, ShieldCheck, Zap, ArrowRight, TrendingDown,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { AuroraBackground } from "@/components/ui/aurora-bg";
import { m } from "@/components/marketing/motion";
import { useLocale } from "@/lib/locale";

interface HostSummary {
  gpu_model: string;
  available: number;
  total: number;
  vram_gb: number;
  price_cad: number;
  spot_cad: number;
  locations: string[];
}

const API = process.env.NEXT_PUBLIC_API_URL ?? "";

type LoadState = "loading" | "ready" | "degraded" | "error";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.5, ease: "easeOut" as const },
  }),
};

function HeroStat({
  i, icon: Icon, accent, value, suffix, label,
}: {
  i: number;
  icon: typeof Cpu;
  accent: "gold" | "emerald" | "cyan";
  value: string;
  suffix: string;
  label: string;
}) {
  const accentMap = {
    gold: "text-accent-gold border-accent-gold/30 bg-accent-gold/10",
    emerald: "text-emerald border-emerald/30 bg-emerald/10",
    cyan: "text-accent-cyan border-accent-cyan/30 bg-accent-cyan/10",
  };
  const iconMap = {
    gold: "text-accent-gold",
    emerald: "text-emerald",
    cyan: "text-accent-cyan",
  };

  return (
    <m.div
      variants={fadeUp}
      custom={i}
      className={`rounded-xl border px-5 py-4 backdrop-blur-sm ${accentMap[accent]}`}
    >
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`h-4 w-4 ${iconMap[accent]}`} />
        <span className="text-xs font-medium uppercase tracking-wider text-text-muted">{label}</span>
      </div>
      <p className="text-2xl font-bold tabular-nums">
        {value}
        <span className="ml-1 text-sm font-normal text-text-muted">{suffix}</span>
      </p>
    </m.div>
  );
}

async function fetchGpuSummaries(): Promise<{
  summaries: HostSummary[];
  liveData: boolean;
  hostsOk: boolean;
  pricingOk: boolean;
}> {
  let hostsOk = false;
  let pricingOk = false;
  const [gpuRes, pricingRes] = await Promise.all([
    fetch(`${API}/api/v2/gpu/available`, { credentials: "omit" }),
    fetch(`${API}/api/pricing/reference`, { credentials: "omit" }),
  ]);

  hostsOk = gpuRes.ok;
  pricingOk = pricingRes.ok;
  const gpuBody = hostsOk ? await gpuRes.json() : { gpus: [] };
  const pricingBody = pricingOk ? await pricingRes.json() : { pricing: {} };

  const offers: Array<Record<string, unknown>> = gpuBody.gpus || [];
  const pricing: Record<string, Record<string, number>> = pricingBody.pricing || {};

  const byModel = new Map<
    string,
    { available: number; total: number; vram: number; locations: Set<string> }
  >();

  for (const row of offers) {
    const model = (row.gpu_model as string) || "Unknown";
    const vram = Number(row.vram_gb) || 0;
    const count = Number(row.count_available) || 0;
    const province = (row.province as string) || "";
    if (!byModel.has(model)) {
      byModel.set(model, { available: 0, total: 0, vram, locations: new Set() });
    }
    const entry = byModel.get(model)!;
    entry.available += count;
    entry.total += count;
    if (province) entry.locations.add(province);
    if (vram > entry.vram) entry.vram = vram;
  }

  for (const [model] of Object.entries(pricing)) {
    if (!byModel.has(model)) {
      const vram = model.includes("H100") ? 80
        : model.includes("A100") ? 80
          : model.includes("L40") ? 48
            : model.includes("4090") ? 24
              : 24;
      byModel.set(model, { available: 0, total: 0, vram, locations: new Set() });
    }
  }

  const summaries: HostSummary[] = [];
  for (const [model, entry] of byModel) {
    const ref = pricing[model] || {};
    const base = (ref.base_rate_cad as number) || 0;
    const spot = (ref.spot_cad as number) || (base ? base * 0.4 : 0);
    summaries.push({
      gpu_model: model,
      available: entry.available,
      total: entry.total,
      vram_gb: entry.vram || (ref.vram_gb as number) || 0,
      price_cad: base,
      spot_cad: spot,
      locations: Array.from(entry.locations),
    });
  }

  summaries.sort(
    (a, b) =>
      (b.available > 0 ? 1 : 0) - (a.available > 0 ? 1 : 0) || a.price_cad - b.price_cad,
  );

  return {
    summaries,
    liveData: hostsOk && offers.length > 0,
    hostsOk,
    pricingOk,
  };
}

export function GPUAvailabilityContent() {
  const { t } = useLocale();
  const [gpus, setGpus] = useState<HostSummary[]>([]);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [liveData, setLiveData] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  async function load(opts?: { refresh?: boolean }) {
    if (opts?.refresh) {
      setLoadState((s) => (s === "ready" ? "ready" : "loading"));
    }
    try {
      const { summaries, liveData: live, hostsOk, pricingOk } = await fetchGpuSummaries();
      setGpus(summaries);
      setLiveData(live);
      setLastUpdated(new Date());
      setLoadState(hostsOk || pricingOk ? (hostsOk ? "ready" : "degraded") : "error");
    } catch {
      setGpus([]);
      setLiveData(false);
      setLoadState("error");
    }
  }

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const { summaries, liveData: live, hostsOk, pricingOk } = await fetchGpuSummaries();
        if (!active) return;
        setGpus(summaries);
        setLiveData(live);
        setLastUpdated(new Date());
        setLoadState(hostsOk || pricingOk ? (hostsOk ? "ready" : "degraded") : "error");
      } catch {
        if (!active) return;
        setGpus([]);
        setLiveData(false);
        setLoadState("error");
      }
    })();
    const interval = setInterval(() => { void load({ refresh: true }); }, 30_000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const spotPrices = gpus.filter((g) => g.spot_cad > 0).map((g) => g.spot_cad);
  const cheapestSpot = spotPrices.length ? Math.min(...spotPrices) : 0.3;
  const totalAvailable = gpus.reduce((sum, g) => sum + g.available, 0);
  const modelsWithStock = gpus.filter((g) => g.available > 0).length;
  const spotGpus = gpus.filter((g) => g.spot_cad > 0);
  const bestSpotModel = spotGpus.length
    ? spotGpus.reduce((a, b) => (a.spot_cad <= b.spot_cad ? a : b)).gpu_model
    : "";

  return (
    <div className="relative mx-auto max-w-7xl px-6 py-28">
      <AuroraBackground className="-z-10 opacity-50" />

      {/* Hero */}
      <m.div
        className="text-center mb-12"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <m.div variants={fadeUp} custom={0} className="mb-6 inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-4 py-1.5 backdrop-blur-sm">
          <Activity className="h-3.5 w-3.5 text-emerald-400" />
          <span className="text-xs font-medium text-emerald-400">
            {liveData ? t("gpus.badge_live") : t("gpus.badge_reference")}
          </span>
        </m.div>
        <m.h1 variants={fadeUp} custom={1} className="text-4xl font-bold md:text-5xl lg:text-6xl tracking-tight">
          {t("gpus.title")}
        </m.h1>
        <m.p variants={fadeUp} custom={2} className="mt-5 text-lg text-text-secondary max-w-2xl mx-auto leading-relaxed">
          {t("gpus.subtitle")}
        </m.p>
      </m.div>

      {/* Stat strip */}
      <m.div
        className="mb-20 grid grid-cols-1 gap-4 sm:grid-cols-3 max-w-3xl mx-auto"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <HeroStat i={0} icon={TrendingDown} accent="gold" value={`$${cheapestSpot.toFixed(2)}`} suffix="CAD/hr" label={t("gpus.stat_cheapest")} />
        <HeroStat i={1} icon={Cpu} accent="cyan" value={String(gpus.length || "—")} suffix={modelsWithStock > 0 ? `· ${totalAvailable} live` : ""} label={t("gpus.stat_models")} />
        <HeroStat i={2} icon={Leaf} accent="emerald" value="100%" suffix="" label={t("gpus.stat_hydro")} />
      </m.div>

      {/* Sovereign story + map */}
      <m.div
        className="mb-24 grid gap-10 lg:grid-cols-2 lg:items-center"
        initial={{ opacity: 0, y: 28 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.55 }}
      >
        <div className="space-y-6">
          <div className="inline-flex items-center gap-2 rounded-full border border-accent-cyan/30 bg-accent-cyan/10 px-3 py-1 text-xs font-medium text-accent-cyan">
            <ShieldCheck className="h-3.5 w-3.5" />
            Sovereign compute
          </div>
          <h2 className="text-3xl font-bold leading-tight">{t("gpus.sovereign_title")}</h2>
          <p className="text-text-secondary leading-relaxed">{t("gpus.sovereign_desc")}</p>
          <ul className="space-y-3">
            {[t("gpus.sovereign_i1"), t("gpus.sovereign_i2"), t("gpus.sovereign_i3")].map((item) => (
              <li key={item} className="flex items-start gap-3 text-sm text-text-secondary">
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald/15 text-emerald">
                  <Zap className="h-3 w-3" />
                </span>
                {item}
              </li>
            ))}
          </ul>
        </div>
        <div className="relative aspect-[4/3] overflow-hidden rounded-2xl border border-border/60 bg-surface/40 backdrop-blur-sm">
          <Image
            src="/canada-map-arc-light.svg"
            alt="Canadian compute network"
            fill
            className="object-contain p-8 dark:hidden"
            priority
          />
          <Image
            src="/canada-map-arc.svg"
            alt="Canadian compute network"
            fill
            className="object-contain p-8 hidden dark:block"
            priority
          />
          <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-transparent pointer-events-none" />
        </div>
      </m.div>

      {/* Fleet section header */}
      <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">{t("gpus.fleet_title")}</h2>
          <p className="mt-1 text-sm text-text-secondary max-w-xl">{t("gpus.fleet_desc")}</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-sm text-text-muted">
            <Clock className="h-3.5 w-3.5" />
            {lastUpdated ? `${t("gpus.updated")} ${lastUpdated.toLocaleTimeString()}` : "…"}
          </div>
          <button
            type="button"
            onClick={() => void load({ refresh: true })}
            disabled={loadState === "loading"}
            className="flex min-h-10 items-center gap-1.5 rounded-lg border border-border px-3 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors disabled:opacity-50"
            aria-label={t("gpus.refresh")}
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loadState === "loading" ? "animate-spin" : ""}`} />
            <span className="hidden sm:inline">{t("gpus.refresh")}</span>
          </button>
        </div>
      </div>

      {loadState === "degraded" && (
        <div role="status" className="mb-6 flex items-start gap-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <p>{t("gpus.degraded")}</p>
        </div>
      )}

      {loadState === "error" && (
        <div role="alert" className="mb-6 flex items-start gap-3 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <p>{t("gpus.error")}</p>
        </div>
      )}

      {loadState === "loading" && gpus.length === 0 ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-56 rounded-2xl border border-border bg-surface/30 animate-pulse" />
          ))}
        </div>
      ) : gpus.length === 0 ? (
        <div className="text-center py-24 text-text-muted rounded-2xl border border-dashed border-border">
          <Cpu className="h-12 w-12 mx-auto mb-4 opacity-40" />
          <p>{t("gpus.empty")}</p>
        </div>
      ) : (
        <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
          {gpus.map((gpu, i) => {
            const spotOff = gpu.price_cad > 0
              ? Math.round((1 - gpu.spot_cad / gpu.price_cad) * 100)
              : 0;
            const isBest = gpu.gpu_model === bestSpotModel && gpu.spot_cad > 0;
            return (
              <m.div
                key={gpu.gpu_model}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-40px" }}
                transition={{ delay: (i % 6) * 0.05, duration: 0.4 }}
                className={`group relative overflow-hidden rounded-2xl border bg-surface/50 backdrop-blur-sm p-6 transition-all hover:border-border hover:shadow-lg hover:shadow-accent-cyan/5 ${
                  isBest ? "border-emerald/40 ring-1 ring-emerald/20" : "border-border/70"
                }`}
              >
                {isBest && (
                  <span className="absolute top-4 right-4 rounded-full bg-emerald/15 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald">
                    Best value
                  </span>
                )}
                <div className="flex items-start gap-4 mb-5">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-accent-red/10 ring-1 ring-accent-red/20">
                    <Image src="/gpu-light.svg" alt="" width={28} height={28} className="dark:hidden" />
                    <Image src="/gpu.svg" alt="" width={28} height={28} className="hidden dark:block" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h3 className="font-semibold text-lg truncate">{gpu.gpu_model}</h3>
                    <p className="text-xs text-text-muted">{gpu.vram_gb} GB VRAM</p>
                    <span
                      className={`mt-2 inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                        gpu.available > 0
                          ? "bg-emerald-500/10 text-emerald-400"
                          : "bg-yellow-500/10 text-yellow-400"
                      }`}
                    >
                      <span className={`h-1.5 w-1.5 rounded-full ${gpu.available > 0 ? "bg-emerald-400 animate-pulse" : "bg-yellow-400"}`} />
                      {gpu.available > 0 ? `${gpu.available} ${t("gpus.available")}` : t("gpus.on_request")}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="rounded-xl bg-background/60 border border-border/50 p-3">
                    <p className="text-[10px] font-medium uppercase tracking-wider text-text-muted mb-1">{t("gpus.ondemand")}</p>
                    <p className="text-xl font-bold tabular-nums">
                      {gpu.price_cad > 0 ? `$${gpu.price_cad.toFixed(2)}` : "—"}
                      <span className="text-xs font-normal text-text-muted">/hr</span>
                    </p>
                  </div>
                  <div className="rounded-xl bg-emerald/[0.06] border border-emerald/20 p-3">
                    <p className="text-[10px] font-medium uppercase tracking-wider text-emerald/80 mb-1">{t("gpus.spot")}</p>
                    <p className="text-xl font-bold tabular-nums text-emerald">
                      {gpu.spot_cad > 0 ? `$${gpu.spot_cad.toFixed(2)}` : "—"}
                      <span className="text-xs font-normal text-text-muted">/hr</span>
                      {spotOff > 0 && (
                        <span className="ml-1 text-[10px] font-semibold text-emerald">−{spotOff}%</span>
                      )}
                    </p>
                  </div>
                </div>

                {gpu.locations.length > 0 && (
                  <div className="flex items-center gap-1.5 text-xs text-text-muted mb-4">
                    <MapPin className="h-3 w-3 shrink-0" />
                    <span className="truncate">{gpu.locations.join(" · ")}</span>
                  </div>
                )}

                <Link
                  href="/register"
                  className="flex min-h-11 w-full items-center justify-center gap-2 rounded-xl bg-accent-red px-4 text-sm font-medium text-white hover:bg-accent-red-hover transition-colors"
                >
                  {t("gpus.deploy")}
                  <ArrowRight className="h-3.5 w-3.5 opacity-80 group-hover:translate-x-0.5 transition-transform" />
                </Link>
              </m.div>
            );
          })}
        </div>
      )}

      {/* CTA */}
      <m.div
        className="mt-28 text-center rounded-2xl border border-border/60 bg-surface/40 backdrop-blur-sm px-8 py-16"
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-3xl font-bold mb-3">{t("gpus.cta_title")}</h2>
        <p className="text-text-secondary max-w-lg mx-auto mb-8">{t("gpus.cta_desc")}</p>
        <div className="flex flex-wrap justify-center gap-4">
          <Link href="/register">
            <Button size="lg" className="px-10 shadow-lg shadow-accent-red/15">
              {t("gpus.cta_start")} <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
          <Link href="/pricing">
            <Button variant="outline" size="lg">
              {t("gpus.cta_pricing")}
            </Button>
          </Link>
        </div>
      </m.div>
    </div>
  );
}