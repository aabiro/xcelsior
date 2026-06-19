"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import {
  Cpu, Activity, MapPin, Clock, RefreshCw, AlertCircle,
  Leaf, ShieldCheck, Zap, ArrowRight, TrendingDown,
  Brain, Layers, Bot, Sparkles, ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { AuroraBackground } from "@/components/ui/aurora-bg";
import { m } from "@/components/marketing/motion";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import {
  curateMarketingGpus,
  marketingGpuLabel,
  normalizeGpuModel,
  type MarketingGpuRow,
} from "@/lib/marketing-gpu";

const API = process.env.NEXT_PUBLIC_API_URL ?? "";

type LoadState = "loading" | "ready" | "degraded" | "error";
type PriceMode = "spot" | "ondemand";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.5, ease: "easeOut" as const },
  }),
};

const WORKLOADS = [
  { key: "training", icon: Brain, art: "/gpu-fleet/workload-training.svg" },
  { key: "inference", icon: Layers, art: "/gpu-fleet/workload-inference.svg" },
] as const;

const FLAGSHIP = [
  { key: "b200", tier: "B200", accent: "from-accent-gold/20 to-accent-red/10 border-accent-gold/30" },
  { key: "h100", tier: "H100", accent: "from-accent-cyan/20 to-accent-violet/10 border-accent-cyan/30" },
  { key: "a100", tier: "A100", accent: "from-emerald/15 to-accent-cyan/5 border-emerald/25" },
] as const;

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
      className={`rounded-2xl border px-6 py-5 backdrop-blur-sm ${accentMap[accent]}`}
    >
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`h-4 w-4 ${iconMap[accent]}`} />
        <span className="text-xs font-medium uppercase tracking-wider text-text-muted">{label}</span>
      </div>
      <p className="text-3xl font-bold tabular-nums tracking-tight">
        {value}
        <span className="ml-1.5 text-sm font-normal text-text-muted">{suffix}</span>
      </p>
    </m.div>
  );
}

async function fetchGpuSummaries(): Promise<{
  summaries: MarketingGpuRow[];
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
    const raw = (row.gpu_model as string) || "Unknown";
    const model = normalizeGpuModel(raw, Number(row.vram_gb) || 0);
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

  for (const [rawModel, rates] of Object.entries(pricing)) {
    const model = normalizeGpuModel(rawModel, (rates.vram_gb as number) || 0);
    if (!byModel.has(model)) {
      const vram = (rates.vram_gb as number) || 0;
      byModel.set(model, { available: 0, total: 0, vram, locations: new Set() });
    }
  }

  const summaries: MarketingGpuRow[] = [];
  for (const [model, entry] of byModel) {
    const ref = pricing[model] || pricing[Object.keys(pricing).find((k) => normalizeGpuModel(k) === model) || ""] || {};
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

  return {
    summaries: curateMarketingGpus(summaries),
    liveData: hostsOk && offers.length > 0,
    hostsOk,
    pricingOk,
  };
}

function GpuPriceDock({
  gpu,
  priceMode,
  onToggleMode,
  t,
}: {
  gpu: MarketingGpuRow;
  priceMode: PriceMode;
  onToggleMode: () => void;
  t: (key: string) => string;
}) {
  const spotOff = gpu.price_cad > 0
    ? Math.round((1 - gpu.spot_cad / gpu.price_cad) * 100)
    : 0;
  const active = priceMode === "spot" ? gpu.spot_cad : gpu.price_cad;
  const label = priceMode === "spot" ? t("gpus.spot") : t("gpus.ondemand");

  return (
    <div
      className="fixed bottom-0 left-0 right-0 z-40 border-t border-border/70 bg-background/92 backdrop-blur-xl shadow-[0_-12px_40px_rgba(0,0,0,0.35)]"
      style={{ paddingBottom: "max(0.75rem, env(safe-area-inset-bottom))" }}
    >
      <div className="mx-auto flex max-w-6xl items-center gap-4 px-4 py-3 sm:px-6 pr-24 sm:pr-28">
        <button
          type="button"
          onClick={onToggleMode}
          className="group flex min-w-0 flex-1 items-center gap-4 rounded-xl border border-border/60 bg-surface/50 px-4 py-2.5 text-left transition-colors hover:border-accent-cyan/40"
        >
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-accent-red/10 ring-1 ring-accent-red/20">
            <Cpu className="h-5 w-5 text-accent-red" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold">{marketingGpuLabel(gpu.gpu_model)}</p>
            <p className="text-xs text-text-muted">{gpu.vram_gb} GB · {label}</p>
          </div>
          <div className="shrink-0 text-right">
            <p className={cn(
              "text-2xl font-bold tabular-nums transition-all duration-300",
              priceMode === "spot" ? "text-emerald" : "text-text-primary",
            )}>
              {active > 0 ? `$${active.toFixed(2)}` : "—"}
              <span className="text-xs font-normal text-text-muted">/hr</span>
            </p>
            {/* Always reserve this line's height so cycling spot/on-demand
                doesn't change the dock height (no jitter). */}
            <p className={cn(
              "text-[10px] font-semibold text-emerald",
              !(priceMode === "spot" && spotOff > 0) && "invisible",
            )}>
              −{spotOff > 0 ? spotOff : 0}% vs on-demand
            </p>
          </div>
          <ChevronRight className="h-4 w-4 shrink-0 text-text-muted transition-transform group-hover:translate-x-0.5" />
        </button>
        <Link href="/register" className="hidden sm:block shrink-0">
          <Button size="sm" className="gap-1.5 shadow-lg shadow-accent-red/15">
            {t("gpus.deploy")} <ArrowRight className="h-3.5 w-3.5" />
          </Button>
        </Link>
      </div>
    </div>
  );
}

export function GPUAvailabilityContent() {
  const { t } = useLocale();
  const [gpus, setGpus] = useState<MarketingGpuRow[]>([]);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [liveData, setLiveData] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [hoveredModel, setHoveredModel] = useState<string | null>(null);
  const [priceMode, setPriceMode] = useState<PriceMode>("spot");
  const [cyclePrices, setCyclePrices] = useState(false);
  const cardsRef = useRef<HTMLDivElement>(null);
  const [cardsInView, setCardsInView] = useState(false);

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
  const bestSpotGpu = useMemo(() => {
    const withSpot = gpus.filter((g) => g.spot_cad > 0);
    if (!withSpot.length) return gpus[0] ?? null;
    return withSpot.reduce((a, b) => (a.spot_cad <= b.spot_cad ? a : b));
  }, [gpus]);

  const activeModel = hoveredModel ?? selectedModel ?? bestSpotGpu?.gpu_model ?? gpus[0]?.gpu_model ?? null;
  const activeGpu = gpus.find((g) => g.gpu_model === activeModel) ?? null;

  useEffect(() => {
    if (!cyclePrices) return;
    const id = setInterval(() => {
      setPriceMode((m) => (m === "spot" ? "ondemand" : "spot"));
    }, 2200);
    return () => clearInterval(id);
  }, [cyclePrices]);

  // Only show the sticky price dock while the GPU cards are in view.
  useEffect(() => {
    const el = cardsRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => setCardsInView(entry.isIntersecting),
      { rootMargin: "0px 0px -10% 0px" },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [gpus.length]);

  const handleCardInteract = useCallback((model: string) => {
    setSelectedModel(model);
    setCyclePrices(true);
    setPriceMode("spot");
  }, []);

  const togglePriceMode = useCallback(() => {
    setPriceMode((m) => (m === "spot" ? "ondemand" : "spot"));
    setCyclePrices(false);
  }, []);

  return (
    <div className="relative mx-auto max-w-6xl px-6 py-28 pb-36">
      <AuroraBackground className="-z-10 opacity-50" />
      <Image
        src="/gpu-fleet/accent-flare.svg"
        alt=""
        width={120}
        height={120}
        className="pointer-events-none absolute right-8 top-32 opacity-40 hidden lg:block"
        aria-hidden
      />

      {/* Hero */}
      <m.div
        className="mb-20 grid gap-12 lg:grid-cols-[1fr_1.05fr] lg:items-center"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <div className="text-center lg:text-left">
          <m.div variants={fadeUp} custom={0} className="mb-6 inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-4 py-1.5 backdrop-blur-sm">
            <Activity className="h-3.5 w-3.5 text-emerald-400" />
            <span className="text-xs font-medium text-emerald-400">
              {liveData ? t("gpus.badge_live") : t("gpus.badge_reference")}
            </span>
          </m.div>
          <m.h1 variants={fadeUp} custom={1} className="text-4xl font-bold md:text-5xl lg:text-6xl tracking-tight leading-[1.08]">
            {t("gpus.title")}
          </m.h1>
          <m.p variants={fadeUp} custom={2} className="mt-6 text-lg text-text-secondary max-w-xl mx-auto lg:mx-0 leading-relaxed">
            {t("gpus.subtitle")}
          </m.p>
          <m.div variants={fadeUp} custom={3} className="mt-8 flex flex-wrap justify-center lg:justify-start gap-3">
            <Link href="/register">
              <Button size="lg" className="px-8 shadow-lg shadow-accent-red/15">
                {t("gpus.deploy")} <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
            <Link href="/pricing">
              <Button variant="outline" size="lg">{t("gpus.cta_pricing")}</Button>
            </Link>
          </m.div>
        </div>
        <m.div
          variants={fadeUp}
          custom={2}
          className="relative aspect-[3/2] overflow-hidden rounded-3xl border border-border/50 bg-surface/30 shadow-2xl shadow-accent-cyan/5"
        >
          <Image src="/gpu-fleet/hero-power.svg" alt="" fill className="object-cover" priority />
        </m.div>
      </m.div>

      {/* MCP cross-promo */}
      <m.div
        className="mb-16 rounded-2xl border border-accent-violet/30 bg-gradient-to-r from-accent-violet/10 via-surface/40 to-accent-cyan/10 p-6 md:p-8"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        variants={fadeUp}
        custom={0}
      >
        <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div className="flex items-start gap-4">
            <div className="rounded-xl border border-accent-violet/30 bg-accent-violet/10 p-3">
              <Bot className="h-6 w-6 text-accent-violet" />
            </div>
            <div>
              <p className="text-sm font-medium uppercase tracking-wider text-accent-violet">{t("gpus.mcp_badge")}</p>
              <h2 className="mt-1 text-xl font-semibold tracking-tight md:text-2xl">{t("gpus.mcp_title")}</h2>
              <p className="mt-2 max-w-xl text-sm text-text-secondary leading-relaxed">{t("gpus.mcp_desc")}</p>
            </div>
          </div>
          <Link href="/mcp" className="shrink-0">
            <Button size="lg" variant="outline" className="border-accent-violet/40 hover:bg-accent-violet/10">
              {t("gpus.mcp_cta")} <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </m.div>

      {/* Stat strip */}
      <m.div
        className="mb-28 grid grid-cols-1 gap-5 sm:grid-cols-3"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <HeroStat i={0} icon={TrendingDown} accent="gold" value={`$${cheapestSpot.toFixed(2)}`} suffix="CAD/hr" label={t("gpus.stat_cheapest")} />
        <HeroStat i={1} icon={Cpu} accent="cyan" value={String(gpus.length || "—")} suffix={modelsWithStock > 0 ? `· ${totalAvailable} live` : ""} label={t("gpus.stat_models")} />
        <HeroStat i={2} icon={Leaf} accent="emerald" value="100%" suffix="" label={t("gpus.stat_hydro")} />
      </m.div>

      {/* Workloads — training + inference only */}
      <m.div
        className="mb-28"
        initial={{ opacity: 0, y: 28 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.55 }}
      >
        <div className="mb-10 max-w-2xl">
          <h2 className="text-3xl font-bold tracking-tight">{t("gpus.power_title")}</h2>
          <p className="mt-4 text-text-secondary leading-relaxed">{t("gpus.power_desc")}</p>
        </div>
        <div className="grid gap-6 md:grid-cols-2">
          {WORKLOADS.map(({ key, icon: Icon, art }, i) => (
            <m.div
              key={key}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08, duration: 0.45 }}
              className="group overflow-hidden rounded-2xl border border-border/60 bg-surface/40 backdrop-blur-sm transition-colors hover:border-accent-cyan/30"
            >
              <div className="relative h-28 overflow-hidden bg-gradient-to-br from-surface/80 to-background/40">
                <Image src={art} alt="" fill className="object-cover opacity-90 transition-transform duration-500 group-hover:scale-[1.03]" />
              </div>
              <div className="p-6">
                <span className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-accent-cyan/10 ring-1 ring-accent-cyan/20">
                  <Icon className="h-5 w-5 text-accent-cyan" />
                </span>
                <h3 className="text-lg font-semibold">{t(`gpus.workload_${key}`)}</h3>
                <p className="mt-2 text-sm text-text-secondary leading-relaxed">{t(`gpus.workload_${key}_desc`)}</p>
              </div>
            </m.div>
          ))}
        </div>
      </m.div>

      {/* Flagship strip */}
      <m.div
        className="mb-28 rounded-3xl border border-border/60 bg-gradient-to-br from-surface/60 via-background/40 to-surface/30 p-8 md:p-10 backdrop-blur-sm"
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <div className="mb-8 max-w-2xl">
          <p className="text-xs font-semibold uppercase tracking-widest text-accent-gold mb-2">{t("gpus.featured_title")}</p>
          <p className="text-text-secondary leading-relaxed">{t("gpus.featured_desc")}</p>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {FLAGSHIP.map(({ key, tier, accent }, i) => (
            <div key={key} className={`rounded-2xl border bg-gradient-to-br p-6 ${accent}`}>
              <p className="text-2xl font-bold tracking-tight">{tier}</p>
              <p className="mt-2 text-sm text-text-secondary">{t(`gpus.featured_${key}`)}</p>
              {i === 0 && (
                <span className="mt-4 inline-flex rounded-full bg-accent-gold/15 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-accent-gold">
                  Latest gen
                </span>
              )}
            </div>
          ))}
        </div>
      </m.div>

      {/* Sovereign story + glam map */}
      <m.div
        className="mb-32 grid gap-14 lg:grid-cols-2 lg:items-center"
        initial={{ opacity: 0, y: 28 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.55 }}
      >
        <div className="space-y-6">
          <div className="inline-flex items-center gap-2 rounded-full border border-accent-cyan/30 bg-accent-cyan/10 px-3 py-1 text-xs font-medium text-accent-cyan">
            <ShieldCheck className="h-3.5 w-3.5" />
            {t("gpus.sovereign_badge")}
          </div>
          <h2 className="text-3xl font-bold leading-tight tracking-tight">{t("gpus.sovereign_title")}</h2>
          <p className="text-text-secondary leading-relaxed text-lg">{t("gpus.sovereign_desc")}</p>
          <ul className="space-y-4 pt-2">
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
        <div className="relative aspect-[4/3] overflow-hidden rounded-3xl border border-accent-cyan/20 bg-[#060a14] shadow-2xl shadow-accent-cyan/10">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_30%_20%,rgba(0,212,255,0.18),transparent_55%),radial-gradient(ellipse_at_80%_60%,rgba(124,58,237,0.14),transparent_50%)]" />
          <Image
            src="/gpu-fleet/canada-sovereign.svg"
            alt={t("gpus.sovereign_map_alt")}
            fill
            className="object-contain p-6 text-text-primary"
          />
          <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between rounded-xl border border-border/50 bg-background/70 px-4 py-2.5 backdrop-blur-md">
            <span className="text-xs font-medium text-text-secondary">{t("gpus.sovereign_map_caption")}</span>
            <span className="flex items-center gap-1.5 text-xs text-emerald">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald animate-pulse" />
              {t("gpus.sovereign_map_live")}
            </span>
          </div>
        </div>
      </m.div>

      {/* Fleet section */}
      <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">{t("gpus.fleet_title")}</h2>
          <p className="mt-3 text-text-secondary max-w-xl leading-relaxed">{t("gpus.fleet_desc")}</p>
          <p className="mt-2 text-xs text-text-muted">{t("gpus.fleet_hint")}</p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
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
        <div role="status" className="mb-8 flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <p>{t("gpus.degraded")}</p>
        </div>
      )}

      {loadState === "error" && (
        <div role="alert" className="mb-8 flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <p>{t("gpus.error")}</p>
        </div>
      )}

      {loadState === "loading" && gpus.length === 0 ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-36 rounded-2xl border border-border bg-surface/30 animate-pulse" />
          ))}
        </div>
      ) : gpus.length === 0 ? (
        <div className="text-center py-28 text-text-muted rounded-2xl border border-dashed border-border">
          <Cpu className="h-12 w-12 mx-auto mb-4 opacity-40" />
          <p>{t("gpus.empty")}</p>
        </div>
      ) : (
        <div ref={cardsRef} className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {gpus.map((gpu, i) => {
            const isActive = gpu.gpu_model === activeModel;
            const isBest = gpu.gpu_model === bestSpotGpu?.gpu_model && gpu.spot_cad > 0;
            return (
              <m.button
                key={gpu.gpu_model}
                type="button"
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-40px" }}
                transition={{ delay: (i % 6) * 0.05, duration: 0.4 }}
                onClick={() => handleCardInteract(gpu.gpu_model)}
                onMouseEnter={() => {
                  setHoveredModel(gpu.gpu_model);
                  setCyclePrices(true);
                }}
                onMouseLeave={() => setHoveredModel(null)}
                className={cn(
                  "group relative flex flex-col rounded-2xl border bg-surface/40 p-5 text-left backdrop-blur-sm transition-all hover:shadow-lg hover:shadow-accent-cyan/5",
                  isActive
                    ? "border-accent-cyan/50 ring-2 ring-accent-cyan/25 shadow-lg shadow-accent-cyan/10"
                    : "border-border/70 hover:border-border",
                  isBest && !isActive && "border-emerald/30",
                )}
              >
                <div className="mb-4 flex items-start justify-between gap-3">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-accent-red/10 ring-1 ring-accent-red/20">
                    <Image src="/gpu-light.svg" alt="" width={28} height={28} className="dark:hidden" />
                    <Image src="/gpu.svg" alt="" width={28} height={28} className="hidden dark:block" />
                  </div>
                  {isBest && (
                    <span className="rounded-full bg-emerald/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald">
                      Best value
                    </span>
                  )}
                </div>
                <h3 className="text-lg font-semibold tracking-tight">{marketingGpuLabel(gpu.gpu_model)}</h3>
                <p className="mt-0.5 text-sm text-text-muted">{gpu.vram_gb} GB VRAM</p>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <span
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium",
                      gpu.available > 0 ? "bg-emerald-500/10 text-emerald-400" : "bg-yellow-500/10 text-yellow-400",
                    )}
                  >
                    <span className={cn("h-1.5 w-1.5 rounded-full", gpu.available > 0 ? "bg-emerald-400 animate-pulse" : "bg-yellow-400")} />
                    {gpu.available > 0 ? `${gpu.available} ${t("gpus.available")}` : t("gpus.on_request")}
                  </span>
                  {gpu.locations.length > 0 && (
                    <span className="flex items-center gap-1 text-xs text-text-muted">
                      <MapPin className="h-3 w-3 shrink-0" />
                      {gpu.locations.slice(0, 2).join(" · ")}
                      {gpu.locations.length > 2 ? ` +${gpu.locations.length - 2}` : ""}
                    </span>
                  )}
                </div>
                <p className="mt-4 flex items-center gap-1 text-xs text-accent-cyan opacity-0 transition-opacity group-hover:opacity-100">
                  <Sparkles className="h-3 w-3" />
                  {t("gpus.card_price_hint")}
                </p>
              </m.button>
            );
          })}
        </div>
      )}

      {/* Deploy velocity — under fleet */}
      <m.div
        className="mt-20 overflow-hidden rounded-3xl border border-border/60 bg-gradient-to-br from-surface/50 via-background/30 to-accent-violet/5"
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <div className="grid gap-0 lg:grid-cols-[1fr_1.1fr] lg:items-center">
          <div className="p-8 md:p-10">
            <p className="text-xs font-semibold uppercase tracking-widest text-accent-violet mb-3">{t("gpus.velocity_badge")}</p>
            <h3 className="text-2xl font-bold tracking-tight md:text-3xl">{t("gpus.velocity_title")}</h3>
            <p className="mt-4 text-text-secondary leading-relaxed">{t("gpus.velocity_desc")}</p>
            <ol className="mt-8 space-y-4">
              {(["pick", "provision", "pulse"] as const).map((step, idx) => (
                <li key={step} className="flex gap-4">
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent-cyan/15 text-sm font-bold text-accent-cyan">
                    {idx + 1}
                  </span>
                  <div>
                    <p className="font-medium">{t(`gpus.velocity_${step}`)}</p>
                    <p className="text-sm text-text-muted">{t(`gpus.velocity_${step}_desc`)}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
          <div className="relative min-h-[200px] border-t border-border/40 lg:border-l lg:border-t-0">
            <Image src="/gpu-fleet/deploy-pipeline.svg" alt="" fill className="object-cover" />
            <div className="absolute bottom-4 right-4 hidden md:block w-48 opacity-80">
              <Image src="/gpu-fleet/spot-pulse.svg" alt="" width={480} height={120} className="w-full h-auto rounded-xl" />
            </div>
          </div>
        </div>
      </m.div>

      {/* CTA */}
      <m.div
        className="mt-36 text-center rounded-3xl border border-border/60 bg-surface/40 backdrop-blur-sm px-8 py-20"
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-3xl md:text-4xl font-bold mb-4 tracking-tight">{t("gpus.cta_title")}</h2>
        <p className="text-text-secondary max-w-lg mx-auto mb-10 leading-relaxed">{t("gpus.cta_desc")}</p>
        <div className="flex flex-wrap justify-center gap-4">
          <Link href="/register">
            <Button size="lg" className="px-10 shadow-lg shadow-accent-red/15">
              {t("gpus.cta_start")} <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
          <Link href="/pricing">
            <Button variant="outline" size="lg">{t("gpus.cta_pricing")}</Button>
          </Link>
        </div>
      </m.div>

      {activeGpu && cardsInView && (
        <GpuPriceDock
          gpu={activeGpu}
          priceMode={priceMode}
          onToggleMode={togglePriceMode}
          t={t}
        />
      )}
    </div>
  );
}