"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { AlertCircle, ArrowRight, ChevronRight, Clock, RefreshCw } from "lucide-react";
import { SITE_ASSETS, siteIcon } from "@/lib/brand-assets";
import { useLocale } from "@/lib/locale";
import {
  curateMarketingGpus,
  gpuTierBadge,
  marketingGpuLabel,
  normalizeGpuModel,
  type MarketingGpuRow,
} from "@/lib/marketing-gpu";

const API = process.env.NEXT_PUBLIC_API_URL ?? "";

type LoadState = "loading" | "ready" | "degraded" | "error";
type PriceMode = "spot" | "ondemand";
type SiteTone = "cyan" | "coral" | "gold" | "green" | "violet";

const WORKLOADS = [
  { key: "training", icon: "activity", art: "/gpu-fleet/workload-training.svg", tone: "violet" as const },
  { key: "inference", icon: "sparkle", art: "/gpu-fleet/workload-inference.svg", tone: "cyan" as const },
] as const;

const FLAGSHIP = [
  { key: "b200", tier: "B200", tone: "gold" as const },
  { key: "h100", tier: "H100", tone: "cyan" as const },
  { key: "a100", tier: "A100", tone: "green" as const },
] as const;

function ThemeIcon({ name }: { name: string }) {
  return (
    <>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={siteIcon(name, "dark")} className="site-theme-dark" alt="" aria-hidden />
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={siteIcon(name, "light")} className="site-theme-light" alt="" aria-hidden />
    </>
  );
}

function SectionMarker({ code, label }: { code: string; label: string }) {
  return (
    <div className="site-marker">
      <span className="site-marker-code">[ {code} ]</span>
      <span className="site-marker-line" />
      <span>{label}</span>
    </div>
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

function tierColor(tier: ReturnType<typeof gpuTierBadge>) {
  switch (tier) {
    case "flagship":
      return "var(--gold)";
    case "datacenter":
      return "var(--cyan)";
    case "pro":
      return "var(--violet)";
    default:
      return "var(--green)";
  }
}

function formatPrice(value: number) {
  return value > 0 ? `$${value.toFixed(2)}` : "—";
}

function availabilityMeta(gpu: MarketingGpuRow, t: (key: string) => string) {
  const tier = gpuTierBadge(gpu.gpu_model);
  const isFlagshipOrDc = tier === "flagship" || tier === "datacenter";

  if (gpu.available > 0) {
    return {
      color: "var(--green)",
      label: `${gpu.available} ${t("gpus.available")}`,
      pulse: true,
      tone: "green" as const,
    };
  }

  return {
    color: isFlagshipOrDc ? "var(--cyan)" : "var(--gold)",
    label: isFlagshipOrDc ? t("gpus.on_request_reserved") : t("gpus.on_request"),
    pulse: false,
    tone: isFlagshipOrDc ? ("cyan" as const) : ("gold" as const),
  };
}

function barWidth(value: number, total: number, fallback: number) {
  if (total <= 0) return `${fallback}%`;
  return `${Math.max(18, Math.min(100, Math.round((value / total) * 100)))}%`;
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
      className="site-price-dock"
      style={{ paddingBottom: "max(0.75rem, env(safe-area-inset-bottom))" }}
    >
      <div className="site-container">
        <div className="site-rails site-price-dock-shell">
          <button
            type="button"
            onClick={onToggleMode}
            className="site-price-dock-card"
          >
            <span className="site-icon-box site-price-dock-icon" aria-hidden>
              <ThemeIcon name="gpu" />
            </span>
            <span className="site-price-dock-main">
              <span className="site-price-dock-title">{marketingGpuLabel(gpu.gpu_model)}</span>
              <span className="site-price-dock-copy">{gpu.vram_gb} GB · {label}</span>
            </span>
            <span className="site-price-dock-value">
              <strong>
                {formatPrice(active)}
                <span>/hr</span>
              </strong>
              <span className={priceMode === "spot" && spotOff > 0 ? "site-price-dock-note" : "site-price-dock-note site-price-dock-note-hidden"}>
                −{spotOff > 0 ? spotOff : 0}% vs on-demand
              </span>
            </span>
            <ChevronRight className="site-price-dock-chevron" />
          </button>
          <Link href="/register" className="site-button site-button-primary site-price-dock-cta">
            {t("gpus.deploy")} <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
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
  const regionsLive = useMemo(
    () => new Set(gpus.flatMap((gpu) => gpu.locations)).size,
    [gpus],
  );
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

  const handleRowKeyDown = useCallback((event: React.KeyboardEvent<HTMLTableRowElement>, model: string) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    handleCardInteract(model);
  }, [handleCardInteract]);

  const telemetryBars = [
    {
      label: t("gpus.stat_cheapest"),
      value: `${formatPrice(cheapestSpot)}/hr`,
      width: "100%",
    },
    {
      label: t("gpus.stat_models"),
      value: `${modelsWithStock}/${gpus.length || 0}`,
      width: barWidth(modelsWithStock, gpus.length || 1, 44),
    },
    {
      label: t("gpus.updated"),
      value: lastUpdated ? lastUpdated.toLocaleTimeString() : "—",
      width: liveData ? "82%" : "48%",
    },
  ];

  return (
    <>
      <section className="site-hero">
        <div className="site-grid-bg" aria-hidden />
        <div className="site-container">
          <div className="site-rails site-hero-rails">
            <div style={{ animation: "heroUp .7s ease both" }}>
              <div className="site-pill">
                <span className="site-live-dot" />
                <span>{liveData ? t("gpus.badge_live") : t("gpus.badge_reference")}</span>
              </div>
              <h1 className="site-hero-title">
                <span className="site-gradient-text">{t("gpus.title")}</span>
              </h1>
              <p className="site-hero-copy">{t("gpus.subtitle")}</p>
              <div className="site-hero-actions">
                <Link href="/register" className="site-button site-button-primary">
                  {t("gpus.deploy")} <ArrowRight className="h-3.5 w-3.5" />
                </Link>
                <Link href="/pricing" className="site-button site-button-ghost">
                  {t("gpus.cta_pricing")}
                </Link>
              </div>
            </div>

            <div className="site-telemetry-wrap" aria-label="Live GPU market preview" aria-live="polite">
              <div className="site-telemetry-card">
                <div className="site-telemetry-head">
                  <div className="site-telemetry-model">
                    <span className="site-telemetry-mark">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={SITE_ASSETS.iconGradient} style={{ width: 20, height: 20 }} alt="" aria-hidden />
                    </span>
                    <div>
                      <div className="site-mono" style={{ color: "var(--text)", fontSize: 13, fontWeight: 600 }}>
                        {bestSpotGpu ? marketingGpuLabel(bestSpotGpu.gpu_model) : "RTX 4090"}
                      </div>
                      <div className="site-mono" style={{ color: "var(--text-4)", fontSize: 11 }}>
                        {bestSpotGpu ? `${bestSpotGpu.vram_gb} GB VRAM` : "Live inventory"}
                      </div>
                    </div>
                  </div>
                  <span className="site-live-badge">
                    <span className="site-live-dot" />
                    {liveData ? t("gpus.badge_live") : t("gpus.badge_reference")}
                  </span>
                </div>

                {telemetryBars.map((bar) => (
                  <div key={bar.label} className="site-meter">
                    <div className="site-meter-label">
                      <span style={{ color: "var(--text-4)" }}>{bar.label}</span>
                      <span style={{ color: "var(--text-2)" }}>{bar.value}</span>
                    </div>
                    <div className="site-meter-track">
                      <div className="site-meter-bar" style={{ width: bar.width }} />
                    </div>
                  </div>
                ))}

                <div className="site-telemetry-price">
                  <span className="site-mono" style={{ color: "var(--text-4)", fontSize: 11, textTransform: "uppercase" }}>
                    {t("gpus.stat_models")}
                  </span>
                  <strong style={{ color: "var(--text)", fontSize: 25 }}>{regionsLive || 0} regions</strong>
                </div>
              </div>
            </div>
          </div>

          <div className="site-rails site-kpi-strip">
            <div className="site-kpi">
              <div className="site-kpi-label">{t("gpus.stat_cheapest")}</div>
              <div className="site-kpi-value">${cheapestSpot.toFixed(2)}</div>
            </div>
            <div className="site-kpi">
              <div className="site-kpi-label">{t("gpus.stat_models")}</div>
              <div className="site-kpi-value">{gpus.length || "—"}</div>
            </div>
            <div className="site-kpi">
              <div className="site-kpi-label">{t("gpus.stat_hydro")}</div>
              <div className="site-kpi-value">100%</div>
            </div>
          </div>
        </div>
      </section>

      <div className="site-container">
        <section className="site-rails site-section">
          <div className="site-callout site-callout-grid site-callout-violet">
            <div>
              <div className="site-product-badge" style={{ color: "var(--violet)" }}>{t("gpus.mcp_badge")}</div>
              <h2 className="site-callout-title">{t("gpus.mcp_title")}</h2>
              <p className="site-callout-copy">{t("gpus.mcp_desc")}</p>
            </div>
            <div className="site-callout-actions">
              <span className="site-icon-box" aria-hidden>
                <ThemeIcon name="bot" />
              </span>
              <Link href="/mcp" className="site-button site-button-ghost site-callout-button">
                {t("gpus.mcp_cta")} <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>
        </section>

        <section className="site-rails site-section" style={{ paddingBottom: 0 }}>
          <SectionMarker code="01" label={t("gpus.power_title")} />
          <h2 className="site-section-heading">{t("gpus.power_title")}</h2>
          <p className="site-section-copy">{t("gpus.power_desc")}</p>
          <div className="site-feature-grid site-duo-grid site-section-flush">
            {WORKLOADS.map(({ key, icon, art, tone }) => (
              <article key={key} className="site-feature-card site-media-card" data-tone={tone}>
                <div className="site-media-art">
                  <Image src={art} alt="" fill className="site-media-art-image" />
                </div>
                <div className="site-media-card-body">
                  <div className="site-icon-box">
                    <ThemeIcon name={icon} />
                  </div>
                  <h3 className="site-card-title">{t(`gpus.workload_${key}`)}</h3>
                  <p className="site-card-copy">{t(`gpus.workload_${key}_desc`)}</p>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="site-rails site-section" style={{ paddingBottom: 0 }}>
          <SectionMarker code="02" label={t("gpus.featured_title")} />
          <h2 className="site-section-heading">{t("gpus.featured_title")}</h2>
          <p className="site-section-copy">{t("gpus.featured_desc")}</p>
          <div className="site-foundation-grid site-section-flush">
            {FLAGSHIP.map(({ key, tier, tone }, index) => (
              <article key={key} className="site-foundation-card site-flagship-card" data-tone={tone}>
                <div className="site-product-badge" style={{ color: `var(--${tone})` }}>{tier}</div>
                <h3 className="site-card-title">{tier}</h3>
                <p className="site-card-copy">{t(`gpus.featured_${key}`)}</p>
                {index === 0 ? (
                  <span className="site-chip" data-tone="gold">Latest gen</span>
                ) : null}
              </article>
            ))}
          </div>
        </section>

        <section className="site-rails site-section">
          <SectionMarker code="03" label={t("gpus.sovereign_badge")} />
          <div className="site-feature-grid site-duo-grid">
            <article className="site-feature-card site-story-card">
              <div className="site-icon-box">
                <ThemeIcon name="shield-check" />
              </div>
              <div className="site-product-badge" style={{ color: "var(--cyan)" }}>{t("gpus.sovereign_badge")}</div>
              <h2 className="site-card-title site-story-title">{t("gpus.sovereign_title")}</h2>
              <p className="site-card-copy site-story-copy">{t("gpus.sovereign_desc")}</p>
              <div className="site-story-points">
                {[t("gpus.sovereign_i1"), t("gpus.sovereign_i2"), t("gpus.sovereign_i3")].map((item) => (
                  <p key={item} className="site-product-point">
                    <span style={{ color: "var(--green)" }}>+</span>
                    <span>{item}</span>
                  </p>
                ))}
              </div>
            </article>
            <div className="site-media-panel site-map-panel">
              <Image
                src="/gpu-fleet/canada-sovereign.svg"
                alt={t("gpus.sovereign_map_alt")}
                fill
                className="site-map-image"
              />
              <div className="site-map-footer">
                <span>{t("gpus.sovereign_map_caption")}</span>
                <span className="site-map-live">
                  <span className="site-live-dot" />
                  {t("gpus.sovereign_map_live")}
                </span>
              </div>
            </div>
          </div>
        </section>

        <section className="site-rails site-section">
          <SectionMarker code="04" label={t("gpus.fleet_title")} />
          <div className="site-section-head-row">
            <div>
              <h2 className="site-section-heading site-section-heading-compact">{t("gpus.fleet_title")}</h2>
              <p className="site-section-copy">{t("gpus.fleet_desc")}</p>
              <p className="site-fleet-hint">{t("gpus.fleet_hint")}</p>
            </div>
            <div className="site-fleet-controls" aria-live="polite">
              <div className="site-fleet-updated">
                <Clock className="h-3.5 w-3.5" />
                <span>{lastUpdated ? `${t("gpus.updated")} ${lastUpdated.toLocaleTimeString()}` : "…"}</span>
              </div>
              <button
                type="button"
                onClick={() => void load({ refresh: true })}
                disabled={loadState === "loading"}
                className="site-button site-button-ghost site-refresh-button"
                aria-label={t("gpus.refresh")}
              >
                <RefreshCw className={`h-3.5 w-3.5 ${loadState === "loading" ? "animate-spin" : ""}`} />
                <span>{t("gpus.refresh")}</span>
              </button>
            </div>
          </div>

          {loadState === "degraded" ? (
            <div role="status" className="site-status-banner" data-tone="gold">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <p>{t("gpus.degraded")}</p>
            </div>
          ) : null}

          {loadState === "error" ? (
            <div role="alert" className="site-status-banner" data-tone="coral">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <p>{t("gpus.error")}</p>
            </div>
          ) : null}

          {loadState === "loading" && gpus.length === 0 ? (
            <div className="site-fleet-loading" aria-hidden>
              {[...Array(6)].map((_, index) => (
                <div key={index} className="site-fleet-skeleton" />
              ))}
            </div>
          ) : gpus.length === 0 ? (
            <div className="site-empty-state">
              <span className="site-icon-box" aria-hidden>
                <ThemeIcon name="gpu" />
              </span>
              <p>{t("gpus.empty")}</p>
            </div>
          ) : (
            <div ref={cardsRef} className="site-table-wrap" style={{ marginTop: 36 }}>
              <table className="site-table" aria-label={t("pricing.table_label")}>
                <thead>
                  <tr>
                    <th>{t("pricing.col_gpu")}</th>
                    <th>{t("gpus.available")}</th>
                    <th>{t("pricing.col_vram")}</th>
                    <th>{t("pricing.col_spot")}</th>
                    <th>{t("pricing.col_ondemand")}</th>
                    <th>{t("gpus.updated")}</th>
                  </tr>
                </thead>
                <tbody>
                  {gpus.map((gpu) => {
                    const isActive = gpu.gpu_model === activeModel;
                    const isBest = gpu.gpu_model === bestSpotGpu?.gpu_model && gpu.spot_cad > 0;
                    const availability = availabilityMeta(gpu, t);
                    return (
                      <tr
                        key={gpu.gpu_model}
                        role="button"
                        tabIndex={0}
                        aria-pressed={isActive}
                        onClick={() => handleCardInteract(gpu.gpu_model)}
                        onKeyDown={(event) => handleRowKeyDown(event, gpu.gpu_model)}
                        onMouseEnter={() => {
                          setHoveredModel(gpu.gpu_model);
                          setCyclePrices(true);
                        }}
                        onMouseLeave={() => setHoveredModel(null)}
                        onFocus={() => {
                          setHoveredModel(gpu.gpu_model);
                          setCyclePrices(true);
                        }}
                        onBlur={() => setHoveredModel(null)}
                        className={`site-fleet-row${isActive ? " site-fleet-row-active" : ""}${isBest ? " site-fleet-row-best" : ""}`}
                      >
                        <td className="site-table-feature">
                          <div className="site-fleet-model">
                            <span className="site-fleet-dot" style={{ color: availability.color }} />
                            <div>
                              <div>{marketingGpuLabel(gpu.gpu_model)}</div>
                              <div className="site-table-subtle">
                                {gpu.locations.length > 0 ? gpu.locations.join(" · ") : t("gpus.on_request")}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td>
                          <span className="site-chip" data-tone={availability.tone}>{availability.label}</span>
                        </td>
                        <td>{gpu.vram_gb} GB</td>
                        <td className={priceMode === "spot" ? "site-table-x" : undefined}>{formatPrice(gpu.spot_cad)}</td>
                        <td className={priceMode === "ondemand" ? "site-table-x" : undefined}>{formatPrice(gpu.price_cad)}</td>
                        <td>
                          <span className="site-row-action">
                            {isBest ? "Best value" : t("gpus.card_price_hint")}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="site-rails site-section">
          <SectionMarker code="05" label={t("gpus.velocity_badge")} />
          <div className="site-callout site-callout-split">
            <div>
              <div className="site-product-badge" style={{ color: "var(--violet)" }}>{t("gpus.velocity_badge")}</div>
              <h2 className="site-callout-title">{t("gpus.velocity_title")}</h2>
              <p className="site-callout-copy">{t("gpus.velocity_desc")}</p>
              <div className="site-callout-visual">
                <Image src="/gpu-fleet/deploy-pipeline.svg" alt="" fill className="site-callout-visual-image" />
              </div>
            </div>
            <div className="site-timeline">
              {(["pick", "provision", "pulse"] as const).map((step, index) => (
                <div key={step} className="site-timeline-item">
                  <span className="site-timeline-dot" />
                  <span className="site-timeline-year">0{index + 1}</span>
                  <strong className="site-card-title" style={{ fontSize: 20, marginBottom: 8 }}>{t(`gpus.velocity_${step}`)}</strong>
                  <p className="site-timeline-copy">{t(`gpus.velocity_${step}_desc`)}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="site-rails site-cta">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={SITE_ASSETS.iconGradient} className="site-cta-mark" alt="" aria-hidden />
          <h2 className="site-cta-title">{t("gpus.cta_title")}</h2>
          <p className="site-section-copy" style={{ marginBottom: 28 }}>{t("gpus.cta_desc")}</p>
          <div className="site-hero-actions">
            <Link href="/register" className="site-button site-button-primary" style={{ padding: "15px 28px" }}>
              {t("gpus.cta_start")} <ArrowRight className="h-3.5 w-3.5" />
            </Link>
            <Link href="/pricing" className="site-button site-button-ghost" style={{ padding: "15px 28px" }}>
              {t("gpus.cta_pricing")}
            </Link>
          </div>
        </section>
      </div>

      {activeGpu && cardsInView ? (
        <GpuPriceDock
          gpu={activeGpu}
          priceMode={priceMode}
          onToggleMode={togglePriceMode}
          t={t}
        />
      ) : null}
    </>
  );
}
