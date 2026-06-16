"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowRight, Check, DollarSign, Zap, CalendarClock, ShieldCheck, TrendingDown, Leaf, Cpu } from "lucide-react";
import { m } from "@/components/marketing/motion";
import { AuroraBackground } from "@/components/ui/aurora-bg";
import { useLocale } from "@/lib/locale";
import posthog from "posthog-js";
import { cn } from "@/lib/utils";
import { gpuTierBadge, marketingGpuLabel } from "@/lib/marketing-gpu";

const SavingsCalculator = dynamic(
  () => import("./calculator").then((mod) => mod.SavingsCalculator),
  { loading: () => <div className="h-48 animate-pulse rounded-xl bg-surface/50" aria-hidden /> },
);

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5, ease: "easeOut" as const },
  }),
};

interface GpuRow {
  model: string;
  vram: number;
  onDemand: number;
  spot: number;
  reserved1m: number;
  reserved1y: number;
}

const TIER_STYLES = {
  flagship: "border-accent-gold/30 bg-gradient-to-br from-accent-gold/10 to-surface/40",
  datacenter: "border-accent-cyan/25 bg-gradient-to-br from-accent-cyan/8 to-surface/40",
  pro: "border-accent-violet/25 bg-gradient-to-br from-accent-violet/8 to-surface/40",
  value: "border-border/60 bg-surface/40",
} as const;

export function PricingContent({ gpus }: { gpus: GpuRow[] }) {
  const { t } = useLocale();
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    posthog.capture("pricing_page_viewed", { gpu_count: gpus.length });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const cheapestSpot = gpus.length ? Math.min(...gpus.map((g) => g.spot)) : 0.3;
  const maxSpotSaving = gpus.reduce((best, g) => {
    if (!g.onDemand) return best;
    return Math.max(best, Math.round((1 - g.spot / g.onDemand) * 100));
  }, 0);
  const maxReservedSaving = gpus.reduce((best, g) => {
    if (!g.onDemand) return best;
    return Math.max(best, Math.round((1 - g.reserved1y / g.onDemand) * 100));
  }, 0);
  const bestModel = gpus.length
    ? gpus.reduce((a, b) => (a.spot <= b.spot ? a : b)).model
    : "";

  const grouped = useMemo(() => {
    const order = ["flagship", "datacenter", "pro", "value"] as const;
    const buckets: Record<(typeof order)[number], GpuRow[]> = {
      flagship: [],
      datacenter: [],
      pro: [],
      value: [],
    };
    for (const gpu of gpus) {
      buckets[gpuTierBadge(gpu.model)].push(gpu);
    }
    return order
      .map((tier) => ({ tier, rows: buckets[tier] }))
      .filter((g) => g.rows.length > 0);
  }, [gpus]);

  return (
    <div className="relative mx-auto max-w-7xl px-6 py-28">
      <AuroraBackground className="-z-10 opacity-60" />

      <m.div
        className="text-center mb-10"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <m.div variants={fadeUp} custom={0} className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent-gold/30 bg-accent-gold/10 px-4 py-1.5 backdrop-blur-sm">
          <DollarSign className="h-3 w-3 text-accent-gold" />
          <span className="text-xs font-medium text-accent-gold">{t("pricing.badge")}</span>
        </m.div>
        <m.h1 variants={fadeUp} custom={1} className="text-4xl font-bold md:text-5xl">{t("pricing.title")}</m.h1>
        <m.p variants={fadeUp} custom={2} className="mt-4 text-lg text-text-secondary max-w-2xl mx-auto">
          {t("pricing.subtitle")}
        </m.p>
      </m.div>

      <m.div
        className="mb-16 grid grid-cols-1 gap-4 sm:grid-cols-3 max-w-3xl mx-auto"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <HeroStat i={0} icon={DollarSign} accent="gold" value={`$${cheapestSpot.toFixed(2)}`} suffix="CAD/hr" label={t("pricing.col_spot")} />
        <HeroStat i={1} icon={TrendingDown} accent="emerald" value={`${maxSpotSaving || 70}%`} suffix="off" label={t("pricing.stat_spot_save")} />
        <HeroStat i={2} icon={Leaf} accent="cyan" value="100%" suffix="hydro" label="Clean power" />
      </m.div>

      {/* Curated GPU cards */}
      <m.div
        className="mb-20 space-y-12"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <div className="text-center max-w-2xl mx-auto">
          <h2 className="text-2xl font-bold tracking-tight">{t("pricing.fleet_title")}</h2>
          <p className="mt-2 text-sm text-text-secondary">{t("pricing.fleet_desc")}</p>
        </div>

        {grouped.map(({ tier, rows }) => (
          <div key={tier}>
            <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-text-muted">
              {t(`pricing.tier_${tier}`)}
            </p>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {rows.map((gpu) => {
                const spotOff = gpu.onDemand ? Math.round((1 - gpu.spot / gpu.onDemand) * 100) : 0;
                const isBest = gpu.model === bestModel;
                const isOpen = expanded === gpu.model;
                return (
                  <button
                    key={gpu.model}
                    type="button"
                    onClick={() => setExpanded(isOpen ? null : gpu.model)}
                    className={cn(
                      "group relative rounded-2xl border p-5 text-left transition-all hover:-translate-y-0.5 hover:shadow-lg hover:shadow-accent-cyan/5",
                      TIER_STYLES[tier],
                      isBest && "ring-1 ring-emerald/30",
                      isOpen && "ring-2 ring-accent-cyan/30",
                    )}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-background/50 ring-1 ring-border/60">
                          <Cpu className="h-4 w-4 text-accent-cyan" />
                        </span>
                        <div>
                          <p className="font-semibold tracking-tight">{marketingGpuLabel(gpu.model)}</p>
                          <p className="text-xs text-text-muted">{gpu.vram} GB VRAM</p>
                        </div>
                      </div>
                      {isBest && (
                        <span className="rounded-full bg-emerald/15 px-2 py-0.5 text-[10px] font-semibold uppercase text-emerald">
                          Best value
                        </span>
                      )}
                    </div>

                    <div className="mt-5 flex items-end justify-between gap-4">
                      <div>
                        <p className="text-[10px] font-medium uppercase tracking-wider text-emerald/90">{t("pricing.col_spot")}</p>
                        <p className="text-3xl font-bold tabular-nums text-emerald">
                          ${gpu.spot.toFixed(2)}
                          <span className="text-sm font-normal text-text-muted">/hr</span>
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-[10px] font-medium uppercase tracking-wider text-text-muted">{t("pricing.col_ondemand")}</p>
                        <p className="text-lg font-mono tabular-nums text-text-secondary">
                          ${gpu.onDemand.toFixed(2)}/hr
                        </p>
                        {spotOff > 0 && (
                          <p className="text-[10px] font-semibold text-emerald">−{spotOff}%</p>
                        )}
                      </div>
                    </div>

                    <div
                      className={cn(
                        "grid transition-all duration-300 ease-out",
                        isOpen ? "mt-4 grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
                      )}
                    >
                      <div className="overflow-hidden">
                        <div className="flex gap-3 border-t border-border/50 pt-4">
                          <div className="flex-1 rounded-lg bg-background/40 px-3 py-2">
                            <p className="text-[10px] uppercase tracking-wider text-accent-cyan">{t("pricing.col_reserved1")}</p>
                            <p className="font-mono text-sm">${gpu.reserved1m.toFixed(2)}/hr</p>
                          </div>
                          <div className="flex-1 rounded-lg bg-background/40 px-3 py-2">
                            <p className="text-[10px] uppercase tracking-wider text-accent-gold">{t("pricing.col_reserved12")}</p>
                            <p className="font-mono text-sm">${gpu.reserved1y.toFixed(2)}/hr</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <p className="mt-3 text-[11px] text-text-muted opacity-70 group-hover:opacity-100">
                      {isOpen ? t("pricing.card_collapse") : t("pricing.card_expand")}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </m.div>

      {/* Plans */}
      <m.div
        className="grid grid-cols-1 gap-8 md:grid-cols-3 mb-20"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <PlanCard
          t={t}
          i={0}
          icon={Zap}
          accent="cyan"
          name={t("pricing.tier_ondemand_title")}
          description={t("pricing.tier_ondemand_desc")}
          price={t("pricing.tier_ondemand_from")}
          unit={t("pricing.tier_ondemand_unit")}
          features={[t("pricing.tier_ondemand_i1"), t("pricing.tier_ondemand_i2"), t("pricing.tier_ondemand_i3"), t("pricing.tier_ondemand_i4")]}
        />
        <PlanCard
          t={t}
          i={1}
          icon={CalendarClock}
          accent="gold"
          name={t("pricing.tier_reserved_title")}
          description={t("pricing.tier_reserved_desc")}
          price={t("pricing.tier_reserved_from")}
          unit={t("pricing.tier_ondemand_unit")}
          features={[t("pricing.tier_reserved_i1"), t("pricing.tier_reserved_i2"), t("pricing.tier_reserved_i3"), t("pricing.tier_reserved_i4")]}
          highlighted
          badge={maxReservedSaving > 0 ? `${t("pricing.tier_reserved_badge")} · −${maxReservedSaving}%` : t("pricing.tier_reserved_badge")}
        />
        <PlanCard
          t={t}
          i={2}
          icon={ShieldCheck}
          accent="violet"
          name={t("pricing.tier_sovereign_title")}
          description={t("pricing.tier_sovereign_desc")}
          price={t("pricing.tier_sovereign_from")}
          unit=""
          features={[t("pricing.tier_sovereign_i1"), t("pricing.tier_sovereign_i2"), t("pricing.tier_sovereign_i3"), t("pricing.tier_sovereign_i4")]}
        />
      </m.div>

      <div className="mb-20">
        <h2 className="text-2xl font-bold text-center mb-8">{t("pricing.savings_calculator")}</h2>
        <SavingsCalculator gpus={gpus.map((g) => ({ model: g.model, onDemand: g.onDemand }))} />
      </div>

      <m.div
        className="text-center"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <Link href="/register">
          <Button size="lg" className="text-base px-10 shadow-lg shadow-accent-cyan/10">
            {t("pricing.cta_start")} <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </m.div>
    </div>
  );
}

const ACCENT_RING: Record<string, string> = {
  cyan: "text-accent-cyan bg-accent-cyan/10 ring-accent-cyan/20",
  gold: "text-accent-gold bg-accent-gold/10 ring-accent-gold/20",
  violet: "text-accent-violet bg-accent-violet/10 ring-accent-violet/20",
  emerald: "text-emerald bg-emerald/10 ring-emerald/20",
};

function HeroStat({
  i, icon: Icon, accent, value, suffix, label,
}: {
  i: number;
  icon: typeof DollarSign;
  accent: string;
  value: string;
  suffix: string;
  label: string;
}) {
  return (
    <m.div variants={fadeUp} custom={i} className="glow-card flex items-center gap-3 rounded-xl p-4">
      <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ring-1 ${ACCENT_RING[accent]}`}>
        <Icon className="h-5 w-5" />
      </span>
      <div className="min-w-0">
        <p className="font-mono text-xl font-bold leading-none">
          {value} <span className="text-xs font-normal text-text-muted">{suffix}</span>
        </p>
        <p className="mt-1 truncate text-xs text-text-muted">{label}</p>
      </div>
    </m.div>
  );
}

function PlanCard({
  t, i, icon: Icon, accent, name, description, price, unit, features, highlighted, badge,
}: {
  t: (key: string) => string;
  i: number;
  icon: typeof DollarSign;
  accent: string;
  name: string;
  description: string;
  price: string;
  unit: string;
  features: string[];
  highlighted?: boolean;
  badge?: string;
}) {
  return (
    <m.div
      variants={fadeUp}
      custom={i}
      className={`glow-card rounded-xl p-8 relative transition-transform hover:-translate-y-1 ${highlighted ? "ring-1 ring-accent-gold/40 brand-top-accent" : ""}`}
      style={{
        "--glow-color": highlighted ? "rgba(245,158,11,0.15)" : "rgba(0,212,255,0.08)",
      } as React.CSSProperties}
    >
      <div className="mb-4 flex items-center justify-between">
        <span className={`flex h-11 w-11 items-center justify-center rounded-xl ring-1 ${ACCENT_RING[accent]}`}>
          <Icon className="h-5 w-5" />
        </span>
        {badge && (
          <span className="rounded-full bg-accent-gold/20 px-3 py-0.5 text-xs font-medium text-accent-gold">{badge}</span>
        )}
      </div>
      <h3 className="text-xl font-bold">{name}</h3>
      <p className="mt-1 text-sm text-text-secondary">{description}</p>
      <div className="my-6">
        <span className="text-3xl font-bold font-mono">{price}</span>
        <span className="text-text-muted">{unit}</span>
      </div>
      <ul className="space-y-2 mb-6">
        {features.map((f) => (
          <li key={f} className="flex items-center gap-2 text-sm text-text-secondary">
            <Check className="h-4 w-4 text-emerald shrink-0" />
            {f}
          </li>
        ))}
      </ul>
      <Link href="/register">
        <Button variant={highlighted ? "gold" : "outline"} className="w-full">
          {t("pricing.cta_get_started")}
        </Button>
      </Link>
    </m.div>
  );
}