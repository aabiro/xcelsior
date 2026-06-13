"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowRight, Check, DollarSign, Zap, CalendarClock, ShieldCheck, TrendingDown, Leaf } from "lucide-react";
import { m } from "@/components/marketing/motion";
import { AuroraBackground } from "@/components/ui/aurora-bg";
import { useLocale } from "@/lib/locale";

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

export function PricingContent({ gpus }: { gpus: GpuRow[] }) {
  const { t } = useLocale();

  // Headline numbers computed from live data so the hero never drifts from the table.
  const cheapestSpot = gpus.length ? Math.min(...gpus.map((g) => g.spot)) : 0.3;
  const maxSpotSaving = gpus.reduce((best, g) => {
    if (!g.onDemand) return best;
    return Math.max(best, Math.round((1 - g.spot / g.onDemand) * 100));
  }, 0);
  const maxReservedSaving = gpus.reduce((best, g) => {
    if (!g.onDemand) return best;
    return Math.max(best, Math.round((1 - g.reserved1y / g.onDemand) * 100));
  }, 0);
  // Cheapest spot row gets a "best value" highlight.
  const bestModel = gpus.length
    ? gpus.reduce((a, b) => (a.spot <= b.spot ? a : b)).model
    : "";

  return (
    <div className="relative mx-auto max-w-7xl px-6 py-28">
      <AuroraBackground className="-z-10 opacity-60" />

      {/* Header */}
      <m.div
        className="text-center mb-10"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <m.div variants={fadeUp} custom={0} className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent-gold/30 bg-accent-gold/10 px-4 py-1.5 backdrop-blur-sm">
          <DollarSign className="h-3 w-3 text-accent-gold" />
          <span className="text-xs font-medium text-accent-gold">
            {t("pricing.badge")}
          </span>
        </m.div>
        <m.h1 variants={fadeUp} custom={1} className="text-4xl font-bold md:text-5xl">
          {t("pricing.title")}
        </m.h1>
        <m.p variants={fadeUp} custom={2} className="mt-4 text-lg text-text-secondary max-w-2xl mx-auto">
          {t("pricing.subtitle")}
        </m.p>
      </m.div>

      {/* Headline stat strip — pure pizzazz, numbers derived from live pricing */}
      <m.div
        className="mb-16 grid grid-cols-1 gap-4 sm:grid-cols-3 max-w-3xl mx-auto"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <HeroStat i={0} icon={DollarSign} accent="gold" value={`$${cheapestSpot.toFixed(2)}`} suffix="CAD/hr" label={t("pricing.col_spot")} />
        <HeroStat i={1} icon={TrendingDown} accent="emerald" value={`${maxSpotSaving || 70}%`} suffix="off" label={t("pricing.col_spot")} />
        <HeroStat i={2} icon={Leaf} accent="cyan" value="100%" suffix="hydro" label="Clean power" />
      </m.div>

      <p className="mb-3 text-center text-sm text-text-muted md:hidden" role="note">
        {t("pricing.scroll_hint")}
      </p>

      {/* GPU Pricing Table */}
      <m.div
        className="overflow-x-auto mb-20 rounded-xl border border-border bg-surface/50 backdrop-blur-sm scroll-smooth"
        tabIndex={0}
        aria-label={t("pricing.table_label")}
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="py-4 pl-6 pr-4 text-left font-medium text-text-secondary">{t("pricing.col_gpu")}</th>
              <th className="py-4 px-4 text-center font-medium text-text-secondary">{t("pricing.col_vram")}</th>
              <th className="py-4 px-4 text-center font-medium text-text-primary">{t("pricing.col_ondemand")}</th>
              <th className="py-4 px-4 text-center font-medium text-emerald">{t("pricing.col_spot")}</th>
              <th className="py-4 px-4 text-center font-medium text-accent-cyan">{t("pricing.col_reserved1")}</th>
              <th className="py-4 px-4 text-center font-medium text-accent-gold">{t("pricing.col_reserved12")}</th>
            </tr>
          </thead>
          <tbody>
            {gpus.map((gpu) => {
              const spotOff = gpu.onDemand ? Math.round((1 - gpu.spot / gpu.onDemand) * 100) : 0;
              const isBest = gpu.model === bestModel;
              return (
                <tr
                  key={gpu.model}
                  className={`border-b border-border/50 transition-colors ${isBest ? "bg-emerald/[0.06] hover:bg-emerald/10" : "hover:bg-surface-hover"}`}
                >
                  <td className="py-4 pl-6 pr-4 font-medium">
                    <span className="inline-flex items-center gap-2">
                      {gpu.model}
                      {isBest && (
                        <span className="rounded-full bg-emerald/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald">
                          Best value
                        </span>
                      )}
                    </span>
                  </td>
                  <td className="py-4 px-4 text-center text-text-secondary">{gpu.vram}GB</td>
                  <td className="py-4 px-4 text-center font-mono">${gpu.onDemand.toFixed(2)}/hr</td>
                  <td className="py-4 px-4 text-center font-mono text-emerald">
                    ${gpu.spot.toFixed(2)}/hr
                    {spotOff > 0 && (
                      <span className="ml-1.5 rounded bg-emerald/12 px-1.5 py-0.5 text-[10px] font-semibold text-emerald align-middle">
                        −{spotOff}%
                      </span>
                    )}
                  </td>
                  <td className="py-4 px-4 text-center font-mono text-accent-cyan">${gpu.reserved1m.toFixed(2)}/hr</td>
                  <td className="py-4 px-4 text-center font-mono text-accent-gold">${gpu.reserved1y.toFixed(2)}/hr</td>
                </tr>
              );
            })}
          </tbody>
        </table>
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

      {/* Savings Calculator */}
      <div className="mb-20">
        <h2 className="text-2xl font-bold text-center mb-8">{t("pricing.savings_calculator")}</h2>
        <SavingsCalculator gpus={gpus.map((g) => ({ model: g.model, onDemand: g.onDemand }))} />
      </div>

      {/* CTA */}
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
    <m.div
      variants={fadeUp}
      custom={i}
      className="glow-card flex items-center gap-3 rounded-xl p-4"
    >
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
  t,
  i,
  icon: Icon,
  accent,
  name,
  description,
  price,
  unit,
  features,
  highlighted,
  badge,
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
        "--glow-color": highlighted
          ? "rgba(245,158,11,0.15)"
          : "rgba(0,212,255,0.08)",
      } as React.CSSProperties}
    >
      <div className="mb-4 flex items-center justify-between">
        <span className={`flex h-11 w-11 items-center justify-center rounded-xl ring-1 ${ACCENT_RING[accent]}`}>
          <Icon className="h-5 w-5" />
        </span>
        {badge && (
          <span className="rounded-full bg-accent-gold/20 px-3 py-0.5 text-xs font-medium text-accent-gold">
            {badge}
          </span>
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
