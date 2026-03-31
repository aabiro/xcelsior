"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowRight, Check, DollarSign } from "lucide-react";
import { motion } from "framer-motion";
import { SavingsCalculator } from "./calculator";
import { useLocale } from "@/lib/locale";

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

  return (
    <div className="mx-auto max-w-7xl px-6 py-28">
      {/* Header */}
      <motion.div
        className="text-center mb-16"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <motion.div variants={fadeUp} custom={0} className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent-gold/30 bg-accent-gold/10 px-4 py-1.5 backdrop-blur-sm">
          <DollarSign className="h-3 w-3 text-accent-gold" />
          <span className="text-xs font-medium text-accent-gold">
            {t("pricing.badge")}
          </span>
        </motion.div>
        <motion.h1 variants={fadeUp} custom={1} className="text-4xl font-bold md:text-5xl">
          {t("pricing.title")}
        </motion.h1>
        <motion.p variants={fadeUp} custom={2} className="mt-4 text-lg text-text-secondary max-w-2xl mx-auto">
          {t("pricing.subtitle")}
        </motion.p>
      </motion.div>

      {/* GPU Pricing Table */}
      <motion.div
        className="overflow-x-auto mb-20 rounded-xl border border-border bg-surface/50 backdrop-blur-sm"
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
            {gpus.map((gpu) => (
              <tr key={gpu.model} className="border-b border-border/50 hover:bg-surface-hover transition-colors">
                <td className="py-4 pl-6 pr-4 font-medium">{gpu.model}</td>
                <td className="py-4 px-4 text-center text-text-secondary">{gpu.vram}GB</td>
                <td className="py-4 px-4 text-center font-mono">${gpu.onDemand.toFixed(2)}/hr</td>
                <td className="py-4 px-4 text-center font-mono text-emerald">${gpu.spot.toFixed(2)}/hr</td>
                <td className="py-4 px-4 text-center font-mono text-accent-cyan">${gpu.reserved1m.toFixed(2)}/hr</td>
                <td className="py-4 px-4 text-center font-mono text-accent-gold">${gpu.reserved1y.toFixed(2)}/hr</td>
              </tr>
            ))}
          </tbody>
        </table>
      </motion.div>

      {/* AI Compute Fund Callout */}
      <motion.div
        className="glow-card rounded-xl p-8 md:p-12 mb-20"
        style={{ "--glow-color": "rgba(245,158,11,0.12)" } as React.CSSProperties}
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex flex-col md:flex-row items-start gap-8">
          <div className="flex-1">
            <h2 className="text-2xl font-bold mb-3">{t("pricing.fund_title")}</h2>
            <p className="text-text-secondary leading-relaxed mb-4">
              {t("pricing.fund_desc")}
            </p>
            <ul className="space-y-2">
              <li className="flex items-center gap-2 text-sm text-text-secondary">
                <Check className="h-4 w-4 text-emerald" /> {t("pricing.fund_i1")}
              </li>
              <li className="flex items-center gap-2 text-sm text-text-secondary">
                <Check className="h-4 w-4 text-emerald" /> {t("pricing.fund_i2")}
              </li>
              <li className="flex items-center gap-2 text-sm text-text-secondary">
                <Check className="h-4 w-4 text-emerald" /> {t("pricing.fund_i3")}
              </li>
            </ul>
          </div>
          <div className="rounded-xl border border-accent-gold/30 bg-accent-gold/5 p-6 text-center min-w-[200px]">
            <p className="text-sm text-text-secondary mb-1">{t("pricing.fund_effective")}</p>
            <p className="text-4xl font-bold font-mono bg-gradient-to-r from-accent-gold to-accent-red bg-clip-text text-transparent">{t("pricing.fund_effective_price")}</p>
            <p className="text-xs text-text-muted">{t("pricing.fund_effective_unit")}</p>
          </div>
        </div>
      </motion.div>

      {/* Plans */}
      <motion.div
        className="grid grid-cols-1 gap-8 md:grid-cols-3 mb-20"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <PlanCard
          t={t}
          i={0}
          name={t("pricing.tier_ondemand_title")}
          description={t("pricing.tier_ondemand_desc")}
          price={t("pricing.tier_ondemand_from")}
          unit={t("pricing.tier_ondemand_unit")}
          features={[t("pricing.tier_ondemand_i1"), t("pricing.tier_ondemand_i2"), t("pricing.tier_ondemand_i3"), t("pricing.tier_ondemand_i4")]}
        />
        <PlanCard
          t={t}
          i={1}
          name={t("pricing.tier_reserved_title")}
          description={t("pricing.tier_reserved_desc")}
          price={t("pricing.tier_reserved_from")}
          unit={t("pricing.tier_ondemand_unit")}
          features={[t("pricing.tier_reserved_i1"), t("pricing.tier_reserved_i2"), t("pricing.tier_reserved_i3"), t("pricing.tier_reserved_i4")]}
          highlighted
          badge={t("pricing.tier_reserved_badge")}
        />
        <PlanCard
          t={t}
          i={2}
          name={t("pricing.tier_sovereign_title")}
          description={t("pricing.tier_sovereign_desc")}
          price={t("pricing.tier_sovereign_from")}
          unit=""
          features={[t("pricing.tier_sovereign_i1"), t("pricing.tier_sovereign_i2"), t("pricing.tier_sovereign_i3"), t("pricing.tier_sovereign_i4")]}
        />
      </motion.div>

      {/* Savings Calculator */}
      <div className="mb-20">
        <h2 className="text-2xl font-bold text-center mb-8">{t("pricing.savings_calculator")}</h2>
        <SavingsCalculator gpus={gpus.map((g) => ({ model: g.model, onDemand: g.onDemand }))} />
      </div>

      {/* CTA */}
      <motion.div
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
      </motion.div>
    </div>
  );
}

function PlanCard({
  t,
  i,
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
  name: string;
  description: string;
  price: string;
  unit: string;
  features: string[];
  highlighted?: boolean;
  badge?: string;
}) {
  return (
    <motion.div
      variants={fadeUp}
      custom={i}
      className={`glow-card rounded-xl p-8 relative ${highlighted ? "ring-1 ring-accent-gold/40" : ""}`}
      style={{
        "--glow-color": highlighted
          ? "rgba(245,158,11,0.15)"
          : "rgba(0,212,255,0.08)",
      } as React.CSSProperties}
    >
      {badge && (
        <span className="mb-4 inline-block rounded-full bg-accent-gold/20 px-3 py-0.5 text-xs font-medium text-accent-gold">
          {badge}
        </span>
      )}
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
    </motion.div>
  );
}
