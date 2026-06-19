"use client";

import Image from "next/image";
import Link from "next/link";
import {
  ArrowRight, BadgeCheck, BarChart3, Check, DollarSign, FileCheck, Globe,
  Leaf, Lock, Scale, Server, Shield, Users, Zap,
} from "lucide-react";
import { m } from "@/components/marketing/motion";
import { cn } from "@/lib/utils";
import { useLocale } from "@/lib/locale";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: Math.min(i, 4) * 0.06, duration: 0.5, ease: "easeOut" as const },
  }),
};

type Accent = "cyan" | "violet" | "emerald";

const ACCENT: Record<Accent, { badge: string; text: string; glow: string }> = {
  cyan: { badge: "bg-accent-cyan/10 text-accent-cyan ring-accent-cyan/25", text: "text-accent-cyan", glow: "bg-accent-cyan/20" },
  violet: { badge: "bg-accent-violet/10 text-accent-violet ring-accent-violet/25", text: "text-accent-violet", glow: "bg-accent-violet/20" },
  emerald: { badge: "bg-emerald/10 text-emerald ring-emerald/25", text: "text-emerald", glow: "bg-emerald/20" },
};

const PRODUCTS: { key: string; href: string; art: string; accent: Accent }[] = [
  { key: "gpus", href: "/gpu-availability", art: "/gpu-fleet/hero-power.svg", accent: "cyan" },
  { key: "mcp", href: "/mcp", art: "/mcp/hero-agent-gpu.svg", accent: "violet" },
  { key: "serverless", href: "/register", art: "/features/serverless.svg", accent: "cyan" },
  { key: "instances", href: "/register", art: "/features/instances.svg", accent: "violet" },
  { key: "hosting", href: "/register", art: "/features/hosting.svg", accent: "emerald" },
  { key: "xcelai", href: "/register", art: "/features/xcel-ai.svg", accent: "violet" },
  { key: "volumes", href: "/register", art: "/features/volumes.svg", accent: "cyan" },
];

const categoryColors: Record<string, { bg: string; text: string; glow: string }> = {
  "features.cat_sovereignty": { bg: "bg-accent-red/10", text: "text-accent-red", glow: "rgba(220,38,38,0.12)" },
  "features.cat_compute": { bg: "bg-accent-cyan/10", text: "text-accent-cyan", glow: "rgba(0,212,255,0.12)" },
  "features.cat_trust": { bg: "bg-accent-violet/10", text: "text-accent-violet", glow: "rgba(124,58,237,0.12)" },
  "features.cat_billing": { bg: "bg-accent-gold/10", text: "text-accent-gold", glow: "rgba(245,158,11,0.12)" },
};

const featureKeys = [
  { icon: Shield, title: "features.sovereignty_title", desc: "features.sovereignty_desc", cat: "features.cat_sovereignty" },
  { icon: Globe, title: "features.jurisdiction_title", desc: "features.jurisdiction_desc", cat: "features.cat_sovereignty" },
  { icon: Lock, title: "features.cloud_act_title", desc: "features.cloud_act_desc", cat: "features.cat_sovereignty" },
  { icon: Server, title: "features.marketplace_title", desc: "features.marketplace_desc", cat: "features.cat_compute" },
  { icon: Zap, title: "features.spot_title", desc: "features.spot_desc", cat: "features.cat_compute" },
  { icon: BarChart3, title: "features.telemetry_title", desc: "features.telemetry_desc", cat: "features.cat_compute" },
  { icon: BadgeCheck, title: "features.trust_title", desc: "features.trust_desc", cat: "features.cat_trust" },
  { icon: Scale, title: "features.sla_title", desc: "features.sla_desc", cat: "features.cat_trust" },
  { icon: FileCheck, title: "features.compliance_title", desc: "features.compliance_desc", cat: "features.cat_trust" },
  { icon: DollarSign, title: "features.billing_title", desc: "features.billing_desc", cat: "features.cat_billing" },
  { icon: Users, title: "features.payouts_title", desc: "features.payouts_desc", cat: "features.cat_billing" },
  { icon: Leaf, title: "features.green_title", desc: "features.green_desc", cat: "features.cat_billing" },
];

export function FeaturesContent() {
  const { t } = useLocale();

  return (
    <div className="relative overflow-hidden">
      {/* Hero */}
      <div className="mx-auto max-w-7xl px-6 pt-28 pb-10 text-center">
        <m.div initial="hidden" whileInView="visible" viewport={{ once: true }}>
          <m.p variants={fadeUp} custom={0} className="text-sm font-semibold uppercase tracking-widest text-accent-cyan">
            {t("features.platform_eyebrow")}
          </m.p>
          <m.h1 variants={fadeUp} custom={1} className="mt-3 text-4xl font-bold md:text-5xl">
            {t("features.title_1")}{" "}
            <span className="bg-gradient-to-r from-accent-cyan via-accent-violet to-accent-red bg-clip-text text-transparent">
              {t("features.title_accent")}
            </span>
          </m.h1>
          <m.p variants={fadeUp} custom={2} className="mx-auto mt-4 max-w-2xl text-lg text-text-secondary">
            {t("features.platform_subtitle")}
          </m.p>
        </m.div>
      </div>

      {/* Product spotlights */}
      <div className="mx-auto max-w-6xl space-y-24 px-6 py-16 sm:space-y-28">
        {PRODUCTS.map((p, i) => {
          const a = ACCENT[p.accent];
          const reverse = i % 2 === 1;
          return (
            <m.section
              key={p.key}
              variants={fadeUp}
              custom={0}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: "-80px" }}
              className="grid items-center gap-10 lg:grid-cols-2 lg:gap-16"
            >
              <div className={cn("relative", reverse && "lg:order-2")}>
                <div className={cn("pointer-events-none absolute -inset-6 rounded-[2rem] opacity-40 blur-3xl", a.glow)} aria-hidden />
                <Image
                  src={p.art}
                  alt=""
                  width={480}
                  height={320}
                  className="relative w-full rounded-2xl border border-border/60 shadow-2xl shadow-black/40"
                />
              </div>
              <div>
                <span className={cn("inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wider ring-1", a.badge)}>
                  {t(`features.prod_${p.key}_badge`)}
                </span>
                <h2 className="mt-4 text-2xl font-bold tracking-tight sm:text-3xl">{t(`features.prod_${p.key}_title`)}</h2>
                <p className="mt-3 leading-relaxed text-text-secondary">{t(`features.prod_${p.key}_desc`)}</p>
                <ul className="mt-5 space-y-2.5">
                  {(["b1", "b2"] as const).map((b) => (
                    <li key={b} className="flex items-start gap-2.5 text-sm text-text-primary">
                      <Check className={cn("mt-0.5 h-4 w-4 shrink-0", a.text)} />
                      {t(`features.prod_${p.key}_${b}`)}
                    </li>
                  ))}
                </ul>
                <Link href={p.href} className={cn("mt-6 inline-flex items-center gap-1.5 text-sm font-semibold transition-transform hover:gap-2.5", a.text)}>
                  {t(`features.prod_${p.key}_cta`)} <ArrowRight className="h-4 w-4" />
                </Link>
              </div>
            </m.section>
          );
        })}
      </div>

      {/* Platform foundations grid */}
      <div className="mx-auto max-w-7xl px-6 py-20">
        <h2 className="mb-10 text-center text-2xl font-bold tracking-tight sm:text-3xl">
          {t("features.everything_title")}
        </h2>
        <m.div
          className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          {featureKeys.map((f, idx) => {
            const colors = categoryColors[f.cat] ?? { bg: "bg-accent-cyan/10", text: "text-accent-cyan", glow: "rgba(0,212,255,0.08)" };
            return (
              <m.div
                key={f.title}
                variants={fadeUp}
                custom={idx}
                className="group glow-card rounded-xl p-6"
                style={{ "--glow-color": colors.glow } as React.CSSProperties}
              >
                <div className="mb-4 flex items-center gap-3">
                  <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${colors.bg} transition-transform group-hover:scale-110`}>
                    <f.icon className={`h-5 w-5 ${colors.text}`} />
                  </div>
                  <span className={`text-xs font-medium uppercase tracking-wider ${colors.text}`}>{t(f.cat)}</span>
                </div>
                <h3 className="mb-2 text-lg font-semibold">{t(f.title)}</h3>
                <p className="text-sm leading-relaxed text-text-secondary">{t(f.desc)}</p>
              </m.div>
            );
          })}
        </m.div>
      </div>
    </div>
  );
}
