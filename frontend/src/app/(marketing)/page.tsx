"use client";

import Link from "next/link";
import {
  Shield,
  Zap,
  Scale,
  Server,
  BadgeCheck,
  BarChart3,
  Globe,
  Leaf,
  DollarSign,
  ArrowRight,
  MapPin,
  Check,
} from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { useLocale } from "@/lib/locale";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5, ease: "easeOut" as const },
  }),
};

export default function HomePage() {
  const { t } = useLocale();
  return (
    <>
      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <section className="aurora-gradient relative overflow-hidden">
        {/* Floating aurora particles */}
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute left-[10%] top-[15%] h-64 w-64 rounded-full bg-accent-cyan/8 blur-[100px] animate-[aurora-drift_8s_ease-in-out_infinite]" />
          <div className="absolute right-[15%] top-[25%] h-48 w-48 rounded-full bg-accent-violet/6 blur-[80px] animate-[aurora-drift_10s_ease-in-out_infinite_2s]" />
          <div className="absolute left-[40%] bottom-[10%] h-56 w-56 rounded-full bg-accent-red/5 blur-[90px] animate-[aurora-drift_12s_ease-in-out_infinite_4s]" />
        </div>

        <div className="relative mx-auto max-w-7xl px-6 py-28 md:py-40">
          <motion.div
            className="max-w-3xl"
            initial="hidden"
            animate="visible"
          >
            <motion.div variants={fadeUp} custom={0} className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent-gold/30 bg-accent-gold/10 px-4 py-1.5 backdrop-blur-sm">
              <MapPin className="h-3 w-3 text-accent-gold" />
              <span className="text-xs font-medium text-accent-gold">
                {t("home.badge")}
              </span>
            </motion.div>

            <motion.h1 variants={fadeUp} custom={1} className="text-5xl font-bold leading-[1.08] tracking-tight md:text-6xl lg:text-7xl">
              {t("home.hero_line1")}
              <br />
              {t("home.hero_line2")}{" "}
              <span className="bg-gradient-to-r from-accent-cyan via-accent-violet to-accent-red bg-clip-text text-transparent">
                {t("home.hero_accent")}
              </span>
            </motion.h1>

            <motion.p variants={fadeUp} custom={2} className="mt-6 text-lg text-text-secondary leading-relaxed max-w-2xl">
              {t("home.hero_desc")}
            </motion.p>

            <motion.div variants={fadeUp} custom={3} className="mt-10 flex flex-wrap gap-4">
              <Link href="/register">
                <Button size="lg" className="text-base px-8 shadow-lg shadow-accent-cyan/10">
                  {t("home.cta_start")}
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link href="/pricing">
                <Button variant="outline" size="lg" className="text-base px-8 border-border-light">
                  {t("home.cta_pricing")}
                </Button>
              </Link>
            </motion.div>

            <motion.div variants={fadeUp} custom={4} className="mt-14 flex flex-wrap gap-8 text-sm text-text-secondary">
              <div className="flex items-center gap-2">
                <div className="flex h-7 w-7 items-center justify-center rounded-full bg-accent-gold/10">
                  <DollarSign className="h-3.5 w-3.5 text-accent-gold" />
                </div>
                {t("home.stat_price")}
              </div>
              <div className="flex items-center gap-2">
                <div className="flex h-7 w-7 items-center justify-center rounded-full bg-emerald/10">
                  <Shield className="h-3.5 w-3.5 text-emerald" />
                </div>
                {t("home.stat_pipeda")}
              </div>
              <div className="flex items-center gap-2">
                <div className="flex h-7 w-7 items-center justify-center rounded-full bg-emerald/10">
                  <Leaf className="h-3.5 w-3.5 text-emerald" />
                </div>
                {t("home.stat_hydro")}
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ── Value Propositions ────────────────────────────────────────── */}
      <div className="brand-line" />
      <section className="bg-navy py-28">
        <div className="mx-auto max-w-7xl px-6">
          <motion.div
            className="text-center mb-16"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            <motion.h2 variants={fadeUp} custom={0} className="text-3xl font-bold md:text-4xl">
              {t("home.why_title")}
            </motion.h2>
            <motion.p variants={fadeUp} custom={1} className="mt-4 text-text-secondary max-w-2xl mx-auto">
              {t("home.why_desc")}
            </motion.p>
          </motion.div>

          <motion.div
            className="grid grid-cols-1 gap-8 md:grid-cols-3"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            <ValueCard i={0} icon={Shield} title={t("home.val_sovereignty_title")} description={t("home.val_sovereignty_desc")} glowColor="rgba(220,38,38,0.12)" accent="accent-red" />
            <ValueCard i={1} icon={Scale} title={t("home.val_compliance_title")} description={t("home.val_compliance_desc")} glowColor="rgba(245,158,11,0.12)" accent="accent-gold" />
            <ValueCard i={2} icon={Zap} title={t("home.val_pricing_title")} description={t("home.val_pricing_desc")} glowColor="rgba(16,185,129,0.12)" accent="emerald" />
          </motion.div>
        </div>
      </section>

      {/* ── Feature Grid ─────────────────────────────────────────────── */}
      <section className="border-t border-border bg-navy-light/30 py-28 relative overflow-hidden">
        {/* Subtle grid background */}
        <div className="pointer-events-none absolute inset-0 opacity-[0.03]" style={{
          backgroundImage: "linear-gradient(currentColor 1px, transparent 1px), linear-gradient(90deg, currentColor 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }} />

        <div className="relative mx-auto max-w-7xl px-6">
          <motion.h2
            className="text-3xl font-bold text-center mb-16 md:text-4xl"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            {t("home.built_title")}
          </motion.h2>
          <motion.div
            className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            <FeatureCard i={0} icon={Server} title={t("home.feat_marketplace_title")} description={t("home.feat_marketplace_desc")} />
            <FeatureCard i={1} icon={BadgeCheck} title={t("home.feat_trust_title")} description={t("home.feat_trust_desc")} />
            <FeatureCard i={2} icon={BarChart3} title={t("home.feat_telemetry_title")} description={t("home.feat_telemetry_desc")} />
            <FeatureCard i={3} icon={Globe} title={t("home.feat_jurisdiction_title")} description={t("home.feat_jurisdiction_desc")} />
            <FeatureCard i={4} icon={DollarSign} title={t("home.feat_spot_title")} description={t("home.feat_spot_desc")} />
            <FeatureCard i={5} icon={Leaf} title={t("home.feat_green_title")} description={t("home.feat_green_desc")} />
          </motion.div>
        </div>
      </section>

      {/* ── Comparison ────────────────────────────────────────────────── */}
      <section className="border-t border-border bg-navy py-28">
        <div className="mx-auto max-w-7xl px-6">
          <motion.h2
            className="text-3xl font-bold text-center mb-16 md:text-4xl"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            {t("home.compare_title")}
          </motion.h2>
          <motion.div
            className="overflow-x-auto rounded-xl border border-border bg-surface/50 backdrop-blur-sm"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.15 }}
          >
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th scope="col" className="py-4 pl-6 pr-4 text-left text-text-secondary font-medium">{t("home.compare_feature")}</th>
                  <th scope="col" className="py-4 px-4 text-center font-bold text-accent-cyan">{t("home.compare_xcelsior")}</th>
                  <th scope="col" className="py-4 px-4 text-center text-text-secondary">{t("home.compare_aws")}</th>
                  <th scope="col" className="py-4 px-4 text-center text-text-secondary">{t("home.compare_vast")}</th>
                  <th scope="col" className="py-4 px-4 text-center text-text-secondary">{t("home.compare_runpod")}</th>
                </tr>
              </thead>
              <tbody className="text-text-secondary">
                <CompRow feature={t("home.cmp_sovereignty")} xcelsior={t("home.cmp_sovereignty_x")} aws={t("home.cmp_sovereignty_aws")} vast={t("home.cmp_sovereignty_vast")} runpod={t("home.cmp_sovereignty_rp")} />
                <CompRow feature={t("home.cmp_pipeda")} xcelsior={t("home.cmp_pipeda_x")} aws={t("home.cmp_pipeda_aws")} vast={t("home.cmp_pipeda_vast")} runpod={t("home.cmp_pipeda_rp")} />
                <CompRow feature={t("home.cmp_price")} xcelsior={t("home.cmp_price_x")} aws={t("home.cmp_price_aws")} vast={t("home.cmp_price_vast")} runpod={t("home.cmp_price_rp")} />
                <CompRow feature={t("home.cmp_cad")} xcelsior={t("home.cmp_cad_x")} aws={t("home.cmp_cad_aws")} vast={t("home.cmp_cad_vast")} runpod={t("home.cmp_cad_rp")} />
                <CompRow feature={t("home.cmp_rebate")} xcelsior={t("home.cmp_rebate_x")} aws={t("home.cmp_rebate_aws")} vast={t("home.cmp_rebate_vast")} runpod={t("home.cmp_rebate_rp")} />
                <CompRow feature={t("home.cmp_verification")} xcelsior={t("home.cmp_verification_x")} aws={t("home.cmp_verification_aws")} vast={t("home.cmp_verification_vast")} runpod={t("home.cmp_verification_rp")} />
                <CompRow feature={t("home.cmp_green")} xcelsior={t("home.cmp_green_x")} aws={t("home.cmp_green_aws")} vast={t("home.cmp_green_vast")} runpod={t("home.cmp_green_rp")} />
              </tbody>
            </table>
          </motion.div>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────────────── */}
      <div className="brand-line" />
      <section className="relative overflow-hidden py-32">
        {/* Animated aurora sweep */}
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute left-0 top-0 h-full w-full aurora-gradient" />
          <div className="absolute left-[20%] top-[20%] h-72 w-72 rounded-full bg-accent-cyan/10 blur-[120px] animate-[aurora-drift_6s_ease-in-out_infinite]" />
          <div className="absolute right-[20%] bottom-[15%] h-64 w-64 rounded-full bg-accent-violet/8 blur-[100px] animate-[aurora-drift_8s_ease-in-out_infinite_3s]" />
        </div>

        <motion.div
          className="relative mx-auto max-w-3xl px-6 text-center"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          <motion.h2 variants={fadeUp} custom={0} className="text-4xl font-bold md:text-5xl lg:text-6xl">
            {t("home.cta_title_1")}
            <br />
            <span className="bg-gradient-to-r from-accent-cyan via-accent-violet to-accent-red bg-clip-text text-transparent">
              {t("home.cta_title_2")}
            </span>
          </motion.h2>
          <motion.p variants={fadeUp} custom={1} className="mt-6 text-lg text-text-secondary max-w-xl mx-auto">
            {t("home.cta_desc")}
          </motion.p>
          <motion.div variants={fadeUp} custom={2} className="mt-10">
            <Link href="/register">
              <Button variant="gold" size="lg" className="text-base px-10 shadow-lg shadow-accent-gold/15">
                {t("home.cta_button")}
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </motion.div>
        </motion.div>
      </section>
    </>
  );
}

function ValueCard({
  icon: Icon,
  title,
  description,
  accent,
  glowColor,
  i,
}: {
  icon: typeof Shield;
  title: string;
  description: string;
  accent: string;
  glowColor: string;
  i: number;
}) {
  return (
    <motion.div
      variants={fadeUp}
      custom={i}
      className="glow-card rounded-xl p-8"
      style={{ "--glow-color": glowColor } as React.CSSProperties}
    >
      <div
        className={`mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-${accent}/10`}
      >
        <Icon className={`h-6 w-6 text-${accent}`} />
      </div>
      <h3 className="mb-2 text-xl font-semibold">{title}</h3>
      <p className="text-sm text-text-secondary leading-relaxed">{description}</p>
    </motion.div>
  );
}

function FeatureCard({
  icon: Icon,
  title,
  description,
  i,
}: {
  icon: typeof Shield;
  title: string;
  description: string;
  i: number;
}) {
  return (
    <motion.div
      variants={fadeUp}
      custom={i}
      className="group glow-card rounded-xl p-6"
    >
      <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-accent-cyan/10 transition-colors group-hover:bg-accent-cyan/20">
        <Icon className="h-5 w-5 text-accent-cyan" />
      </div>
      <h3 className="mb-1 font-semibold">{title}</h3>
      <p className="text-sm text-text-secondary leading-relaxed">{description}</p>
    </motion.div>
  );
}

function CompRow({
  feature,
  xcelsior,
  aws,
  vast,
  runpod,
}: {
  feature: string;
  xcelsior: string;
  aws: string;
  vast: string;
  runpod: string;
}) {
  const isCheck = (val: string) => val === "✓" || val === "Yes" || val === "Oui";
  const isX = (val: string) => val === "✗" || val === "No" || val === "Non";

  return (
    <tr className="border-b border-border/50 hover:bg-surface-hover transition-colors">
      <td className="py-3.5 pl-6 pr-4 font-medium text-text-primary">{feature}</td>
      <td className="py-3.5 px-4 text-center font-medium text-accent-cyan">
        {isCheck(xcelsior) ? <Check className="inline h-4 w-4 text-emerald" aria-label="Yes" /> : xcelsior}
      </td>
      <td className="py-3.5 px-4 text-center">
        {isCheck(aws) ? <Check className="inline h-4 w-4 text-text-muted" aria-label="Yes" /> : isX(aws) ? <span className="text-accent-red" aria-label="No">✗</span> : aws}
      </td>
      <td className="py-3.5 px-4 text-center">
        {isCheck(vast) ? <Check className="inline h-4 w-4 text-text-muted" aria-label="Yes" /> : isX(vast) ? <span className="text-accent-red" aria-label="No">✗</span> : vast}
      </td>
      <td className="py-3.5 px-4 text-center">
        {isCheck(runpod) ? <Check className="inline h-4 w-4 text-text-muted" aria-label="Yes" /> : isX(runpod) ? <span className="text-accent-red" aria-label="No">✗</span> : runpod}
      </td>
    </tr>
  );
}
