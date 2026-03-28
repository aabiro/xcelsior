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
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useLocale } from "@/lib/locale";

export default function HomePage() {
  const { t } = useLocale();
  return (
    <>
      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <section className="aurora-gradient relative overflow-hidden">
        <div className="mx-auto max-w-7xl px-6 py-24 md:py-36">
          <div className="max-w-3xl">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent-gold/30 bg-accent-gold/10 px-4 py-1.5">
              <span className="text-xs font-medium text-accent-gold">
                {t("home.badge")}
              </span>
            </div>

            <h1 className="text-5xl font-bold leading-tight tracking-tight md:text-6xl lg:text-7xl">
              {t("home.hero_line1")}
              <br />
              {t("home.hero_line2")}{" "}
              <span className="bg-gradient-to-r from-accent-red to-accent-gold bg-clip-text text-transparent">
                {t("home.hero_accent")}
              </span>
            </h1>

            <p className="mt-6 text-lg text-text-secondary leading-relaxed max-w-2xl">
              {t("home.hero_desc")}
            </p>

            <div className="mt-8 flex flex-wrap gap-4">
              <Link href="/register">
                <Button size="lg" className="text-base">
                  {t("home.cta_start")}
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link href="/pricing">
                <Button variant="outline" size="lg" className="text-base">
                  {t("home.cta_pricing")}
                </Button>
              </Link>
            </div>

            <div className="mt-12 flex flex-wrap gap-8 text-sm text-text-secondary">
              <div className="flex items-center gap-2">
                <DollarSign className="h-4 w-4 text-accent-gold" />
                {t("home.stat_price")}
              </div>
              <div className="flex items-center gap-2">
                <Shield className="h-4 w-4 text-emerald" />
                {t("home.stat_pipeda")}
              </div>
              <div className="flex items-center gap-2">
                <Leaf className="h-4 w-4 text-emerald" />
                {t("home.stat_hydro")}
              </div>
            </div>
          </div>
        </div>

        {/* Background decoration */}
        <div className="absolute -right-48 top-1/4 h-96 w-96 rounded-full bg-accent-red/5 blur-3xl" />
        <div className="absolute -right-24 top-1/2 h-64 w-64 rounded-full bg-ice-blue/5 blur-3xl" />
      </section>

      {/* ── Value Propositions ────────────────────────────────────────── */}
      <section className="border-t border-border bg-navy py-24">
        <div className="mx-auto max-w-7xl px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold md:text-4xl">
              {t("home.why_title")}
            </h2>
            <p className="mt-4 text-text-secondary max-w-2xl mx-auto">
              {t("home.why_desc")}
            </p>
          </div>

          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            <ValueCard
              icon={Shield}
              title={t("home.val_sovereignty_title")}
              description={t("home.val_sovereignty_desc")}
              accent="accent-red"
            />
            <ValueCard
              icon={Scale}
              title={t("home.val_compliance_title")}
              description={t("home.val_compliance_desc")}
              accent="accent-gold"
            />
            <ValueCard
              icon={Zap}
              title={t("home.val_pricing_title")}
              description={t("home.val_pricing_desc")}
              accent="emerald"
            />
          </div>
        </div>
      </section>

      {/* ── Feature Grid ─────────────────────────────────────────────── */}
      <section className="border-t border-border bg-navy-light/30 py-24">
        <div className="mx-auto max-w-7xl px-6">
          <h2 className="text-3xl font-bold text-center mb-16 md:text-4xl">
            {t("home.built_title")}
          </h2>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
            <FeatureCard icon={Server} title={t("home.feat_marketplace_title")} description={t("home.feat_marketplace_desc")} />
            <FeatureCard icon={BadgeCheck} title={t("home.feat_trust_title")} description={t("home.feat_trust_desc")} />
            <FeatureCard icon={BarChart3} title={t("home.feat_telemetry_title")} description={t("home.feat_telemetry_desc")} />
            <FeatureCard icon={Globe} title={t("home.feat_jurisdiction_title")} description={t("home.feat_jurisdiction_desc")} />
            <FeatureCard icon={DollarSign} title={t("home.feat_spot_title")} description={t("home.feat_spot_desc")} />
            <FeatureCard icon={Leaf} title={t("home.feat_green_title")} description={t("home.feat_green_desc")} />
          </div>
        </div>
      </section>

      {/* ── Comparison ────────────────────────────────────────────────── */}
      <section className="border-t border-border bg-navy py-24">
        <div className="mx-auto max-w-7xl px-6">
          <h2 className="text-3xl font-bold text-center mb-16 md:text-4xl">
            {t("home.compare_title")}
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="py-4 pr-4 text-left text-text-secondary font-medium">{t("home.compare_feature")}</th>
                  <th className="py-4 px-4 text-center font-bold text-accent-gold">{t("home.compare_xcelsior")}</th>
                  <th className="py-4 px-4 text-center text-text-secondary">{t("home.compare_aws")}</th>
                  <th className="py-4 px-4 text-center text-text-secondary">{t("home.compare_vast")}</th>
                  <th className="py-4 px-4 text-center text-text-secondary">{t("home.compare_runpod")}</th>
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
          </div>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────────────── */}
      <section className="border-t border-border aurora-gradient py-24">
        <div className="mx-auto max-w-3xl px-6 text-center">
          <h2 className="text-4xl font-bold md:text-5xl">
            {t("home.cta_title_1")}
            <br />
            <span className="bg-gradient-to-r from-accent-red to-accent-gold bg-clip-text text-transparent">
              {t("home.cta_title_2")}
            </span>
          </h2>
          <p className="mt-6 text-lg text-text-secondary">
            {t("home.cta_desc")}
          </p>
          <div className="mt-8">
            <Link href="/register">
              <Button variant="gold" size="lg" className="text-base">
                {t("home.cta_button")}
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}

function ValueCard({
  icon: Icon,
  title,
  description,
  accent,
}: {
  icon: typeof Shield;
  title: string;
  description: string;
  accent: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface p-8 card-hover">
      <div
        className={`mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-${accent}/10`}
      >
        <Icon className={`h-6 w-6 text-${accent}`} />
      </div>
      <h3 className="mb-2 text-xl font-semibold">{title}</h3>
      <p className="text-sm text-text-secondary leading-relaxed">{description}</p>
    </div>
  );
}

function FeatureCard({
  icon: Icon,
  title,
  description,
}: {
  icon: typeof Shield;
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface p-6 card-hover">
      <Icon className="mb-3 h-5 w-5 text-ice-blue" />
      <h3 className="mb-1 font-semibold">{title}</h3>
      <p className="text-sm text-text-secondary leading-relaxed">{description}</p>
    </div>
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
  return (
    <tr className="border-b border-border/50">
      <td className="py-3 pr-4 font-medium text-text-primary">{feature}</td>
      <td className="py-3 px-4 text-center font-medium text-accent-gold">{xcelsior}</td>
      <td className="py-3 px-4 text-center">{aws}</td>
      <td className="py-3 px-4 text-center">{vast}</td>
      <td className="py-3 px-4 text-center">{runpod}</td>
    </tr>
  );
}
