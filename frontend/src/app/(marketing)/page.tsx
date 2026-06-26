"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import {
  Shield,
  DollarSign,
  Leaf,
  ArrowRight,
  MapPin,
} from "lucide-react";
import { m } from "@/components/marketing/motion";
import { Button } from "@/components/ui/button";
import { BRAND_ASSETS } from "@/lib/brand-assets";
import { useLocale } from "@/lib/locale";

const HomeBelowFold = dynamic(
  () => import("@/components/marketing/home-below-fold").then((mod) => mod.HomeBelowFold),
  { loading: () => <div className="min-h-[50vh]" aria-hidden /> },
);

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
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute left-[10%] top-[15%] h-64 w-64 rounded-full bg-accent-cyan/8 blur-[100px] animate-[aurora-drift_8s_ease-in-out_infinite]" />
          <div className="absolute right-[15%] top-[25%] h-48 w-48 rounded-full bg-accent-violet/6 blur-[80px] animate-[aurora-drift_10s_ease-in-out_infinite_2s]" />
          <div className="absolute left-[40%] bottom-[10%] h-56 w-56 rounded-full bg-accent-red/5 blur-[90px] animate-[aurora-drift_12s_ease-in-out_infinite_4s]" />
        </div>

        <div className="relative mx-auto max-w-7xl px-6 py-28 md:py-40">
          <m.div
            className="max-w-3xl"
            initial="hidden"
            animate="visible"
          >
            <m.div variants={fadeUp} custom={0} className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent-gold/30 bg-accent-gold/10 px-4 py-1.5 backdrop-blur-sm">
              <MapPin className="h-3 w-3 text-accent-gold" />
              <span className="text-xs font-medium text-accent-gold">
                {t("home.badge")}
              </span>
            </m.div>

            <m.div variants={fadeUp} custom={1} className="mb-4 flex" aria-label="GPU compute, accelerated">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={BRAND_ASSETS.textTagMedLight} alt="" className="hidden h-5 w-auto dark:block" aria-hidden="true" />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={BRAND_ASSETS.textTagMedDark} alt="" className="block h-5 w-auto dark:hidden" aria-hidden="true" />
            </m.div>

            <m.h1 variants={fadeUp} custom={2} className="text-5xl font-bold leading-[1.08] tracking-tight md:text-6xl lg:text-7xl">
              <span>{t("home.hero_line1")}</span>
              <br />
              <span>
              {t("home.hero_line2")}{" "}
              <span className="bg-gradient-to-r from-accent-cyan via-accent-violet to-accent-red bg-clip-text text-transparent">
                {t("home.hero_accent")}
              </span>
              </span>
            </m.h1>

            <m.p variants={fadeUp} custom={3} className="mt-6 text-lg text-text-secondary leading-relaxed max-w-2xl">
              {t("home.hero_desc")}
            </m.p>

            <m.div variants={fadeUp} custom={4} className="mt-10 flex flex-wrap gap-4">
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
            </m.div>

            <m.div variants={fadeUp} custom={5} className="mt-14 flex flex-wrap gap-4 sm:gap-8 text-sm text-text-secondary">
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
            </m.div>
          </m.div>
        </div>
      </section>

      <HomeBelowFold />
    </>
  );
}
