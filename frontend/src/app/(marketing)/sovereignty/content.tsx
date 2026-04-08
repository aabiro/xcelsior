"use client";

import Link from "next/link";
import { Shield, AlertTriangle, CheckCircle, XCircle, ArrowRight, MapPin } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ProviderLogo } from "@/components/ui/provider-logo";
import { useLocale } from "@/lib/locale";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5, ease: "easeOut" as const },
  }),
};

export function SovereigntyContent() {
  const { t } = useLocale();

  return (
    <div className="mx-auto max-w-7xl px-6 py-28">
      {/* Hero */}
      <motion.div
        className="text-center mb-20"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <motion.div variants={fadeUp} custom={0} className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent-red/30 bg-accent-red/10 px-4 py-1.5 backdrop-blur-sm">
          <Shield className="h-3.5 w-3.5 text-accent-red" />
          <span className="text-xs font-medium text-accent-red">{t("sov.badge")}</span>
        </motion.div>
        <motion.h1 variants={fadeUp} custom={1} className="text-4xl font-bold md:text-5xl lg:text-6xl">
          {t("sov.title")}
        </motion.h1>
        <motion.p variants={fadeUp} custom={2} className="mt-6 text-lg text-text-secondary max-w-3xl mx-auto">
          {t("sov.subtitle")}
        </motion.p>
      </motion.div>

      {/* Comparison */}
      <motion.div
        className="grid grid-cols-1 gap-8 md:grid-cols-2 mb-20"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <motion.div
          variants={fadeUp}
          custom={0}
          className="glow-card rounded-xl p-8"
          style={{ "--glow-color": "rgba(245,158,11,0.1)" } as React.CSSProperties}
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent-gold/10">
              <AlertTriangle className="h-5 w-5 text-accent-gold" />
            </div>
            <h2 className="text-xl font-bold">{t("sov.residency_title")}</h2>
          </div>
          <p className="text-sm text-text-secondary mb-6">
            {t("sov.residency_desc")}
          </p>
          <ul className="space-y-3">
            <Row icon={XCircle} color="text-accent-red" text={t("sov.residency_aws")} logo="aws" />
            <Row icon={XCircle} color="text-accent-red" text={t("sov.residency_google")} logo="google-cloud" />
            <Row icon={XCircle} color="text-accent-red" text={t("sov.residency_azure")} logo="azure" />
          </ul>
        </motion.div>

        <motion.div
          variants={fadeUp}
          custom={1}
          className="glow-card rounded-xl p-8 ring-1 ring-emerald/30"
          style={{ "--glow-color": "rgba(16,185,129,0.12)" } as React.CSSProperties}
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald/10">
              <CheckCircle className="h-5 w-5 text-emerald" />
            </div>
            <h2 className="text-xl font-bold">{t("sov.sovereignty_title")}</h2>
          </div>
          <p className="text-sm text-text-secondary mb-6">
            {t("sov.sovereignty_desc")}
          </p>
          <ul className="space-y-3">
            <Row icon={CheckCircle} color="text-emerald" text={t("sov.sovereignty_i1")} />
            <Row icon={CheckCircle} color="text-emerald" text={t("sov.sovereignty_i2")} />
            <Row icon={CheckCircle} color="text-emerald" text={t("sov.sovereignty_i3")} />
            <Row icon={CheckCircle} color="text-emerald" text={t("sov.sovereignty_i4")} />
          </ul>
        </motion.div>
      </motion.div>

      {/* CLOUD Act Section */}
      <motion.div
        className="glow-card rounded-xl p-8 md:p-12 mb-20"
        style={{ "--glow-color": "rgba(0,212,255,0.08)" } as React.CSSProperties}
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-2xl font-bold mb-4">{t("sov.cloud_title")}</h2>
        <p className="text-text-secondary leading-relaxed mb-6">
          {t("sov.cloud_p1")}
        </p>
        <p className="text-text-secondary leading-relaxed">
          {t("sov.cloud_p2")}
        </p>
      </motion.div>

      {/* Province Compliance */}
      <div className="mb-20">
        <motion.h2
          className="text-2xl font-bold text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          {t("sov.province_title")}
        </motion.h2>
        <motion.div
          className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          <ProvinceCard i={0} province={t("sov.prov_qc")} law={t("sov.prov_qc_law")} description={t("sov.prov_qc_desc")} />
          <ProvinceCard i={1} province={t("sov.prov_on")} law={t("sov.prov_on_law")} description={t("sov.prov_on_desc")} />
          <ProvinceCard i={2} province={t("sov.prov_bc")} law={t("sov.prov_bc_law")} description={t("sov.prov_bc_desc")} />
          <ProvinceCard i={3} province={t("sov.prov_ns")} law={t("sov.prov_ns_law")} description={t("sov.prov_ns_desc")} />
          <ProvinceCard i={4} province={t("sov.prov_fed")} law={t("sov.prov_fed_law")} description={t("sov.prov_fed_desc")} />
          <ProvinceCard i={5} province={t("sov.prov_all")} law={t("sov.prov_all_law")} description={t("sov.prov_all_desc")} />
        </motion.div>
      </div>

      {/* CTA */}
      <motion.div
        className="text-center"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <motion.h2 variants={fadeUp} custom={0} className="text-3xl font-bold mb-4">{t("sov.cta_title")}</motion.h2>
        <motion.p variants={fadeUp} custom={1} className="text-text-secondary mb-8">{t("sov.cta_desc")}</motion.p>
        <motion.div variants={fadeUp} custom={2}>
          <Link href="/register">
            <Button size="lg" className="text-base px-10 shadow-lg shadow-accent-cyan/10">
              {t("sov.cta_button")} <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </motion.div>
      </motion.div>
    </div>
  );
}

function Row({ icon: Icon, color, text, logo }: { icon: typeof CheckCircle; color: string; text: string; logo?: string }) {
  return (
    <li className="flex items-start gap-2">
      <Icon className={`h-4 w-4 mt-0.5 ${color} shrink-0`} />
      {logo && (
        <ProviderLogo
          provider={logo}
          framed
          size={22}
          className="mt-0.5 rounded-lg border-border/60 bg-background/70 shadow-none"
        />
      )}
      <span className="text-sm text-text-secondary">{text}</span>
    </li>
  );
}

function ProvinceCard({ i, province, law, description }: { i: number; province: string; law: string; description: string }) {
  return (
    <motion.div
      variants={fadeUp}
      custom={i}
      className="group glow-card rounded-xl p-5"
      style={{ "--glow-color": "rgba(220,38,38,0.08)" } as React.CSSProperties}
    >
      <div className="flex items-center gap-2 mb-2">
        <MapPin className="h-4 w-4 text-accent-red transition-transform group-hover:scale-110" />
        <span className="font-semibold">{province}</span>
      </div>
      <p className="text-sm font-medium text-accent-gold mb-1">{law}</p>
      <p className="text-xs text-text-secondary leading-relaxed">{description}</p>
    </motion.div>
  );
}
