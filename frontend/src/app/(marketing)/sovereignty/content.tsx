"use client";

import Link from "next/link";
import { Shield, AlertTriangle, CheckCircle, XCircle, ArrowRight, MapPin } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useLocale } from "@/lib/locale";

export function SovereigntyContent() {
  const { t } = useLocale();

  return (
    <div className="mx-auto max-w-7xl px-6 py-24">
      {/* Hero */}
      <div className="text-center mb-20">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent-red/30 bg-accent-red/10 px-4 py-1.5">
          <Shield className="h-3.5 w-3.5 text-accent-red" />
          <span className="text-xs font-medium text-accent-red">{t("sov.badge")}</span>
        </div>
        <h1 className="text-4xl font-bold md:text-5xl lg:text-6xl">
          {t("sov.title")}
        </h1>
        <p className="mt-6 text-lg text-text-secondary max-w-3xl mx-auto">
          {t("sov.subtitle")}
        </p>
      </div>

      {/* Comparison */}
      <div className="grid grid-cols-1 gap-8 md:grid-cols-2 mb-20">
        <div className="rounded-xl border border-border bg-surface p-8">
          <div className="flex items-center gap-3 mb-6">
            <AlertTriangle className="h-6 w-6 text-accent-gold" />
            <h2 className="text-xl font-bold">{t("sov.residency_title")}</h2>
          </div>
          <p className="text-sm text-text-secondary mb-6">
            {t("sov.residency_desc")}
          </p>
          <ul className="space-y-3">
            <Row icon={XCircle} color="text-accent-red" text={t("sov.residency_aws")} />
            <Row icon={XCircle} color="text-accent-red" text={t("sov.residency_google")} />
            <Row icon={XCircle} color="text-accent-red" text={t("sov.residency_azure")} />
          </ul>
        </div>

        <div className="rounded-xl border border-accent-gold/30 bg-surface p-8">
          <div className="flex items-center gap-3 mb-6">
            <CheckCircle className="h-6 w-6 text-emerald" />
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
        </div>
      </div>

      {/* CLOUD Act Section */}
      <div className="rounded-xl border border-border bg-navy-light/30 p-8 md:p-12 mb-20">
        <h2 className="text-2xl font-bold mb-4">{t("sov.cloud_title")}</h2>
        <p className="text-text-secondary leading-relaxed mb-6">
          {t("sov.cloud_p1")}
        </p>
        <p className="text-text-secondary leading-relaxed">
          {t("sov.cloud_p2")}
        </p>
      </div>

      {/* Province Compliance */}
      <div className="mb-20">
        <h2 className="text-2xl font-bold text-center mb-12">{t("sov.province_title")}</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <ProvinceCard province={t("sov.prov_qc")} law={t("sov.prov_qc_law")} description={t("sov.prov_qc_desc")} />
          <ProvinceCard province={t("sov.prov_on")} law={t("sov.prov_on_law")} description={t("sov.prov_on_desc")} />
          <ProvinceCard province={t("sov.prov_bc")} law={t("sov.prov_bc_law")} description={t("sov.prov_bc_desc")} />
          <ProvinceCard province={t("sov.prov_ns")} law={t("sov.prov_ns_law")} description={t("sov.prov_ns_desc")} />
          <ProvinceCard province={t("sov.prov_fed")} law={t("sov.prov_fed_law")} description={t("sov.prov_fed_desc")} />
          <ProvinceCard province={t("sov.prov_all")} law={t("sov.prov_all_law")} description={t("sov.prov_all_desc")} />
        </div>
      </div>

      {/* CTA */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-4">{t("sov.cta_title")}</h2>
        <p className="text-text-secondary mb-8">{t("sov.cta_desc")}</p>
        <Link href="/register">
          <Button size="lg">
            {t("sov.cta_button")} <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </div>
    </div>
  );
}

function Row({ icon: Icon, color, text }: { icon: typeof CheckCircle; color: string; text: string }) {
  return (
    <li className="flex items-start gap-2">
      <Icon className={`h-4 w-4 mt-0.5 ${color} shrink-0`} />
      <span className="text-sm text-text-secondary">{text}</span>
    </li>
  );
}

function ProvinceCard({ province, law, description }: { province: string; law: string; description: string }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-5">
      <div className="flex items-center gap-2 mb-2">
        <MapPin className="h-4 w-4 text-accent-red" />
        <span className="font-semibold">{province}</span>
      </div>
      <p className="text-sm font-medium text-accent-gold mb-1">{law}</p>
      <p className="text-xs text-text-secondary leading-relaxed">{description}</p>
    </div>
  );
}
