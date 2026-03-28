"use client";

import { Shield, Zap, Scale, Server, BadgeCheck, BarChart3, Globe, Leaf, DollarSign, Lock, FileCheck, Users } from "lucide-react";
import { useLocale } from "@/lib/locale";

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
    <div className="mx-auto max-w-7xl px-6 py-24">
      <div className="text-center mb-16">
        <h1 className="text-4xl font-bold md:text-5xl">
          {t("features.title_1")}{" "}
          <span className="bg-gradient-to-r from-accent-red to-accent-gold bg-clip-text text-transparent">
            {t("features.title_accent")}
          </span>
        </h1>
        <p className="mt-4 text-lg text-text-secondary max-w-2xl mx-auto">
          {t("features.subtitle")}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {featureKeys.map((f) => (
          <div key={f.title} className="rounded-xl border border-border bg-surface p-6 card-hover">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-ice-blue/10">
                <f.icon className="h-5 w-5 text-ice-blue" />
              </div>
              <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
                {t(f.cat)}
              </span>
            </div>
            <h3 className="mb-2 text-lg font-semibold">{t(f.title)}</h3>
            <p className="text-sm text-text-secondary leading-relaxed">{t(f.desc)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
