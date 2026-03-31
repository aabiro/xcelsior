"use client";

import { Shield, Zap, Scale, Server, BadgeCheck, BarChart3, Globe, Leaf, DollarSign, Lock, FileCheck, Users } from "lucide-react";
import { motion } from "framer-motion";
import { useLocale } from "@/lib/locale";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.5, ease: "easeOut" as const },
  }),
};

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
    <div className="mx-auto max-w-7xl px-6 py-28 relative overflow-hidden">
      {/* Subtle grid background */}
      <div className="pointer-events-none absolute inset-0 opacity-[0.03]" style={{
        backgroundImage: "linear-gradient(currentColor 1px, transparent 1px), linear-gradient(90deg, currentColor 1px, transparent 1px)",
        backgroundSize: "60px 60px",
      }} />

      <motion.div
        className="relative text-center mb-16"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <motion.h1 variants={fadeUp} custom={0} className="text-4xl font-bold md:text-5xl">
          {t("features.title_1")}{" "}
          <span className="bg-gradient-to-r from-accent-cyan via-accent-violet to-accent-red bg-clip-text text-transparent">
            {t("features.title_accent")}
          </span>
        </motion.h1>
        <motion.p variants={fadeUp} custom={1} className="mt-4 text-lg text-text-secondary max-w-2xl mx-auto">
          {t("features.subtitle")}
        </motion.p>
      </motion.div>

      <motion.div
        className="relative grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        {featureKeys.map((f, idx) => {
          const colors = categoryColors[f.cat] ?? { bg: "bg-accent-cyan/10", text: "text-accent-cyan", glow: "rgba(0,212,255,0.08)" };
          return (
            <motion.div
              key={f.title}
              variants={fadeUp}
              custom={idx}
              className="group glow-card rounded-xl p-6"
              style={{ "--glow-color": colors.glow } as React.CSSProperties}
            >
              <div className="mb-4 flex items-center gap-3">
                <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${colors.bg} transition-colors group-hover:scale-110`}>
                  <f.icon className={`h-5 w-5 ${colors.text}`} />
                </div>
                <span className={`text-xs font-medium uppercase tracking-wider ${colors.text}`}>
                  {t(f.cat)}
                </span>
              </div>
              <h3 className="mb-2 text-lg font-semibold">{t(f.title)}</h3>
              <p className="text-sm text-text-secondary leading-relaxed">{t(f.desc)}</p>
            </motion.div>
          );
        })}
      </motion.div>
    </div>
  );
}
