"use client";

import Link from "next/link";
import { ArrowRight, MapPin, Shield, Leaf, Users, Zap, Heart } from "lucide-react";
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

const valueKeys = [
  { icon: Shield, title: "about.val_sovereignty_title", desc: "about.val_sovereignty_desc", glow: "rgba(220,38,38,0.12)" },
  { icon: Leaf, title: "about.val_green_title", desc: "about.val_green_desc", glow: "rgba(16,185,129,0.12)" },
  { icon: Users, title: "about.val_community_title", desc: "about.val_community_desc", glow: "rgba(124,58,237,0.12)" },
  { icon: Zap, title: "about.val_access_title", desc: "about.val_access_desc", glow: "rgba(0,212,255,0.12)" },
  { icon: Heart, title: "about.val_canada_title", desc: "about.val_canada_desc", glow: "rgba(220,38,38,0.12)" },
  { icon: MapPin, title: "about.val_local_title", desc: "about.val_local_desc", glow: "rgba(245,158,11,0.12)" },
];

const milestoneKeys = [
  { year: "about.journey_2024_title", events: ["about.journey_2024_p1", "about.journey_2024_p2"] },
  { year: "about.journey_2025_title", events: ["about.journey_2025_p1", "about.journey_2025_p2", "about.journey_2025_p3"] },
  { year: "about.journey_2026_title", events: ["about.journey_2026_p1"] },
];

export function AboutContent() {
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
          <MapPin className="h-3.5 w-3.5 text-accent-red" />
          <span className="text-xs font-medium text-accent-red">
            {t("about.badge")}
          </span>
        </motion.div>
        <motion.h1 variants={fadeUp} custom={1} className="text-4xl font-bold md:text-5xl lg:text-6xl">
          {t("about.title").split(/(truly Canadian|véritablement canadien)/i).map((part, i) =>
            /truly Canadian|véritablement canadien/i.test(part) ? (
              <span key={i} className="bg-gradient-to-r from-accent-cyan via-accent-violet to-accent-red bg-clip-text text-transparent">
                {part}
              </span>
            ) : (
              <span key={i}>{part}</span>
            ),
          )}
        </motion.h1>
        <motion.p variants={fadeUp} custom={2} className="mt-6 text-lg text-text-secondary max-w-3xl mx-auto leading-relaxed">
          {t("about.subtitle")}
        </motion.p>
      </motion.div>

      {/* Mission */}
      <motion.div
        className="glow-card rounded-xl p-8 md:p-12 mb-20"
        style={{ "--glow-color": "rgba(245,158,11,0.12)" } as React.CSSProperties}
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-2xl font-bold mb-4">{t("about.mission_title")}</h2>
        <p className="text-lg text-text-secondary leading-relaxed">
          {t("about.mission_p1")}
        </p>
      </motion.div>

      {/* Values */}
      <div className="mb-20">
        <motion.h2
          className="text-2xl font-bold text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          {t("about.values_title")}
        </motion.h2>
        <motion.div
          className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          {valueKeys.map((v, idx) => (
            <motion.div
              key={v.title}
              variants={fadeUp}
              custom={idx}
              className="group glow-card rounded-xl p-6"
              style={{ "--glow-color": v.glow } as React.CSSProperties}
            >
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-accent-cyan/10 transition-transform group-hover:scale-110">
                <v.icon className="h-5 w-5 text-accent-cyan" />
              </div>
              <h3 className="mb-2 text-lg font-semibold">{t(v.title)}</h3>
              <p className="text-sm text-text-secondary leading-relaxed">{t(v.desc)}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Timeline */}
      <div className="mb-20">
        <motion.h2
          className="text-2xl font-bold text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          {t("about.journey_title")}
        </motion.h2>
        <div className="relative mx-auto max-w-2xl">
          <div className="absolute left-4 top-0 bottom-0 w-px bg-gradient-to-b from-accent-cyan via-accent-violet to-accent-gold" />
          <motion.div
            className="space-y-8"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            {milestoneKeys.flatMap((group, gi) =>
              group.events.map((eventKey, ei) => (
                <motion.div key={eventKey} variants={fadeUp} custom={gi * 2 + ei} className="relative pl-12">
                  <div className="absolute left-2.5 top-1.5 h-3 w-3 rounded-full border-2 border-accent-gold bg-navy shadow-[0_0_8px_rgba(245,158,11,0.4)]" />
                  {ei === 0 && <span className="text-xs font-mono text-accent-gold">{t(group.year)}</span>}
                  <p className="text-sm text-text-secondary mt-0.5">{t(eventKey)}</p>
                </motion.div>
              )),
            )}
          </motion.div>
        </div>
      </div>

      {/* CTA */}
      <motion.div
        className="text-center"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <motion.h2 variants={fadeUp} custom={0} className="text-3xl font-bold mb-4">{t("about.cta_title")}</motion.h2>
        <motion.p variants={fadeUp} custom={1} className="text-text-secondary mb-8">{t("about.cta_desc")}</motion.p>
        <motion.div variants={fadeUp} custom={2}>
          <Link href="/register">
            <Button size="lg" className="text-base px-10 shadow-lg shadow-accent-cyan/10">
              {t("about.cta_button")} <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </motion.div>
      </motion.div>
    </div>
  );
}
