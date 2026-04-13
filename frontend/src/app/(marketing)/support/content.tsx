"use client";

import Link from "next/link";
import { MessageCircle, Mail, BookOpen, Shield, Clock, Headphones } from "lucide-react";
import { motion } from "framer-motion";
import { useLocale } from "@/lib/locale";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5, ease: "easeOut" as const },
  }),
};

const channels = [
  {
    icon: MessageCircle,
    titleKey: "support.chat_title",
    descKey: "support.chat_desc",
    glow: "rgba(0,212,255,0.12)",
    actionKey: "support.chat_action",
    type: "chat" as const,
  },
  {
    icon: Mail,
    titleKey: "support.email_title",
    descKey: "support.email_desc",
    glow: "rgba(124,58,237,0.12)",
    actionKey: "support.email_action",
    type: "email" as const,
  },
  {
    icon: BookOpen,
    titleKey: "support.docs_title",
    descKey: "support.docs_desc",
    glow: "rgba(16,185,129,0.12)",
    actionKey: "support.docs_action",
    type: "docs" as const,
  },
];

export function SupportContent() {
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
        <motion.div
          variants={fadeUp}
          custom={0}
          className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent-cyan/30 bg-accent-cyan/10 px-4 py-1.5 backdrop-blur-sm"
        >
          <Headphones className="h-3.5 w-3.5 text-accent-cyan" />
          <span className="text-xs font-medium text-accent-cyan">
            {t("support.badge")}
          </span>
        </motion.div>

        <motion.h1
          variants={fadeUp}
          custom={1}
          className="text-4xl font-bold md:text-5xl lg:text-6xl"
        >
          {t("support.title")}
        </motion.h1>

        <motion.p
          variants={fadeUp}
          custom={2}
          className="mt-6 text-lg text-text-secondary max-w-3xl mx-auto leading-relaxed"
        >
          {t("support.subtitle")}
        </motion.p>
      </motion.div>

      {/* Support Channels */}
      <motion.div
        className="grid gap-8 md:grid-cols-3 mb-20"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        {channels.map((ch, i) => (
          <motion.div
            key={ch.type}
            variants={fadeUp}
            custom={i}
            className="glow-card rounded-xl p-8 text-center"
            style={{ "--glow-color": ch.glow } as React.CSSProperties}
          >
            <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-full bg-surface-elevated">
              <ch.icon className="h-7 w-7 text-accent-cyan" />
            </div>
            <h3 className="text-xl font-semibold mb-3">{t(ch.titleKey)}</h3>
            <p className="text-text-secondary text-sm leading-relaxed mb-6">
              {t(ch.descKey)}
            </p>
            {ch.type === "chat" && (
              <button
                onClick={() => {
                  window.dispatchEvent(new CustomEvent("open-chat-widget"));
                }}
                className="inline-flex items-center gap-2 rounded-lg bg-accent-cyan/10 border border-accent-cyan/30 px-5 py-2.5 text-sm font-medium text-accent-cyan hover:bg-accent-cyan/20 transition-colors"
              >
                <MessageCircle className="h-4 w-4" />
                {t(ch.actionKey)}
              </button>
            )}
            {ch.type === "email" && (
              <a
                href="mailto:support@xcelsior.ca"
                className="inline-flex items-center gap-2 rounded-lg bg-accent-violet/10 border border-accent-violet/30 px-5 py-2.5 text-sm font-medium text-accent-violet hover:bg-accent-violet/20 transition-colors"
              >
                <Mail className="h-4 w-4" />
                {t(ch.actionKey)}
              </a>
            )}
            {ch.type === "docs" && (
              <a
                href="https://docs.xcelsior.ca"
                className="inline-flex items-center gap-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30 px-5 py-2.5 text-sm font-medium text-emerald-400 hover:bg-emerald-500/20 transition-colors"
              >
                <BookOpen className="h-4 w-4" />
                {t(ch.actionKey)}
              </a>
            )}
          </motion.div>
        ))}
      </motion.div>

      {/* Info Cards */}
      <motion.div
        className="grid gap-6 md:grid-cols-2 mb-20"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <motion.div
          variants={fadeUp}
          custom={0}
          className="glow-card rounded-xl p-8"
          style={{ "--glow-color": "rgba(245,158,11,0.12)" } as React.CSSProperties}
        >
          <div className="flex items-center gap-3 mb-4">
            <Clock className="h-5 w-5 text-accent-gold" />
            <h3 className="text-lg font-semibold">{t("support.hours_title")}</h3>
          </div>
          <p className="text-text-secondary text-sm leading-relaxed">
            {t("support.hours_desc")}
          </p>
        </motion.div>

        <motion.div
          variants={fadeUp}
          custom={1}
          className="glow-card rounded-xl p-8"
          style={{ "--glow-color": "rgba(220,38,38,0.12)" } as React.CSSProperties}
        >
          <div className="flex items-center gap-3 mb-4">
            <Shield className="h-5 w-5 text-accent-red" />
            <h3 className="text-lg font-semibold">{t("support.security_title")}</h3>
          </div>
          <p className="text-text-secondary text-sm leading-relaxed">
            {t("support.security_desc")}
          </p>
        </motion.div>
      </motion.div>

      {/* CTA */}
      <motion.div
        className="text-center"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <p className="text-text-secondary mb-6">{t("support.cta")}</p>
        <Link
          href="/pricing"
          className="inline-flex items-center gap-2 rounded-lg bg-accent-cyan px-6 py-3 text-sm font-semibold text-white hover:bg-accent-cyan/90 transition-colors"
        >
          {t("support.cta_button")}
        </Link>
      </motion.div>
    </div>
  );
}
