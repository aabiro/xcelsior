"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Download,
  Monitor,
  Smartphone,
  Globe,
  Bell,
  RefreshCw,
  Link2,
  Shield,
  Terminal,
  Cpu,
  ChevronRight,
} from "lucide-react";
import { useLocale } from "@/lib/locale";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.5, ease: "easeOut" as const },
  }),
};

type Platform = "macos" | "windows" | "linux" | "unknown";

function detectPlatform(): Platform {
  if (typeof navigator === "undefined") return "unknown";
  const ua = navigator.userAgent.toLowerCase();
  if (ua.includes("mac")) return "macos";
  if (ua.includes("win")) return "windows";
  if (ua.includes("linux")) return "linux";
  return "unknown";
}

const DOWNLOAD_BASE = "https://downloads.xcelsior.ca/desktop";

const PLATFORMS = [
  {
    id: "macos" as Platform,
    label: "macOS",
    desc: "Apple Silicon & Intel",
    file: "Xcelsior_0.1.0_aarch64.dmg",
    icon: Monitor,
  },
  {
    id: "windows" as Platform,
    label: "Windows",
    desc: "Windows 10+",
    file: "Xcelsior_0.1.0_x64-setup.exe",
    icon: Monitor,
  },
  {
    id: "linux" as Platform,
    label: "Linux",
    desc: ".deb / .AppImage / .rpm",
    file: "Xcelsior_0.1.0_amd64.AppImage",
    icon: Terminal,
  },
];

const DESKTOP_FEATURES = [
  {
    icon: Bell,
    titleKey: "download.feature_tray_title",
    descKey: "download.feature_tray_desc",
  },
  {
    icon: RefreshCw,
    titleKey: "download.feature_updates_title",
    descKey: "download.feature_updates_desc",
  },
  {
    icon: Link2,
    titleKey: "download.feature_links_title",
    descKey: "download.feature_links_desc",
  },
  {
    icon: Shield,
    titleKey: "download.feature_single_title",
    descKey: "download.feature_single_desc",
  },
  {
    icon: Cpu,
    titleKey: "download.feature_control_title",
    descKey: "download.feature_control_desc",
  },
  {
    icon: Monitor,
    titleKey: "download.feature_login_title",
    descKey: "download.feature_login_desc",
  },
];

export function DownloadContent() {
  const [detected] = useState<Platform>(() => detectPlatform());
  const { t } = useLocale();

  const primary = PLATFORMS.find((p) => p.id === detected) ?? PLATFORMS[0];
  const others = PLATFORMS.filter((p) => p.id !== primary.id);

  return (
    <div className="mx-auto max-w-7xl px-6 py-28 relative overflow-hidden">
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(currentColor 1px, transparent 1px), linear-gradient(90deg, currentColor 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      {/* Hero */}
      <motion.div
        className="relative text-center mb-20"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <motion.h1
          variants={fadeUp}
          custom={0}
          className="text-4xl font-bold md:text-5xl"
        >
          {t("download.hero_title")}
        </motion.h1>
        <motion.p
          variants={fadeUp}
          custom={1}
          className="mt-4 text-lg text-text-secondary max-w-2xl mx-auto"
        >
          {t("download.hero_subtitle")}
        </motion.p>
      </motion.div>

      {/* Download cards */}
      <motion.div
        className="relative mb-24"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        {/* Primary download */}
        <motion.div variants={fadeUp} custom={2} className="mb-8">
          <a
            href={`${DOWNLOAD_BASE}/${primary.file}`}
            className="group glow-card flex flex-col items-center gap-4 rounded-xl p-8 text-center sm:flex-row sm:text-left"
            style={
              { "--glow-color": "rgba(0,212,255,0.12)" } as React.CSSProperties
            }
          >
            <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-accent-cyan/10">
              <Download className="h-7 w-7 text-accent-cyan" />
            </div>
            <div className="flex-1">
              <p className="text-xl font-semibold">
                {t("download.primary_for", { platform: primary.label })}
              </p>
              <p className="text-sm text-text-secondary">{primary.desc}</p>
            </div>
            <span className="rounded-lg bg-accent-cyan px-5 py-2.5 text-sm font-medium text-navy transition-colors group-hover:bg-accent-cyan/80">
              {t("download.primary_cta")}
            </span>
          </a>
        </motion.div>

        {/* Other platforms */}
        <motion.div
          variants={fadeUp}
          custom={3}
          className="grid grid-cols-1 gap-4 sm:grid-cols-2"
        >
          {others.map((p) => (
            <a
              key={p.id}
              href={`${DOWNLOAD_BASE}/${p.file}`}
              className="group glow-card flex items-center gap-4 rounded-xl p-5"
              style={
                {
                  "--glow-color": "rgba(124,58,237,0.08)",
                } as React.CSSProperties
              }
            >
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent-violet/10">
                <p.icon className="h-5 w-5 text-accent-violet" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-semibold">{p.label}</p>
                <p className="text-xs text-text-secondary">{p.desc}</p>
              </div>
              <ChevronRight className="h-4 w-4 text-text-muted group-hover:text-text-secondary transition-colors" />
            </a>
          ))}
        </motion.div>

        <motion.p
          variants={fadeUp}
          custom={4}
          className="mt-4 text-center text-xs text-text-muted"
        >
          {t("download.build_note")}
        </motion.p>
      </motion.div>

      {/* What the desktop app does */}
      <motion.div
        className="relative mb-24"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <motion.h2
          variants={fadeUp}
          custom={0}
          className="text-2xl font-bold mb-2"
        >
          {t("download.section_what_adds")}
        </motion.h2>
        <motion.p
          variants={fadeUp}
          custom={1}
          className="text-text-secondary mb-10 max-w-xl"
        >
          {t("download.section_what_adds_desc")}
        </motion.p>
        <motion.div
          className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          {DESKTOP_FEATURES.map((f, idx) => (
            <motion.div
              key={f.titleKey}
              variants={fadeUp}
              custom={idx}
              className="group glow-card rounded-xl p-6"
              style={
                {
                  "--glow-color": "rgba(0,212,255,0.08)",
                } as React.CSSProperties
              }
            >
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-accent-cyan/10 transition-colors group-hover:scale-110">
                <f.icon className="h-5 w-5 text-accent-cyan" />
              </div>
              <h3 className="mb-2 text-lg font-semibold">{t(f.titleKey)}</h3>
              <p className="text-sm text-text-secondary leading-relaxed">
                {t(f.descKey)}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </motion.div>

      {/* Mobile section */}
      <motion.div
        className="relative"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <motion.div
          variants={fadeUp}
          custom={0}
          className="glow-card rounded-xl p-8 sm:p-10"
          style={
            {
              "--glow-color": "rgba(245,158,11,0.10)",
            } as React.CSSProperties
          }
        >
          <div className="flex flex-col gap-6 sm:flex-row sm:items-start">
            <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl bg-accent-gold/10">
              <Smartphone className="h-6 w-6 text-accent-gold" />
            </div>
            <div className="flex-1">
              <h2 className="text-2xl font-bold mb-3">
                {t("download.mobile_title")}
              </h2>
              <p className="text-text-secondary leading-relaxed mb-4">
                {t("download.mobile_p1")}
              </p>
              <p className="text-text-secondary leading-relaxed mb-6">
                {t("download.mobile_p2")}
              </p>
              <div className="flex flex-wrap gap-4">
                <div className="flex items-center gap-2 text-sm text-text-secondary">
                  <Globe className="h-4 w-4 text-accent-gold" />
                  {t("download.mobile_b1")}
                </div>
                <div className="flex items-center gap-2 text-sm text-text-secondary">
                  <Bell className="h-4 w-4 text-accent-gold" />
                  {t("download.mobile_b2")}
                </div>
                <div className="flex items-center gap-2 text-sm text-text-secondary">
                  <Shield className="h-4 w-4 text-accent-gold" />
                  {t("download.mobile_b3")}
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* CTA */}
        <motion.div
          variants={fadeUp}
          custom={1}
          className="mt-12 text-center"
        >
          <p className="text-text-secondary mb-4">
            {t("download.footer_note")}
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/register"
              className="inline-flex h-10 items-center rounded-lg bg-accent-red px-5 text-sm font-medium text-white hover:bg-accent-red-hover transition-colors"
            >
              {t("download.footer_create")}
            </Link>
            <Link
              href="/login"
              className="inline-flex h-10 items-center rounded-lg border border-border px-5 text-sm font-medium text-text-secondary hover:text-text-primary hover:bg-surface-hover transition-colors"
            >
              {t("download.footer_signin")}
            </Link>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
