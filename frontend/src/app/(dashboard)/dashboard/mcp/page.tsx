"use client";

import Link from "next/link";
import Image from "next/image";
import { ExternalLink } from "lucide-react";
import { FadeIn } from "@/components/ui/motion";
import { PixelField } from "@/components/ui/pixel-field";
import { McpConnectCard } from "@/components/dashboard/mcp-connect-card";
import { SITE_ASSETS } from "@/lib/brand-assets";
import { useLocale } from "@/lib/locale";

export default function McpConnectPage() {
  const { t } = useLocale();

  return (
    <div
      className="dashboard-mcp-page relative isolate h-full min-h-0 overflow-hidden"
      style={{ fontFamily: "var(--font-geist-sans), Geist, system-ui, sans-serif" }}
    >
      <PixelField count={18} className="z-0" />

      {/* Corner API reference link — mirrors a "Developer Platform" affordance */}
      <a
        href="https://docs.xcelsior.ca"
        target="_blank"
        rel="noopener noreferrer"
        className="absolute right-0 top-0 z-20 inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-surface/70 px-3 py-1.5 text-xs text-text-secondary shadow-sm backdrop-blur-sm transition-colors hover:border-border-light hover:text-text-primary"
      >
        {t("dash.mcp.api_ref")}
        <ExternalLink className="h-3 w-3" />
      </a>

      <div className="relative z-10 mx-auto flex h-full max-w-2xl flex-col items-center justify-center px-4 py-2 text-center sm:py-4">
        <FadeIn>
          <Image
            src={SITE_ASSETS.iconGradient}
            alt="Xcelsior"
            width={48}
            height={48}
            className="mb-4 h-12 w-12"
            priority
          />
        </FadeIn>

        <FadeIn delay={0.08}>
          <h1 className="brand-gradient-text mb-2 text-3xl font-bold tracking-tight sm:text-4xl">
            {t("dash.mcp.headline")}
          </h1>
          <p className="mb-5 max-w-md text-sm text-text-secondary">{t("dash.mcp.subhead")}</p>
        </FadeIn>

        <FadeIn delay={0.16} className="w-full">
          <McpConnectCard />
        </FadeIn>

        <FadeIn delay={0.24}>
          <p className="mt-4 text-xs text-text-muted">
            {t("dash.mcp.footnote")}{" "}
            <Link href="/dashboard/settings#mcp" className="text-accent-cyan underline-offset-2 hover:underline">
              {t("dash.mcp.guide")}
            </Link>
          </p>
        </FadeIn>
      </div>
    </div>
  );
}
