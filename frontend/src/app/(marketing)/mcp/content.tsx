"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { AuthAwareLink } from "@/components/marketing/auth-aware-link";
import { ArrowRight, CheckCircle2, Zap } from "lucide-react";
import { CodeBlock } from "@/components/ui/code-block";
import { PixelField } from "@/components/ui/pixel-field";
import { m } from "@/components/marketing/motion";
import { siteIcon } from "@/lib/brand-assets";
import { useLocale } from "@/lib/locale";

const API = process.env.NEXT_PUBLIC_API_URL ?? "";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.5, ease: "easeOut" as const },
  }),
};

const FLOW_STEPS = [
  { key: "discover", art: "/mcp/flow-discover.svg", prompt: "mcp.landing.flow_prompt_discover" },
  { key: "launch", art: "/mcp/flow-launch.svg", prompt: "mcp.landing.flow_prompt_launch" },
  { key: "monitor", art: "/mcp/flow-monitor.svg", prompt: "mcp.landing.flow_prompt_monitor" },
] as const;

const BENTO = [
  { key: "discovery", icon: "globe", tools: 5, accent: "cyan" as const },
  { key: "compute", icon: "server", tools: 2, accent: "violet" as const },
  { key: "serverless", icon: "grid", tools: 4, accent: "green" as const },
  { key: "billing", icon: "coins", tools: 3, accent: "gold" as const },
  { key: "guardrails", icon: "shield-check", tools: 1, accent: "gold" as const },
  { key: "monitoring", icon: "activity", tools: 1, accent: "cyan" as const },
] as const;

const AGENT_TABS = [
  { id: "cursor", art: "/mcp/agent-cursor.svg", labelKey: "mcp.landing.agent_cursor" },
  { id: "claude", art: "/mcp/agent-claude.svg", labelKey: "mcp.landing.agent_claude" },
  { id: "vscode", art: "/mcp/agent-vscode.svg", labelKey: "mcp.landing.agent_vscode" },
] as const;

function ThemeIcon({ name }: { name: string }) {
  return (
    <>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={siteIcon(name, "dark")} className="site-theme-dark" alt="" aria-hidden />
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={siteIcon(name, "light")} className="site-theme-light" alt="" aria-hidden />
    </>
  );
}

function SectionMarker({ code, label }: { code: string; label: string }) {
  return (
    <div className="site-marker">
      <span className="site-marker-code">[ {code} ]</span>
      <span className="site-marker-line" />
      <span>{label}</span>
    </div>
  );
}

// MCP client config formats differ by agent: Cursor uses `mcpServers`+`url`,
// Claude Code adds `type: "http"`, and VS Code uses `servers`+`type: "http"`.
function mcpConfigSnippet(agentId: string): string {
  const url = "https://xcelsior.ca/mcp";
  const headers = { Authorization: "******" };
  if (agentId === "vscode") {
    return JSON.stringify({ servers: { xcelsior: { type: "http", url, headers } } }, null, 2);
  }
  if (agentId === "claude") {
    return JSON.stringify({ mcpServers: { xcelsior: { type: "http", url, headers } } }, null, 2);
  }
  return JSON.stringify({ mcpServers: { xcelsior: { url, headers } } }, null, 2);
}

function mcpConfigPath(agentId: string): string {
  if (agentId === "vscode") return ".vscode/mcp.json";
  if (agentId === "claude") return ".mcp.json (project root)";
  return "~/.cursor/mcp.json";
}

export function McpLandingContent() {
  const { t } = useLocale();
  const [gpuCount, setGpuCount] = useState<number | null>(null);
  const [agentTab, setAgentTab] = useState<(typeof AGENT_TABS)[number]["id"]>("cursor");

  useEffect(() => {
    fetch(`${API}/api/v2/gpu/available`, { credentials: "omit" })
      .then((r) => (r.ok ? r.json() : null))
      .then((body) => {
        const gpus = body?.gpus as Array<{ count_available?: number }> | undefined;
        if (!gpus) return;
        const total = gpus.reduce((s, g) => s + (Number(g.count_available) || 0), 0);
        setGpuCount(total);
      })
      .catch(() => {});
  }, []);

  const configSnippet = mcpConfigSnippet(agentTab);

  return (
    <div className="mcp-particle-page">
      <PixelField position="fixed" className="mcp-particle-field" />
      <div className="mcp-particle-content">
      <section className="site-hero">
        <div className="site-grid-bg" aria-hidden />
        <div className="site-container">
          <div className="site-rails site-hero-rails">
            <m.div initial="hidden" animate="visible" variants={fadeUp} custom={0}>
              <div className="site-pill">
                <span className="site-live-dot" />
                <span>{t("mcp.landing.badge")}</span>
              </div>
              <h1 className="site-hero-title">{t("mcp.landing.headline")}</h1>
              <p className="site-hero-copy">{t("mcp.landing.subheadline")}</p>
              <div className="site-hero-actions">
                <AuthAwareLink intent="mcp" className="site-button site-button-primary">
                  <span>{t("mcp.landing.cta_connect")}</span>
                  <ArrowRight className="site-button-icon" />
                </AuthAwareLink>
                <Link href="#tools" className="site-button site-button-ghost">
                  {t("mcp.landing.cta_tools")}
                </Link>
              </div>
            </m.div>

            <m.div initial="hidden" animate="visible" variants={fadeUp} custom={1} className="site-telemetry-wrap">
              <div className="site-telemetry-card site-hero-visual-card">
                <Image
                  src="/mcp/mcp-agent-dark.svg"
                  alt=""
                  width={640}
                  height={420}
                  priority
                  className="site-hero-illustration site-theme-dark"
                />
                <Image
                  src="/mcp/mcp-agent-light.svg"
                  alt=""
                  width={640}
                  height={420}
                  priority
                  className="site-hero-illustration site-theme-light"
                />
                <div className="site-hero-stat-grid">
                  <div className="site-hero-stat-card">
                    <div className="site-kpi-value">{gpuCount ?? "-"}</div>
                    <div className="site-kpi-label">{t("mcp.landing.stat_gpus")}</div>
                  </div>
                  <div className="site-hero-stat-card site-hero-stat-card-accent">
                    <div className="site-kpi-value">10+</div>
                    <div className="site-kpi-label">{t("mcp.landing.stat_tools")}</div>
                  </div>
                </div>
              </div>
            </m.div>
          </div>
        </div>
      </section>

      <div className="site-container">
        <section className="site-rails site-section">
          <SectionMarker code="01" label={t("mcp.landing.problem_tagline")} />
          <div className="site-contrast-grid" style={{ marginTop: 52 }}>
            {(["pain_1", "pain_2", "pain_3"] as const).map((key, index) => (
              <m.article
                key={key}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                custom={index}
                className="site-contrast-card"
              >
                <p className="site-product-badge">Today</p>
                <p className="site-card-copy">{t(`mcp.landing.${key}`)}</p>
                <div className="site-contrast-divider" />
                <p className="site-product-badge" style={{ color: "var(--cyan)" }}>With Xcelsior</p>
                <p className="site-card-copy" style={{ color: "var(--text-2)" }}>{t(`mcp.landing.solution_${index + 1}`)}</p>
              </m.article>
            ))}
          </div>
        </section>

        <section className="site-rails site-section">
          <SectionMarker code="02" label={t("mcp.landing.flow_title")} />
          <div className="site-flow-grid site-section-flush">
            {FLOW_STEPS.map((step, index) => (
              <m.article
                key={step.key}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                custom={index}
                className="site-feature-card site-flow-card"
              >
                <Image src={step.art} alt="" width={320} height={200} className="site-flow-art" />
                <p className="site-product-badge">{t(`mcp.landing.flow_${step.key}`)}</p>
                <p className="site-flow-prompt">&ldquo;{t(step.prompt)}&rdquo;</p>
              </m.article>
            ))}
          </div>
        </section>

        <section id="tools" className="site-rails site-section">
          <SectionMarker code="03" label={t("mcp.landing.bento_title")} />
          <div className="site-bento-grid" style={{ marginTop: 52 }}>
            {BENTO.map((card, index) => (
              <m.article
                key={card.key}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                custom={index}
                className="site-bento-card"
                data-accent={card.accent}
              >
                <div className="site-bento-head">
                  <div className="site-icon-box">
                    <ThemeIcon name={card.icon} />
                  </div>
                  <span className="site-bento-badge">
                    {card.tools} {t("mcp.landing.tools_label")}
                  </span>
                </div>
                <h3 className="site-card-title">{t(`mcp.landing.bento_${card.key}_title`)}</h3>
                <p className="site-card-copy">{t(`mcp.landing.bento_${card.key}_desc`)}</p>
              </m.article>
            ))}
          </div>
        </section>

        <section className="site-rails site-section">
          <SectionMarker code="04" label={t("mcp.landing.guardrails_title")} />
          <div className="site-split-panel">
            <div className="site-split-panel-media">
              <Image src="/mcp/flow-guardrails.svg" alt="" width={320} height={200} className="site-guardrails-art" />
            </div>
            <div className="site-split-panel-body">
              <h2 className="site-callout-title" style={{ color: "var(--gold)" }}>{t("mcp.landing.guardrails_title")}</h2>
              <p className="site-callout-copy">{t("mcp.landing.guardrails_desc")}</p>
              <ul className="site-checklist">
                {(["guard_1", "guard_2", "guard_3"] as const).map((key) => (
                  <li key={key} className="site-check-item">
                    <CheckCircle2 className="site-check-icon" />
                    <span>{t(`mcp.landing.${key}`)}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <section className="site-rails site-section">
          <SectionMarker code="05" label={t("mcp.landing.setup_title")} />
          <p className="site-section-copy">{t("mcp.landing.setup_desc")}</p>
          <div className="site-setup-shell">
            <div className="site-tab-list" role="tablist" aria-label={t("mcp.landing.setup_title")}>
              {AGENT_TABS.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setAgentTab(tab.id)}
                  className="site-tab"
                  data-active={agentTab === tab.id}
                  role="tab"
                  aria-selected={agentTab === tab.id}
                >
                  <Image src={tab.art} alt="" width={20} height={20} className="site-tab-art" />
                  <span>{t(tab.labelKey)}</span>
                </button>
              ))}
            </div>
            <CodeBlock filename={mcpConfigPath(agentTab)} code={configSnippet} className="site-marketing-code" />
            <div className="site-hero-actions" style={{ marginTop: 24 }}>
              <AuthAwareLink intent="mcp" className="site-button site-button-primary">
                {t("mcp.landing.setup_cta")}
              </AuthAwareLink>
            </div>
          </div>
        </section>

        <section className="site-rails site-cta">
          <h2 className="site-cta-title">{t("mcp.landing.footer_line")}</h2>
          <Link href="/gpu-availability" className="site-button site-button-primary" style={{ padding: "15px 28px" }}>
            <span>{t("mcp.landing.footer_cta")}</span>
            <Zap className="site-button-icon" />
          </Link>
        </section>
      </div>
      </div>
    </div>
  );
}
