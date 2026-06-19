"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import {
  Bot, Shield, Zap, ArrowRight, Sparkles, Wallet, Activity,
  Search, Server, Layers, CheckCircle2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { AuroraBackground } from "@/components/ui/aurora-bg";
import { m } from "@/components/marketing/motion";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";

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
  { key: "discovery", icon: Search, tools: 5, accent: "cyan" as const },
  { key: "compute", icon: Server, tools: 2, accent: "violet" as const },
  { key: "serverless", icon: Layers, tools: 4, accent: "emerald" as const },
  { key: "billing", icon: Wallet, tools: 3, accent: "gold" as const },
  { key: "guardrails", icon: Shield, tools: 1, accent: "gold" as const },
  { key: "monitoring", icon: Activity, tools: 1, accent: "cyan" as const },
] as const;

const AGENT_TABS = [
  { id: "cursor", art: "/mcp/agent-cursor.svg", labelKey: "mcp.landing.agent_cursor" },
  { id: "claude", art: "/mcp/agent-claude.svg", labelKey: "mcp.landing.agent_claude" },
  { id: "vscode", art: "/mcp/agent-vscode.svg", labelKey: "mcp.landing.agent_vscode" },
] as const;

function accentClass(accent: "cyan" | "violet" | "emerald" | "gold") {
  const map = {
    cyan: "border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan",
    violet: "border-accent-violet/30 bg-accent-violet/10 text-accent-violet",
    emerald: "border-emerald/30 bg-emerald/10 text-emerald",
    gold: "border-accent-gold/30 bg-accent-gold/10 text-accent-gold",
  };
  return map[accent];
}

// MCP client config formats differ by agent: Cursor uses `mcpServers`+`url`,
// Claude Code adds `type: "http"`, and VS Code uses `servers`+`type: "http"`.
function mcpConfigSnippet(agentId: string): string {
  const url = "https://xcelsior.ca/mcp";
  const headers = { Authorization: "Bearer YOUR_OAUTH_TOKEN" };
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
    <div className="relative overflow-hidden">
      <AuroraBackground className="-z-10 opacity-50" />
      <section className="relative mx-auto max-w-6xl px-4 pb-20 pt-16 sm:px-6 lg:px-8 min-h-[70vh]">
          <div className="grid items-center gap-12 lg:grid-cols-2">
            <m.div initial="hidden" animate="visible" variants={fadeUp} custom={0}>
              <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-accent-violet/30 bg-accent-violet/10 px-3 py-1 text-xs font-medium text-accent-violet">
                <Sparkles className="h-3.5 w-3.5" />
                {t("mcp.landing.badge")}
              </div>
              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl lg:text-[3.25rem] lg:leading-[1.1]">
                {t("mcp.landing.headline")}
              </h1>
              <p className="mt-5 max-w-xl text-lg text-text-secondary leading-relaxed">
                {t("mcp.landing.subheadline")}
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link href="/dashboard/settings#mcp">
                  <Button size="lg" className="gap-2">
                    {t("mcp.landing.cta_connect")}
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
                <a href="#tools">
                  <Button variant="outline" size="lg">{t("mcp.landing.cta_tools")}</Button>
                </a>
              </div>
              <div className="mt-10 flex flex-wrap gap-4">
                <div className="rounded-xl border border-border/60 bg-surface/40 px-4 py-3 backdrop-blur-sm">
                  <p className="text-2xl font-bold tabular-nums">{gpuCount ?? "—"}</p>
                  <p className="text-xs text-text-muted uppercase tracking-wider">{t("mcp.landing.stat_gpus")}</p>
                </div>
                <div className="rounded-xl border border-accent-violet/25 bg-accent-violet/8 px-4 py-3">
                  <p className="text-2xl font-bold">10+</p>
                  <p className="text-xs text-text-muted uppercase tracking-wider">{t("mcp.landing.stat_tools")}</p>
                </div>
              </div>
            </m.div>
            <m.div initial="hidden" animate="visible" variants={fadeUp} custom={1} className="relative">
              <Image
                src="/mcp/hero-agent-gpu.svg"
                alt=""
                width={480}
                height={280}
                priority
                className="w-full max-w-lg mx-auto drop-shadow-2xl"
              />
            </m.div>
          </div>
      </section>

      <section className="border-y border-border/50 bg-navy-light/40 py-12">
        <div className="mx-auto max-w-4xl px-4 text-center sm:px-6">
          <p className="text-sm font-semibold uppercase tracking-widest text-accent-violet">
            {t("mcp.landing.problem_tagline")}
          </p>
          <ul className="mt-6 grid gap-4 text-left sm:grid-cols-3">
            {(["pain_1", "pain_2", "pain_3"] as const).map((k, i) => (
              <li key={k} className="rounded-xl border border-border/60 bg-surface/30 p-4">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-text-muted">Today</p>
                <p className="mt-1 text-sm text-text-secondary">{t(`mcp.landing.${k}`)}</p>
                <p className="mt-3 text-[11px] font-semibold uppercase tracking-wider text-accent-cyan">With Xcelsior</p>
                <p className="mt-1 text-sm font-medium text-text-primary">{t(`mcp.landing.solution_${i + 1}`)}</p>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
        <h2 className="text-center text-3xl font-bold tracking-tight">{t("mcp.landing.flow_title")}</h2>
        <div className="mt-12 grid gap-8 lg:grid-cols-3">
          {FLOW_STEPS.map((step, i) => (
            <m.div key={step.key} variants={fadeUp} initial="hidden" whileInView="visible" viewport={{ once: true }} custom={i}
              className="rounded-2xl border border-border/60 bg-surface/20 p-6 backdrop-blur-sm">
              <Image src={step.art} alt="" width={320} height={200} className="w-full rounded-xl" />
              <p className="mt-4 text-xs font-medium uppercase tracking-wider text-text-muted">
                {t(`mcp.landing.flow_${step.key}`)}
              </p>
              <p className="mt-2 font-mono text-sm text-accent-cyan/90">&ldquo;{t(step.prompt)}&rdquo;</p>
            </m.div>
          ))}
        </div>
      </section>

      <section id="tools" className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
        <h2 className="text-center text-3xl font-bold tracking-tight">{t("mcp.landing.bento_title")}</h2>
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {BENTO.map((card, i) => {
            const Icon = card.icon;
            return (
              <m.div
                key={card.key}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                custom={i}
                className={cn(
                  "group rounded-2xl border p-6 transition-transform hover:-translate-y-1",
                  accentClass(card.accent),
                )}
              >
                <div className="flex items-center justify-between">
                  <Icon className="h-5 w-5" />
                  <span className="rounded-full bg-black/20 px-2 py-0.5 text-xs font-medium">
                    {card.tools} {t("mcp.landing.tools_label")}
                  </span>
                </div>
                <h3 className="mt-4 text-lg font-semibold text-text-primary">
                  {t(`mcp.landing.bento_${card.key}_title`)}
                </h3>
                <p className="mt-2 text-sm text-text-secondary opacity-90">
                  {t(`mcp.landing.bento_${card.key}_desc`)}
                </p>
              </m.div>
            );
          })}
        </div>
      </section>

      <section className="bg-gradient-to-br from-accent-gold/5 via-transparent to-accent-violet/5 py-20">
        <div className="mx-auto grid max-w-6xl items-center gap-10 px-4 sm:px-6 lg:grid-cols-2">
          <Image src="/mcp/flow-guardrails.svg" alt="" width={320} height={200} className="w-full max-w-md mx-auto" />
          <div>
            <h2 className="text-3xl font-bold tracking-tight text-accent-gold">{t("mcp.landing.guardrails_title")}</h2>
            <p className="mt-4 text-text-secondary leading-relaxed">{t("mcp.landing.guardrails_desc")}</p>
            <ul className="mt-6 space-y-3">
              {(["guard_1", "guard_2", "guard_3"] as const).map((k) => (
                <li key={k} className="flex items-start gap-2 text-sm text-text-secondary">
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-accent-gold" />
                  {t(`mcp.landing.${k}`)}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-3xl px-4 py-20 sm:px-6">
        <h2 className="text-center text-3xl font-bold">{t("mcp.landing.setup_title")}</h2>
        <p className="mt-3 text-center text-text-secondary">{t("mcp.landing.setup_desc")}</p>
        <div className="mt-8 flex justify-center gap-2">
          {AGENT_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setAgentTab(tab.id)}
              className={cn(
                "flex items-center gap-2 rounded-lg border px-3 py-2 text-sm transition-colors",
                agentTab === tab.id
                  ? "border-accent-cyan/50 bg-accent-cyan/10 text-accent-cyan"
                  : "border-border/60 text-text-muted hover:text-text-primary",
              )}
            >
              <Image src={tab.art} alt="" width={20} height={20} />
              {t(tab.labelKey)}
            </button>
          ))}
        </div>
        <p className="mt-6 font-mono text-[11px] text-text-muted">{mcpConfigPath(agentTab)}</p>
        <pre className="mt-2 overflow-x-auto rounded-xl border border-border/60 bg-[#0a0e1a] p-4 text-xs leading-relaxed text-accent-cyan/90">
          {configSnippet}
        </pre>
        <div className="mt-6 flex justify-center">
          <Link href="/dashboard/settings#mcp">
            <Button>{t("mcp.landing.setup_cta")}</Button>
          </Link>
        </div>
      </section>

      <section className="border-t border-border/50 py-16 text-center">
        <p className="text-lg font-medium text-text-secondary">{t("mcp.landing.footer_line")}</p>
        <Link href="/gpu-availability" className="mt-2 inline-flex items-center gap-1 text-accent-cyan hover:underline">
          {t("mcp.landing.footer_cta")}
          <Zap className="h-4 w-4" />
        </Link>
      </section>
    </div>
  );
}