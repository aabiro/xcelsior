"use client";

import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { Check, Copy, KeyRound, Loader2, Lock, RefreshCw, Terminal } from "lucide-react";
import { PillToggle } from "@/components/dashboard/pill-toggle";
import { useLocale } from "@/lib/locale";
import { useAuth } from "@/lib/auth";
import { MCP_CONNECTOR_URL } from "@/lib/mcp";
import * as api from "@/lib/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

type Tab = "mcp" | "cli";

/** What the Agent Skill install step runs. Shown in a code chip, copied verbatim. */
export const SKILL_INSTALL_COMMAND = "npx skills add xcelsior-gpu/skill";

export function McpConnectCard() {
  const { user } = useAuth();
  // Each tab is its own credential, bound to the active workspace. Remount on an
  // account or workspace switch so a revealed key never outlives its context.
  return <WorkspaceConnectCard key={`${user?.user_id ?? user?.email}:${user?.team_id ?? "personal"}`} />;
}

function WorkspaceConnectCard() {
  const { t } = useLocale();
  const [tab, setTab] = useState<Tab>("mcp");
  const [connections, setConnections] = useState<Partial<Record<Tab, api.McpQuickConnect>>>({});
  const [pending, setPending] = useState<Partial<Record<Tab, boolean>>>({});
  const [failed, setFailed] = useState<Partial<Record<Tab, boolean>>>({});
  const [copied, setCopied] = useState(false);
  const inFlight = useRef<Partial<Record<Tab, Promise<void>>>>({});
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const load = useCallback(
    (surface: Tab, regenerate = false) => {
      const running = inFlight.current[surface];
      if (running) return running;
      setPending((s) => ({ ...s, [surface]: true }));
      setFailed((s) => ({ ...s, [surface]: false }));
      const request = api
        .getMcpQuickConnect(regenerate, surface)
        .then((result) => {
          if (!mounted.current) return;
          setConnections((s) => ({ ...s, [surface]: result }));
          if (regenerate) toast.success(t("dash.mcp.regenerated"));
        })
        .catch(() => {
          if (!mounted.current) return;
          setFailed((s) => ({ ...s, [surface]: true }));
          toast.error(t(regenerate ? "dash.mcp.regenerate_failed" : "dash.mcp.load_failed"));
        })
        .finally(() => {
          delete inFlight.current[surface];
          if (mounted.current) setPending((s) => ({ ...s, [surface]: false }));
        });
      inFlight.current[surface] = request;
      return request;
    },
    [t],
  );

  // Lazy per tab: opening the CLI tab is what mints the CLI key.
  useEffect(() => {
    if (!connections[tab] && !failed[tab]) void load(tab);
  }, [tab, connections, failed, load]);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(timer);
  }, [copied]);

  const conn = connections[tab];
  const loading = !!pending[tab] || (!conn && !failed[tab]);
  // Keys are stored as a hash, so an existing key can never be shown again.
  // access_token is present only on a fresh mint; otherwise the masked prefix
  // identifies the key without being something an agent could use.
  const token = conn?.access_token ?? "";
  const masked = !token && !!conn;
  const displayKey = token || conn?.key_prefix || "";
  const values: Record<string, string> = {
    token,
    mcp_url: conn?.mcp_url ?? MCP_CONNECTOR_URL,
    api_url: conn?.api_url ?? "https://xcelsior.ca",
    command: SKILL_INSTALL_COMMAND,
  };
  const template = t(tab === "cli" ? "dash.mcp.cli_prompt" : "dash.mcp.mcp_prompt");
  const promptText = template.replace(/\{(\w+)\}/g, (match, name: string) => values[name] ?? match);

  const copy = async (value: string, message: string, primary = false) => {
    try {
      await navigator.clipboard.writeText(value);
      toast.success(message);
      if (primary) setCopied(true);
    } catch {
      toast.error(t("dash.mcp.copy_failed"));
    }
  };

  // The template's placeholders become chips, so what is on screen is exactly
  // what Copy Prompt puts on the clipboard.
  const renderPrompt = () =>
    template.split(/(\{\w+\})/g).map((part, i) => {
      if (part === "{command}") {
        return (
          <button
            key={i}
            type="button"
            onClick={() => void copy(SKILL_INSTALL_COMMAND, t("dash.mcp.command_copied"))}
            title={t("dash.mcp.copy_command")}
            className="mcp-connect-chip mcp-connect-chip-command mx-0.5 inline-flex max-w-full items-center gap-1.5 whitespace-nowrap rounded-lg px-2 py-0.5 align-baseline font-mono text-[0.82em] font-semibold"
          >
            <span aria-hidden className="opacity-60">$</span>
            {SKILL_INSTALL_COMMAND}
          </button>
        );
      }
      if (part === "{token}") {
        if (masked) {
          return (
            <span
              key={i}
              title={t("dash.mcp.key_masked_hint")}
              className="mcp-connect-chip mcp-connect-chip-masked mx-0.5 inline-flex items-center gap-1.5 whitespace-nowrap rounded-lg px-2 py-0.5 align-baseline font-mono text-[0.82em] font-semibold"
            >
              <Lock aria-hidden className="h-3 w-3" />
              {displayKey}
            </span>
          );
        }
        return (
          <button
            key={i}
            type="button"
            onClick={() => void copy(token, t("dash.mcp.token_copied"))}
            title={t("dash.mcp.copy_token")}
            aria-label={t("dash.mcp.copy_token")}
            className="mcp-connect-chip mcp-connect-chip-key mx-0.5 inline max-w-full break-all rounded-lg px-2 py-0.5 align-baseline font-mono text-[0.82em] font-semibold"
          >
            {displayKey}
          </button>
        );
      }
      if (part === "{mcp_url}" || part === "{api_url}") {
        return (
          <span key={i} className="whitespace-nowrap font-mono text-[0.85em] text-text-primary">
            {values[part.slice(1, -1)]}
          </span>
        );
      }
      return <Fragment key={i}>{part}</Fragment>;
    });

  const keyLabel = t(tab === "cli" ? "dash.mcp.key_label_cli" : "dash.mcp.key_label_mcp");

  return (
    <div className="mcp-connect-card glow-card glass relative mx-auto w-full max-w-2xl rounded-[22px] p-6 sm:p-8">
      <div className="brand-line mb-6 rounded-full" />

      <div className="mb-5 flex justify-center">
        <PillToggle
          size="lg"
          value={tab}
          onChange={(id) => {
            setTab(id as Tab);
            setCopied(false);
          }}
          options={[
            { id: "mcp", label: t("dash.mcp.tab_mcp") },
            { id: "cli", label: t("dash.mcp.tab_cli") },
          ]}
        />
      </div>

      <p className="mb-4 text-center text-sm text-text-secondary">
        {tab === "cli" ? t("dash.mcp.cli_intro") : t("dash.mcp.prompt_intro")}
      </p>

      {/* Prompt surface — more opaque than the glass card behind it, so it reads
          as the focal element rather than blending into the panel. */}
      <div className="mcp-connect-prompt overflow-hidden rounded-2xl border border-border text-left">
        <div className="flex items-center justify-between gap-3 border-b border-border/60 px-4 py-2.5 sm:px-5">
          <span className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-text-muted">
            <Terminal className="h-3.5 w-3.5" />
            {t("dash.mcp.prompt_label")}
          </span>
          {!loading && !failed[tab] && (
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-medium",
                masked
                  ? "border-border/70 text-text-muted"
                  : "border-emerald/30 bg-emerald/10 text-emerald",
              )}
            >
              <KeyRound className="h-3 w-3" />
              {keyLabel} · {masked ? t(conn?.in_use ? "dash.mcp.key_status_in_use" : "dash.mcp.key_status_issued") : t("dash.mcp.key_status_ready")}
            </span>
          )}
        </div>
        <div className="px-4 py-5 sm:px-5">
          {loading ? (
            <div className="space-y-2.5" role="status" aria-label={t("dash.mcp.loading")}>
              <div className="skeleton h-4 w-full rounded" />
              <div className="skeleton h-4 w-11/12 rounded" />
              <div className="skeleton h-4 w-2/3 rounded" />
            </div>
          ) : failed[tab] ? (
            <div className="flex flex-col items-center gap-3 py-2 text-center text-sm text-text-secondary">
              {t("dash.mcp.load_failed")}
              <button
                type="button"
                onClick={() => void load(tab)}
                className="rounded-full border border-border px-4 py-1.5 text-xs font-medium text-text-primary hover:bg-surface-hover"
              >
                {t("dash.mcp.retry")}
              </button>
            </div>
          ) : (
            <p className="text-[15px] font-medium leading-[1.9] text-text-primary">{renderPrompt()}</p>
          )}
        </div>
        {!loading && !failed[tab] && (
          <div className="border-t border-border/60 px-4 py-3 text-xs leading-relaxed text-text-muted sm:px-5">
            {masked
              ? t("dash.mcp.key_in_use_note", { key: keyLabel })
              : tab === "cli"
                ? t("dash.mcp.cli_env_hint")
                : t("dash.mcp.separate_keys")}
          </div>
        )}
      </div>

      <div className="mt-5 flex flex-wrap items-center justify-center gap-3">
        <button
          type="button"
          onClick={() => void copy(promptText, t("dash.mcp.copied"), true)}
          // A masked key would hand the agent a credential that cannot work,
          // so copying is closed until a fresh key is created.
          disabled={loading || masked || !!failed[tab]}
          className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-accent-cyan to-accent-violet px-6 py-2.5 text-sm font-semibold text-white shadow-sm transition-transform hover:scale-[1.02] active:scale-100 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100"
        >
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          {t("dash.mcp.copy_prompt")}
        </button>
        <button
          type="button"
          onClick={() => void load(tab, true)}
          disabled={loading}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full border px-4 py-2.5 text-xs font-medium transition-colors disabled:opacity-50",
            masked
              ? "border-accent-cyan/40 text-accent-cyan hover:bg-accent-cyan/10"
              : "border-border/60 text-text-muted hover:text-text-primary",
          )}
        >
          {pending[tab] ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          {masked ? t("dash.mcp.create_new_key") : t("dash.mcp.regenerate")}
        </button>
      </div>
    </div>
  );
}
