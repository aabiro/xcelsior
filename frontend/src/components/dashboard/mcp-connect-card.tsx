"use client";

import { useCallback, useEffect, useState } from "react";
import { Copy, Check, RefreshCw, Loader2, Terminal } from "lucide-react";
import { PillToggle } from "@/components/dashboard/pill-toggle";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

type Tab = "mcp" | "cli";

export function McpConnectCard() {
  const { t } = useLocale();
  const [tab, setTab] = useState<Tab>("mcp");
  const [conn, setConn] = useState<api.McpQuickConnect | null>(null);
  const [loading, setLoading] = useState(true);
  const [regenerating, setRegenerating] = useState(false);
  const [copied, setCopied] = useState(false);

  const load = useCallback(async (regenerate = false) => {
    try {
      const res = await api.getMcpQuickConnect(regenerate);
      setConn(res);
      if (regenerate) toast.success(t("dash.mcp.regenerated"));
    } catch {
      toast.error(t(regenerate ? "dash.mcp.regenerate_failed" : "dash.mcp.load_failed"));
    } finally {
      setLoading(false);
      setRegenerating(false);
    }
  }, [t]);

  useEffect(() => {
    void load(false);
  }, [load]);

  const handleRegenerate = () => {
    setRegenerating(true);
    void load(true);
  };

  const token = conn?.access_token ?? "";
  const mcpTarget = conn?.mcp_url ?? "https://xcelsior.ca/mcp";

  const promptText = conn
    ? tab === "mcp"
      ? t("dash.mcp.mcp_prompt", { token, mcp_url: mcpTarget })
      : t("dash.mcp.cli_prompt", { token })
    : "";

  const handleCopy = () => {
    if (!promptText) return;
    void navigator.clipboard.writeText(promptText);
    setCopied(true);
    toast.success(t("dash.mcp.copied"));
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopyToken = () => {
    if (!token) return;
    void navigator.clipboard.writeText(token);
    toast.success(t("dash.mcp.token_copied"));
  };

  const tokenIndex = token ? promptText.indexOf(token) : -1;
  const promptBeforeToken = tokenIndex >= 0 ? promptText.slice(0, tokenIndex) : promptText;
  const promptAfterToken = tokenIndex >= 0 ? promptText.slice(tokenIndex + token.length) : "";
  const abbreviatedToken = token.length > 24
    ? `${token.slice(0, 12)}…${token.slice(-8)}`
    : token;

  return (
    <div className="mcp-connect-card glow-card glass relative mx-auto w-full max-w-2xl rounded-[22px] p-6 sm:p-8">
      <div className="brand-line mb-6 rounded-full" />

      <div className="mb-6 flex justify-center">
        <PillToggle
          size="lg"
          value={tab}
          onChange={(id) => setTab(id as Tab)}
          options={[
            { id: "mcp", label: t("dash.mcp.tab_mcp") },
            { id: "cli", label: t("dash.mcp.tab_cli") },
          ]}
        />
      </div>

      <p className="mb-3 text-center text-sm text-text-secondary">{t("dash.mcp.prompt_intro")}</p>

      {/* Prompt surface — deliberately more opaque than the glass card behind it, so it reads
          as the focal element rather than blending into the surrounding panel. */}
      <div className="rounded-xl border border-border bg-surface p-4 shadow-inner">
        <div className="mb-2 flex items-center justify-center gap-2 text-[11px] font-medium uppercase tracking-wider text-text-muted">
          <Terminal className="h-3.5 w-3.5" />
          {t("dash.mcp.prompt_label")}
        </div>
        {loading ? (
          <div className="space-y-2">
            <div className="skeleton h-4 w-full rounded" />
            <div className="skeleton h-4 w-3/4 rounded" />
          </div>
        ) : (
          <p className="whitespace-pre-wrap break-words text-[13px] leading-relaxed text-text-primary">
            {promptBeforeToken}
            {tokenIndex >= 0 && (
              <button
                type="button"
                onClick={handleCopyToken}
                className="inline rounded-sm px-0.5 font-mono text-[10px] text-accent-cyan/70 decoration-accent-cyan/50 underline-offset-4 transition-colors hover:text-accent-cyan hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-cyan/50"
                aria-label={t("dash.mcp.copy_token")}
                title={t("dash.mcp.copy_token")}
              >
                {abbreviatedToken}
              </button>
            )}
            {promptAfterToken}
          </p>
        )}
        {tab === "cli" && !loading && (
          <p className="mt-3 border-t border-border/50 pt-3 text-xs text-text-muted">
            {t("dash.mcp.cli_env_hint")}
          </p>
        )}
      </div>

      {/* Actions */}
      <div className="mt-5 flex items-center justify-center gap-3">
        <button
          type="button"
          onClick={handleCopy}
          disabled={loading}
          className={cn(
            "inline-flex items-center gap-2 rounded-full px-6 py-2.5 text-sm font-semibold text-white shadow-sm transition-transform",
            "bg-gradient-to-r from-accent-cyan to-accent-violet hover:scale-[1.02] active:scale-100",
            "disabled:cursor-not-allowed disabled:opacity-60",
          )}
        >
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          {t("dash.mcp.copy_prompt")}
        </button>
        <button
          type="button"
          onClick={handleRegenerate}
          disabled={loading || regenerating}
          title={t("dash.mcp.regenerate")}
          className="inline-flex items-center gap-1.5 rounded-full border border-border/60 px-4 py-2.5 text-xs text-text-muted transition-colors hover:text-text-primary disabled:opacity-60"
        >
          {regenerating ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <RefreshCw className="h-3.5 w-3.5" />
          )}
          {t("dash.mcp.regenerate")}
        </button>
      </div>
    </div>
  );
}
