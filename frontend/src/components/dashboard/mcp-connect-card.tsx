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
    } catch {
      toast.error(t("dash.mcp.load_failed"));
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

  return (
    <div className="glow-card glass relative mx-auto w-full max-w-2xl rounded-[22px] p-6 sm:p-8">
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

      <p className="mb-3 text-sm text-text-secondary">{t("dash.mcp.prompt_intro")}</p>

      {/* Prompt surface */}
      <div className="rounded-xl border border-border/70 bg-surface/50 p-4">
        <div className="mb-2 flex items-center gap-2 text-[11px] font-medium uppercase tracking-wider text-text-muted">
          <Terminal className="h-3.5 w-3.5" />
          {t("dash.mcp.prompt_label")}
        </div>
        {loading ? (
          <div className="space-y-2">
            <div className="skeleton h-4 w-full rounded" />
            <div className="skeleton h-4 w-3/4 rounded" />
          </div>
        ) : (
          <p className="whitespace-pre-wrap break-words font-mono text-[13px] leading-relaxed text-text-primary">
            {promptText}
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
