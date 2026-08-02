"use client";

import { useCallback, useEffect, useState } from "react";
import { Copy, Check, RefreshCw, Loader2, Terminal } from "lucide-react";
import { PillToggle } from "@/components/dashboard/pill-toggle";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import { MCP_CONNECTOR_URL, oneClickInstalls } from "@/lib/mcp";
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

  // Keys are stored as a hash, so an existing one can never be shown again.
  // access_token is present only on a fresh mint; otherwise we show the masked
  // prefix, which is safe to display but cannot be pasted into a config.
  const token = conn?.access_token ?? "";
  const inUse = conn?.in_use ?? false;
  const displayKey = conn?.access_token ?? conn?.key_prefix ?? "";
  const mcpTarget = conn?.mcp_url ?? "https://xcelsior.ca/mcp";

  const promptText = conn
    ? tab === "mcp"
      ? t("dash.mcp.mcp_prompt", { token: displayKey, mcp_url: mcpTarget })
      : t("dash.mcp.cli_prompt", { token: displayKey })
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

  const tokenIndex = displayKey ? promptText.indexOf(displayKey) : -1;
  const promptBeforeToken = tokenIndex >= 0 ? promptText.slice(0, tokenIndex) : promptText;
  const promptAfterToken =
    tokenIndex >= 0 ? promptText.slice(tokenIndex + displayKey.length) : "";
  // Shown whole: an agent key is 51 characters, so there is nothing to
  // abbreviate and a partial key is worse than useless to paste.
  const abbreviatedToken = displayKey;

  return (
    <div className="mcp-connect-card glow-card glass relative mx-auto w-full max-w-2xl rounded-[22px] p-6 sm:p-8">
      <div className="brand-line mb-6 rounded-full" />

      {/* The default path: paste the URL, sign in, approve. Shown above the
          token prompt because a person should not have to hold a credential to
          connect an assistant — the token below is for automation. */}
      <div className="mb-6 rounded-xl border border-accent-cyan/30 bg-accent-cyan/5 p-4">
        <p className="text-sm font-semibold text-text-primary">{t("dash.mcp.oauth_title")}</p>
        <p className="mt-1 text-xs leading-relaxed text-text-secondary">
          {t("dash.mcp.oauth_body")}
        </p>
        <div className="mt-3 flex items-center gap-2">
          <code className="flex-1 truncate rounded-lg border border-border/60 bg-surface px-3 py-2 font-mono text-xs text-text-primary">
            {MCP_CONNECTOR_URL}
          </code>
          <button
            type="button"
            onClick={() => {
              void navigator.clipboard.writeText(MCP_CONNECTOR_URL);
              toast.success(t("dash.mcp.copied"));
            }}
            aria-label={t("dash.mcp.oauth_copy")}
            title={t("dash.mcp.oauth_copy")}
            className="rounded-lg border border-border/60 p-2 text-text-muted transition-colors hover:text-text-primary"
          >
            <Copy className="h-4 w-4" />
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {oneClickInstalls().map((install) =>
            install.href ? (
              <a
                key={install.id}
                href={install.href}
                className="rounded-full border border-border/60 px-3 py-1.5 text-xs text-text-secondary transition-colors hover:text-text-primary"
              >
                {t("dash.mcp.install_in")} {install.label}
              </a>
            ) : (
              <button
                key={install.id}
                type="button"
                onClick={() => {
                  void navigator.clipboard.writeText(install.command ?? "");
                  toast.success(t("dash.mcp.copied"));
                }}
                className="rounded-full border border-border/60 px-3 py-1.5 text-xs text-text-secondary transition-colors hover:text-text-primary"
              >
                {install.label}
              </button>
            ),
          )}
        </div>
      </div>

      <p className="mb-4 text-center text-[11px] uppercase tracking-wider text-text-muted">
        {t("dash.mcp.automation_heading")}
      </p>

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

      <p className="mb-3 text-center text-sm text-text-secondary">
        {tab === "cli" ? t("dash.mcp.cli_intro") : t("dash.mcp.prompt_intro")}
      </p>

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
          <p className="whitespace-pre-wrap break-words text-sm sm:text-base leading-relaxed text-text-primary font-medium">
            {promptBeforeToken}
            {tokenIndex >= 0 &&
              (token ? (
                <button
                  type="button"
                  onClick={handleCopyToken}
                  className="inline rounded px-1 py-0.5 bg-accent-cyan/10 font-mono text-xs sm:text-sm text-accent-cyan decoration-accent-cyan/50 underline-offset-4 transition-colors hover:bg-accent-cyan/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-cyan/50 font-bold"
                  aria-label={t("dash.mcp.copy_token")}
                  title={t("dash.mcp.copy_token")}
                >
                  {abbreviatedToken}
                </button>
              ) : (
                // Masked prefix: identifies which key is live without being
                // copyable, since there is no secret to copy.
                <span className="inline rounded px-1 py-0.5 bg-surface font-mono text-xs sm:text-sm text-text-muted font-bold">
                  {abbreviatedToken}
                </span>
              ))}
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
          // Copying a masked key would hand the agent a credential that cannot
          // work, so the copy path is closed once a key is live elsewhere.
          disabled={loading || inUse}
          className={cn(
            "inline-flex items-center gap-2 rounded-full px-6 py-2.5 text-sm font-semibold text-white shadow-sm transition-transform",
            "bg-gradient-to-r from-accent-cyan to-accent-violet hover:scale-[1.02] active:scale-100",
            "disabled:cursor-not-allowed disabled:opacity-60",
          )}
        >
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          {t("dash.mcp.copy_prompt")}
        </button>
        {/* Same control throughout — only its label changes. Once a key is in
            use this is the way forward, so it reads as the action it performs
            rather than adding a second primary-styled button. */}
        <button
          type="button"
          onClick={handleRegenerate}
          disabled={loading || regenerating}
          title={inUse ? t("dash.mcp.create_new_key") : t("dash.mcp.regenerate")}
          className="inline-flex items-center gap-1.5 rounded-full border border-border/60 px-4 py-2.5 text-xs text-text-muted transition-colors hover:text-text-primary disabled:opacity-60"
        >
          {regenerating ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <RefreshCw className="h-3.5 w-3.5" />
          )}
          {inUse ? t("dash.mcp.create_new_key") : t("dash.mcp.regenerate")}
        </button>
      </div>

      {inUse && !loading && (
        <p className="mt-3 text-center text-[10px] leading-none text-text-muted">
          {t("dash.mcp.key_in_config")}
        </p>
      )}
    </div>
  );
}
