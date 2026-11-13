"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { AlertTriangle, Check, CircleSlash, Clock, CreditCard, KeyRound, Loader2, ShieldCheck, X } from "lucide-react";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

/**
 * Where a person answers an agent that asked for a permission its key lacks.
 *
 * The agent can only make the request and hand over this link. Nothing is
 * granted until someone signed in to the dashboard presses Approve here, on a
 * page that says in plain words what the key will then be able to do.
 */
export default function McpApprovePage() {
  return (
    <Suspense fallback={null}>
      <ApproveRequest />
    </Suspense>
  );
}

type Phase = "loading" | "missing" | "ready";

function ApproveRequest() {
  const { t } = useLocale();
  const code = (useSearchParams().get("code") ?? "").trim();
  const [request, setRequest] = useState<api.McpPermissionRequest | null>(null);
  const [phase, setPhase] = useState<Phase>(code ? "loading" : "missing");
  const [deciding, setDeciding] = useState<"approve" | "decline" | null>(null);

  useEffect(() => {
    if (!code) return;
    let active = true;
    api
      .getMcpPermissionRequest(code)
      .then((result) => {
        if (!active) return;
        setRequest(result.request);
        setPhase("ready");
      })
      .catch(() => {
        if (active) setPhase("missing");
      });
    return () => {
      active = false;
    };
  }, [code]);

  const decide = useCallback(
    async (approve: boolean) => {
      if (deciding) return;
      setDeciding(approve ? "approve" : "decline");
      try {
        const result = await api.decideMcpPermissionRequest(code, approve);
        setRequest(result.request);
      } catch {
        toast.error(t("dash.mcp.approve.failed"));
      } finally {
        setDeciding(null);
      }
    },
    [code, deciding, t],
  );

  const keyLabel = request
    ? t(request.surface === "cli" ? "dash.mcp.key_label_cli" : "dash.mcp.key_label_mcp")
    : "";
  const spendsMoney = !!request?.requested.some((item) => item.spends_money);

  return (
    <div className="relative mx-auto flex min-h-full w-full max-w-xl flex-col justify-center px-4 py-8">
      <div className="mcp-connect-card glow-card glass rounded-[22px] p-6 sm:p-8">
        <div className="brand-line mb-6 rounded-full" />

        {phase === "loading" && (
          <div className="space-y-3" role="status" aria-label={t("dash.mcp.approve.loading")}>
            <div className="skeleton h-6 w-2/3 rounded" />
            <div className="skeleton h-4 w-full rounded" />
            <div className="skeleton h-24 w-full rounded-xl" />
          </div>
        )}

        {phase === "missing" && (
          <Outcome
            icon={CircleSlash}
            tone="muted"
            title={t("dash.mcp.approve.missing_title")}
            body={t("dash.mcp.approve.missing_body")}
          />
        )}

        {phase === "ready" && request?.status === "expired" && (
          <Outcome
            icon={Clock}
            tone="muted"
            title={t("dash.mcp.approve.expired_title")}
            body={t("dash.mcp.approve.expired_body")}
          />
        )}

        {phase === "ready" && request?.status === "approved" && (
          <Outcome
            icon={Check}
            tone="good"
            title={t("dash.mcp.approve.approved_title")}
            body={t("dash.mcp.approve.approved_body")}
          />
        )}

        {phase === "ready" && request?.status === "declined" && (
          <Outcome
            icon={X}
            tone="muted"
            title={t("dash.mcp.approve.declined_title")}
            body={t("dash.mcp.approve.declined_body")}
          />
        )}

        {phase === "ready" && request?.status === "pending" && (
          <>
            <div className="mb-5 flex items-start gap-3">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-accent-cyan to-accent-violet text-white">
                <ShieldCheck className="h-5 w-5" />
              </span>
              <div className="min-w-0">
                <h1 className="text-lg font-semibold text-text-primary">{t("dash.mcp.approve.title")}</h1>
                <p className="mt-1 text-sm text-text-secondary">{t("dash.mcp.approve.subtitle")}</p>
              </div>
            </div>

            <div className="mb-4 flex flex-wrap items-center gap-2 text-xs text-text-muted">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-border/70 px-2.5 py-1 font-medium">
                <KeyRound className="h-3 w-3" />
                {keyLabel}
                {request.key_prefix && <code className="font-mono">{request.key_prefix}</code>}
              </span>
              <span className="inline-flex items-center gap-1.5">
                <Clock className="h-3 w-3" />
                {t("dash.mcp.approve.expires", {
                  time: new Date(request.expires_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                })}
              </span>
            </div>

            <ul
              aria-label={t("dash.mcp.approve.list_label")}
              className="mcp-connect-prompt divide-y divide-border/40 overflow-hidden rounded-2xl border border-border"
            >
              {request.requested.map((item) => (
                <li key={item.scope} className="flex items-start gap-3 px-4 py-3">
                  <span
                    className={cn(
                      "mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-lg",
                      item.spends_money
                        ? "bg-accent-gold/15 text-accent-gold"
                        : "bg-accent-violet/15 text-accent-violet",
                    )}
                  >
                    {item.spends_money ? <CreditCard className="h-3.5 w-3.5" /> : <Check className="h-3.5 w-3.5" />}
                  </span>
                  <div className="min-w-0">
                    <p className="text-sm font-medium leading-snug text-text-primary">{item.description}</p>
                    <code className="mt-0.5 block font-mono text-[11px] text-text-muted">{item.scope}</code>
                  </div>
                </li>
              ))}
            </ul>

            {request.reason && (
              // Written by the agent, not by Xcelsior: shown as a quotation so
              // it cannot pass for part of this page's own wording.
              <figure className="mt-4 rounded-xl border border-border/60 bg-surface/40 px-4 py-3">
                <figcaption className="text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                  {t("dash.mcp.approve.reason_label")}
                </figcaption>
                <blockquote className="mt-1 break-words text-sm text-text-secondary">“{request.reason}”</blockquote>
              </figure>
            )}

            {spendsMoney && (
              <p className="mt-4 flex items-start gap-2 rounded-xl border border-accent-gold/30 bg-accent-gold/10 px-4 py-3 text-xs leading-relaxed text-accent-gold">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                {t("dash.mcp.approve.money_warning")}
              </p>
            )}

            <div className="mt-6 flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
              <button
                type="button"
                onClick={() => void decide(false)}
                disabled={!!deciding}
                className="inline-flex items-center justify-center gap-2 rounded-full border border-border/70 px-6 py-2.5 text-sm font-medium text-text-secondary transition-colors hover:text-text-primary disabled:opacity-50"
              >
                {deciding === "decline" && <Loader2 className="h-4 w-4 animate-spin" />}
                {t("dash.mcp.approve.decline")}
              </button>
              <button
                type="button"
                onClick={() => void decide(true)}
                disabled={!!deciding}
                className="inline-flex items-center justify-center gap-2 rounded-full bg-gradient-to-r from-accent-cyan to-accent-violet px-6 py-2.5 text-sm font-semibold text-white shadow-sm transition-transform hover:scale-[1.02] disabled:opacity-50 disabled:hover:scale-100"
              >
                {deciding === "approve" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
                {t("dash.mcp.approve.approve")}
              </button>
            </div>
            <p className="mt-4 text-center text-xs text-text-muted">{t("dash.mcp.approve.revoke_hint")}</p>
          </>
        )}

        {(phase === "missing" || (phase === "ready" && request?.status !== "pending")) && (
          <div className="mt-6 text-center">
            <Link
              href="/dashboard/mcp"
              className="inline-flex items-center gap-1.5 rounded-full border border-border/70 px-5 py-2 text-sm font-medium text-text-secondary transition-colors hover:text-text-primary"
            >
              {t("dash.mcp.approve.manage")}
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}

function Outcome({
  icon: Icon,
  tone,
  title,
  body,
}: {
  icon: typeof Check;
  tone: "good" | "muted";
  title: string;
  body: string;
}) {
  return (
    <div className="flex flex-col items-center gap-3 py-4 text-center">
      <span
        className={cn(
          "flex h-12 w-12 items-center justify-center rounded-2xl",
          tone === "good" ? "bg-emerald/15 text-emerald" : "bg-surface-hover text-text-muted",
        )}
      >
        <Icon className="h-6 w-6" />
      </span>
      <h1 className="text-lg font-semibold text-text-primary">{title}</h1>
      <p className="max-w-sm text-sm leading-relaxed text-text-secondary">{body}</p>
    </div>
  );
}
