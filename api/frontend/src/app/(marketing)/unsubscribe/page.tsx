"use client";

import { Suspense, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { CheckCircle, Loader2, MailX, XCircle } from "lucide-react";
import { SiteAuthCard, SiteAuthHeader } from "@/components/marketing/SiteAuthShell";
import { unsubscribeWithToken } from "@/lib/api";
import { useLocale } from "@/lib/locale";

/**
 * Where the Unsubscribe link in a product-update email lands.
 *
 * It asks before acting: mail scanners and link previews open every link in
 * a message, and opening this page must not unsubscribe anyone. (A mail
 * client's own one-click unsubscribe posts straight to the API instead.)
 * No sign-in: the token in the link is the authority.
 */
function UnsubscribeContent() {
  const { t } = useLocale();
  const token = useSearchParams().get("token");
  const [status, setStatus] = useState<"ask" | "working" | "done" | "error">(token ? "ask" : "error");
  const [error, setError] = useState("");

  const confirm = async () => {
    if (!token) return;
    setStatus("working");
    try {
      await unsubscribeWithToken(token);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "");
      setStatus("error");
    }
  };

  return (
    <SiteAuthCard>
      {status === "ask" || status === "working" ? (
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader title={t("auth.unsub_title")} subtitle={t("auth.unsub_desc")} />
          <div className="site-auth-status-icon" style={{ margin: "0 auto" }}>
            <MailX className="h-7 w-7" />
          </div>
          <button
            type="button"
            onClick={() => void confirm()}
            disabled={status === "working"}
            className="site-button site-button-primary site-auth-button"
          >
            {status === "working" ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            {t(status === "working" ? "auth.unsub_working" : "auth.unsub_confirm")}
          </button>
          <Link href="/" className="site-button site-button-ghost site-auth-button">
            {t("auth.unsub_home")}
          </Link>
        </div>
      ) : null}
      {status === "done" ? (
        <div className="site-auth-section site-auth-stack" role="status">
          <SiteAuthHeader title={t("auth.unsub_done")} subtitle={t("auth.unsub_done_desc")} />
          <div className="site-auth-status-icon" data-tone="success" style={{ margin: "0 auto" }}>
            <CheckCircle className="h-7 w-7" />
          </div>
          <Link href="/" className="site-button site-button-ghost site-auth-button">
            {t("auth.unsub_home")}
          </Link>
        </div>
      ) : null}
      {status === "error" ? (
        <div className="site-auth-section site-auth-stack" role="alert">
          <SiteAuthHeader title={t("auth.unsub_failed")} subtitle={error || t("auth.unsub_missing")} />
          <div className="site-auth-status-icon" data-tone="error" style={{ margin: "0 auto" }}>
            <XCircle className="h-7 w-7" />
          </div>
          {token ? (
            <button type="button" onClick={() => void confirm()} className="site-button site-button-primary site-auth-button">
              {t("auth.unsub_confirm")}
            </button>
          ) : null}
          <Link href="/" className="site-button site-button-ghost site-auth-button">
            {t("auth.unsub_home")}
          </Link>
        </div>
      ) : null}
    </SiteAuthCard>
  );
}

export default function UnsubscribePage() {
  return (
    <Suspense fallback={<div className="site-auth-loading"><Loader2 className="site-auth-spinner h-8 w-8 animate-spin" /></div>}>
      <UnsubscribeContent />
    </Suspense>
  );
}
