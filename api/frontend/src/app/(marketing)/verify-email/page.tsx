"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { CheckCircle, Loader2, XCircle } from "lucide-react";
import { SiteAuthCard, SiteAuthHeader } from "@/components/marketing/SiteAuthShell";
import { verifyEmail } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";

function VerifyEmailContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { login } = useAuth();
  const { t } = useLocale();
  const token = searchParams.get("token");
  const missingToken = !token;
  const [status, setStatus] = useState<"loading" | "success" | "error">(missingToken ? "error" : "loading");
  const [errorMsg, setErrorMsg] = useState(missingToken ? "No verification token provided." : "");

  useEffect(() => {
    if (missingToken || !token) {
      return;
    }

    verifyEmail(token)
      .then(async () => {
        setStatus("success");
        await login().catch(() => {});
        setTimeout(() => router.push("/dashboard"), 2000);
      })
      .catch((err) => {
        setStatus("error");
        setErrorMsg(err instanceof Error ? err.message : "Verification failed");
      });
  }, [missingToken, token, login, router]);

  return (
    <SiteAuthCard>
      {status === "loading" ? (
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader title={t("auth.verify_verifying")} />
          <div className="site-auth-loading" style={{ minHeight: 160 }}>
            <Loader2 className="site-auth-spinner h-10 w-10 animate-spin" />
          </div>
        </div>
      ) : null}
      {status === "success" ? (
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader title={t("auth.verify_success")} subtitle={t("auth.verify_redirecting")} />
          <div className="site-auth-status-icon" data-tone="success" style={{ margin: "0 auto" }}>
            <CheckCircle className="h-7 w-7" />
          </div>
          <button type="button" onClick={() => router.push("/dashboard")} className="site-button site-button-primary site-auth-button">
            {t("auth.verify_go_dashboard")}
          </button>
        </div>
      ) : null}
      {status === "error" ? (
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader title={t("auth.verify_failed")} subtitle={errorMsg} />
          <div className="site-auth-status-icon" data-tone="error" style={{ margin: "0 auto" }}>
            <XCircle className="h-7 w-7" />
          </div>
          <Link href="/login" className="site-button site-button-ghost site-auth-button">
            {t("auth.verify_back_login")}
          </Link>
        </div>
      ) : null}
    </SiteAuthCard>
  );
}

export default function VerifyEmailPage() {
  return (
    <Suspense fallback={<div className="site-auth-loading"><Loader2 className="site-auth-spinner h-8 w-8 animate-spin" /></div>}>
      <VerifyEmailContent />
    </Suspense>
  );
}
