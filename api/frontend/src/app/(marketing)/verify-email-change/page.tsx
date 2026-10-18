"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2, MailCheck, XCircle } from "lucide-react";
import { toast } from "sonner";
import { SiteAuthCard, SiteAuthHeader } from "@/components/marketing/SiteAuthShell";
import { confirmEmailChange } from "@/lib/api";
import { useAuth } from "@/lib/auth";

function VerifyEmailChangeContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { refreshUser } = useAuth();
  const token = searchParams.get("token");
  const missingToken = !token;
  const [status, setStatus] = useState<"loading" | "success" | "error">(missingToken ? "error" : "loading");
  const [newEmail, setNewEmail] = useState("");
  const [errorMsg, setErrorMsg] = useState(missingToken ? "No confirmation token provided." : "");
  const ran = useRef(false);

  useEffect(() => {
    if (missingToken || !token || ran.current) return;
    ran.current = true;

    confirmEmailChange(token)
      .then(async (res) => {
        setNewEmail(res.email);
        setStatus("success");
        toast.success(res.message || "Email address updated");
        await refreshUser().catch(() => {});
      })
      .catch((err) => {
        setStatus("error");
        setErrorMsg(err instanceof Error ? err.message : "Confirmation failed");
      });
  }, [missingToken, token, refreshUser]);

  return (
    <SiteAuthCard>
      {status === "loading" ? (
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader title="Confirming your new email…" />
          <div className="site-auth-loading" style={{ minHeight: 160 }}>
            <Loader2 className="site-auth-spinner h-10 w-10 animate-spin" />
          </div>
        </div>
      ) : null}
      {status === "success" ? (
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader title="Email updated" subtitle={<><span>Your account email is now</span><br /><span className="site-auth-emphasis">{newEmail}</span></>} />
          <div className="site-auth-status-icon" data-tone="success" style={{ margin: "0 auto" }}>
            <MailCheck className="h-7 w-7" />
          </div>
          <button
            type="button"
            onClick={() => router.push("/dashboard/settings#profile")}
            className="site-button site-button-primary site-auth-button"
          >
            Back to settings
          </button>
        </div>
      ) : null}
      {status === "error" ? (
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader title="Couldn't confirm" subtitle={errorMsg} />
          <div className="site-auth-status-icon" data-tone="error" style={{ margin: "0 auto" }}>
            <XCircle className="h-7 w-7" />
          </div>
          <Link href="/dashboard/settings#profile" className="site-button site-button-ghost site-auth-button">
            Back to settings
          </Link>
        </div>
      ) : null}
    </SiteAuthCard>
  );
}

export default function VerifyEmailChangePage() {
  return (
    <Suspense fallback={<div className="site-auth-loading"><Loader2 className="site-auth-spinner h-8 w-8 animate-spin" /></div>}>
      <VerifyEmailChangeContent />
    </Suspense>
  );
}
