"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CheckCircle, XCircle, Loader2 } from "lucide-react";
import { beginBrowserOAuthLogin, verifyEmail } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import Link from "next/link";

function VerifyEmailContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { login } = useAuth();
  const { t } = useLocale();
  const token = searchParams.get("token");
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    if (!token) {
      setStatus("error");
      setErrorMsg("No verification token provided.");
      return;
    }

    verifyEmail(token)
      .then(async () => {
        try {
          await beginBrowserOAuthLogin("/dashboard");
          return;
        } catch {
          setStatus("success");
          await login();
          setTimeout(() => router.push("/dashboard"), 2000);
        }
      })
      .catch((err) => {
        setStatus("error");
        setErrorMsg(err instanceof Error ? err.message : "Verification failed");
      });
  }, [token, login, router]);

  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
      <Card className="w-full max-w-md p-8 text-center">
        {status === "loading" && (
          <>
            <Loader2 className="mx-auto h-12 w-12 animate-spin text-ice-blue mb-4" />
            <h1 className="text-xl font-bold">{t("auth.verify_verifying")}</h1>
          </>
        )}
        {status === "success" && (
          <>
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-green-500/10">
              <CheckCircle className="h-8 w-8 text-green-400" />
            </div>
            <h1 className="text-2xl font-bold mb-2">{t("auth.verify_success")}</h1>
            <p className="text-text-secondary mb-4">{t("auth.verify_redirecting")}</p>
            <Button onClick={() => router.push("/dashboard")} className="w-full">
              {t("auth.verify_go_dashboard")}
            </Button>
          </>
        )}
        {status === "error" && (
          <>
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-accent-red/10">
              <XCircle className="h-8 w-8 text-accent-red" />
            </div>
            <h1 className="text-2xl font-bold mb-2">{t("auth.verify_failed")}</h1>
            <p className="text-text-secondary mb-4">{errorMsg}</p>
            <div className="space-y-3">
              <Link href="/login">
                <Button variant="outline" className="w-full">{t("auth.verify_back_login")}</Button>
              </Link>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

export default function VerifyEmailPage() {
  return (
    <Suspense fallback={
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-ice-blue" />
      </div>
    }>
      <VerifyEmailContent />
    </Suspense>
  );
}
