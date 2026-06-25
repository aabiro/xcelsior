"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { MailCheck, XCircle, Loader2 } from "lucide-react";
import { confirmEmailChange } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import Link from "next/link";
import { toast } from "sonner";

function VerifyEmailChangeContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { refreshUser } = useAuth();
  const token = searchParams.get("token");
  const missingToken = !token;
  const [status, setStatus] = useState<"loading" | "success" | "error">(missingToken ? "error" : "loading");
  const [newEmail, setNewEmail] = useState("");
  const [errorMsg, setErrorMsg] = useState(missingToken ? "No confirmation token provided." : "");
  // Guard against the effect firing twice (React strict mode) — the token is
  // single-use, so a second call would error after the first succeeds.
  const ran = useRef(false);

  useEffect(() => {
    if (missingToken || !token || ran.current) return;
    ran.current = true;

    confirmEmailChange(token)
      .then(async (res) => {
        setNewEmail(res.email);
        setStatus("success");
        toast.success(res.message || "Email address updated");
        // Refresh the in-context user so the dashboard reflects the new email.
        await refreshUser().catch(() => {});
      })
      .catch((err) => {
        setStatus("error");
        setErrorMsg(err instanceof Error ? err.message : "Confirmation failed");
      });
  }, [missingToken, token, refreshUser]);

  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
      <Card className="w-full max-w-md p-8 text-center">
        {status === "loading" && (
          <>
            <Loader2 className="mx-auto mb-4 h-12 w-12 animate-spin text-ice-blue" />
            <h1 className="text-xl font-bold">Confirming your new email…</h1>
          </>
        )}
        {status === "success" && (
          <>
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-emerald/10">
              <MailCheck className="h-8 w-8 text-emerald" />
            </div>
            <h1 className="mb-2 text-2xl font-bold">Email updated</h1>
            <p className="mb-1 text-text-secondary">Your account email is now</p>
            <p className="mb-5 font-medium text-text-primary break-all">{newEmail}</p>
            <Button onClick={() => router.push("/dashboard/settings#profile")} className="w-full">
              Back to settings
            </Button>
          </>
        )}
        {status === "error" && (
          <>
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-accent-red/10">
              <XCircle className="h-8 w-8 text-accent-red" />
            </div>
            <h1 className="mb-2 text-2xl font-bold">Couldn&apos;t confirm</h1>
            <p className="mb-4 text-text-secondary">{errorMsg}</p>
            <Link href="/dashboard/settings#profile">
              <Button variant="outline" className="w-full">Back to settings</Button>
            </Link>
          </>
        )}
      </Card>
    </div>
  );
}

export default function VerifyEmailChangePage() {
  return (
    <Suspense fallback={
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-ice-blue" />
      </div>
    }>
      <VerifyEmailChangeContent />
    </Suspense>
  );
}
