"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { completeBrowserOAuthLogin } from "@/lib/api";
import { useAuth } from "@/lib/auth";

function OAuthCallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login } = useAuth();
  const code = searchParams.get("code");
  const state = searchParams.get("state");
  const oauthError = searchParams.get("error");
  const oauthErrorDescription = searchParams.get("error_description");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    if (oauthError) {
      setError(oauthErrorDescription || oauthError);
      return () => {
        cancelled = true;
      };
    }
    if (!code || !state) {
      setError("OAuth callback is missing required parameters.");
      return () => {
        cancelled = true;
      };
    }

    (async () => {
      try {
        const result = await completeBrowserOAuthLogin(code, state);
        await login();
        if (!cancelled) {
          router.replace(result.redirectPath);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "OAuth sign-in failed");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [code, state, oauthError, oauthErrorDescription, login, router]);

  if (error) {
    return (
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
        <Card className="w-full max-w-md p-8 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-accent-red/10">
            <ShieldAlert className="h-7 w-7 text-accent-red" />
          </div>
          <h1 className="mb-2 text-2xl font-bold">Sign-in failed</h1>
          <p className="mb-6 text-sm text-text-secondary">{error}</p>
          <div className="space-y-3">
            <Link href="/login" className="block">
              <Button className="w-full">Back to Login</Button>
            </Link>
            <Link href="/" className="block">
              <Button variant="outline" className="w-full">Go Home</Button>
            </Link>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
      <Card className="w-full max-w-md p-8 text-center">
        <Loader2 className="mx-auto mb-4 h-10 w-10 animate-spin text-ice-blue" />
        <h1 className="mb-2 text-2xl font-bold">Finishing sign-in</h1>
        <p className="text-sm text-text-secondary">
          Completing the secure browser login flow and redirecting you back.
        </p>
      </Card>
    </div>
  );
}

export default function OAuthCallbackPage() {
  return (
    <Suspense fallback={
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-ice-blue" />
      </div>
    }>
      <OAuthCallbackContent />
    </Suspense>
  );
}
