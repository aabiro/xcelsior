"use client";

import { useState } from "react";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { ArrowLeft, Mail, CheckCircle } from "lucide-react";
import { requestPasswordReset } from "@/lib/api";
import { useLocale } from "@/lib/locale";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState("");
  const { t } = useLocale();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await requestPasswordReset(email);
      if (res.account_exists === false) {
        setError(t("auth.forgot_no_account"));
      } else {
        setSent(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send reset email");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
      <Card className="w-full max-w-md p-8">
        {sent ? (
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald/20">
              <CheckCircle className="h-6 w-6 text-emerald" />
            </div>
            <h1 className="text-2xl font-bold">{t("auth.forgot_success_title")}</h1>
            <p className="mt-2 text-sm text-text-secondary">
              {t("auth.forgot_success_desc", { email })}
            </p>
            <Link href="/login">
              <Button variant="outline" className="mt-6">
                <ArrowLeft className="h-4 w-4" /> {t("auth.forgot_back")}
              </Button>
            </Link>
          </div>
        ) : (
          <>
            <div className="mb-8 text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-accent-red">
                <Mail className="h-5 w-5 text-white" />
              </div>
              <h1 className="text-2xl font-bold">{t("auth.forgot_title")}</h1>
              <p className="mt-1 text-sm text-text-secondary">
                {t("auth.forgot_subtitle")}
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <div className="rounded-lg bg-accent-red/10 border border-accent-red/30 p-3 text-sm text-accent-red">
                  {error}
                </div>
              )}
              <div className="space-y-2">
                <Label htmlFor="email">{t("auth.forgot_email_label")}</Label>
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? t("auth.forgot_loading") : t("auth.forgot_button")}
              </Button>
            </form>

            <p className="mt-6 text-center text-sm text-text-secondary">
              {t("auth.forgot_remember")}{" "}
              <Link href="/login" className="text-ice-blue hover:underline">
                {t("auth.forgot_signin_link")}
              </Link>
            </p>
          </>
        )}
      </Card>
    </div>
  );
}
