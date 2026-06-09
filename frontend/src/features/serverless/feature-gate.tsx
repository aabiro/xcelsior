"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Loader2, Zap } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";

export function ServerlessFeatureGate({ children }: { children: React.ReactNode }) {
  const { t } = useLocale();
  const [enabled, setEnabled] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    void api.getServerlessEnabled()
      .then((res) => { if (!cancelled) setEnabled(!!res.enabled); })
      .catch(() => { if (!cancelled) setEnabled(false); });
    return () => { cancelled = true; };
  }, []);

  if (enabled === null) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-accent-violet" />
      </div>
    );
  }

  if (!enabled) {
    return (
      <div className="max-w-lg mx-auto mt-16">
        <Card className="border-accent-violet/20 bg-gradient-to-br from-accent-violet/5 via-surface to-surface overflow-hidden">
          <CardContent className="pt-10 pb-10 text-center space-y-4">
            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-accent-violet/15">
              <Zap className="h-7 w-7 text-accent-violet" />
            </div>
            <h1 className="text-xl font-semibold">{t("dash.serverless.feature_off_title")}</h1>
            <p className="text-sm text-text-muted max-w-sm mx-auto">{t("dash.serverless.feature_off_desc")}</p>
            <Link href="/dashboard">
              <Button variant="outline">{t("dash.overview")}</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return <>{children}</>;
}