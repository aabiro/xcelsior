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
      <Card className="max-w-lg mx-auto mt-16 border-border/60">
        <CardContent className="pt-8 pb-8 text-center space-y-4">
          <Zap className="h-10 w-10 mx-auto text-text-muted" />
          <h1 className="text-xl font-semibold">{t("dash.serverless.feature_off_title")}</h1>
          <p className="text-sm text-text-muted">{t("dash.serverless.feature_off_desc")}</p>
          <Button asChild variant="outline">
            <Link href="/dashboard">{t("dash.overview")}</Link>
          </Button>
        </CardContent>
      </Card>
    );
  }

  return <>{children}</>;
}