"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { ArrowLeft, Loader2 } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { GpuAvailability } from "@/lib/api";
import { getTeamContext } from "@/lib/team-context";
import { TeamContextBanner } from "@/components/team/team-context-banner";
import { DeployStudio } from "@/features/serverless/deploy-studio";
import { toast } from "sonner";

export default function DeployStudioPage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const team = getTeamContext(user);
  const canWrite = team.canWriteInstances;
  const [gpus, setGpus] = useState<GpuAvailability[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoading(true);
    try {
      const res = await api.fetchAvailableGPUs();
      setGpus(res.gpus || []);
    } catch {
      // GPU catalog optional for Deploy Studio — falls back to gpu-models.ts
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(false); }, [load]);

  useEffect(() => {
    const onTeamChanged = () => { void load(false); };
    window.addEventListener("xcelsior-team-changed", onTeamChanged);
    return () => window.removeEventListener("xcelsior-team-changed", onTeamChanged);
  }, [load]);

  if (!canWrite) {
    return (
      <div className="space-y-4">
        <TeamContextBanner team={team} variant="general" />
        <Link href="/dashboard/inference" className="inline-flex items-center gap-1 text-sm text-text-muted hover:text-text-primary">
          <ArrowLeft className="h-4 w-4" /> {t("dash.serverless.back_list")}
        </Link>
        <p className="text-text-muted py-12 text-center">{t("dash.serverless.viewer_blocked")}</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex justify-center py-24">
        <Loader2 className="h-8 w-8 animate-spin text-text-muted" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <TeamContextBanner team={team} variant="general" />
      <Link href="/dashboard/inference" className="inline-flex items-center gap-1 text-sm text-text-muted hover:text-text-primary">
        <ArrowLeft className="h-4 w-4" /> {t("dash.serverless.back_list")}
      </Link>
      <DeployStudio gpus={gpus} canWrite={canWrite} />
    </div>
  );
}