"use client";

import { useEffect, useState, useCallback } from "react";
import { Loader2 } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import { getTeamContext } from "@/lib/team-context";
import { TeamContextBanner } from "@/components/team/team-context-banner";
import { DeployStudio } from "@/features/serverless/deploy-studio";
import { EMPTY_DEPLOY_GPU_DATA, loadDeployGpuData, type DeployGpuData } from "@/features/serverless/gpu-options";


export default function DeployStudioPage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const team = getTeamContext(user);
  const canWrite = team.canWriteInstances;
  const [gpuData, setGpuData] = useState<DeployGpuData>(EMPTY_DEPLOY_GPU_DATA);
  const [loading, setLoading] = useState(true);

  // Live inventory only. A failed read is shown as a failure, never as a catalog.
  const load = useCallback(() => {
    void loadDeployGpuData(api.fetchAvailableGPUs).then((data) => {
      setGpuData(data);
      setLoading(false);
    });
  }, []);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    const onTeamChanged = () => { load(); };
    window.addEventListener("xcelsior-team-changed", onTeamChanged);
    return () => window.removeEventListener("xcelsior-team-changed", onTeamChanged);
  }, [load]);

  if (!canWrite) {
    return (
      <div className="space-y-4">
        <TeamContextBanner team={team} variant="general" />
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
      <DeployStudio gpuData={gpuData} canWrite={canWrite} />
    </div>
  );
}
