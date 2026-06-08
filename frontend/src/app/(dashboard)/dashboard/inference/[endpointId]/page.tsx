"use client";

import { use } from "react";
import { useAuth } from "@/lib/auth";
import { getTeamContext } from "@/lib/team-context";
import { TeamContextBanner } from "@/components/team/team-context-banner";
import { EndpointDetail } from "@/features/serverless/endpoint-detail";

export default function EndpointDetailPage({
  params,
}: {
  params: Promise<{ endpointId: string }>;
}) {
  const { endpointId } = use(params);
  const { user } = useAuth();
  const team = getTeamContext(user);
  const canWrite = team.canWriteInstances;

  return (
    <div className="space-y-6">
      <TeamContextBanner team={team} variant="general" />
      <EndpointDetail endpointId={endpointId} canWrite={canWrite} />
    </div>
  );
}