"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";

import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Cpu, RefreshCw, Plus, Trash2, Loader2, Zap, BarChart3, DollarSign,
  Server, Globe, ChevronRight, Rocket,
} from "lucide-react";
import { FadeIn, StaggerList, StaggerItem } from "@/components/ui/motion";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { ServerlessEndpoint } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { getTeamContext } from "@/lib/team-context";
import { TeamContextBanner } from "@/components/team/team-context-banner";
import { CopyableText } from "@/features/serverless/copyable-text";
import { EngineBadge, ServerlessEmptyState, ServerlessHero } from "@/features/serverless/serverless-ui";
import { toast } from "sonner";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";

function serverlessActionError(err: unknown, viewerMessage: string): string {
  const msg = err instanceof Error ? err.message : "Request failed";
  return /team viewers cannot/i.test(msg) ? viewerMessage : msg;
}

export default function InferencePage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const team = getTeamContext(user);
  const canWrite = team.canWriteInstances;
  const [endpoints, setEndpoints] = useState<ServerlessEndpoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);

  const load = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoading(true);
    try {
      const res = await api.listServerlessEndpoints();
      setEndpoints(res.endpoints || []);
    } catch {
      toast.error(t("dash.serverless.load_failed"));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => { void load(false); }, [load]);

  useEffect(() => {
    const onTeamChanged = () => { void load(false); };
    window.addEventListener("xcelsior-team-changed", onTeamChanged);
    return () => window.removeEventListener("xcelsior-team-changed", onTeamChanged);
  }, [load]);

  useEventStream({
    eventTypes: [
      "serverless_endpoint.created",
      "serverless_endpoint.deleted",
      "serverless_endpoint.updated",
      "serverless_job.completed",
      "serverless_worker.scaled",
    ],
    onEvent: () => { void load(false); },
  });

  const requestDelete = (e: React.MouseEvent, endpointId: string) => {
    e.preventDefault();
    e.stopPropagation();
    if (!canWrite) return toast.error(t("dash.serverless.viewer_blocked"));
    setPendingDeleteId(endpointId);
  };

  const confirmDelete = async () => {
    if (!pendingDeleteId) return;
    const endpointId = pendingDeleteId;
    setPendingDeleteId(null);
    try {
      await api.deleteServerlessEndpoint(endpointId);
      toast.success(t("dash.serverless.deleted"));
      void load(false);
    } catch (err) {
      toast.error(serverlessActionError(err, t("dash.serverless.viewer_blocked")));
    }
  };

  const activeCount = endpoints.filter((e) => e.status === "active").length;
  const totalRequests = endpoints.reduce((sum, e) => sum + (e.total_requests || 0), 0);
  const totalCost = endpoints.reduce((sum, e) => sum + (e.total_cost_cad || 0), 0);

  return (
    <div className="space-y-6">
      <TeamContextBanner team={team} variant="general" />

      <FadeIn>
        <ServerlessHero
          icon={Zap}
          badge="GPU-seconds"
          title={t("dash.serverless.title")}
          description={t("dash.serverless.subtitle")}
          accent="cyan"
        >
          <Button variant="outline" size="sm" onClick={() => void load(true)}>
            <RefreshCw className="h-3.5 w-3.5" />
          </Button>
          {canWrite ? (
            <Link href="/dashboard/inference/new">
              <Button size="sm">
                <Rocket className="h-3.5 w-3.5" /> {t("dash.serverless.open_studio")}
              </Button>
            </Link>
          ) : (
            <Button size="sm" disabled title={t("dash.serverless.viewer_blocked")}>
              <Plus className="h-3.5 w-3.5" /> {t("dash.serverless.view_only")}
            </Button>
          )}
        </ServerlessHero>
      </FadeIn>

      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard label={t("dash.serverless.stat_active")} value={activeCount} icon={Zap} glow="emerald" />
        <StatCard label={t("dash.serverless.stat_requests")} value={totalRequests.toLocaleString()} icon={BarChart3} glow="violet" />
        <StatCard label={t("dash.serverless.stat_cost")} value={`$${totalCost.toFixed(2)} CAD`} icon={DollarSign} glow="cyan" />
      </div>

      <div className="space-y-3">
        {loading ? (
          <div className="flex justify-center py-16">
            <Loader2 className="h-7 w-7 animate-spin text-text-muted" />
          </div>
        ) : endpoints.length === 0 ? (
          <ServerlessEmptyState
            icon={Cpu}
            title={t("dash.serverless.empty")}
            accent="cyan"
          >
            {canWrite && (
              <Link href="/dashboard/inference/new" className="inline-block mt-2">
                <Button>
                  <Rocket className="h-4 w-4" /> {t("dash.serverless.open_studio")}
                </Button>
              </Link>
            )}
          </ServerlessEmptyState>
        ) : (
          <StaggerList className="space-y-3">
            {endpoints.map((ep) => (
              <StaggerItem key={ep.endpoint_id}>
                <Link href={`/dashboard/inference/${ep.endpoint_id}`} className="block group">
                  <div className="glow-card brand-top-accent rounded-xl border border-border bg-surface p-4 transition-all group-hover:border-accent-violet/30 group-hover:shadow-[0_0_24px_rgba(139,92,246,0.08)]">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-semibold truncate">
                            {ep.name || ep.model_name || ep.model_id}
                          </span>
                          <Badge variant={ep.status === "active" ? "active" : "warning"}>{ep.status}</Badge>
                          {ep.mode && <Badge variant="default" className="text-[10px]">{ep.mode}</Badge>}
                          {ep.mode === "preset" && <EngineBadge engine={ep.managed_engine} />}
                        </div>
                        <CopyableText text={ep.endpoint_id} />
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        {canWrite && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-red-500 opacity-70 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity"
                            onClick={(e) => requestDelete(e, ep.endpoint_id)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        )}
                        <ChevronRight className="h-5 w-5 text-text-muted group-hover:text-accent-violet transition-colors" />
                      </div>
                    </div>
                    <div className="mt-3 grid gap-x-6 gap-y-1 sm:grid-cols-2 lg:grid-cols-4 text-xs text-text-muted">
                      <span className="flex items-center gap-1"><Server className="h-3 w-3" /> {ep.gpu_type || "Auto"}</span>
                      <span className="flex items-center gap-1"><Globe className="h-3 w-3" /> {ep.region}</span>
                      <span className="flex items-center gap-1"><Cpu className="h-3 w-3" /> {ep.min_workers}-{ep.max_workers} workers</span>
                      <span className="flex items-center gap-1"><DollarSign className="h-3 w-3" /> ${(ep.total_cost_cad || 0).toFixed(2)}</span>
                    </div>
                  </div>
                </Link>
              </StaggerItem>
            ))}
          </StaggerList>
        )}
      </div>

      <ConfirmDialog
        open={pendingDeleteId !== null}
        title={t("dash.serverless.delete_title") || "Delete endpoint?"}
        description={t("dash.serverless.delete_confirm")}
        confirmLabel={t("common.delete") || "Delete"}
        cancelLabel={t("common.cancel")}
        variant="danger"
        onConfirm={confirmDelete}
        onCancel={() => setPendingDeleteId(null)}
      />
    </div>
  );
}