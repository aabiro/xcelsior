"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog } from "@/components/ui/dialog";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { Cpu, Loader2, Plus, RefreshCw, Rocket, Sparkles, Zap } from "lucide-react";
import { FadeIn } from "@/components/ui/motion";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { GpuAvailability, ServerlessEndpoint } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { getTeamContext } from "@/lib/team-context";
import { PRESET_MODELS, formatTokenRateFromPricing, type TokenPricingQuote } from "@/features/serverless/constants";
import { TokenPricingTable } from "@/features/serverless/token-pricing-table";
import { ServerlessEmptyState, ServerlessHero } from "@/features/serverless/serverless-ui";
import { formatModelDisplayName, formatServerlessChip } from "@/features/serverless/format";
import { ServerlessEndpointManagement } from "@/features/serverless/endpoint-management";
import { DeployStudio } from "@/features/serverless/deploy-studio";
import { toast } from "sonner";

function serverlessActionError(err: unknown, viewerMessage: string): string {
  const msg = err instanceof Error ? err.message : "Request failed";
  return /team viewers cannot/i.test(msg) ? viewerMessage : msg;
}

export default function InferencePage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const router = useRouter();
  const searchParams = useSearchParams();
  const team = getTeamContext(user);
  const canWrite = team.canWriteInstances;
  const [endpoints, setEndpoints] = useState<ServerlessEndpoint[]>([]);
  const [gpus, setGpus] = useState<GpuAvailability[]>([]);
  const [loading, setLoading] = useState(true);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [tokenQuotes, setTokenQuotes] = useState<Record<string, TokenPricingQuote>>({});
  const [activeTab, setActiveTab] = useState<"endpoints" | "catalog">("endpoints");
  const [createOpen, setCreateOpen] = useState(false);

  const selectedEndpointId = searchParams.get("endpoint") || endpoints[0]?.endpoint_id || "";

  const load = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoading(true);
    try {
      const [endpointRes, gpuRes] = await Promise.all([
        api.listServerlessEndpoints(),
        api.fetchAvailableGPUs().catch(() => ({ gpus: [] })),
      ]);
      setEndpoints(endpointRes.endpoints || []);
      setGpus(gpuRes.gpus || []);
    } catch {
      toast.error(t("dash.serverless.load_failed"));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => { void load(false); }, [load]);

  useEffect(() => {
    void (async () => {
      try {
        const res = await fetch("/api/v2/serverless/preset-token-pricing", { credentials: "include" });
        if (!res.ok) return;
        const data = (await res.json()) as { quotes?: Record<string, TokenPricingQuote> };
        if (data.quotes) setTokenQuotes(data.quotes);
      } catch {
        /* token table is optional */
      }
    })();
  }, []);

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
      "serverless_worker.ready",
    ],
    onEvent: () => { void load(false); },
  });

  const selectEndpoint = (endpointId: string) => {
    setActiveTab("endpoints");
    router.replace(`/dashboard/inference?endpoint=${encodeURIComponent(endpointId)}`, { scroll: false });
  };

  const confirmDelete = async () => {
    if (!pendingDeleteId) return;
    const endpointId = pendingDeleteId;
    setPendingDeleteId(null);
    try {
      await api.deleteServerlessEndpoint(endpointId);
      toast.success(t("dash.serverless.deleted"));
      await load(false);
      const next = endpoints.find((endpoint) => endpoint.endpoint_id !== endpointId);
      if (next) selectEndpoint(next.endpoint_id);
      else router.replace("/dashboard/inference", { scroll: false });
    } catch (err) {
      toast.error(serverlessActionError(err, t("dash.serverless.viewer_blocked")));
    }
  };

  return (
    <div className="space-y-6">
      <FadeIn>
        <ServerlessHero
          icon={Zap}
          badge="¢/worker/sec"
          title={t("dash.serverless.title")}
          description={t("dash.serverless.subtitle")}
          accent="cyan"
        >
          <Button variant="outline" size="sm" onClick={() => void load(true)}>
            <RefreshCw className="h-3.5 w-3.5" />
          </Button>
          {canWrite ? (
            <Button size="sm" onClick={() => setCreateOpen(true)}>
              <Plus className="h-3.5 w-3.5" /> + ENDPOINT
            </Button>
          ) : (
            <Button size="sm" disabled title={t("dash.serverless.viewer_blocked")}>
              <Rocket className="h-3.5 w-3.5" /> {t("dash.serverless.view_only")}
            </Button>
          )}
        </ServerlessHero>
      </FadeIn>

      <div className="inline-flex rounded-lg border border-border bg-surface-hover/40 p-1">
        <button
          type="button"
          onClick={() => setActiveTab("endpoints")}
          className={`rounded-md px-3 py-1.5 text-sm ${activeTab === "endpoints" ? "bg-accent-violet/15 text-accent-violet" : "text-text-muted"}`}
        >
          My Endpoints
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("catalog")}
          className={`rounded-md px-3 py-1.5 text-sm ${activeTab === "catalog" ? "bg-accent-violet/15 text-accent-violet" : "text-text-muted"}`}
        >
          Model Catalogue
        </button>
      </div>

      {activeTab === "catalog" && (
        <div className="space-y-4">
          {Object.keys(tokenQuotes).length > 0 && (
            <FadeIn>
              <div className="space-y-2">
                <p className="text-xs font-medium text-text-muted uppercase tracking-wide">Preset LLM token rates (CAD / 1M)</p>
                <TokenPricingTable quotes={tokenQuotes} />
              </div>
            </FadeIn>
          )}
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {PRESET_MODELS.map((model) => (
              <div key={model.id} className="rounded-xl border border-border bg-surface p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="font-semibold truncate">{formatModelDisplayName(model.id)}</p>
                    <p className="mt-1 text-xs text-text-muted">{formatServerlessChip(model.task)} · {model.vram} GB VRAM</p>
                  </div>
                  <Badge variant="default">{formatServerlessChip(model.task)}</Badge>
                </div>
                {model.task !== "rerank" && formatTokenRateFromPricing(model.id, tokenQuotes[model.id]) && (
                  <p className="mt-3 text-xs font-mono text-accent-cyan">{formatTokenRateFromPricing(model.id, tokenQuotes[model.id])}</p>
                )}
                {canWrite && (
                  <Button size="sm" variant="outline" className="mt-4" onClick={() => setCreateOpen(true)}>
                    <Rocket className="h-3.5 w-3.5" /> Create endpoint
                  </Button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === "endpoints" && (
        loading && endpoints.length === 0 ? (
          <div className="flex justify-center py-16">
            <Loader2 className="h-7 w-7 animate-spin text-text-muted" />
          </div>
        ) : endpoints.length === 0 ? (
          <ServerlessEmptyState icon={Cpu} title={t("dash.serverless.empty")} accent="cyan">
            {canWrite && (
              <Button className="mt-2" onClick={() => setCreateOpen(true)}>
                <Rocket className="h-4 w-4" /> Create endpoint
              </Button>
            )}
          </ServerlessEmptyState>
        ) : (
          <ServerlessEndpointManagement
            endpoints={endpoints}
            selectedEndpointId={selectedEndpointId}
            canWrite={canWrite}
            loading={loading}
            onSelectEndpoint={selectEndpoint}
            onRefresh={() => void load(false)}
            onCreateEndpoint={() => setCreateOpen(true)}
            onDeleteEndpoint={(endpointId) => {
              if (!canWrite) return toast.error(t("dash.serverless.viewer_blocked"));
              setPendingDeleteId(endpointId);
            }}
          />
        )
      )}

      <Dialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        title="Create endpoint"
        maxWidth="max-w-6xl"
        className="h-[88vh]"
        bodyClassName="min-h-0 flex-1 overflow-hidden px-0 pb-0"
      >
        <DeployStudio gpus={gpus} canWrite={canWrite} />
      </Dialog>

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
