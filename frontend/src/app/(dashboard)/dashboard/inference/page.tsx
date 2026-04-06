"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Cpu, RefreshCw, Plus, Trash2, Loader2, Zap, Clock, BarChart3,
  Globe, Server, Copy, Check, DollarSign, ChevronDown,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { InferenceEndpoint, GpuAvailability } from "@/lib/api";
import { toast } from "sonner";

const DOCKER_IMAGES = [
  { label: "vLLM (Recommended)", value: "xcelsior/vllm:latest" },
  { label: "Text Generation Inference", value: "xcelsior/tgi:latest" },
  { label: "Triton Inference Server", value: "xcelsior/triton:latest" },
  { label: "Custom Image", value: "custom" },
];

function CopyableText({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <button
      onClick={copy}
      className="inline-flex items-center gap-1 font-mono text-xs text-text-muted hover:text-text-primary transition-colors cursor-pointer"
      title="Copy to clipboard"
    >
      {text}
      {copied ? <Check className="h-3 w-3 text-green-500" /> : <Copy className="h-3 w-3" />}
    </button>
  );
}

export default function InferencePage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const [endpoints, setEndpoints] = useState<InferenceEndpoint[]>([]);
  const [gpus, setGpus] = useState<GpuAvailability[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [showForm, setShowForm] = useState(false);

  // Form state
  const [modelName, setModelName] = useState("");
  const [gpuType, setGpuType] = useState("");
  const [region, setRegion] = useState("ca-east");
  const [dockerImage, setDockerImage] = useState("xcelsior/vllm:latest");
  const [customImage, setCustomImage] = useState("");
  const [minWorkers, setMinWorkers] = useState(0);
  const [maxWorkers, setMaxWorkers] = useState(3);
  const [mode, setMode] = useState<"sync" | "async">("sync");
  const [scaledownSec, setScaledownSec] = useState(300);
  const [healthEndpoint, setHealthEndpoint] = useState("/health");

  const load = useCallback(() => {
    setLoading(true);
    Promise.all([
      api.listInferenceEndpoints().catch(() => ({ endpoints: [] })),
      api.fetchAvailableGPUs().catch(() => ({ gpus: [] })),
    ]).then(([epRes, gpuRes]) => {
      setEndpoints(epRes.endpoints || []);
      setGpus(gpuRes.gpus || []);
    }).finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  // Unique GPU types and regions
  const gpuTypes = [...new Set(gpus.map((g) => g.gpu_model))];
  const regions = [...new Set(gpus.map((g) => g.region))];

  // Selected GPU price
  const selectedGpu = gpus.find((g) => g.gpu_model === gpuType && g.region === region);
  const costPerHour = selectedGpu?.price_per_hour_cad ?? 0;

  const handleCreate = async () => {
    if (!modelName.trim()) return toast.error("Model name is required");
    if (!gpuType) return toast.error("Select a GPU type");

    setCreating(true);
    try {
      const image = dockerImage === "custom" ? customImage : dockerImage;
      await api.createInferenceEndpoint({
        model_name: modelName.trim(),
        gpu_type: gpuType,
        region,
        docker_image: image,
        min_workers: minWorkers,
        max_workers: maxWorkers,
        mode,
        scaledown_window_sec: scaledownSec,
        health_endpoint: healthEndpoint || undefined,
      });
      toast.success("Endpoint deployed");
      setShowForm(false);
      setModelName("");
      setGpuType("");
      load();
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to create endpoint");
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (endpointId: string) => {
    try {
      await api.deleteInferenceEndpoint(endpointId);
      toast.success("Endpoint deleted");
      load();
    } catch {
      toast.error("Failed to delete endpoint");
    }
  };

  const activeCount = endpoints.filter((e) => e.status === "active").length;
  const totalRequests = endpoints.reduce((sum, e) => sum + (e.total_requests || 0), 0);
  const totalCost = endpoints.reduce((sum, e) => sum + (e.total_cost_cad || 0), 0);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Serverless Inference</h1>
          <p className="text-sm text-text-muted">
            Deploy models on dedicated GPUs. OpenAI-compatible at /v1/chat/completions.
            Billed in real-time from your credits.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </Button>
          <Button size="sm" onClick={() => setShowForm(!showForm)}>
            <Plus className="h-3.5 w-3.5" /> Deploy
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard label="Active Endpoints" value={activeCount} icon={Zap} glow="emerald" />
        <StatCard label="Total Requests" value={totalRequests.toLocaleString()} icon={BarChart3} glow="violet" />
        <StatCard label="Total Cost" value={`$${totalCost.toFixed(2)} CAD`} icon={DollarSign} glow="cyan" />
      </div>

      {/* Create Endpoint Form */}
      {showForm && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Deploy New Serverless Endpoint</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Model Name */}
            <div>
              <label className="block text-sm font-medium mb-1">Model</label>
              <Input
                placeholder="e.g. meta-llama/Llama-3-8B-Instruct"
                value={modelName}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setModelName(e.target.value)}
              />
            </div>

            {/* GPU Type + Region */}
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <label className="block text-sm font-medium mb-1">GPU Type</label>
                <div className="relative">
                  <select
                    value={gpuType}
                    onChange={(e) => setGpuType(e.target.value)}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm appearance-none pr-8"
                  >
                    <option value="">Select GPU...</option>
                    {gpuTypes.map((g) => {
                      const info = gpus.find((gpu) => gpu.gpu_model === g);
                      return (
                        <option key={g} value={g}>
                          {g} ({info?.vram_gb ?? "?"}GB) — ${info?.price_per_hour_cad?.toFixed(2) ?? "?"}/hr
                          {info && info.count_available > 0 ? ` (${info.count_available} avail)` : " (on-demand)"}
                        </option>
                      );
                    })}
                  </select>
                  <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-text-muted pointer-events-none" />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Region</label>
                <div className="relative">
                  <select
                    value={region}
                    onChange={(e) => setRegion(e.target.value)}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm appearance-none pr-8"
                  >
                    {regions.length > 0 ? regions.map((r) => (
                      <option key={r} value={r}>{r}</option>
                    )) : (
                      <option value="ca-east">ca-east</option>
                    )}
                  </select>
                  <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-text-muted pointer-events-none" />
                </div>
              </div>
            </div>

            {/* Docker Image */}
            <div>
              <label className="block text-sm font-medium mb-1">Container Image</label>
              <div className="relative">
                <select
                  value={dockerImage}
                  onChange={(e) => setDockerImage(e.target.value)}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm appearance-none pr-8"
                >
                  {DOCKER_IMAGES.map((img) => (
                    <option key={img.value} value={img.value}>{img.label}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-text-muted pointer-events-none" />
              </div>
              {dockerImage === "custom" && (
                <Input
                  className="mt-2"
                  placeholder="your-registry/your-image:tag"
                  value={customImage}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setCustomImage(e.target.value)}
                />
              )}
            </div>

            {/* Workers + Mode */}
            <div className="grid gap-4 sm:grid-cols-4">
              <div>
                <label className="block text-sm font-medium mb-1">Min Workers</label>
                <Input
                  type="number"
                  min={0}
                  max={10}
                  value={minWorkers}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setMinWorkers(Number(e.target.value))}
                />
                <p className="text-xs text-text-muted mt-0.5">0 = scale to zero</p>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Max Workers</label>
                <Input
                  type="number"
                  min={1}
                  max={10}
                  value={maxWorkers}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setMaxWorkers(Number(e.target.value))}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Mode</label>
                <div className="flex gap-1 rounded-md border border-border p-0.5">
                  <button
                    onClick={() => setMode("sync")}
                    className={`flex-1 rounded px-2 py-1.5 text-xs font-medium transition-colors ${mode === "sync" ? "bg-primary text-primary-foreground" : "text-text-muted hover:text-text-primary"}`}
                  >
                    Sync
                  </button>
                  <button
                    onClick={() => setMode("async")}
                    className={`flex-1 rounded px-2 py-1.5 text-xs font-medium transition-colors ${mode === "async" ? "bg-primary text-primary-foreground" : "text-text-muted hover:text-text-primary"}`}
                  >
                    Async
                  </button>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Scaledown</label>
                <div className="relative">
                  <select
                    value={scaledownSec}
                    onChange={(e) => setScaledownSec(Number(e.target.value))}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm appearance-none pr-8"
                  >
                    <option value={60}>1 min</option>
                    <option value={300}>5 min</option>
                    <option value={900}>15 min</option>
                    <option value={1800}>30 min</option>
                    <option value={3600}>1 hour</option>
                  </select>
                  <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-text-muted pointer-events-none" />
                </div>
              </div>
            </div>

            {/* Health Endpoint */}
            <div>
              <label className="block text-sm font-medium mb-1">Health Endpoint</label>
              <Input
                value={healthEndpoint}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                  let v = e.target.value;
                  if (v && !v.startsWith("/")) v = "/" + v;
                  setHealthEndpoint(v);
                }}
                placeholder="/health"
              />
              <p className="text-xs text-text-muted mt-0.5">Path the scheduler pings to verify your container is ready.</p>
            </div>

            {/* Cost Estimate */}
            {gpuType && (
              <div className="rounded-lg border border-border bg-muted/50 p-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-text-muted">Estimated Cost</span>
                  <span className="font-medium">
                    ${costPerHour.toFixed(2)} CAD/hr per worker
                    {minWorkers > 0 && (
                      <span className="text-text-muted"> · ~${(costPerHour * minWorkers * 24).toFixed(2)}/day</span>
                    )}
                  </span>
                </div>
                <p className="text-xs text-text-muted mt-1">
                  Compute billed per-second from your credits. Token usage billed separately ($0.50/1M input, $1.50/1M output).
                  {minWorkers === 0 && " Scale-to-zero: no charge when idle."}
                </p>
              </div>
            )}

            {/* Actions */}
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowForm(false)}>Cancel</Button>
              <Button onClick={handleCreate} disabled={creating}>
                {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
                Deploy Endpoint
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Endpoints List */}
      <div className="space-y-3">
        {loading ? (
          <div className="flex justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
          </div>
        ) : endpoints.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center text-text-muted">
              <Cpu className="mx-auto h-8 w-8 mb-2 opacity-50" />
              No serverless endpoints yet. Deploy a model to get started.
            </CardContent>
          </Card>
        ) : (
          endpoints.map((ep) => (
            <Card key={ep.endpoint_id}>
              <CardContent className="py-4 space-y-3">
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold">{ep.model_id || ep.model_name}</span>
                      <Badge variant={ep.status === "active" ? "active" : "warning"}>
                        {ep.status}
                      </Badge>
                      {ep.mode && (
                        <Badge variant="default" className="text-xs">{ep.mode}</Badge>
                      )}
                    </div>
                    <CopyableText text={ep.endpoint_id} />
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDelete(ep.endpoint_id)}
                    className="text-red-500 hover:text-red-600"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
                <div className="grid gap-x-6 gap-y-1 sm:grid-cols-2 lg:grid-cols-4 text-xs text-text-muted">
                  <div className="flex items-center gap-1">
                    <Server className="h-3 w-3" /> GPU: {ep.gpu_type || "Auto"}
                  </div>
                  <div className="flex items-center gap-1">
                    <Globe className="h-3 w-3" /> Region: {ep.region || "ca-east"}
                  </div>
                  <div className="flex items-center gap-1">
                    <Cpu className="h-3 w-3" /> Workers: {ep.min_workers}–{ep.max_workers}
                  </div>
                  <div className="flex items-center gap-1">
                    <DollarSign className="h-3 w-3" /> Cost: ${(ep.total_cost_cad || 0).toFixed(2)} CAD
                  </div>
                  <div className="flex items-center gap-1">
                    <BarChart3 className="h-3 w-3" /> Requests: {(ep.total_requests || 0).toLocaleString()}
                  </div>
                  <div className="flex items-center gap-1">
                    <Zap className="h-3 w-3" /> Tokens: {(ep.total_tokens_generated || 0).toLocaleString()}
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="h-3 w-3" /> Image: {(ep.docker_image || "vllm").split("/").pop()}
                  </div>
                  {ep.cost_per_hour_cad !== undefined && ep.cost_per_hour_cad > 0 && (
                    <div className="flex items-center gap-1">
                      <DollarSign className="h-3 w-3" /> ${ep.cost_per_hour_cad.toFixed(2)}/hr
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}
