"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Cpu, RefreshCw, Plus, Trash2, Loader2, Zap, Clock, BarChart3,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { InferenceEndpoint } from "@/lib/api";
import { toast } from "sonner";

export default function InferencePage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const [endpoints, setEndpoints] = useState<InferenceEndpoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newModel, setNewModel] = useState("");
  const [newMaxWorkers, setNewMaxWorkers] = useState(3);

  const load = useCallback(() => {
    setLoading(true);
    api.listInferenceEndpoints()
      .then((res) => setEndpoints(res.endpoints || []))
      .catch(() => toast.error("Failed to load endpoints"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleCreate = async () => {
    if (!newModel.trim()) return toast.error("Model name required");
    setCreating(true);
    try {
      await api.createInferenceEndpoint({
        model_name: newModel.trim(),
        max_workers: newMaxWorkers,
      });
      toast.success("Endpoint created");
      setNewModel("");
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
  const avgLatency = endpoints.length
    ? Math.round(endpoints.reduce((sum, e) => sum + (e.avg_latency_ms || 0), 0) / endpoints.length)
    : 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Serverless Inference</h1>
          <p className="text-sm text-text-muted">
            Deploy models as API endpoints. OpenAI-compatible at /v1/chat/completions.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-3.5 w-3.5" /> Refresh
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard label="Active Endpoints" value={activeCount} icon={Zap} />
        <StatCard label="Total Requests" value={totalRequests.toLocaleString()} icon={BarChart3} />
        <StatCard label="Avg Latency" value={`${avgLatency}ms`} icon={Clock} />
      </div>

      {/* Create Endpoint */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Deploy New Endpoint</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-3 sm:flex-row">
            <Input
              placeholder="Model name (e.g. meta-llama/Llama-3-8B)"
              value={newModel}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewModel(e.target.value)}
              className="flex-1"
            />
            <Input
              type="number"
              placeholder="Max workers"
              value={newMaxWorkers}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewMaxWorkers(Number(e.target.value))}
              className="w-32"
            />
            <Button onClick={handleCreate} disabled={creating}>
              {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
              Deploy
            </Button>
          </div>
        </CardContent>
      </Card>

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
              No inference endpoints yet. Deploy a model above.
            </CardContent>
          </Card>
        ) : (
          endpoints.map((ep) => (
            <Card key={ep.endpoint_id}>
              <CardContent className="flex items-center justify-between py-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{ep.model_name}</span>
                    <Badge variant={ep.status === "active" ? "active" : "warning"}>
                      {ep.status}
                    </Badge>
                  </div>
                  <div className="flex gap-4 text-xs text-text-muted">
                    <span>Workers: {ep.min_workers}–{ep.max_workers}</span>
                    <span>Requests: {ep.total_requests?.toLocaleString() || 0}</span>
                    <span>Avg: {Math.round(ep.avg_latency_ms || 0)}ms</span>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDelete(ep.endpoint_id)}
                  className="text-red-500 hover:text-red-600"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}
