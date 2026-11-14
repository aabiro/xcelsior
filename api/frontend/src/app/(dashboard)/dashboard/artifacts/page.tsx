"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input, Label, Select } from "@/components/ui/input";
import {
  Package, RefreshCw, Download, File, Upload, Trash2, MapPin, Clock,
  Loader2, CheckCircle, X, FileUp,
} from "lucide-react";
import * as api from "@/lib/api";
import type { ArtifactEntry } from "@/lib/api";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";
import { useAuth } from "@/lib/auth";
import { getTeamContext } from "@/lib/team-context";
import { TeamContextBanner } from "@/components/team/team-context-banner";

const REGIONS = [
  { value: "primary", label: "Primary (durable)" },
  { value: "any", label: "Any region (lowest cost)" },
];

const ARTIFACT_TYPES = [
  { value: "model_weights", label: "Model Weights" },
  { value: "dataset", label: "Dataset" },
  { value: "checkpoint", label: "Checkpoint" },
  { value: "job_output", label: "Session Output" },
  { value: "log", label: "Log File" },
  { value: "telemetry", label: "Telemetry" },
  { value: "container_image", label: "Container Image" },
];

function artifactTypeLabel(value: string | undefined) {
  if (!value) return "";
  const plain = value.includes(".") ? value.split(".").pop()!.toLowerCase() : value;
  return ARTIFACT_TYPES.find((item) => item.value === plain)?.label || plain;
}

const EXPIRY_OPTIONS = [
  { value: "7", label: "7 days" },
  { value: "30", label: "30 days" },
  { value: "90", label: "90 days" },
  { value: "365", label: "1 year" },
  { value: "0", label: "No expiry" },
];

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)}GB`;
}

function formatWhen(value: string | number | undefined) {
  if (value == null || value === "") return "";
  const numeric = typeof value === "number" ? value : Number(value);
  if (Number.isFinite(numeric)) {
    const ms = numeric < 1e12 ? numeric * 1000 : numeric;
    return new Date(ms).toLocaleDateString();
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "" : parsed.toLocaleDateString();
}

function browserUrl(url: string | undefined) {
  return !!url && /^https?:\/\//i.test(url);
}

export default function ArtifactsPage() {
  const { t } = useLocale();
  const { user } = useAuth();
  const team = getTeamContext(user);
  const canWrite = team.canWriteInstances;
  const [artifacts, setArtifacts] = useState<ArtifactEntry[]>([]);
  const [loading, setLoading] = useState(true);

  // Upload state
  const [showUpload, setShowUpload] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [region, setRegion] = useState("any");
  const [expiryDays, setExpiryDays] = useState("30");
  const [artifactType, setArtifactType] = useState("model_weights");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    setLoading(true);
    api.fetchArtifacts()
      .then((d) => setArtifacts(d.artifacts || []))
      .catch(() => toast.error("Failed to load artifacts"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    const onTeamChanged = () => { load(); };
    window.addEventListener("xcelsior-team-changed", onTeamChanged);
    return () => window.removeEventListener("xcelsior-team-changed", onTeamChanged);
  }, [load]);

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = () => setDragOver(false);
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) { setUploadFile(file); setShowUpload(true); }
  };
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) { setUploadFile(file); setShowUpload(true); }
  };

  const handleUpload = async () => {
    if (!uploadFile) return;
    setUploading(true);
    setUploadProgress(0);

    try {
      setUploadProgress(15);
      const ticket = await api.uploadArtifact({
        job_id: "",
        filename: uploadFile.name,
        artifact_type: artifactType,
        storage_policy: region,
        retain_days: Number(expiryDays),
      });
      if (ticket.max_upload_bytes && uploadFile.size > ticket.max_upload_bytes) {
        throw new Error(`File exceeds the ${formatSize(ticket.max_upload_bytes)} upload limit`);
      }
      const sessionId = ticket.upload_session_id;
      if (!sessionId) throw new Error("Upload did not return a session");

      setUploadProgress(40);
      const direct = ticket.upload_url || ticket.url;
      let stored = false;
      if (direct && browserUrl(direct)) {
        try {
          const put = await fetch(direct, {
            method: ticket.method || "PUT",
            body: uploadFile,
            headers: ticket.headers || { "Content-Type": uploadFile.type || "application/octet-stream" },
          });
          stored = put.ok;
        } catch {
          stored = false;
        }
      }
      if (!stored) {
        await api.uploadArtifactBytes(
          sessionId,
          uploadFile,
          uploadFile.type || "application/octet-stream",
        );
      }

      setUploadProgress(80);
      await api.finalizeArtifact(sessionId);
      setUploadProgress(100);
      toast.success("Artifact uploaded successfully");
      setUploadFile(null);
      setShowUpload(false);
      load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleDownload = async (artifact: ArtifactEntry) => {
    try {
      const res = await api.downloadArtifact({
        artifact_id: artifact.artifact_id,
        job_id: artifact.job_id || "",
        filename: artifact.filename,
        artifact_type: artifact.artifact_type || "job_output",
      });
      const href = res.download_url || res.url;
      if (browserUrl(href)) {
        window.open(href, "_blank");
        return;
      }
      const fileRes = await fetch(
        `/api/artifacts/content/${encodeURIComponent(artifact.artifact_id)}`,
        { credentials: "include" },
      );
      if (!fileRes.ok) throw new Error("Download failed");
      const blob = await fileRes.blob();
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = artifact.filename || "artifact";
      link.click();
      URL.revokeObjectURL(objectUrl);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Download failed");
    }
  };

  return (
    <div className="space-y-6">
      <TeamContextBanner team={team} variant="artifacts" />
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.artifacts.title")}</h1>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowUpload(!showUpload)} disabled={!canWrite}>
            <Upload className="h-3.5 w-3.5" /> {t("dash.artifacts.upload_btn")}
          </Button>
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
          </Button>
        </div>
      </div>

      {/* Upload Area */}
      {showUpload && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><FileUp className="h-4 w-4" /> {t("dash.artifacts.upload_title")}</CardTitle>
            <CardDescription>{t("dash.artifacts.upload_desc")}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Drop zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`cursor-pointer rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
                dragOver ? "border-ice-blue bg-ice-blue/5" : "border-border hover:border-text-muted"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileSelect}
                className="hidden"
              />
              {uploadFile ? (
                <div className="flex items-center justify-center gap-3">
                  <CheckCircle className="h-5 w-5 text-emerald" />
                  <div>
                    <p className="text-sm font-medium">{uploadFile.name}</p>
                    <p className="text-xs text-text-muted">{formatSize(uploadFile.size)}</p>
                  </div>
                  <button onClick={(e) => { e.stopPropagation(); setUploadFile(null); }} className="text-text-muted hover:text-text-primary">
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ) : (
                <>
                  <Upload className="mx-auto h-8 w-8 text-text-muted mb-2" />
                  <p className="text-sm text-text-secondary">Drop a file here or click to browse</p>
                  <p className="text-xs text-text-muted mt-1">The server enforces the upload size limit</p>
                </>
              )}
            </div>

            {/* Options */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label className="text-xs">Storage Policy</Label>
                <Select value={region} onChange={(e) => setRegion(e.target.value)}>
                  {REGIONS.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Artifact Type</Label>
                <Select value={artifactType} onChange={(e) => setArtifactType(e.target.value)}>
                  {ARTIFACT_TYPES.map((item) => (
                    <option key={item.value} value={item.value}>{item.label}</option>
                  ))}
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Expiry</Label>
                <Select value={expiryDays} onChange={(e) => setExpiryDays(e.target.value)}>
                  {EXPIRY_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                </Select>
              </div>
            </div>

            {/* Progress */}
            {uploading && (
              <div className="space-y-1">
                <div className="h-2 rounded-full bg-border overflow-hidden">
                  <div
                    className="h-full rounded-full bg-ice-blue transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <p className="text-xs text-text-muted text-center">{uploadProgress}%</p>
              </div>
            )}

            <div className="flex gap-2 justify-end">
              <Button variant="outline" size="sm" onClick={() => { setShowUpload(false); setUploadFile(null); }}>
                Cancel
              </Button>
              <Button size="sm" onClick={handleUpload} disabled={!uploadFile || uploading}>
                {uploading ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Uploading…</> : "Upload Artifact"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Artifact List */}
      {loading ? (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => <div key={i} className="h-16 rounded-xl bg-surface skeleton-pulse" />)}
        </div>
      ) : artifacts.length === 0 ? (
        <Card
          className="p-12 text-center cursor-pointer hover:border-text-muted transition-colors"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <Package className="mx-auto h-12 w-12 text-text-muted mb-4" />
          <h3 className="text-lg font-semibold mb-1">{t("dash.artifacts.empty")}</h3>
          <p className="text-sm text-text-secondary mb-3">{t("dash.artifacts.empty_desc")}</p>
          <Button size="sm" onClick={() => setShowUpload(true)} disabled={!canWrite}>
            <Upload className="h-3.5 w-3.5" /> {t("dash.artifacts.empty_cta")}
          </Button>
        </Card>
      ) : (
        <div className="space-y-2">
          {artifacts.map((a) => (
            <div key={a.artifact_id} className="flex items-center justify-between rounded-lg border border-border p-4 hover:bg-surface-hover transition-colors">
              <div className="flex items-center gap-3 min-w-0">
                <File className="h-5 w-5 text-ice-blue shrink-0" />
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate">{a.filename}</p>
                  <div className="flex items-center gap-2 text-xs text-text-muted flex-wrap">
                    {a.job_id && <span>Job: {a.job_id.slice(0, 8)}</span>}
                    {a.size_bytes != null && <span>· {formatSize(a.size_bytes)}</span>}
                    {a.storage_policy && (
                      <span className="flex items-center gap-0.5">
                        · <MapPin className="h-3 w-3" /> {a.storage_policy}
                      </span>
                    )}
                    {a.artifact_type && <span>· {artifactTypeLabel(a.artifact_type)}</span>}
                    {a.created_at && <span>· {formatWhen(a.created_at)}</span>}
                  </div>
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={() => handleDownload(a)}>
                <Download className="h-3.5 w-3.5" /> Download
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
