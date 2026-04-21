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

const REGIONS = [
  { value: "canada_only", label: "Canada only (strict residency)" },
  { value: "canada_preferred", label: "Canada preferred (cache elsewhere)" },
  { value: "any", label: "Any region (lowest cost)" },
];

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

export default function ArtifactsPage() {
  const { t } = useLocale();
  const [artifacts, setArtifacts] = useState<ArtifactEntry[]>([]);
  const [loading, setLoading] = useState(true);

  // Upload state
  const [showUpload, setShowUpload] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [region, setRegion] = useState("ca-central-1");
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
      // Step 1: Get presigned upload URL
      setUploadProgress(20);
      const res = await api.uploadArtifact({
        job_id: "",
        filename: uploadFile.name,
        artifact_type: artifactType,
        residency_policy: region,
      });

      // Step 2: Upload file to presigned URL
      setUploadProgress(50);
      if (res.upload_url) {
        await fetch(res.upload_url, {
          method: "PUT",
          body: uploadFile,
          headers: { "Content-Type": uploadFile.type || "application/octet-stream" },
        });
      }

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
        job_id: artifact.job_id || "",
        filename: artifact.filename,
        artifact_type: artifact.artifact_type || "job_output",
      });
      if (res.download_url) {
        window.open(res.download_url, "_blank");
      }
    } catch {
      toast.error("Download failed");
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.artifacts.title")}</h1>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowUpload(!showUpload)}>
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
                  <p className="text-xs text-text-muted mt-1">Max 10GB per file</p>
                </>
              )}
            </div>

            {/* Options */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label className="text-xs">Data Residency Region</Label>
                <Select value={region} onChange={(e) => setRegion(e.target.value)}>
                  {REGIONS.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Artifact Type</Label>
                <Select value={artifactType} onChange={(e) => setArtifactType(e.target.value)}>
                  <option value="model_weights">Model Weights</option>
                  <option value="dataset">Dataset</option>
                  <option value="checkpoint">Checkpoint</option>
                  <option value="job_output">Job Output</option>
                  <option value="log">Log File</option>
                  <option value="telemetry">Telemetry</option>
                  <option value="container_image">Container Image</option>
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
          <Button size="sm" onClick={() => setShowUpload(true)}>
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
                    {a.residency_policy && (
                      <span className="flex items-center gap-0.5">
                        · <MapPin className="h-3 w-3" /> {a.residency_policy}
                      </span>
                    )}
                    {a.artifact_type && <span>· {a.artifact_type}</span>}
                    {a.created_at && <span>· {new Date(a.created_at).toLocaleDateString()}</span>}
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
