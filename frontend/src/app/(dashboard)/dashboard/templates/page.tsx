"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import NextLink from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input, Select, Label } from "@/components/ui/input";
import { Dialog } from "@/components/ui/dialog";
import {
  Layers, RefreshCw, Trash2, Loader2, Search, Star, StarOff,
  Globe, Lock, Rocket, Pencil, X, Check, Copy, Filter, Tag,
  ChevronDown, ChevronUp, ArrowUpDown,
} from "lucide-react";
import { FadeIn } from "@/components/ui/motion";
import { useAuth } from "@/lib/auth";
import * as api from "@/lib/api";
import type { UserImage, UserImageScope } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

// Phase D — /dashboard/templates
//
// Single-page surface to organize + manage every saved pod template
// (user_images rows). Three tabs: Mine / Community / All (admin).
// Every action (star, edit metadata, delete, launch) is one click.
//
// Data model: see `UserImage` in lib/api.ts. Templates come from
// `docker commit` of a running instance (POST /instances/{id}/snapshot);
// this page is the downstream organization layer on top of that.

type SortKey = "name" | "size_bytes" | "created_at" | "status";
type SortDir = "asc" | "desc";

const STATUS_BADGES: Record<UserImage["status"], { label: string; variant: "default" | "warning" | "failed" | "running" }> = {
  pending: { label: "Building", variant: "warning" },
  ready: { label: "Ready", variant: "running" },
  failed: { label: "Failed", variant: "failed" },
};

function formatBytes(n: number): string {
  if (!n) return "—";
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(0)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function formatRelative(ts: number): string {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function TemplatesPage() {
  const { user } = useAuth();
  const isAdmin = !!user?.is_admin;

  const [images, setImages] = useState<UserImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [scope, setScope] = useState<UserImageScope>("mine");
  const [starredOnly, setStarredOnly] = useState(false);
  const [query, setQuery] = useState("");
  const [labelFilter, setLabelFilter] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [editing, setEditing] = useState<UserImage | null>(null);
  const [confirmBulkDelete, setConfirmBulkDelete] = useState(false);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const res = await api.listUserImages({
        scope,
        starred: starredOnly,
        label: labelFilter || undefined,
        q: query || undefined,
        limit: 500,
      });
      setImages(res.images);
    } catch (e) {
      toast.error(`Failed to load templates: ${(e as Error).message}`);
    } finally {
      setRefreshing(false);
      setLoading(false);
    }
  }, [scope, starredOnly, labelFilter, query]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Live updates — SSE invalidates the list on create/update/delete.
  useEventStream({
    eventTypes: ["user_image_created", "user_image_updated", "user_image_deleted", "user_image_ready"],
    onEvent: () => refresh(),
  });

  // Union of every distinct label across the visible images — powers
  // the filter chip row. Sorted alphabetically for determinism.
  const allLabels = useMemo(() => {
    const s = new Set<string>();
    for (const img of images) for (const l of img.labels || []) s.add(l);
    return Array.from(s).sort();
  }, [images]);

  const sorted = useMemo(() => {
    const copy = [...images];
    copy.sort((a, b) => {
      // Starred always floats to the top within the current scope/filter,
      // regardless of the secondary sort choice. Matches the server-side
      // ORDER BY so the client doesn't reorder differently on re-renders.
      if (scope === "mine") {
        if (a.starred && !b.starred) return -1;
        if (!a.starred && b.starred) return 1;
      }
      let cmp = 0;
      switch (sortKey) {
        case "name":
          cmp = `${a.name}:${a.tag}`.localeCompare(`${b.name}:${b.tag}`);
          break;
        case "size_bytes":
          cmp = a.size_bytes - b.size_bytes;
          break;
        case "created_at":
          cmp = a.created_at - b.created_at;
          break;
        case "status":
          cmp = a.status.localeCompare(b.status);
          break;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
    return copy;
  }, [images, sortKey, sortDir, scope]);

  function toggleSort(k: SortKey) {
    if (sortKey === k) setSortDir(d => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(k); setSortDir("desc"); }
  }

  function toggleSelect(id: string) {
    setSelected(s => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id); else n.add(id);
      return n;
    });
  }

  const selectedMine = useMemo(
    () => sorted.filter(i => selected.has(i.image_id) && i.is_mine),
    [sorted, selected],
  );

  async function bulkDelete() {
    if (selectedMine.length === 0) return;
    setConfirmBulkDelete(false);
    const count = selectedMine.length;
    try {
      await Promise.all(
        selectedMine.map(img => api.deleteUserImage(img.image_id)),
      );
      toast.success(`Deleted ${count} template${count === 1 ? "" : "s"}`);
      setSelected(new Set());
      refresh();
    } catch (e) {
      toast.error(`Bulk delete failed: ${(e as Error).message}`);
      refresh();
    }
  }

  async function toggleStar(img: UserImage) {
    try {
      await api.patchUserImage(img.image_id, { starred: !img.starred });
      // Optimistic update — the SSE invalidation will repaint anyway,
      // but this avoids the 100-300 ms flash of un-toggled state.
      setImages(prev =>
        prev.map(i =>
          i.image_id === img.image_id
            ? { ...i, starred: !i.starred, starred_at: !i.starred ? Date.now() / 1000 : null }
            : i,
        ),
      );
    } catch (e) {
      toast.error(`Could not star: ${(e as Error).message}`);
    }
  }

  async function deleteOne(img: UserImage) {
    if (!confirm(`Delete template ${img.name}:${img.tag}? This cannot be undone.`)) return;
    try {
      await api.deleteUserImage(img.image_id);
      toast.success("Template deleted");
      refresh();
    } catch (e) {
      toast.error(`Delete failed: ${(e as Error).message}`);
    }
  }

  const mineCount = images.filter(i => i.is_mine).length;

  return (
    <FadeIn>
      <div className="space-y-6 p-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <Layers className="h-6 w-6 text-ice-blue" />
              Templates
            </h1>
            <p className="text-sm text-text-secondary mt-1">
              Save, organize, and launch from your pod snapshots. Snapshots are created from the
              {" "}<NextLink href="/dashboard/instances" className="text-ice-blue hover:underline">
                instances page
              </NextLink>{" "}
              via the "Save as template" action.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={refresh}
              disabled={refreshing}
              title="Refresh"
            >
              {refreshing
                ? <Loader2 className="h-4 w-4 animate-spin" />
                : <RefreshCw className="h-4 w-4" />}
            </Button>
          </div>
        </div>

        {/* Scope tabs */}
        <div className="flex items-center gap-1 border-b border-border">
          {(["mine", "public"] as const).map(s => (
            <button
              key={s}
              onClick={() => { setScope(s); setSelected(new Set()); }}
              className={cn(
                "px-4 py-2 text-sm font-medium border-b-2 transition-colors -mb-px",
                scope === s
                  ? "border-ice-blue text-text-primary"
                  : "border-transparent text-text-muted hover:text-text-primary",
              )}
            >
              {s === "mine" ? "My Templates" : "Community"}
              {s === "mine" && mineCount > 0 && (
                <span className="ml-2 text-xs text-text-muted">({mineCount})</span>
              )}
            </button>
          ))}
          {isAdmin && (
            <button
              onClick={() => { setScope("all"); setSelected(new Set()); }}
              className={cn(
                "px-4 py-2 text-sm font-medium border-b-2 transition-colors -mb-px",
                scope === "all"
                  ? "border-ice-blue text-text-primary"
                  : "border-transparent text-text-muted hover:text-text-primary",
              )}
            >
              All <Badge variant="info" className="ml-1 text-[10px]">admin</Badge>
            </button>
          )}
        </div>

        {/* Filter row */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative flex-1 min-w-[200px] max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-text-muted pointer-events-none" />
            <Input
              placeholder="Search name, tag, description…"
              value={query}
              onChange={e => setQuery(e.target.value)}
              className="pl-9"
            />
          </div>
          {scope === "mine" && (
            <Button
              variant={starredOnly ? "default" : "outline"}
              size="sm"
              onClick={() => setStarredOnly(s => !s)}
            >
              <Star className={cn("h-4 w-4 mr-1", starredOnly && "fill-yellow-400 text-yellow-400")} />
              Starred only
            </Button>
          )}
          {allLabels.length > 0 && (
            <div className="flex items-center gap-1 flex-wrap">
              <Tag className="h-4 w-4 text-text-muted" />
              <button
                onClick={() => setLabelFilter("")}
                className={cn(
                  "text-xs px-2 py-1 rounded-full border",
                  labelFilter === ""
                    ? "border-ice-blue bg-ice-blue/10 text-text-primary"
                    : "border-border text-text-muted hover:border-border-hover",
                )}
              >
                all
              </button>
              {allLabels.map(l => (
                <button
                  key={l}
                  onClick={() => setLabelFilter(labelFilter === l ? "" : l)}
                  className={cn(
                    "text-xs px-2 py-1 rounded-full border",
                    labelFilter === l
                      ? "border-ice-blue bg-ice-blue/10 text-text-primary"
                      : "border-border text-text-muted hover:border-border-hover",
                  )}
                >
                  {l}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Bulk action bar */}
        {selectedMine.length > 0 && (
          <div className="flex items-center justify-between px-4 py-2 rounded-lg bg-ice-blue/5 border border-ice-blue/30">
            <span className="text-sm">
              {selectedMine.length} selected
            </span>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                className="bg-accent-red text-white hover:bg-accent-red-hover"
                onClick={() => setConfirmBulkDelete(true)}
              >
                <Trash2 className="h-4 w-4 mr-1" />
                Delete selected
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setSelected(new Set())}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}

        {/* Table */}
        <Card>
          <CardContent className="p-0">
            {loading ? (
              <div className="flex items-center justify-center py-16">
                <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
              </div>
            ) : sorted.length === 0 ? (
              <div className="text-center py-16 px-6">
                <Layers className="h-10 w-10 mx-auto text-text-muted/50" />
                <p className="mt-3 text-sm text-text-secondary">
                  {scope === "mine"
                    ? "No templates yet. Save your first snapshot from a running instance."
                    : "No public templates match."}
                </p>
                {scope === "mine" && (
                  <NextLink
                    href="/dashboard/instances"
                    className="inline-block mt-3 text-sm text-ice-blue hover:underline"
                  >
                    Go to instances →
                  </NextLink>
                )}
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-xs text-text-muted border-b border-border">
                    <tr>
                      <th className="w-10 px-3 py-3">
                        {scope === "mine" && (
                          <input
                            type="checkbox"
                            checked={
                              selectedMine.length > 0 &&
                              selectedMine.length === sorted.filter(i => i.is_mine).length
                            }
                            onChange={e => {
                              if (e.target.checked) {
                                setSelected(new Set(sorted.filter(i => i.is_mine).map(i => i.image_id)));
                              } else {
                                setSelected(new Set());
                              }
                            }}
                          />
                        )}
                      </th>
                      <th className="w-8 px-2 py-3" />
                      <th className="text-left px-3 py-3 font-medium cursor-pointer" onClick={() => toggleSort("name")}>
                        Name <ArrowUpDown className="inline h-3 w-3 ml-1" />
                      </th>
                      <th className="text-left px-3 py-3 font-medium">Labels</th>
                      <th className="text-left px-3 py-3 font-medium cursor-pointer" onClick={() => toggleSort("status")}>
                        Status
                      </th>
                      <th className="text-right px-3 py-3 font-medium cursor-pointer" onClick={() => toggleSort("size_bytes")}>
                        Size
                      </th>
                      <th className="text-left px-3 py-3 font-medium cursor-pointer" onClick={() => toggleSort("created_at")}>
                        Created
                      </th>
                      <th className="text-left px-3 py-3 font-medium">Visibility</th>
                      <th className="text-right px-3 py-3 font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.map(img => {
                      const isMine = img.is_mine;
                      const statusBadge = STATUS_BADGES[img.status];
                      return (
                        <tr
                          key={img.image_id}
                          className={cn(
                            "border-b border-border/50 hover:bg-surface-hover/50 transition-colors",
                            selected.has(img.image_id) && "bg-ice-blue/5",
                          )}
                        >
                          <td className="px-3 py-3">
                            {isMine && (
                              <input
                                type="checkbox"
                                checked={selected.has(img.image_id)}
                                onChange={() => toggleSelect(img.image_id)}
                              />
                            )}
                          </td>
                          <td className="px-2 py-3">
                            {isMine ? (
                              <button
                                onClick={() => toggleStar(img)}
                                title={img.starred ? "Unstar" : "Star"}
                                className="text-text-muted hover:text-yellow-400 transition-colors"
                              >
                                <Star
                                  className={cn(
                                    "h-4 w-4",
                                    img.starred && "fill-yellow-400 text-yellow-400",
                                  )}
                                />
                              </button>
                            ) : null}
                          </td>
                          <td className="px-3 py-3">
                            <div className="font-medium">{img.name}<span className="text-text-muted">:{img.tag}</span></div>
                            {img.description && (
                              <div className="text-xs text-text-muted mt-0.5 line-clamp-1 max-w-md">
                                {img.description}
                              </div>
                            )}
                            <div className="text-[10px] font-mono text-text-muted/70 mt-0.5 truncate max-w-md">
                              {img.image_ref}
                            </div>
                          </td>
                          <td className="px-3 py-3">
                            <div className="flex flex-wrap gap-1">
                              {(img.labels || []).slice(0, 3).map(l => (
                                <span
                                  key={l}
                                  className="text-[10px] px-1.5 py-0.5 rounded bg-surface border border-border text-text-secondary"
                                >
                                  {l}
                                </span>
                              ))}
                              {(img.labels || []).length > 3 && (
                                <span className="text-[10px] text-text-muted">
                                  +{img.labels.length - 3}
                                </span>
                              )}
                            </div>
                          </td>
                          <td className="px-3 py-3">
                            <Badge variant={statusBadge.variant as any}>{statusBadge.label}</Badge>
                          </td>
                          <td className="px-3 py-3 text-right tabular-nums">
                            {formatBytes(img.size_bytes)}
                          </td>
                          <td className="px-3 py-3 text-text-muted whitespace-nowrap">
                            {formatRelative(img.created_at)}
                          </td>
                          <td className="px-3 py-3">
                            {img.is_public ? (
                              <span className="inline-flex items-center gap-1 text-xs text-green-500">
                                <Globe className="h-3 w-3" /> Public
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-xs text-text-muted">
                                <Lock className="h-3 w-3" /> Private
                              </span>
                            )}
                          </td>
                          <td className="px-3 py-3">
                            <div className="flex items-center justify-end gap-1">
                              <NextLink
                                href={`/dashboard/instances?template=${encodeURIComponent(img.image_id)}`}
                                className="p-1.5 rounded hover:bg-surface-hover text-text-muted hover:text-ice-blue transition-colors"
                                title="Launch from template"
                              >
                                <Rocket className="h-4 w-4" />
                              </NextLink>
                              {isMine && (
                                <>
                                  <button
                                    onClick={() => setEditing(img)}
                                    className="p-1.5 rounded hover:bg-surface-hover text-text-muted hover:text-text-primary transition-colors"
                                    title="Edit"
                                  >
                                    <Pencil className="h-4 w-4" />
                                  </button>
                                  <button
                                    onClick={() => deleteOne(img)}
                                    className="p-1.5 rounded hover:bg-surface-hover text-text-muted hover:text-red-400 transition-colors"
                                    title="Delete"
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </button>
                                </>
                              )}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Edit modal */}
        {editing && (
          <EditTemplateDialog
            image={editing}
            onClose={() => setEditing(null)}
            onSaved={() => { setEditing(null); refresh(); }}
          />
        )}

        {/* Bulk delete confirm */}
        <Dialog
          open={confirmBulkDelete}
          onClose={() => setConfirmBulkDelete(false)}
          title={`Delete ${selectedMine.length} template${selectedMine.length === 1 ? "" : "s"}?`}
          description="Soft-deleted templates can be re-created with the same name immediately. The underlying registry image is not removed by this action."
        >
          <div className="px-6 py-4 flex justify-end gap-2">
            <Button variant="outline" onClick={() => setConfirmBulkDelete(false)}>Cancel</Button>
            <Button className="bg-accent-red text-white hover:bg-accent-red-hover" onClick={bulkDelete}>Delete</Button>
          </div>
        </Dialog>
      </div>
    </FadeIn>
  );
}

// ── Edit modal ────────────────────────────────────────────────────────
function EditTemplateDialog({
  image,
  onClose,
  onSaved,
}: {
  image: UserImage;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [description, setDescription] = useState(image.description || "");
  const [isPublic, setIsPublic] = useState(image.is_public);
  const [labels, setLabels] = useState<string[]>(image.labels || []);
  const [labelInput, setLabelInput] = useState("");
  const [saving, setSaving] = useState(false);

  function addLabel() {
    const v = labelInput.trim().toLowerCase();
    if (!v) return;
    if (!/^[a-z0-9][a-z0-9._-]{0,31}$/.test(v)) {
      toast.error("Labels must be lowercase [a-z0-9._-], ≤32 chars");
      return;
    }
    if (labels.includes(v)) return;
    if (labels.length >= 20) {
      toast.error("At most 20 labels per template");
      return;
    }
    setLabels([...labels, v]);
    setLabelInput("");
  }

  async function save() {
    setSaving(true);
    try {
      await api.patchUserImage(image.image_id, {
        description: description.trim(),
        is_public: isPublic,
        labels,
      });
      toast.success("Template updated");
      onSaved();
    } catch (e) {
      toast.error(`Save failed: ${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Dialog
      open
      onClose={onClose}
      title={`Edit ${image.name}:${image.tag}`}
      description={image.image_ref}
      maxWidth="max-w-xl"
    >
      <div className="px-6 py-4 space-y-4">
        <div>
          <Label>Description</Label>
          <textarea
            value={description}
            onChange={e => setDescription(e.target.value)}
            maxLength={512}
            rows={3}
            className="w-full mt-1 rounded-lg border border-border bg-navy px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-ice-blue focus:border-transparent"
            placeholder="What's in this template?"
          />
          <p className="text-xs text-text-muted mt-1">{description.length}/512</p>
        </div>

        <div>
          <Label>Labels</Label>
          <div className="flex gap-2 mt-1">
            <Input
              value={labelInput}
              onChange={e => setLabelInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  addLabel();
                }
              }}
              placeholder="pytorch, experiment-42, prod…"
            />
            <Button type="button" variant="outline" onClick={addLabel}>Add</Button>
          </div>
          <div className="flex flex-wrap gap-1 mt-2">
            {labels.map(l => (
              <span
                key={l}
                className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-surface border border-border"
              >
                {l}
                <button
                  onClick={() => setLabels(labels.filter(x => x !== l))}
                  className="hover:text-red-400"
                >
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))}
          </div>
        </div>

        <div>
          <Label>Visibility</Label>
          <div className="flex items-center gap-3 mt-2">
            <button
              type="button"
              onClick={() => setIsPublic(false)}
              className={cn(
                "flex-1 rounded-lg border px-3 py-2 text-sm text-left transition-colors",
                !isPublic
                  ? "border-ice-blue bg-ice-blue/10"
                  : "border-border hover:border-border-hover",
              )}
            >
              <div className="flex items-center gap-2 font-medium">
                <Lock className="h-4 w-4" /> Private
              </div>
              <p className="text-xs text-text-muted mt-0.5">Only you can see + launch.</p>
            </button>
            <button
              type="button"
              onClick={() => setIsPublic(true)}
              className={cn(
                "flex-1 rounded-lg border px-3 py-2 text-sm text-left transition-colors",
                isPublic
                  ? "border-ice-blue bg-ice-blue/10"
                  : "border-border hover:border-border-hover",
              )}
            >
              <div className="flex items-center gap-2 font-medium">
                <Globe className="h-4 w-4" /> Public
              </div>
              <p className="text-xs text-text-muted mt-0.5">Appears in Community tab.</p>
            </button>
          </div>
        </div>
      </div>

      <div className="px-6 pb-5 flex justify-end gap-2">
        <Button variant="outline" onClick={onClose}>Cancel</Button>
        <Button onClick={save} disabled={saving}>
          {saving ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : <Check className="h-4 w-4 mr-1" />}
          Save
        </Button>
      </div>
    </Dialog>
  );
}
