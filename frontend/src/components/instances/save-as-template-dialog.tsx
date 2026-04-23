"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Camera } from "lucide-react";
import { snapshotInstance } from "@/lib/api";
import { toast } from "sonner";

interface SaveAsTemplateDialogProps {
  open: boolean;
  instanceId: string;
  defaultName?: string;
  onClose: () => void;
  onSaved?: (imageRef: string) => void;
}

const NAME_RE = /^[a-z0-9][a-z0-9._-]*$/;

function sanitizeName(raw: string): string {
  return raw.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+/, "").slice(0, 63);
}

export function SaveAsTemplateDialog({
  open,
  instanceId,
  defaultName,
  onClose,
  onSaved,
}: SaveAsTemplateDialogProps) {
  const [name, setName] = useState("");
  const [tag, setTag] = useState("latest");
  const [description, setDescription] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setName(sanitizeName(defaultName || `template-${Date.now().toString(36).slice(-6)}`));
      setTag("latest");
      setDescription("");
      setError(null);
    }
  }, [open, defaultName]);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape" && !submitting) onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose, submitting]);

  if (!open) return null;

  const trimmedName = name.trim();
  const trimmedTag = tag.trim() || "latest";
  const nameValid = NAME_RE.test(trimmedName) && trimmedName.length <= 63;
  const tagValid = NAME_RE.test(trimmedTag) && trimmedTag.length <= 63;
  const canSubmit = nameValid && tagValid && !submitting;

  async function handleSubmit() {
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await snapshotInstance(instanceId, {
        name: trimmedName,
        tag: trimmedTag,
        description: description.trim() || undefined,
      });
      toast.success(
        res.status === "queued_registry_down"
          ? "Snapshot queued — registry temporarily unreachable, will retry"
          : "Snapshot started — image will appear in Templates when ready",
      );
      onSaved?.(res.image_ref);
      onClose();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Snapshot failed";
      setError(msg);
      toast.error(`Snapshot failed: ${msg}`);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => !submitting && onClose()} />
      <div className="relative z-10 w-full max-w-md rounded-xl border border-border bg-surface p-6 shadow-2xl mx-4">
        <div className="flex items-start gap-4">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-accent-cyan/10">
            <Camera className="h-5 w-5 text-accent-cyan" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold">Save as Template</h3>
            <p className="mt-1 text-sm text-text-secondary">
              Snapshot this running instance into a reusable image. The container is committed and pushed to your registry.
            </p>
          </div>
        </div>

        <div className="mt-5 space-y-4">
          <div>
            <label className="mb-1 block text-xs font-medium text-text-secondary">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my-template"
              disabled={submitting}
              className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm focus:border-accent-cyan focus:outline-none"
              autoFocus
            />
            {!nameValid && trimmedName.length > 0 && (
              <p className="mt-1 text-xs text-accent-red">
                Lowercase letters, digits, dot/dash/underscore. Must start with letter or digit. Max 63 chars.
              </p>
            )}
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-text-secondary">Tag</label>
            <input
              type="text"
              value={tag}
              onChange={(e) => setTag(e.target.value)}
              placeholder="latest"
              disabled={submitting}
              className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm focus:border-accent-cyan focus:outline-none"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-text-secondary">Description (optional)</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value.slice(0, 512))}
              placeholder="What's in this template?"
              disabled={submitting}
              rows={3}
              className="w-full resize-none rounded-md border border-border bg-bg px-3 py-2 text-sm focus:border-accent-cyan focus:outline-none"
            />
            <p className="mt-1 text-right text-[10px] text-text-secondary">{description.length}/512</p>
          </div>
          {error && <p className="text-xs text-accent-red">{error}</p>}
        </div>

        <div className="mt-6 flex justify-end gap-3">
          <Button variant="outline" size="sm" onClick={onClose} disabled={submitting}>
            Cancel
          </Button>
          <Button size="sm" onClick={handleSubmit} disabled={!canSubmit}>
            {submitting ? "Snapshotting…" : "Save Template"}
          </Button>
        </div>
      </div>
    </div>
  );
}
