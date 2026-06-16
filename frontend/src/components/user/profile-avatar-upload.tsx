"use client";

import { useCallback, useRef, useState } from "react";
import { Camera, Loader2, Trash2, Upload } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { UserAvatar } from "@/components/user/user-avatar";
import { deleteAvatar, uploadAvatar } from "@/lib/api";
import type { User } from "@/lib/auth";

const MAX_MB = 2;
const ACCEPT = "image/jpeg,image/png,image/webp";

export function ProfileAvatarUpload({
  user,
  onUpdated,
}: {
  user: User;
  onUpdated: () => Promise<void>;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(
    async (file: File | null | undefined) => {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        toast.error("Choose a JPEG, PNG, or WebP image");
        return;
      }
      if (file.size > MAX_MB * 1024 * 1024) {
        toast.error(`Image must be under ${MAX_MB} MB`);
        return;
      }
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);
      setUploading(true);
      try {
        await uploadAvatar(file);
        setPreview(null);
        await onUpdated();
        toast.success("Profile photo updated");
      } catch (err) {
        setPreview(null);
        toast.error(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setUploading(false);
        URL.revokeObjectURL(objectUrl);
      }
    },
    [onUpdated],
  );

  const handleRemove = async () => {
    setUploading(true);
    try {
      await deleteAvatar();
      setPreview(null);
      await onUpdated();
      toast.success("Profile photo removed");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not remove photo");
    } finally {
      setUploading(false);
    }
  };

  const displaySrc = preview ?? user.avatar_url ?? null;
  const hasPhoto = !!(displaySrc || user.avatar_url);

  return (
    <div className="flex flex-col sm:flex-row items-center sm:items-start gap-5">
      <div className="relative group">
        <UserAvatar user={user} size="xl" src={displaySrc} />
        <button
          type="button"
          disabled={uploading}
          onClick={() => inputRef.current?.click()}
          className={cn(
            "absolute inset-0 flex items-center justify-center rounded-full",
            "bg-navy/60 opacity-0 group-hover:opacity-100 transition-opacity",
            "focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-cyan",
          )}
          aria-label="Change profile photo"
        >
          {uploading ? (
            <Loader2 className="h-7 w-7 text-accent-cyan animate-spin" />
          ) : (
            <Camera className="h-7 w-7 text-white drop-shadow" />
          )}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="sr-only"
          onChange={(e) => {
            void handleFile(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>

      <div className="flex-1 w-full space-y-3">
        <p className="text-sm font-medium text-text-primary">Profile photo</p>
        <ul className="text-xs text-text-muted space-y-1 max-w-md">
          <li>Recommended: 400×400 px, square</li>
          <li>Displayed as a circle (center crop)</li>
          <li>JPEG, PNG, or WebP · max {MAX_MB} MB</li>
        </ul>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            void handleFile(e.dataTransfer.files?.[0]);
          }}
          className={cn(
            "rounded-xl border border-dashed px-4 py-5 text-center transition-colors",
            dragOver
              ? "border-accent-cyan bg-accent-cyan/8"
              : "border-border/70 bg-surface/40 hover:border-border-light",
          )}
        >
          <Upload className="h-5 w-5 mx-auto mb-2 text-text-muted" />
          <p className="text-sm text-text-secondary">
            Drag an image here, or{" "}
            <button
              type="button"
              className="text-accent-cyan hover:underline font-medium"
              onClick={() => inputRef.current?.click()}
              disabled={uploading}
            >
              browse
            </button>
          </p>
        </div>

        {hasPhoto && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={uploading}
            onClick={() => void handleRemove()}
            className="text-text-muted border-border hover:text-accent-red hover:border-accent-red/40"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Remove photo
          </Button>
        )}
      </div>
    </div>
  );
}