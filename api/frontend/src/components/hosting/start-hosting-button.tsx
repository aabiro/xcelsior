"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Server } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { updateProfile } from "@/lib/api";
import { hostingEnabled } from "@/lib/hosting";
import { cn } from "@/lib/utils";

export function StartHostingButton({ className, compact = false }: { className?: string; compact?: boolean }) {
  const { user, refreshUser } = useAuth();
  const router = useRouter();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  async function start() {
    if (!user || pending) return;
    setPending(true); setError("");
    try {
      if (!hostingEnabled(user)) {
        // Preserve platform authority. Rent/Host navigation never changes this role.
        await updateProfile({ onboarding_intent: "both", ...(!user.is_admin && user.role !== "admin" ? { role: "provider" } : {}) });
        await refreshUser({ throwOnError: true });
      }
      router.push("/dashboard/hosting");
    } catch (error) {
      setError(error instanceof Error ? error.message : "Could not enable hosting. Please try again.");
    } finally { setPending(false); }
  }
  return <div>
    <button type="button" onClick={start} disabled={pending || !user} title={compact ? "Start hosting" : undefined} aria-label={compact ? "Start hosting" : undefined}
      className={cn("inline-flex items-center justify-center gap-2 rounded-xl border border-accent-cyan/30 bg-accent-cyan/10 px-4 py-2.5 text-sm font-medium text-accent-cyan transition-colors hover:bg-accent-cyan/20 disabled:opacity-60", className)}>
      {pending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Server className="h-4 w-4 shrink-0" />}
      {!compact && (pending ? "Enabling hosting…" : "Start hosting")}
    </button>
    {error && <p role="alert" className="mt-2 text-sm text-red-400">{error}</p>}
  </div>;
}
