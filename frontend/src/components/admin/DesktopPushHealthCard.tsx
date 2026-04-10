"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { BellRing, CheckCircle2, ExternalLink, Loader2, Send, TriangleAlert } from "lucide-react";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  fetchAdminOverview,
  sendAdminWebPushTestNotification,
} from "@/lib/api";
import { isWebPushLifecycleMessage } from "@/lib/pwa/web-push";

type WebPushSnapshot = Awaited<ReturnType<typeof fetchAdminOverview>>["web_push"];

type SmokeRun = {
  smokeId: string;
  notificationId: string;
  actionUrl: string;
  sentAt: number;
  receivedAt?: number;
  clickedAt?: number;
};

function formatEpoch(epoch: number | undefined) {
  if (!epoch) return "Never";
  return new Date(epoch * 1000).toLocaleString();
}

function formatLocalTimestamp(timestamp: number | undefined) {
  if (!timestamp) return "Waiting";
  return new Date(timestamp).toLocaleTimeString();
}

function statusTone(value: number, invert = false) {
  if (invert) {
    return value > 0 ? "text-amber-300" : "text-emerald";
  }
  return value > 0 ? "text-emerald" : "text-text-secondary";
}

interface DesktopPushHealthCardProps {
  snapshot: WebPushSnapshot;
  onSnapshotChange: (snapshot: WebPushSnapshot) => void;
}

export function DesktopPushHealthCard({ snapshot, onSnapshotChange }: DesktopPushHealthCardProps) {
  const searchParams = useSearchParams();
  const [sending, setSending] = useState(false);
  const [smokeRun, setSmokeRun] = useState<SmokeRun | null>(null);

  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;

    const handleMessage = (event: MessageEvent) => {
      if (!isWebPushLifecycleMessage(event.data)) return;
      if (!event.data.data?.smoke_test) return;

      const smokeId = typeof event.data.data?.smoke_id === "string" ? event.data.data.smoke_id : "";
      const nextTimestamp = event.data.timestamp;

      setSmokeRun((current) => {
        const base =
          current && current.smokeId === smokeId
            ? current
            : {
                smokeId,
                notificationId: event.data.notificationId ?? "",
                actionUrl: event.data.url,
                sentAt: nextTimestamp,
              };

        return event.data.event === "received"
          ? {
              ...base,
              receivedAt: current?.receivedAt ?? nextTimestamp,
            }
          : {
              ...base,
              clickedAt: current?.clickedAt ?? nextTimestamp,
            };
      });

      if (event.data.event === "received") {
        toast.success("Desktop push smoke notification reached this browser.");
      } else {
        toast.success("Desktop push click-through confirmed.");
      }
    };

    navigator.serviceWorker.addEventListener("message", handleMessage);
    return () => navigator.serviceWorker.removeEventListener("message", handleMessage);
  }, []);

  useEffect(() => {
    const smokeFlag = searchParams.get("push_smoke");
    const smokeId = searchParams.get("smoke_id");
    if (smokeFlag !== "1" || !smokeId) return;

    setSmokeRun((current) => {
      if (current && current.smokeId !== smokeId) return current;
      return {
        smokeId,
        notificationId: current?.notificationId ?? "",
        actionUrl: current?.actionUrl ?? `/dashboard/admin?push_smoke=1&smoke_id=${smokeId}`,
        sentAt: current?.sentAt ?? Date.now(),
        receivedAt: current?.receivedAt,
        clickedAt: current?.clickedAt ?? Date.now(),
      };
    });
  }, [searchParams]);

  const handleSendTest = async () => {
    setSending(true);
    try {
      const result = await sendAdminWebPushTestNotification();
      setSmokeRun({
        smokeId: result.smoke_id,
        notificationId: result.notification_id,
        actionUrl: result.action_url,
        sentAt: Date.now(),
      });
      onSnapshotChange({
        ...snapshot,
        ...result.web_push,
        current_user_subscription_count: result.active_subscription_count,
      });

      if (result.active_subscription_count > 0) {
        toast.success("Desktop push smoke notification sent.");
      } else {
        toast.error("No active desktop push subscription for this admin session. Enable notifications first.");
      }
    } catch (error) {
      console.error("Failed to send desktop push smoke notification", error);
      toast.error("Failed to send desktop push smoke notification.");
    } finally {
      setSending(false);
    }
  };

  const received = Boolean(smokeRun?.receivedAt);
  const clicked = Boolean(smokeRun?.clickedAt);

  return (
    <Card className="glow-card brand-top-accent">
      <CardHeader className="pb-3">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2 text-sm">
              <BellRing className="h-4 w-4 text-accent-gold" />
              Desktop Push Health
            </CardTitle>
            <p className="max-w-2xl text-xs text-text-secondary">
              Live delivery health for the desktop PWA, plus a real admin-only smoke notification flow for staging verification.
            </p>
          </div>

          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={handleSendTest}
            disabled={sending || snapshot.configured !== 1}
            className="min-w-48"
          >
            {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            Send Test Notification
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
          <div className="rounded-xl border border-border/70 bg-background/60 p-3">
            <p className="text-[11px] uppercase tracking-wide text-text-muted">Configured</p>
            <p className={`mt-2 text-lg font-semibold ${snapshot.configured === 1 ? "text-emerald" : "text-amber-300"}`}>
              {snapshot.configured === 1 ? "Yes" : "No"}
            </p>
          </div>
          <div className="rounded-xl border border-border/70 bg-background/60 p-3">
            <p className="text-[11px] uppercase tracking-wide text-text-muted">Active Subs</p>
            <p className={`mt-2 text-lg font-semibold ${statusTone(snapshot.active_subscriptions)}`}>
              {snapshot.active_subscriptions}
            </p>
          </div>
          <div className="rounded-xl border border-border/70 bg-background/60 p-3">
            <p className="text-[11px] uppercase tracking-wide text-text-muted">Revoked Subs</p>
            <p className={`mt-2 text-lg font-semibold ${statusTone(snapshot.revoked_subscriptions, true)}`}>
              {snapshot.revoked_subscriptions}
            </p>
          </div>
          <div className="rounded-xl border border-border/70 bg-background/60 p-3">
            <p className="text-[11px] uppercase tracking-wide text-text-muted">Stale Subs</p>
            <p className={`mt-2 text-lg font-semibold ${statusTone(snapshot.stale_subscriptions, true)}`}>
              {snapshot.stale_subscriptions}
            </p>
          </div>
          <div className="rounded-xl border border-border/70 bg-background/60 p-3">
            <p className="text-[11px] uppercase tracking-wide text-text-muted">Your Browser</p>
            <p className={`mt-2 text-lg font-semibold ${statusTone(snapshot.current_user_subscription_count)}`}>
              {snapshot.current_user_subscription_count}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <div className="rounded-xl border border-border/60 bg-surface/40 p-3">
            <p className="text-[11px] uppercase tracking-wide text-text-muted">Delivered</p>
            <p className="mt-2 text-base font-semibold text-emerald">{snapshot.delivery_success_total}</p>
          </div>
          <div className="rounded-xl border border-border/60 bg-surface/40 p-3">
            <p className="text-[11px] uppercase tracking-wide text-text-muted">Failures</p>
            <p className={`mt-2 text-base font-semibold ${statusTone(snapshot.delivery_failure_total, true)}`}>
              {snapshot.delivery_failure_total}
            </p>
          </div>
          <div className="rounded-xl border border-border/60 bg-surface/40 p-3">
            <p className="text-[11px] uppercase tracking-wide text-text-muted">Revoked on Send</p>
            <p className={`mt-2 text-base font-semibold ${statusTone(snapshot.delivery_revoked_total, true)}`}>
              {snapshot.delivery_revoked_total}
            </p>
          </div>
          <div className="rounded-xl border border-border/60 bg-surface/40 p-3">
            <p className="text-[11px] uppercase tracking-wide text-text-muted">Retained Notifications</p>
            <p className="mt-2 text-base font-semibold text-text-primary">{snapshot.notification_retained_total}</p>
          </div>
        </div>

        <div className="rounded-xl border border-border/70 bg-background/50 p-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
            <div className="space-y-2">
              <p className="text-sm font-medium text-text-primary">Smoke Status</p>
              <p className="text-xs text-text-secondary">
                Send a notification from this page, confirm it reaches the browser, then click it to reopen the admin dashboard through the service worker route.
              </p>
            </div>
            {smokeRun?.actionUrl ? (
              <Link
                href={smokeRun.actionUrl}
                className="inline-flex items-center gap-1 text-xs text-ice-blue hover:underline"
              >
                Open target route
                <ExternalLink className="h-3.5 w-3.5" />
              </Link>
            ) : null}
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <div className="rounded-lg border border-border/60 bg-surface/30 p-3">
              <p className="text-[11px] uppercase tracking-wide text-text-muted">Sent</p>
              <p className="mt-2 text-sm font-medium text-text-primary">{formatLocalTimestamp(smokeRun?.sentAt)}</p>
            </div>
            <div className="rounded-lg border border-border/60 bg-surface/30 p-3">
              <p className="text-[11px] uppercase tracking-wide text-text-muted">Received</p>
              <p className={`mt-2 text-sm font-medium ${received ? "text-emerald" : "text-text-secondary"}`}>
                {received ? formatLocalTimestamp(smokeRun?.receivedAt) : "Waiting for browser delivery"}
              </p>
            </div>
            <div className="rounded-lg border border-border/60 bg-surface/30 p-3">
              <p className="text-[11px] uppercase tracking-wide text-text-muted">Clicked</p>
              <p className={`mt-2 text-sm font-medium ${clicked ? "text-emerald" : "text-text-secondary"}`}>
                {clicked ? formatLocalTimestamp(smokeRun?.clickedAt) : "Click the desktop notification"}
              </p>
            </div>
          </div>

          {clicked ? (
            <div className="mt-4 flex items-start gap-2 rounded-lg border border-emerald/30 bg-emerald/10 px-3 py-2 text-sm text-emerald">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
              <span>Desktop push click-through was confirmed for this smoke run.</span>
            </div>
          ) : snapshot.configured !== 1 ? (
            <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
              <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
              <span>Web push is not configured on this deployment, so staging smoke cannot complete yet.</span>
            </div>
          ) : null}
        </div>

        <div className="grid gap-2 text-xs text-text-secondary md:grid-cols-3">
          <p>Last attempt: <span className="text-text-primary">{formatEpoch(snapshot.last_delivery_attempt_at)}</span></p>
          <p>Last success: <span className="text-text-primary">{formatEpoch(snapshot.last_delivery_success_at)}</span></p>
          <p>
            Last failure:
            <span className="text-text-primary">
              {" "}
              {snapshot.last_delivery_failure_at
                ? `${formatEpoch(snapshot.last_delivery_failure_at)} (${snapshot.last_failure_status_code || "unknown"})`
                : "Never"}
            </span>
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
