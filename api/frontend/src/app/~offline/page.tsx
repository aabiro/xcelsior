"use client";

import Link from "next/link";
import { WifiOff, RefreshCcw, LayoutDashboard, Globe } from "lucide-react";

export default function OfflinePage() {
  return (
    <main className="min-h-screen bg-navy px-6 py-12 text-text-primary">
      <div className="mx-auto flex min-h-[calc(100vh-6rem)] max-w-4xl items-center justify-center">
        <section className="w-full rounded-3xl border border-border bg-surface/95 p-8 shadow-2xl shadow-black/20 backdrop-blur md:p-10">
          <div className="flex flex-col gap-8 md:flex-row md:items-center md:justify-between">
            <div className="max-w-2xl space-y-4">
              <div className="inline-flex h-14 w-14 items-center justify-center rounded-2xl border border-border bg-navy-light text-accent-cyan">
                <WifiOff className="h-7 w-7" />
              </div>
              <div className="space-y-2">
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-text-muted">
                  Offline
                </p>
                <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
                  Xcelsior is waiting for your connection to come back.
                </h1>
                <p className="max-w-xl text-base text-text-secondary">
                  Cached pages and assets are still available, but live actions like launches,
                  billing updates, and settings changes need a network connection before they can
                  complete safely.
                </p>
              </div>
            </div>

            <div className="rounded-2xl border border-border bg-navy-light/70 p-5 text-sm text-text-secondary">
              <p className="font-medium text-text-primary">What you can do now</p>
              <ul className="mt-3 space-y-2">
                <li>Reconnect and retry to refresh live marketplace, billing, and notification data.</li>
                <li>Use cached routes you already opened in this browser or installed app window.</li>
                <li>Keep destructive or billing-related actions for when the app is back online.</li>
              </ul>
            </div>
          </div>

          <div className="mt-8 flex flex-col gap-3 sm:flex-row">
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-accent-red px-5 text-sm font-medium text-white transition-colors hover:bg-accent-red-hover"
            >
              <RefreshCcw className="h-4 w-4" />
              Retry Connection
            </button>
            <Link
              href="/"
              className="inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-border px-5 text-sm font-medium text-text-primary transition-colors hover:bg-navy-light"
            >
              <Globe className="h-4 w-4" />
              Return Home
            </Link>
            <Link
              href="/dashboard"
              className="inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-border px-5 text-sm font-medium text-text-primary transition-colors hover:bg-navy-light"
            >
              <LayoutDashboard className="h-4 w-4" />
              Open Dashboard
            </Link>
          </div>
        </section>
      </div>
    </main>
  );
}
