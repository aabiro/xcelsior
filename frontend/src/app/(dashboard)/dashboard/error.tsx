"use client";

import { useEffect } from "react";

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Dashboard error:", error);
  }, [error]);

  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="mx-auto max-w-md">
        <div className="mb-6 flex h-14 w-14 mx-auto items-center justify-center rounded-xl bg-accent-red/20">
          <span className="text-2xl">⚠</span>
        </div>
        <h2 className="text-xl font-bold text-text-primary">Dashboard Error</h2>
        <p className="mt-2 text-sm text-text-secondary">
          Something went wrong loading this section. Try again or navigate to another page.
        </p>
        {error.digest && (
          <p className="mt-1 text-xs text-text-muted">Error ID: {error.digest}</p>
        )}
        <button
          onClick={reset}
          className="mt-6 rounded-lg bg-accent-red px-5 py-2 text-sm font-medium text-white hover:bg-accent-red/90 transition-colors"
        >
          Try Again
        </button>
      </div>
    </div>
  );
}
