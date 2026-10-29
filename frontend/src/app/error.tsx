"use client";

import { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Unhandled error:", error);
  }, [error]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-navy px-4 text-center">
      <div className="mx-auto max-w-md">
        <div className="mb-6 flex h-16 w-16 mx-auto items-center justify-center rounded-2xl bg-accent-red/20">
          <span className="text-3xl">⚠</span>
        </div>
        <h1 className="text-2xl font-bold text-text-primary">Something went wrong</h1>
        <p className="mt-2 text-text-secondary">
          An unexpected error occurred. Our team has been notified.
        </p>
        {error.digest && (
          <p className="mt-1 text-xs text-text-muted">Error ID: {error.digest}</p>
        )}
        <button
          onClick={reset}
          className="mt-6 rounded-lg bg-accent-red px-6 py-2.5 text-sm font-medium text-white hover:bg-accent-red/90 transition-colors"
        >
          Try Again
        </button>
      </div>
    </div>
  );
}
