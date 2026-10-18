import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-navy px-4 text-center">
      <div className="mx-auto max-w-md">
        <h1 className="text-8xl font-bold text-accent-red">404</h1>
        <h2 className="mt-4 text-2xl font-bold text-text-primary">Page not found</h2>
        <p className="mt-2 text-text-secondary">
          The page you&apos;re looking for doesn&apos;t exist or has been moved.
        </p>
        <div className="mt-8 flex items-center justify-center gap-4">
          <Link
            href="/"
            className="rounded-lg bg-accent-red px-6 py-2.5 text-sm font-medium text-white hover:bg-accent-red/90 transition-colors"
          >
            Go Home
          </Link>
          <Link
            href="/dashboard"
            className="rounded-lg border border-border px-6 py-2.5 text-sm font-medium text-text-primary hover:bg-surface-hover transition-colors"
          >
            Dashboard
          </Link>
        </div>
      </div>
    </div>
  );
}
