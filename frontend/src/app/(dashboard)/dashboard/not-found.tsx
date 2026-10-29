import Link from "next/link";

export default function DashboardNotFound() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="mx-auto max-w-md">
        <h2 className="text-5xl font-bold text-accent-red">404</h2>
        <h3 className="mt-4 text-xl font-bold text-text-primary">Page not found</h3>
        <p className="mt-2 text-sm text-text-secondary">
          This dashboard page doesn&apos;t exist or you don&apos;t have access.
        </p>
        <Link
          href="/dashboard"
          className="mt-6 inline-block rounded-lg bg-accent-red px-5 py-2 text-sm font-medium text-white hover:bg-accent-red/90 transition-colors"
        >
          Back to Dashboard
        </Link>
      </div>
    </div>
  );
}
