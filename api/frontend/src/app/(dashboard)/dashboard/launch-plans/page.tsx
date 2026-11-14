import Link from "next/link";
import { ArrowLeft, ShieldCheck } from "lucide-react";

export default function LaunchPlansIndexPage() {
  return (
    <div className="mx-auto max-w-xl py-16">
      <div className="rounded-2xl border border-border/70 bg-surface/50 p-6">
        <div className="mb-2 flex items-center gap-2 text-sm text-accent-cyan">
          <ShieldCheck className="h-4 w-4" /> Server-bound approval
        </div>
        <h1 className="text-xl font-semibold">Open an approval link</h1>
        <p className="mt-2 text-sm text-text-secondary">
          A launch plan does not have a list. Your agent sends a link to one
          plan, and that page is where you approve it.
        </p>
        <Link
          href="/dashboard/mcp"
          className="mt-5 inline-flex h-10 items-center rounded-md border border-border px-4 text-sm font-medium"
        >
          <ArrowLeft className="mr-2 h-4 w-4" /> Back to MCP
        </Link>
      </div>
    </div>
  );
}
