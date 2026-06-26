"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, BookOpen, Gift, Sparkles, X } from "lucide-react";
import { CodeBlock } from "@/components/ui/code-block";
import { fetchWallet } from "@/lib/api";
import posthog from "posthog-js";

const DISMISS_KEY = "xcelsior.first_run_dismissed";

// Illustrates the core value prop: it's just OpenAI with a different base_url.
// (The base_url is filled in once they deploy a model — see "Deploy a model".)
const SNIPPET = `from openai import OpenAI

client = OpenAI(
    # 1. point at your Xcelsior endpoint  ·  2. keep everything else
    base_url="https://xcelsior.ca/v1/serverless/<your-endpoint>/openai/v1",
    api_key="YOUR_API_KEY",
)

resp = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)`;

/**
 * First-run activation card — shows on the dashboard until the user has run
 * something. Leads with developer experience ("just change your base_url") and
 * the free credit, then sends them to deploy a model.
 */
export function FirstRunCard({ customerId, show }: { customerId?: string; show: boolean }) {
  // Start hidden to avoid a flash before we read localStorage.
  const [dismissed, setDismissed] = useState(true);
  const [balance, setBalance] = useState<number | null>(null);

  useEffect(() => {
    try {
      setDismissed(localStorage.getItem(DISMISS_KEY) === "1");
    } catch {
      setDismissed(false);
    }
  }, []);

  useEffect(() => {
    if (!customerId || !show) return;
    let cancelled = false;
    fetchWallet(customerId)
      .then((r) => { if (!cancelled) setBalance(r.wallet.balance_cad); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [customerId, show]);

  const dismiss = useCallback(() => {
    setDismissed(true);
    try { localStorage.setItem(DISMISS_KEY, "1"); } catch { /* noop */ }
    posthog.capture("first_run_card_dismissed");
  }, []);

  if (!show || dismissed) return null;

  return (
    <div className="relative overflow-hidden rounded-2xl border border-accent-cyan/25 bg-gradient-to-br from-accent-cyan/10 via-surface to-accent-violet/10 p-5 sm:p-6">
      <button
        type="button"
        onClick={dismiss}
        aria-label="Dismiss"
        className="absolute right-3 top-3 flex h-7 w-7 items-center justify-center rounded-lg text-text-muted transition-colors hover:bg-surface-hover hover:text-text-primary"
      >
        <X className="h-4 w-4" />
      </button>

      <div className="flex items-start gap-3 pr-8">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent-cyan/15 ring-1 ring-accent-cyan/25">
          <Sparkles className="h-4 w-4 text-accent-cyan" />
        </div>
        <div className="min-w-0">
          <h2 className="text-lg font-semibold">Run your first inference</h2>
          <p className="mt-0.5 text-sm text-text-secondary">
            OpenAI-compatible — just change your <code className="rounded bg-surface px-1 py-0.5 font-mono text-xs text-accent-cyan">base_url</code> and the rest of your code works exactly like OpenAI.
          </p>
        </div>
      </div>

      {balance !== null && balance > 0 && (
        <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-emerald/30 bg-emerald/10 px-3 py-1 text-xs font-medium text-emerald">
          <Gift className="h-3.5 w-3.5" />
          Your first run is covered by your free credit (${balance.toFixed(2)}).
        </div>
      )}

      <div className="mt-4">
        <CodeBlock filename="first_run.py" code={SNIPPET} />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Link
          href="/dashboard/inference"
          onClick={() => posthog.capture("first_run_card_deploy_clicked")}
          className="inline-flex min-h-10 items-center gap-1.5 rounded-lg bg-accent-cyan px-4 text-sm font-medium text-[#06121a] transition-colors hover:bg-accent-cyan/90"
        >
          Deploy a model <ArrowRight className="h-4 w-4" />
        </Link>
        <a
          href="https://docs.xcelsior.ca"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 text-sm text-text-secondary transition-colors hover:text-text-primary"
        >
          <BookOpen className="h-4 w-4" /> Read the docs
        </a>
      </div>
    </div>
  );
}
