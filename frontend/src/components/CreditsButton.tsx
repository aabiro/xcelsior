"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { Wallet, ChevronDown, ArrowUpRight, CreditCard, History, Loader2, AlertTriangle } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { fetchWallet } from "@/lib/api";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";

const LOW_BALANCE_THRESHOLD = 10; // CAD

export function CreditsButton() {
  const { user } = useAuth();
  const [balance, setBalance] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState(false);
  const [alertDismissed, setAlertDismissed] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const customerId = user?.customer_id || user?.user_id || "";
  const isLow = balance !== null && balance < LOW_BALANCE_THRESHOLD;

  const loadBalance = useCallback(async () => {
    if (!customerId) return;
    try {
      const res = await fetchWallet(customerId);
      setBalance(res.wallet.balance_cad);
    } catch {
      setBalance(null);
    } finally {
      setLoading(false);
    }
  }, [customerId]);

  useEffect(() => {
    loadBalance();
    const interval = setInterval(loadBalance, 30_000);
    return () => clearInterval(interval);
  }, [loadBalance]);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (open) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const formatted = balance !== null ? `$${balance.toFixed(2)}` : "—";

  return (
    <div className="relative" ref={ref}>
      {/* Low-balance alert banner */}
      <AnimatePresence>
        {isLow && !alertDismissed && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            className="absolute right-0 top-full mt-2 w-72 rounded-xl border border-amber-500/30 bg-amber-500/5 shadow-lg shadow-amber-500/10 z-50 p-3"
          >
            <div className="flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-amber-500">Low balance</p>
                <p className="text-xs text-text-muted mt-0.5">
                  Your balance is {formatted}. Top up to avoid service interruption — instances may be suspended when credits run out.
                </p>
                <div className="flex items-center gap-2 mt-2">
                  <Link
                    href="/dashboard/billing?topup=true"
                    onClick={() => setAlertDismissed(true)}
                    className="text-xs font-medium text-amber-500 hover:text-amber-400 underline underline-offset-2"
                  >
                    Top up now
                  </Link>
                  <button
                    onClick={() => setAlertDismissed(true)}
                    className="text-xs text-text-muted hover:text-text-secondary ml-auto"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <button
        onClick={() => { setOpen(!open); if (isLow) setAlertDismissed(true); }}
        className={cn(
          "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm transition-colors",
          isLow
            ? "bg-amber-500/10 hover:bg-amber-500/15 border border-amber-500/30 text-amber-500"
            : "bg-emerald/5 hover:bg-emerald/10 border border-emerald/20 text-emerald"
        )}
      >
        {isLow ? (
          <>
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <span className="h-1.5 w-1.5 rounded-full bg-amber-500 animate-pulse" />
          </>
        ) : (
          <Wallet className="h-4 w-4 shrink-0" />
        )}
        <span className="font-medium">
          {loading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin inline" />
          ) : (
            formatted
          )}
        </span>
        <ChevronDown
          className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 top-full mt-1 w-56 rounded-xl border border-border/60 bg-surface shadow-xl z-50 overflow-hidden"
          >
            {/* Balance header */}
            <div className="px-3 py-3 border-b border-border bg-emerald/5">
              <p className="text-xs text-text-muted uppercase tracking-wider font-medium">Credits</p>
              <p className="text-lg font-bold text-emerald mt-0.5">
                {loading ? "..." : formatted}
                <span className="text-xs font-normal text-text-muted ml-1">CAD</span>
              </p>
            </div>

            {/* Actions */}
            <div className="py-1">
              <Link
                href="/dashboard/billing"
                onClick={() => setOpen(false)}
                className="flex items-center gap-2.5 px-3 py-2 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
              >
                <CreditCard className="h-4 w-4" />
                Billing & Payments
              </Link>
              <Link
                href="/dashboard/billing#history"
                onClick={() => setOpen(false)}
                className="flex items-center gap-2.5 px-3 py-2 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
              >
                <History className="h-4 w-4" />
                Transaction History
              </Link>
            </div>

            {/* Top Up button */}
            <div className="border-t border-border p-2">
              <Link
                href="/dashboard/billing?topup=true"
                onClick={() => setOpen(false)}
                className="flex items-center justify-center gap-1.5 w-full rounded-lg bg-emerald/10 hover:bg-emerald/20 text-emerald text-sm font-medium py-2 transition-colors"
              >
                <ArrowUpRight className="h-4 w-4" />
                Top Up Credits
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
