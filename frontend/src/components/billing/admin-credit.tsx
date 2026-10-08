"use client";

import { useState } from "react";
import { Loader2, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Input, Label } from "@/components/ui/input";
import { adminCreditWallet } from "@/lib/api";
import { toast } from "sonner";

const PRESETS = [25, 50, 100, 500];
const MAX_CAD = 10000;

function newIdempotencyKey(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

/**
 * Platform-admin credit: funds a wallet with no payment taken. The server
 * records it as an admin credit (never a deposit, so it is not counted as
 * money received) and audits who granted it and why.
 */
export function AdminCreditForm({
  customerId,
  recipient,
  onCredited,
  onCancel,
}: {
  customerId: string;
  /** Shown in the confirmation copy, e.g. the customer's email. */
  recipient?: string;
  onCredited: (balanceCad: number, amountCad: number) => void;
  onCancel: () => void;
}) {
  const [amount, setAmount] = useState("50");
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  // One key per attempt: a double click or a retried request grants once.
  const [idempotencyKey, setIdempotencyKey] = useState(newIdempotencyKey);

  const numeric = Number(amount);
  const amountValid = /^\d+(?:\.\d{1,2})?$/.test(amount) && Number.isFinite(numeric) && numeric > 0 && numeric <= MAX_CAD;
  const reasonValid = reason.trim().length >= 3;

  const submit = async () => {
    if (!amountValid || !reasonValid || submitting) return;
    setSubmitting(true);
    try {
      const res = await adminCreditWallet(customerId, numeric, reason.trim(), idempotencyKey);
      toast.success(`$${res.amount_cad.toFixed(2)} CAD credited`, {
        description: recipient ? `Granted to ${recipient}. No payment was taken.` : "No payment was taken.",
      });
      setIdempotencyKey(newIdempotencyKey());
      window.dispatchEvent(new CustomEvent("xcelsior-wallet-changed"));
      onCredited(res.balance_cad, res.amount_cad);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Couldn't credit the wallet");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-xl border border-accent-gold/30 bg-accent-gold/[0.07] px-3.5 py-3">
        <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-accent-gold" />
        <p className="text-xs leading-relaxed text-text-secondary">
          Platform admin only. Adds credit without taking a payment, works in test and production, and is recorded as an
          admin credit with your name and reason. It never counts as revenue.
        </p>
      </div>

      <div>
        <Label className="mb-2 block text-xs text-text-secondary">Amount (CAD)</Label>
        <div className="mb-2 grid grid-cols-4 gap-2">
          {PRESETS.map((p) => (
            <button
              key={p}
              type="button"
              disabled={submitting}
              onClick={() => { setAmount(String(p)); setIdempotencyKey(newIdempotencyKey()); }}
              className={`rounded-lg border px-3 py-2 font-mono text-sm transition-colors ${
                amount === String(p)
                  ? "border-accent-gold bg-accent-gold/10 text-accent-gold"
                  : "border-border bg-background text-text-secondary hover:border-text-muted"
              }`}
            >
              ${p}
            </button>
          ))}
        </div>
        <div className="relative">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 font-mono text-sm text-text-muted">$</span>
          <Input
            aria-label="Credit amount in CAD"
            type="text"
            inputMode="decimal"
            value={amount}
            disabled={submitting}
            onChange={(e) => { setAmount(e.target.value); setIdempotencyKey(newIdempotencyKey()); }}
            className="pl-8 font-mono"
          />
        </div>
        {amount && !amountValid && (
          <p className="mt-1 text-xs text-accent-red">Enter an amount above $0 and up to ${MAX_CAD.toLocaleString("en-CA")}.</p>
        )}
      </div>

      <div>
        <Label htmlFor="admin-credit-reason" className="mb-1.5 block text-xs text-text-secondary">
          Reason <span className="text-text-muted">(shown in the customer&apos;s history)</span>
        </Label>
        <Input
          id="admin-credit-reason"
          value={reason}
          maxLength={200}
          placeholder="e.g. End-to-end launch test, goodwill for outage"
          disabled={submitting}
          onChange={(e) => { setReason(e.target.value); setIdempotencyKey(newIdempotencyKey()); }}
          onKeyDown={(e) => {
            if (e.key === "Enter") void submit();
          }}
        />
      </div>

      <div className="flex gap-3 pt-1">
        <Button variant="outline" className="flex-1" onClick={onCancel} disabled={submitting}>
          Cancel
        </Button>
        <Button className="flex-1" onClick={() => void submit()} disabled={!amountValid || !reasonValid || submitting}>
          {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
          Credit ${amountValid ? numeric.toFixed(2) : "0.00"}
        </Button>
      </div>
    </div>
  );
}

export function AdminCreditDialog({
  open,
  customerId,
  recipient,
  onClose,
  onCredited,
}: {
  open: boolean;
  customerId: string;
  recipient: string;
  onClose: () => void;
  onCredited: (balanceCad: number, amountCad: number) => void;
}) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Add credits"
      description={`Credit ${recipient}'s wallet without a payment.`}
      maxWidth="max-w-md"
    >
      <div className="pt-4">
        {open && (
          <AdminCreditForm
            customerId={customerId}
            recipient={recipient}
            onCancel={onClose}
            onCredited={(balance, amount) => {
              onCredited(balance, amount);
              onClose();
            }}
          />
        )}
      </div>
    </Dialog>
  );
}
