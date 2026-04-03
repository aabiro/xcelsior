"use client";

import { useEffect, useState } from "react";
import {
  useAppKit,
  useAppKitAccount,
  useAppKitProvider,
  useWalletInfo,
} from "@reown/appkit/react";
import type { BitcoinConnector } from "@reown/appkit-utils/bitcoin";
import { CheckCircle2, Link2, Loader2, Wallet } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { isWalletConnectConfigured } from "@/lib/wallet-connect";

interface BitcoinWalletConnectActionProps {
  amountBtc: number;
  recipient: string;
  disabled?: boolean;
}

function formatShortValue(value?: string) {
  if (!value || value.length <= 16) {
    return value ?? "";
  }

  return `${value.slice(0, 8)}...${value.slice(-6)}`;
}

function btcToSatsString(amountBtc: number) {
  const [whole = "0", fraction = ""] = amountBtc.toFixed(8).split(".");
  const sats = `${whole}${fraction.padEnd(8, "0")}`.replace(/^0+(?=\d)/, "");
  return sats || "0";
}

export function BitcoinWalletConnectAction({
  amountBtc,
  recipient,
  disabled = false,
}: BitcoinWalletConnectActionProps) {
  if (!isWalletConnectConfigured) {
    return null;
  }

  const { open } = useAppKit();
  const { address, isConnected } = useAppKitAccount({ namespace: "bip122" });
  const { walletInfo } = useWalletInfo("bip122");
  const { walletProvider } = useAppKitProvider<BitcoinConnector>("bip122");
  const [submitting, setSubmitting] = useState(false);
  const [submittedTxId, setSubmittedTxId] = useState("");

  useEffect(() => {
    setSubmittedTxId("");
  }, [amountBtc, recipient]);

  const handleClick = async () => {
    if (disabled || submitting || submittedTxId) {
      return;
    }

    try {
      if (!isConnected || !walletProvider) {
        await open({ view: "Connect" });
        return;
      }

      setSubmitting(true);
      const txId = await walletProvider.sendTransfer({
        amount: btcToSatsString(amountBtc),
        recipient,
      });
      setSubmittedTxId(txId);
      toast.success("Bitcoin transaction submitted");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "WalletConnect payment failed";
      toast.error(message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="mb-4 rounded-xl border border-emerald/20 bg-emerald/5 p-4">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-9 w-9 items-center justify-center rounded-xl bg-emerald/10 text-emerald">
            {submittedTxId ? (
              <CheckCircle2 className="h-4 w-4" />
            ) : isConnected ? (
              <Wallet className="h-4 w-4" />
            ) : (
              <Link2 className="h-4 w-4" />
            )}
          </div>
          <div>
            <p className="text-sm font-semibold text-text-primary">
              WalletConnect
            </p>
            <p className="text-xs text-text-muted">
              {submittedTxId
                ? `Transaction ${formatShortValue(submittedTxId)} is on the way.`
                : disabled
                  ? "Refresh the quote before sending from a connected wallet."
                  : isConnected
                    ? `Connected ${walletInfo?.name ? `to ${walletInfo.name}` : "wallet"}${address ? ` | ${formatShortValue(address)}` : ""}`
                    : "Connect a compatible Bitcoin wallet and send from checkout."}
            </p>
          </div>
        </div>
        {isConnected && !submittedTxId ? (
          <span className="rounded-full border border-emerald/20 bg-emerald/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-emerald">
            Connected
          </span>
        ) : null}
      </div>

      <Button
        variant="success"
        className="h-11 w-full rounded-xl"
        onClick={handleClick}
        disabled={disabled || submitting || Boolean(submittedTxId)}
      >
        {submitting ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>
              {isConnected ? "Awaiting wallet approval..." : "Opening WalletConnect..."}
            </span>
          </>
        ) : submittedTxId ? (
          <>
            <CheckCircle2 className="h-4 w-4" />
            <span>Transaction Submitted</span>
          </>
        ) : isConnected ? (
          <>
            <Wallet className="h-4 w-4" />
            <span>Pay with {walletInfo?.name ?? "Connected Wallet"}</span>
          </>
        ) : (
          <>
            <Link2 className="h-4 w-4" />
            <span>Connect with WalletConnect</span>
          </>
        )}
      </Button>
    </div>
  );
}
