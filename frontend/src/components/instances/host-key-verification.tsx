"use client";

/**
 * The host-key fingerprint, shown to the human exactly as the agent sees it.
 *
 * Gate P2: *"Connection details in the instance view that match exactly what
 * the tool returns — same host, same fingerprint, same expiry. A human and an
 * agent looking at the same instance must see the same truth."*
 *
 * They did not. `routes/instances.py` `_enrich_instance` has served
 * `host_key_fingerprint` since the host-key rollout, `open_instance_access`
 * returns it to the model with a verification command beside it — and the
 * instance page dropped it at the type boundary. So an agent connecting to an
 * instance could verify the host key and a person looking at the same instance
 * could not. The person got `ssh root@host -p port` and nothing to check it
 * against, which leaves accepting the key unread as the only way forward.
 *
 * ## Why the command is built here and not sent from the server
 *
 * It is built from the host and port **this page is already displaying**. A
 * verification command fetched separately could name a different endpoint than
 * the `ssh` command directly above it, and the user would verify one host while
 * connecting to another — a check that returns "verified" without having
 * verified the thing being trusted.
 *
 * `tests/test_host_key_verification_parity.py` pins the construction against
 * `mcp/src/tools/compute.ts` so the two cannot drift.
 *
 * ## Why the null state renders instead of hiding
 *
 * A missing fingerprint is permanent for whole classes of instance — the tool's
 * own comment lists them: *"older instances, non-interactive launches,
 * proxy-terminated hosts where the container legitimately holds no keys, and
 * agents that have not yet reported"*. Hiding the row for those leaves a user
 * with no signal at all, which reads as "nothing to check here". Saying plainly
 * that it cannot be verified is the honest state, and it is the same thing the
 * tool instructs the model to say.
 *
 * What neither surface does is tell the user to accept the key anyway.
 */

import { useState } from "react";
import { Copy, ShieldCheck, ShieldAlert } from "lucide-react";
import { toast } from "sonner";

export interface HostKeyVerificationProps {
  /** The gateway host shown in the `ssh` command beside this. */
  host: string;
  /** The gateway port shown in the `ssh` command beside this. */
  port: number;
  /**
   * The API's value, verbatim. Never derived and never defaulted — a
   * fingerprint this component invented would be worse than none, because
   * `null` makes a user say "I cannot verify this" and a wrong value makes them
   * say "verified".
   */
  fingerprint?: string | null;
}

/**
 * The check a user runs to confirm the host they are about to trust.
 *
 * Character-for-character the command `open_instance_access` gives the agent.
 * `2>/dev/null` drops ssh-keyscan's progress chatter on stderr, which otherwise
 * interleaves with the fingerprint and makes the comparison harder to read.
 */
export function hostKeyVerifyCommand(host: string, port: number): string {
  return `ssh-keyscan -p ${port} ${host} 2>/dev/null | ssh-keygen -lf -`;
}

export function HostKeyVerification({ host, port, fingerprint }: HostKeyVerificationProps) {
  const [copied, setCopied] = useState<"fp" | "cmd" | null>(null);

  const copy = (value: string, which: "fp" | "cmd", label: string) => {
    navigator.clipboard.writeText(value);
    setCopied(which);
    toast.success(label);
    window.setTimeout(() => setCopied((c) => (c === which ? null : c)), 1500);
  };

  if (!fingerprint) {
    return (
      <div
        data-testid="host-key-unverifiable"
        className="rounded-md border border-accent-gold/25 bg-accent-gold/[0.06] px-2.5 py-2"
      >
        <div className="flex items-start gap-2">
          <ShieldAlert className="h-3.5 w-3.5 text-accent-gold shrink-0 mt-0.5" />
          <div className="min-w-0">
            <p className="text-xs font-medium text-accent-gold">Host key not published</p>
            <p className="mt-0.5 text-[11px] leading-relaxed text-text-muted">
              Xcelsior has not observed a host key for this endpoint, so this first
              connection cannot be checked against one. SSH will show you a fingerprint
              and ask whether to trust it — there is nothing here to compare it to.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const command = hostKeyVerifyCommand(host, port);

  return (
    <div
      data-testid="host-key-verification"
      className="rounded-md border border-emerald/25 bg-emerald/[0.06] px-2.5 py-2 space-y-1.5"
    >
      <div className="flex items-center gap-2">
        <ShieldCheck className="h-3.5 w-3.5 text-emerald shrink-0" />
        <span className="text-xs font-medium text-emerald">Verify this host before you trust it</span>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-[11px] text-text-muted shrink-0 w-[72px]">Fingerprint</span>
        <code
          data-testid="host-key-fingerprint"
          className="flex-1 min-w-0 truncate text-[11px] font-mono text-text-secondary bg-background rounded px-2 py-1 select-all border border-border"
        >
          {fingerprint}
        </code>
        <button
          onClick={() => copy(fingerprint, "fp", "Copied fingerprint")}
          className="text-text-muted hover:text-emerald transition-colors shrink-0"
          title="Copy fingerprint"
          aria-label="Copy fingerprint"
        >
          <Copy className={`h-3 w-3 ${copied === "fp" ? "text-emerald" : ""}`} />
        </button>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-[11px] text-text-muted shrink-0 w-[72px]">Check with</span>
        <code
          data-testid="host-key-verify-command"
          className="flex-1 min-w-0 truncate text-[11px] font-mono text-text-secondary bg-background rounded px-2 py-1 select-all border border-border"
        >
          {command}
        </code>
        <button
          onClick={() => copy(command, "cmd", "Copied verification command")}
          className="text-text-muted hover:text-emerald transition-colors shrink-0"
          title="Copy verification command"
          aria-label="Copy verification command"
        >
          <Copy className={`h-3 w-3 ${copied === "cmd" ? "text-emerald" : ""}`} />
        </button>
      </div>

      <p className="text-[11px] leading-relaxed text-text-muted">
        The SHA256 value it prints must match the fingerprint above. If it does not,
        do not connect — report it.
      </p>
    </div>
  );
}
