"use client";

import { Suspense, useState, type FormEvent } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { CheckCircle, Loader2, ShieldCheck, Terminal, XCircle } from "lucide-react";
import { SiteAuthAlert, SiteAuthCard, SiteAuthHeader } from "@/components/marketing/SiteAuthShell";
import { ApiError, approveDeviceCode } from "@/lib/api";
import { useAuth } from "@/lib/auth";

/**
 * Where the CLI wizard sends people to approve its sign-in. The browser used
 * to land on a developer placeholder at /api/auth/verify that said to submit
 * the code with a POST request, so nobody could finish signing in from a
 * browser. It never approves on its own, even with the code in the link: the
 * person compares it with their terminal and presses Approve (RFC 8628 §5.4).
 */

/** Device codes look like 3F9A-0B1C: eight letters and digits, a dash after four. */
function formatCode(raw: string): string {
  const clean = raw.toUpperCase().replace(/[^0-9A-Z]/g, "").slice(0, 8);
  return clean.length > 4 ? `${clean.slice(0, 4)}-${clean.slice(4)}` : clean;
}

const isComplete = (code: string) => /^[0-9A-Z]{4}-[0-9A-Z]{4}$/.test(code);

function approvalError(err: unknown): string {
  if (err instanceof ApiError && err.status === 404) {
    return "That code has expired or doesn't exist. Check it against your terminal, or start sign-in again in the wizard.";
  }
  if (err instanceof ApiError && err.status === 400) {
    return "This code has already been used. If your terminal is still waiting, start sign-in again there.";
  }
  return err instanceof Error && err.message ? err.message : "Couldn't approve this code. Try again.";
}

function DeviceCodePanel({ code }: { code: string }) {
  return (
    <div className="site-auth-device-code">
      <span className="site-auth-device-code-value" aria-label={`Code ${code.split("").join(" ")}`}>
        {code}
      </span>
      <span className="site-auth-device-code-hint">Only approve if this matches the code in your terminal.</span>
    </div>
  );
}

function DeviceContent() {
  const searchParams = useSearchParams();
  const { user, loading } = useAuth();
  const fromLink = formatCode(searchParams.get("user_code") ?? "");
  const [code, setCode] = useState(isComplete(fromLink) ? fromLink : "");
  const [draft, setDraft] = useState(fromLink);
  const [status, setStatus] = useState<"idle" | "approving" | "approved" | "declined">("idle");
  const [error, setError] = useState("");

  const here = code ? `/device?user_code=${encodeURIComponent(code)}` : "/device";

  async function approve() {
    setStatus("approving");
    setError("");
    try {
      await approveDeviceCode(code);
      setStatus("approved");
    } catch (err) {
      setStatus("idle");
      setError(approvalError(err));
    }
  }

  function submitDraft(event: FormEvent) {
    event.preventDefault();
    if (!isComplete(draft)) return;
    setError("");
    setCode(draft);
  }

  if (loading) {
    return (
      <div className="site-auth-loading">
        <Loader2 className="site-auth-spinner h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (status === "approved") {
    return (
      <SiteAuthCard>
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader
            title="Your terminal is signed in"
            subtitle="Head back to your terminal; the wizard carries on by itself. You can close this tab."
          />
          <div className="site-auth-status-icon" data-tone="success" style={{ margin: "0 auto" }}>
            <CheckCircle className="h-7 w-7" />
          </div>
        </div>
      </SiteAuthCard>
    );
  }

  if (status === "declined") {
    return (
      <SiteAuthCard>
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader
            title="Nothing was approved"
            subtitle="Your account was not connected to anything. The code stops working on its own in a few minutes."
          />
          <div className="site-auth-status-icon" data-tone="info" style={{ margin: "0 auto" }}>
            <XCircle className="h-7 w-7" />
          </div>
          <Link href="/dashboard" className="site-button site-button-ghost site-auth-button">
            Go to dashboard
          </Link>
        </div>
      </SiteAuthCard>
    );
  }

  if (!code) {
    return (
      <SiteAuthCard>
        <SiteAuthHeader
          eyebrow="Xcelsior CLI"
          title="Connect your terminal"
          subtitle="Enter the code the Xcelsior setup wizard is showing in your terminal."
        />
        <form className="site-auth-section site-auth-form" onSubmit={submitDraft}>
          <label htmlFor="device-code" className="site-field-wrap site-auth-field-wrap">
            <span className="site-auth-label">Code</span>
            <input
              id="device-code"
              value={draft}
              onChange={(e) => setDraft(formatCode(e.target.value))}
              placeholder="XXXX-XXXX"
              className="site-field site-auth-field site-auth-otp"
              maxLength={9}
              autoComplete="one-time-code"
              autoCapitalize="characters"
              spellCheck={false}
              autoFocus
            />
          </label>
          {error ? <SiteAuthAlert>{error}</SiteAuthAlert> : null}
          <button type="submit" className="site-button site-button-primary site-auth-button" disabled={!isComplete(draft)}>
            Continue
          </button>
        </form>
      </SiteAuthCard>
    );
  }

  if (!user) {
    return (
      <SiteAuthCard>
        <SiteAuthHeader
          eyebrow="Xcelsior CLI"
          title="Sign in to connect your terminal"
          subtitle="The Xcelsior setup wizard is waiting for you to approve this code."
        />
        <div className="site-auth-section site-auth-stack">
          <DeviceCodePanel code={code} />
          <div className="site-auth-actions">
            <Link href={`/login?redirect=${encodeURIComponent(here)}`} className="site-button site-button-primary site-auth-button">
              Sign in
            </Link>
            <Link href={`/register?redirect=${encodeURIComponent(here)}`} className="site-button site-button-ghost site-auth-button">
              Create an account
            </Link>
          </div>
        </div>
      </SiteAuthCard>
    );
  }

  return (
    <SiteAuthCard>
      <SiteAuthHeader
        eyebrow="Xcelsior CLI"
        title="Approve this sign-in?"
        subtitle={
          <>
            A terminal is asking to sign in to Xcelsior as <span className="site-auth-emphasis">{user.email}</span>.
          </>
        }
      />
      <div className="site-auth-section site-auth-stack">
        <DeviceCodePanel code={code} />
        {error ? <SiteAuthAlert>{error}</SiteAuthAlert> : null}
        <div className="site-auth-actions">
          <button
            type="button"
            className="site-button site-button-primary site-auth-button"
            onClick={() => void approve()}
            disabled={status === "approving"}
          >
            {status === "approving" ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ShieldCheck className="mr-2 h-4 w-4" />}
            Approve
          </button>
          <button
            type="button"
            className="site-button site-button-ghost site-auth-button"
            onClick={() => setStatus("declined")}
            disabled={status === "approving"}
          >
            This wasn&apos;t me
          </button>
        </div>
        <p className="site-auth-note" style={{ textAlign: "center" }}>
          <Terminal className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />
          Different code in your terminal?{" "}
          <button
            type="button"
            className="site-auth-text-button"
            onClick={() => {
              setDraft("");
              setError("");
              setCode("");
            }}
          >
            Enter it instead
          </button>
        </p>
      </div>
    </SiteAuthCard>
  );
}

export default function DevicePage() {
  return (
    <Suspense fallback={<div className="site-auth-loading"><Loader2 className="site-auth-spinner h-8 w-8 animate-spin" /></div>}>
      <DeviceContent />
    </Suspense>
  );
}
