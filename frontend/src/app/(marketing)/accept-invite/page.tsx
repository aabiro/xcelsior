"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { CheckCircle, Loader2, Users, XCircle } from "lucide-react";
import { SiteAuthAlert, SiteAuthCard, SiteAuthHeader } from "@/components/marketing/SiteAuthShell";
import { useAuth } from "@/lib/auth";
import { apiFetch } from "@/lib/api";

function AcceptInviteContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { user, loading: authLoading, refreshUser } = useAuth();
  const token = searchParams.get("token") ?? "";

  const [status, setStatus] = useState<"loading" | "pending" | "accepted" | "error">(
    () => (token ? "loading" : "error"),
  );
  const [teamName, setTeamName] = useState("");
  const [inviteEmail, setInviteEmail] = useState("");
  const [role, setRole] = useState("member");
  const [errorMsg, setErrorMsg] = useState(() => (token ? "" : "Invalid invite link."));

  useEffect(() => {
    if (!token) return;
    apiFetch<{ ok: boolean; pending?: boolean; accepted?: boolean; team_name: string; email: string; role: string; token?: string }>(
      `/api/teams/invite/${encodeURIComponent(token)}`,
    )
      .then((res) => {
        setTeamName(res.team_name);
        setInviteEmail(res.email);
        setRole(res.role);
        if (res.accepted) {
          setStatus("accepted");
        } else {
          setStatus("pending");
        }
      })
      .catch((err) => {
        setStatus("error");
        setErrorMsg(err?.message || "This invitation is invalid or has expired.");
      });
  }, [token]);

  async function handleAccept() {
    setStatus("loading");
    try {
      await apiFetch(`/api/teams/invite/${encodeURIComponent(token)}/accept`, { method: "POST" });
      await refreshUser();
      setStatus("accepted");
    } catch (err: unknown) {
      setStatus("error");
      setErrorMsg((err as Error)?.message || "Failed to accept invitation.");
    }
  }

  if (authLoading || status === "loading") {
    return (
      <div className="site-auth-loading">
        <Loader2 className="site-auth-spinner h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (status === "error") {
    return (
      <SiteAuthCard>
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader title="Invitation Not Found" subtitle={errorMsg} />
          <div className="site-auth-status-icon" data-tone="error" style={{ margin: "0 auto" }}>
            <XCircle className="h-7 w-7" />
          </div>
          <Link href="/dashboard" className="site-button site-button-primary site-auth-button">
            Go to Dashboard
          </Link>
        </div>
      </SiteAuthCard>
    );
  }

  if (status === "accepted") {
    return (
      <SiteAuthCard>
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader
            title="You're In!"
            subtitle={
              <>
                You&apos;ve joined <span className="site-auth-emphasis">{teamName}</span> as a {role}.
              </>
            }
          />
          <div className="site-auth-status-icon" data-tone="success" style={{ margin: "0 auto" }}>
            <CheckCircle className="h-7 w-7" />
          </div>
          <Link href="/dashboard/settings#team" className="site-button site-button-primary site-auth-button">
            View Team
          </Link>
        </div>
      </SiteAuthCard>
    );
  }

  return (
    <SiteAuthCard>
      <SiteAuthHeader
        title="Team Invitation"
        subtitle={
          <>
            You&apos;ve been invited to join <span className="site-auth-emphasis">{teamName}</span>
          </>
        }
      />
      <div className="site-auth-section site-auth-stack">
        <div className="site-auth-status-icon" data-tone="info" style={{ margin: "0 auto" }}>
          <Users className="h-7 w-7" />
        </div>
        <div className="site-auth-alert" data-tone="info">
          Role: <span className="site-auth-emphasis" style={{ textTransform: "capitalize" }}>{role}</span>
        </div>

        {user ? (
          user.email.toLowerCase() === inviteEmail.toLowerCase() ? (
            <button type="button" className="site-button site-button-primary site-auth-button" onClick={handleAccept}>
              Accept Invitation
            </button>
          ) : (
            <div className="site-auth-stack">
              <SiteAuthAlert tone="warn">
                This invitation is for <strong>{inviteEmail}</strong>. You&apos;re signed in as <strong>{user.email}</strong>.
              </SiteAuthAlert>
              <Link href={`/login?redirect=/accept-invite?token=${encodeURIComponent(token)}`} className="site-button site-button-ghost site-auth-button">
                Sign in with the correct account
              </Link>
            </div>
          )
        ) : (
          <div className="site-auth-actions">
            <Link
              href={`/register?email=${encodeURIComponent(inviteEmail)}&invite=${encodeURIComponent(token)}`}
              className="site-button site-button-primary site-auth-button"
            >
              Create Account &amp; Join
            </Link>
            <Link
              href={`/login?redirect=/accept-invite?token=${encodeURIComponent(token)}`}
              className="site-button site-button-ghost site-auth-button"
            >
              Sign In to Accept
            </Link>
          </div>
        )}
      </div>
    </SiteAuthCard>
  );
}

export default function AcceptInvitePage() {
  return (
    <Suspense>
      <AcceptInviteContent />
    </Suspense>
  );
}
