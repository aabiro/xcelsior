"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CheckCircle, Loader2, Users, XCircle } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { apiFetch } from "@/lib/api";

function AcceptInviteContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { user, loading: authLoading } = useAuth();
  const token = searchParams.get("token") ?? "";

  const [status, setStatus] = useState<"loading" | "pending" | "accepted" | "error">("loading");
  const [teamName, setTeamName] = useState("");
  const [inviteEmail, setInviteEmail] = useState("");
  const [role, setRole] = useState("member");
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    if (!token) {
      setStatus("error");
      setErrorMsg("Invalid invite link.");
      return;
    }
    // Peek at the invite
    apiFetch<{ ok: boolean; pending?: boolean; accepted?: boolean; team_name: string; email: string; role: string; token?: string }>(
      `/api/teams/invite/${encodeURIComponent(token)}`
    ).then((res) => {
      setTeamName(res.team_name);
      setInviteEmail(res.email);
      setRole(res.role);
      if (res.accepted) {
        setStatus("accepted");
      } else {
        setStatus("pending");
      }
    }).catch((err) => {
      setStatus("error");
      setErrorMsg(err?.message || "This invitation is invalid or has expired.");
    });
  }, [token]);

  async function handleAccept() {
    setStatus("loading");
    try {
      await apiFetch(`/api/teams/invite/${encodeURIComponent(token)}/accept`, { method: "POST" });
      setStatus("accepted");
    } catch (err: unknown) {
      setStatus("error");
      setErrorMsg((err as Error)?.message || "Failed to accept invitation.");
    }
  }

  if (authLoading || status === "loading") {
    return (
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-accent-cyan" />
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4">
        <Card className="w-full max-w-md p-8 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-accent-red/10">
            <XCircle className="h-8 w-8 text-accent-red" />
          </div>
          <h1 className="text-2xl font-bold mb-2">Invitation Not Found</h1>
          <p className="text-text-secondary mb-6">{errorMsg}</p>
          <Link href="/dashboard">
            <Button className="w-full">Go to Dashboard</Button>
          </Link>
        </Card>
      </div>
    );
  }

  if (status === "accepted") {
    return (
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4">
        <Card className="w-full max-w-md p-8 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-emerald/10">
            <CheckCircle className="h-8 w-8 text-emerald" />
          </div>
          <h1 className="text-2xl font-bold mb-2">You're In!</h1>
          <p className="text-text-secondary mb-6">
            You've joined <strong className="text-text-primary">{teamName}</strong> as a {role}.
          </p>
          <Link href="/dashboard/settings?tab=team">
            <Button className="w-full">View Team</Button>
          </Link>
        </Card>
      </div>
    );
  }

  // status === "pending"
  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4">
      <Card className="w-full max-w-md p-8 text-center">
        <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-accent-cyan/10">
          <Users className="h-8 w-8 text-accent-cyan" />
        </div>
        <div className="flex items-center justify-center gap-2 mb-4">
          <Image src="/xcelsior-logo-rounded.svg" alt="Xcelsior" width={32} height={32} className="rounded-lg" />
          <span className="text-lg font-bold">Xcelsior</span>
        </div>
        <h1 className="text-2xl font-bold mb-2">Team Invitation</h1>
        <p className="text-text-secondary mb-1">
          You've been invited to join
        </p>
        <p className="text-lg font-semibold text-text-primary mb-4">{teamName}</p>
        <p className="text-sm text-text-muted mb-6">
          Role: <span className="capitalize font-medium text-text-secondary">{role}</span>
        </p>

        {user ? (
          user.email.toLowerCase() === inviteEmail.toLowerCase() ? (
            <Button className="w-full bg-accent-cyan text-white hover:bg-accent-cyan/90" onClick={handleAccept}>
              Accept Invitation
            </Button>
          ) : (
            <div className="space-y-3">
              <p className="text-sm text-accent-gold">
                This invitation is for <strong>{inviteEmail}</strong>.
                You're signed in as <strong>{user.email}</strong>.
              </p>
              <Link href={`/login?redirect=/accept-invite?token=${encodeURIComponent(token)}`}>
                <Button variant="outline" className="w-full">Sign in with the correct account</Button>
              </Link>
            </div>
          )
        ) : (
          <div className="space-y-3">
            <Link href={`/register?email=${encodeURIComponent(inviteEmail)}&invite=${encodeURIComponent(token)}`}>
              <Button className="w-full bg-accent-cyan text-white hover:bg-accent-cyan/90">
                Create Account &amp; Join
              </Button>
            </Link>
            <Link href={`/login?redirect=/accept-invite?token=${encodeURIComponent(token)}`}>
              <Button variant="outline" className="w-full">
                Sign In to Accept
              </Button>
            </Link>
          </div>
        )}
      </Card>
    </div>
  );
}

export default function AcceptInvitePage() {
  return (
    <Suspense>
      <AcceptInviteContent />
    </Suspense>
  );
}
